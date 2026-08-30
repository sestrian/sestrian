"""Page fraud proofs — the dispute game (Sharding Road, Phase 1).

A `PageFraudProof` convicts a block of committing a WRONG post-state for one
page, verifiable from first principles by a node that holds NOTHING but the
two block headers. This is the mechanism that lets a future paged validator
trust pages it does not hold: any single honest holder can convict a bad
aggregation with one bounded proof.

The proof is a plain dict (JSON-friendly — it is golden-vectored and travels
over gossip):

    header          accused block's header fields
    parent_header   its parent's header fields
    parent_model_json  the parent ModelState's CANONICAL json — bound by
                    sha256(json) == parent_header.model_root, so page spans
                    and dim need no trusted context
    committed_leaves   hex page-leaf hashes the accused block committed —
                    bound by fold(leaves) == header.state_root (the accused
                    root itself authenticates them; post-v5 blocks carry them
                    in the body, but the verifier never cares where they came
                    from)
    page_id         the disputed page
    txids           ALL txids in the block (recomputes header.txset_root, so
                    the claimant set is provably complete)
    txs             all tx field-dicts (small; sigs verified)
    bodies          txid -> sparse body {n, idx: [u32], val: [i64]} for every
                    tx CLAIMING the page — delta_hash binds each to its tx
                    (the hash is over the dense byte image; sparse is
                    transport)
    parent_page     the parent state's page values (ints)
    parent_path     merkle inclusion proof of parent_page in
                    parent_header.state_root: [(side, hex)]

verify() returns (fraud_proven, reason). Reasons distinguish an INVALID proof
(never act on it) from a VALID one (the block is provably fraudulent).
Scope v1: aggregation fraud on existing pages of non-growth blocks; growth
and fold disputes are checkable from the (tiny) ModelState alone and get
their own game later.
"""

import hashlib
import json

import numpy as np

from rig import merkle
from rig.blockchain import Header
from rig.chain import trimmed_mean_int
from rig.crypto import BackpropTx, delta_hash


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _dense(body: dict, dim: int) -> np.ndarray | None:
    idx = body.get("idx", [])
    val = body.get("val", [])
    if body.get("n") != dim or len(idx) != len(val):
        return None
    out = np.zeros(dim, dtype=np.int64)
    last = -1
    for i, v in zip(idx, val):
        if not (0 <= int(i) < dim) or int(i) <= last:
            return None  # coords must be sorted unique in range
        last = int(i)
        out[int(i)] = int(v)
    return out


def build(header: Header, parent_header: Header, parent_model_json: str,
          committed_leaves: list[str], page_id: int, txs: list[BackpropTx],
          bodies_dense: dict[str, np.ndarray],
          parent_state: np.ndarray) -> dict:
    """Assemble a proof from a prover's local data (a full validator that just
    rejected the block). Sparse-encodes only the claimant bodies."""
    span_pages = json.loads(parent_model_json)["pages"]
    s, e = span_pages[page_id][0], span_pages[page_id][1]
    dim = span_pages[-1][1]
    proof_bodies = {}
    for t in txs:
        if page_id in t.canonical_pages():
            d = bodies_dense[t.txid()]
            nz = np.nonzero(d)[0]
            proof_bodies[t.txid()] = {
                "n": dim,
                "idx": [int(i) for i in nz],
                "val": [int(d[i]) for i in nz],
            }
    parent_leaves = [np.asarray(parent_state[p[0]:p[1]], dtype=np.int64)
                     for p in span_pages]
    levels = merkle.build([pl.tobytes() for pl in parent_leaves])
    return {
        "header": dict(header.__dict__),
        "parent_header": dict(parent_header.__dict__),
        "parent_model_json": parent_model_json,
        "committed_leaves": committed_leaves,
        "page_id": page_id,
        "txids": sorted(t.txid() for t in txs),
        "txs": [{"miner": t.miner, "base_height": t.base_height,
                 "delta_hash": t.delta_hash, "da_pointer": t.da_pointer,
                 "bond": t.bond, "pages": t.canonical_pages(),
                 "data_refs": t.canonical_refs(), "sig_hex": t.sig.hex()}
                for t in txs],
        "bodies": proof_bodies,
        "parent_page": [int(x) for x in parent_state[s:e]],
        "parent_path": [(side, sib.hex())
                        for side, sib in merkle.proof(levels, page_id)],
    }


def verify(proof: dict) -> tuple[bool, str]:
    """First-principles verification. (True, ...) == the block is fraudulent."""
    try:
        h = Header(**proof["header"])
        ph = Header(**proof["parent_header"])
    except TypeError:
        return False, "invalid: malformed headers"
    if h.prev_hash != ph.block_hash():
        return False, "invalid: header does not extend parent"
    # parent ModelState binds spans + dim through model_root
    pmj = proof["parent_model_json"]
    if _sha(pmj.encode()) != ph.model_root:
        return False, "invalid: parent model json does not match model_root"
    pages = json.loads(pmj)["pages"]
    pid = proof["page_id"]
    if not (0 <= pid < len(pages)):
        return False, "invalid: page id out of range"
    s, e = pages[pid][0], pages[pid][1]
    dim = pages[-1][1]
    # committed leaves bind through the accused state_root itself
    leaves = proof["committed_leaves"]
    if len(leaves) != len(pages):
        return False, "invalid: growth blocks are not disputable in fraud v1"
    level = [bytes.fromhex(x) for x in leaves]
    while len(level) > 1:
        level = [merkle._node_hash(level[i],
                                   level[i + 1] if i + 1 < len(level)
                                   else level[i])
                 for i in range(0, len(level), 2)]
    if level[0].hex() != h.state_root:
        return False, "invalid: committed leaves do not fold to state_root"
    # tx set completeness through txset_root
    txids = proof["txids"]
    if _sha("|".join(sorted(txids)).encode()) != h.txset_root:
        return False, "invalid: txids do not reproduce txset_root"
    if len(proof["txs"]) != len(txids) or len(txids) != h.n_txs:
        return False, "invalid: tx list incomplete"
    txs = []
    for t in proof["txs"]:
        tx = BackpropTx(miner=t["miner"], base_height=t["base_height"],
                        delta_hash=t["delta_hash"], da_pointer=t["da_pointer"],
                        bond=t["bond"], pages=list(t["pages"]),
                        data_refs=list(t["data_refs"]))
        tx.sig = bytes.fromhex(t["sig_hex"])
        if tx.txid() not in txids:
            return False, "invalid: tx not in committed set"
        if not tx.verify():
            return False, "invalid: bad tx signature"
        if tx.base_height != ph.height:
            return False, "invalid: tx base height mismatch"
        txs.append(tx)
    # claimant bodies bind through delta_hash (over the dense byte image)
    claim_cols = []
    for tx in txs:
        if pid not in tx.canonical_pages():
            continue
        body = proof["bodies"].get(tx.txid())
        if body is None:
            return False, "invalid: missing claimant body"
        d = _dense(body, dim)
        if d is None:
            return False, "invalid: malformed body"
        if delta_hash(d.tobytes()) != tx.delta_hash:
            return False, "invalid: body does not match delta_hash"
        claim_cols.append(d[s:e])
    # parent page binds through parent state_root
    parent_page = np.asarray(proof["parent_page"], dtype=np.int64)
    if len(parent_page) != e - s:
        return False, "invalid: parent page span mismatch"
    path = [(side, bytes.fromhex(sib)) for side, sib in proof["parent_path"]]
    if not merkle.verify(parent_page.tobytes(), pid, path,
                         bytes.fromhex(ph.state_root)):
        return False, "invalid: parent page not in parent state_root"
    # THE RECOMPUTATION — the exact consensus rule, one page wide
    if claim_cols:
        new_page = parent_page + trimmed_mean_int(claim_cols)
    else:
        new_page = parent_page
    recomputed = merkle.leaf_hash(new_page.astype(np.int64).tobytes())
    committed = bytes.fromhex(leaves[pid])
    if recomputed == committed:
        return False, "invalid: page aggregates correctly — no fraud"
    return True, (f"FRAUD: page {pid} committed leaf {committed.hex()[:12]}… "
                  f"but honest aggregation gives {recomputed.hex()[:12]}…")
