"""Bitcoin-style blocks: hash-linked headers, independent validation, fork choice.

The rig's `rig/chain.py` is a linear list an authority appends to. A real chain
must let any node, holding no special trust, (1) validate a block from first
principles and (2) choose between competing histories. This module adds those:

  * a **header** committing prev_hash, the weights-state root, the tx-set root,
    height, and cumulative work — hashed into the block id (Bitcoin's header);
  * **full validation** of a block against its parent state: every tx signature
    checks, the state transition reproduces the committed root, the tx-set root
    matches (§3.4, §5);
  * a **BlockTree** with **heaviest-valid-chain fork choice** — the same
    Nakamoto rule that lets Bitcoin nodes agree without a coordinator.

Model weights are the state, exactly as before; this layer is about *who gets to
say what the history is* — and the answer is now "the heaviest valid chain",
not "the coordinator".
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .chain import (dequantize, paged_transition, quantize, state_root,
                    trimmed_mean_int)
from .crypto import BackpropTx, verify
from .model_state import (GenesisParams, ModelState, fold as model_fold,
                          page_init, page_state_root)
from .token import (PROPOSER_LOOKBACK, TokenLedger, TransferTx, address,
                    canonical_account_txs, data_root as dta_root,
                    transfer_root as xfer_root)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── PROTOCOL VERSION (v1) ─────────────────────────────────────────────────
# The entire upgrade mechanism, deliberately minimal: the header commits a
# version, and validation requires it to equal the scheduled version for its
# height. A future upgrade appends (activation_height, version) here; nodes
# that don't know a version reject its blocks with "upgrade your node".
VERSION_SCHEDULE: tuple = ((0, 2),)   # protocol v2: the delta envelope


def expected_version(height: int) -> int:
    v = VERSION_SCHEDULE[0][1]
    for h0, ver in VERSION_SCHEDULE:
        if height >= h0:
            v = ver
    return v


def txset_root(txs: list[BackpropTx]) -> str:
    """Order-independent commitment to the accepted tx set (§3.2)."""
    return _sha(("|".join(sorted(t.txid() for t in txs))).encode())


@dataclass
class Header:
    height: int
    prev_hash: str
    state_root: str            # Merkle/hash of the weights AFTER this block
    txset_root: str
    n_txs: int
    work: int                  # fork-choice weight = vrf_work(vrf_proof), non-forgeable
    proposer: str              # pubkey of the block proposer
    # the TRANSFER LANE (protocol rev 2): the token ledger is consensus state
    transfer_root: str = ""    # order-independent commitment to the transfer set
    ledger_root: str = ""      # token-ledger root AFTER this block (rewards+transfers)
    # the DATA LANE (protocol rev 3): staked data registry + challenge market
    data_root: str = ""        # commitment to the block's data-lane tx set
    # the PROPOSER LOTTERY (rev 4): the proposer's VRF proof over the height seed.
    # header.work is derived from it (non-forgeable), replacing free-form work.
    vrf_proof: str = ""        # hex of the deterministic-Ed25519 VRF signature
    # DELTA SCORING (rev 7): commitment to the proposer's held-out-loss scores
    # for the block's deltas. Scores are BLOCK DATA (committed, not recomputed),
    # so consensus stays deterministic across GPUs; truthfulness is bonded and
    # challengeable (the commit-reveal committee is the testnet upgrade).
    score_root: str = ""
    # INFLUENCE SKETCHES (rev 8): commitment to each delta's quantized gradient
    # projection — the corpus-attribution primitive (§8). Same committed-inputs
    # trust tier as scores; recomputable from the DA delta body + public seed.
    sketch_root: str = ""
    # PROTOCOL v1: the ModelState commitment (page table + capacity fold) AFTER
    # this block. Recomputed AND committed, so a fold divergence is a loud
    # validation error instead of a silent fork.
    model_root: str = ""
    # PROTOCOL v1: the proposer's sortition attempt (0..ATTEMPT_MAX). The seed
    # binds to it; header.work = attempt_work(proof, attempt) — low attempts
    # strictly tend to dominate fork choice.
    vrf_attempt: int = 0
    # PROTOCOL v1: header version, checked against the VERSION_SCHEDULE.
    version: int = 1

    def block_hash(self) -> str:
        return _sha(json.dumps(self.__dict__, sort_keys=True).encode())


# rev 7: a delta's score = its held-out loss improvement in micro-nats, >= 0,
# clamped by consensus so a lying proposer can't mint unbounded weight.
SCORE_CAP = 10**9

# rev 8: each delta carries its INFLUENCE SKETCH — the gradient projected
# through the shared seeded ±1/√d matrix (attribution.py Projector), quantized.
# A corpus's accumulated sketch is then pure ledger arithmetic (Σ sketches of
# the deltas that named it), recomputable from DA delta bodies + the public
# seed — so usage royalties are independently checkable and challengeable.
SKETCH_DIM = 256
SKETCH_SEED = 1234                 # the published projection seed (Projector)
SKETCH_SCALE = 10_000              # fixed-point quantization of sketch entries
I32 = 2**31                        # per-entry bound for a committed delta sketch
I64_MAX = 2**63 - 1                # saturating bound for ledger accumulators


def scores_root(scores: dict) -> str:
    """Canonical commitment to {txid: score}: sorted compact JSON, hashed."""
    blob = json.dumps({k: int(v) for k, v in scores.items()},
                      sort_keys=True, separators=(",", ":")).encode()
    return _sha(blob)


def sketch_root(sketches: dict) -> str:
    """Canonical commitment to {txid: [int; SKETCH_DIM]}: sorted compact JSON."""
    blob = json.dumps({k: [int(x) for x in v] for k, v in sketches.items()},
                      sort_keys=True, separators=(",", ":")).encode()
    return _sha(blob)


def _sat64(x: int) -> int:
    """Saturate to i64 — accumulators must stay in the Rust mirror's range."""
    return max(-I64_MAX - 1, min(I64_MAX, x))


def effective_scores(txs: list, scores: dict) -> dict:
    """Consensus scores used for reward weighting: the committed score per txid,
    with a UNIFORM fallback (all 1) when every score is zero — an unscored block
    (bootstrap, eval timeout) still splits rewards equally rather than burning
    them. Deterministic from block content only."""
    eff = {t.txid(): int(scores.get(t.txid(), 0)) for t in txs}
    if eff and sum(eff.values()) == 0:
        eff = {k: 1 for k in eff}
    return eff


@dataclass
class Block:
    header: Header
    txs: list                  # list[BackpropTx]
    bodies: dict               # da_pointer -> int64 delta array (carried for replay)
    transfers: list = field(default_factory=list)   # list[TransferTx]
    data_txs: list = field(default_factory=list)    # rev 3: Data{Submit,Challenge,Vote}Tx
    scores: dict = field(default_factory=dict)      # rev 7: txid -> micro-nat score
    sketches: dict = field(default_factory=dict)    # rev 8: txid -> [int; SKETCH_DIM]

    @property
    def hash(self) -> str:
        return self.header.block_hash()


class ValidationError(Exception):
    pass


def apply_ledger(parent_ledger: TokenLedger, block: Block,
                 data_contributor: str | None,
                 recent_proposers: set[str] = frozenset()) -> TokenLedger:
    """The deterministic token-state transition for one block, in this exact
    order (the Rust node mirrors it):
      1. settle every challenge whose window closed at/before this height;
      2. mint the block reward (miners/proposer; data share across the weighted
         registry);
      3. apply ALL account txs (data lane + transfers) in one merged canonical
         sequence — (sender address, nonce, txid) — so each wallet's nonce
         sequence totally orders everything it does.
    Raises if any tx is invalid — a block carrying one is invalid."""
    led = parent_ledger.copy()
    h = block.header.height
    led.resolve_expired_challenges(h)
    led.resolve_expired_bonds(h)        # return matured delta bonds first
    # PROVENANCE (rev 5): every delta must name data that is staked + active in the
    # registry (the on-chain availability proxy). A delta naming no active corpus
    # is rejected — the model is only ever trained on auditable data. The genesis
    # data entry (data_hash "genesis") is always active, so bootstrap deltas can
    # name it until real corpora are staked.
    active_hashes = {e["data_hash"] for e in led.registry.values()
                     if e["status"] == "active"}
    for tx in block.txs:
        refs = tx.canonical_refs()
        if not any(r in active_hashes for r in refs):
            raise ValidationError(
                f"delta {tx.txid()[:8]} names no staked/available data "
                f"(provenance required)")
    # DELTA SCORING (rev 7): rewards are weighted by each delta's committed
    # held-out-loss score. Miners: pool split ∝ their deltas' scores. Data: each
    # delta's score splits equally across its named active corpora (scaled by
    # 10_000 so integer division doesn't vanish small scores). All-zero scores
    # fall back to uniform — deterministic from block content alone.
    eff = effective_scores(block.txs, block.scores)
    miner_weights: dict[str, int] = {}
    data_credits: dict[str, int] = {}
    active_set = {e["data_hash"] for e in led.registry.values()
                  if e["status"] == "active" and e["weight"] > 0}
    hash_to_entry = {e["data_hash"]: e for e in led.registry.values()
                     if e["status"] == "active"}
    for tx in block.txs:
        s = eff[tx.txid()]
        miner_weights[tx.miner] = miner_weights.get(tx.miner, 0) + s
        named = [r for r in tx.canonical_refs() if r in active_set]
        for r in named:
            data_credits[r] = data_credits.get(r, 0) + s * 10_000 // len(named)
        # rev 8: accrue this delta's committed influence sketch onto the corpora
        # it named — a corpus's ledger sketch = Σ (its deltas' sketches), the
        # projection of its total contribution to the weights. Saturating i64
        # so the Rust mirror can never overflow/diverge.
        sk = block.sketches.get(tx.txid())
        if sk and any(sk):
            named_e = [hash_to_entry[r] for r in tx.canonical_refs()
                       if r in hash_to_entry]
            for e in named_e:
                acc = e.get("sketch") or [0] * SKETCH_DIM
                e["sketch"] = [_sat64(a + x * 10_000 // len(named_e))
                               for a, x in zip(acc, sk)]
    led.apply_reward(h, miner_pubs=[tx.miner for tx in block.txs],
                     proposer_pub=block.header.proposer,
                     data_credits=data_credits,
                     miner_weights=miner_weights,
                     data_addrs=[data_contributor] if data_contributor else [])
    # lock each included delta's admission bond from its miner's balance (after
    # the reward, so this block's reward can fund this block's bond). A miner who
    # can't afford the bond it committed to makes the block invalid.
    from .token import address as _addr
    for tx in block.txs:
        if not led.lock_bond(tx.txid(), _addr(tx.miner), getattr(tx, "bond", 0), h):
            raise ValidationError(f"miner cannot afford delta bond {tx.txid()[:8]}")
    for tx in canonical_account_txs(block.data_txs, block.transfers):
        if isinstance(tx, TransferTx):
            if not tx.verify():
                raise ValidationError(f"bad signature on transfer {tx.txid()[:8]}")
            if not led.apply_transfer(tx):
                raise ValidationError(f"invalid transfer {tx.txid()[:8]} (nonce/balance)")
        else:
            if not led.apply_data_tx(tx, h, recent_proposers):
                raise ValidationError(f"invalid data tx {tx.txid()[:8]}")
    return led


def validate_block(block: Block, parent_w_int: np.ndarray, parent_height: int,
                   parent_ledger: TokenLedger | None = None,
                   data_contributor: str | None = None,
                   recent_proposers: set[str] = frozenset(),
                   parent_model: ModelState | None = None,
                   params: GenesisParams | None = None):
    """Validate a block from first principles against its parent state.

    Returns (post-state weights, post-ledger, post-model) if valid; raises
    ValidationError otherwise. Any node can run this — no trust in the proposer
    required (§5).

    PROTOCOL v1 (parent_model + params given — the live protocol): enforces the
    header version, proposer ELIGIBILITY (stake-weighted VRF sortition with the
    attempt-widening liveness fallback), page claims (existence, active status,
    body zero outside claims), the WORK QUOTA, the per-page state transition,
    growth activation, and the ModelState fold + model_root commitment.

    Legacy mode (parent_model=None) preserves the pre-v1 dense rules for older
    rig callers only; `parent_ledger=None` likewise skips ledger validation."""
    h = block.header
    v1 = parent_model is not None
    dim = int(parent_w_int.shape[0])
    # 0. STRUCTURAL invariants that bind the header to its parent and body.
    #    v1: the version must be the scheduled version for this height — the
    #    whole upgrade mechanism (unknown-to-us versions fail loudly upstream).
    if v1 and h.version != expected_version(h.height):
        raise ValidationError(
            f"header version {h.version} != scheduled {expected_version(h.height)}")
    #    height must advance by exactly one — otherwise a miner could pin a low
    #    height on every block and mint the height-keyed reward forever (the
    #    halving/sunset are only meaningful if height is monotone), and a
    #    height-0 non-genesis block would underflow h.height-1 below.
    if h.height != parent_height + 1:
        raise ValidationError("height must be parent height + 1")
    #    n_txs must match the actual tx count (it is committed in the block hash;
    #    a mismatch means the header misrepresents the block).
    if h.n_txs != len(block.txs):
        raise ValidationError("n_txs does not match tx count")
    #    PROPOSER LOTTERY: the VRF proof must be a valid signature by the
    #    proposer over this (height, attempt) seed, and header.work must be the
    #    attempt-discounted weight derived from it — so work is NON-FORGEABLE.
    #    v1 ADDS the eligibility gate itself: the proof must clear the
    #    stake-weighted threshold for its attempt (cold-start and ATTEMPT_MAX
    #    rules inside lottery.eligible). Genesis is constructed and exempt.
    from . import lottery
    if h.proposer != "genesis":
        proof = bytes.fromhex(h.vrf_proof) if h.vrf_proof else b""
        if v1:
            if not 0 <= h.vrf_attempt <= lottery.ATTEMPT_MAX:
                raise ValidationError("vrf_attempt out of range")
            if parent_ledger is not None:
                stake = parent_ledger.balance(address(h.proposer))
                total = parent_ledger.supply()
                if not lottery.eligible(h.proposer, proof, h.prev_hash, h.height,
                                        h.vrf_attempt, stake, total):
                    raise ValidationError("proposer not eligible at this attempt")
            elif not verify(h.proposer,
                            lottery.seed(h.prev_hash, h.height, h.vrf_attempt),
                            proof):
                raise ValidationError("invalid proposer VRF proof")
            if h.work != lottery.attempt_work(proof, h.vrf_attempt):
                raise ValidationError("header.work is not the VRF-derived weight")
        else:
            if not verify(h.proposer, lottery.seed(h.prev_hash, h.height), proof):
                raise ValidationError("invalid proposer VRF proof")
            if h.work != lottery.vrf_work(proof):
                raise ValidationError("header.work is not the VRF-derived weight")
    # 1. every tx is well-formed and correctly signed; its delta body must have
    #    the model dimension so aggregation cannot be made to panic/diverge by a
    #    short or long body (all bodies share `dim`, checked here before use).
    #    v1 ADDS the page-claim rules: the claim set is canonical and nonempty,
    #    every claimed page exists and is ACTIVE (frozen pages reject deltas),
    #    the body is EXACTLY ZERO outside the claimed spans (a non-claimant's
    #    zero is absence, not a vote for zero), and the claimed region carries
    #    at least the quota's worth of nonzero work (required_nnz).
    for tx in block.txs:
        if not tx.verify():
            raise ValidationError(f"bad signature on tx {tx.txid()[:8]}")
        if tx.base_height != h.height - 1:
            raise ValidationError("tx base_height does not match parent")
        body = block.bodies.get(tx.da_pointer)
        if body is None:
            raise ValidationError(f"missing DA body for {tx.da_pointer}")
        if int(body.shape[0]) != dim:
            raise ValidationError("delta body length != model dimension")
        if _sha(body.tobytes()) != tx.delta_hash:
            raise ValidationError("delta body hash mismatch (DA withholding/forgery)")
        if v1:
            pages = tx.canonical_pages()
            if not pages or list(tx.pages) != pages:
                raise ValidationError("tx pages must be canonical and nonempty")
            for p in pages:
                if not parent_model.is_active(p):
                    raise ValidationError(
                        f"tx claims missing/frozen page {p}")
            mask = np.zeros(dim, dtype=bool)
            for p in pages:
                s, e = parent_model.page_span(p)
                mask[s:e] = True
            if np.any(body[~mask] != 0):
                raise ValidationError("delta body nonzero outside claimed pages")
            nnz = int(np.count_nonzero(body))
            if nnz < parent_model.required_nnz(pages):
                raise ValidationError("delta below work quota")
            # v2 ENVELOPE: the payload never scales with quota. A delta over
            # the cap is invalid no matter how much work it carries — a rising
            # quota narrows the claimable span instead of fattening the wire.
            if params is not None and nnz > params.delta_max_nnz:
                raise ValidationError("delta exceeds the envelope (max nnz)")
    # 2. tx-set root matches
    if txset_root(block.txs) != h.txset_root:
        raise ValidationError("txset_root mismatch")
    # 2b. DELTA SCORES (rev 7): exactly one committed score per included tx,
    #     integer in [0, SCORE_CAP], and the commitment reproduces. Scores are
    #     block data — validators never recompute the float eval (cross-GPU
    #     nondeterminism stays outside consensus); a fraudulent score is a
    #     bonded, challengeable claim.
    txids = {t.txid() for t in block.txs}
    if set(block.scores.keys()) != txids:
        raise ValidationError("scores must cover exactly the included txs")
    for k, v in block.scores.items():
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= SCORE_CAP:
            raise ValidationError(f"score out of range for {k[:8]}")
    if scores_root(block.scores) != h.score_root:
        raise ValidationError("score_root mismatch")
    # 2c. INFLUENCE SKETCHES (rev 8): one committed sketch per included tx,
    #     SKETCH_DIM ints each within i32 (an all-zero sketch = "unsketched",
    #     contributing nothing to attribution), and the commitment reproduces.
    if set(block.sketches.keys()) != txids:
        raise ValidationError("sketches must cover exactly the included txs")
    for k, v in block.sketches.items():
        if len(v) != SKETCH_DIM or any(
                not isinstance(x, int) or isinstance(x, bool)
                or not -I32 <= x < I32 for x in v):
            raise ValidationError(f"sketch malformed for {k[:8]}")
    if sketch_root(block.sketches) != h.sketch_root:
        raise ValidationError("sketch_root mismatch")
    # 3. the state transition reproduces the committed root (deterministic, §3.4)
    if v1:
        # v1: per-page trimmed mean over each page's actual claimants, computed
        # against the PARENT page table; then the ModelState fold — any growth
        # event due this block appends its deterministically-initialized expert
        # page(s) AFTER aggregation and BEFORE the root; state_root commits the
        # page-Merkle root over the (possibly extended) page set. THE ORDER OF
        # THESE THREE STEPS IS CONSENSUS — the Rust mirror must match exactly.
        bodies = [block.bodies[tx.da_pointer] for tx in block.txs]
        claims = [tx.canonical_pages() for tx in block.txs]
        spans = [(p[0], p[1]) for p in parent_model.pages]
        w = paged_transition(parent_w_int, bodies, claims, spans)
        zero_scored = sum(1 for t in block.txs
                          if int(block.scores.get(t.txid(), 0)) == 0)
        post_model, activations = model_fold(parent_model, params, h.height,
                                             len(block.txs), zero_scored,
                                             h.prev_hash)
        for page_id, _layer, _expert, trigger in activations:
            w = np.concatenate([w, page_init(trigger, page_id, params.spec)])
        if page_state_root(w, post_model) != h.state_root:
            raise ValidationError("state_root does not reproduce from txs")
        if post_model.model_root() != h.model_root:
            raise ValidationError("model_root does not reproduce (fold divergence)")
    else:
        deltas = [block.bodies[tx.da_pointer] for tx in block.txs]
        w = parent_w_int + trimmed_mean_int(deltas) if deltas else parent_w_int.copy()
        if state_root(w) != h.state_root:
            raise ValidationError("state_root does not reproduce from txs")
        post_model = None
    # 4. the TRANSFER + DATA LANES: set roots + full token-ledger transition
    led = None
    if parent_ledger is not None:
        if xfer_root(block.transfers) != h.transfer_root:
            raise ValidationError("transfer_root mismatch")
        if dta_root(block.data_txs) != h.data_root:
            raise ValidationError("data_root mismatch")
        led = apply_ledger(parent_ledger, block, data_contributor, recent_proposers)
        if led.root() != h.ledger_root:
            raise ValidationError("ledger_root does not reproduce from block")
    return w, led, post_model


class BlockTree:
    """All known blocks, with heaviest-valid-chain selection (Nakamoto fork choice)."""

    def __init__(self, genesis_w_int: np.ndarray, prune_depth: int | None = None,
                 data_contributor: str | None = None,
                 params: GenesisParams | None = None):
        self.genesis_w = genesis_w_int.copy()
        # PROTOCOL v1 iff genesis params (the ModelSpec + retarget constants)
        # are supplied: the state commitment is the page-Merkle root and the
        # header carries the ModelState commitment from block 0.
        self.params = params
        if params is not None:
            model0 = ModelState.genesis(params.spec)
            assert model0.dim() == int(genesis_w_int.shape[0]), \
                "genesis weight length must equal the ModelSpec page table"
            groot = page_state_root(genesis_w_int, model0)
            gh = Header(0, "0" * 64, groot, _sha(b""), 0, 0, "genesis",
                        model_root=model0.model_root())
        else:
            model0 = None
            gh = Header(0, "0" * 64, state_root(genesis_w_int), _sha(b""), 0, 0,
                        "genesis")
        self.genesis = Block(gh, [], {})
        self.blocks = {self.genesis.hash: self.genesis}
        self.state = {self.genesis.hash: genesis_w_int.copy()}     # per-block post-state
        # per-block ModelState (v1) — small, kept forever like ledgers/headers
        self.model = {self.genesis.hash: model0}
        # the token ledger is chain state too: EMPTY balances at genesis (fair
        # launch), advanced deterministically by every block. data_contributor
        # is a GENESIS PARAMETER (identical on every node): the founding corpus
        # becomes registry entry zero, owned by that wallet, earning the data
        # share under the same rules as any staked entry.
        genesis_ledger = TokenLedger()
        if data_contributor:
            genesis_ledger.seed_genesis_data(data_contributor)
        self.ledger = {self.genesis.hash: genesis_ledger}
        self.data_contributor = data_contributor
        self.cum_work = {self.genesis.hash: 0}
        self.head = self.genesis.hash
        # prune_depth: keep full state + bodies only within this many blocks of the
        # head (plus genesis). Essential at real-model scale — an 86M state is
        # ~0.7GB, so retaining one per block OOMs in minutes. Headers, txs and
        # cum_work are kept forever (fork choice needs them); a reorg deeper than
        # prune_depth would need replay from genesis (Bitcoin prunes the same way).
        self.prune_depth = prune_depth

    def add_block(self, block: Block) -> bool:
        """Validate and attach a block. Returns True if it became the new head."""
        if block.hash in self.blocks:
            return False
        parent = block.header.prev_hash
        if parent not in self.blocks:
            raise ValidationError("orphan: parent unknown")
        # ledger validation only when the block commits one (rev-2+ blocks always
        # do; legacy rev-1 blocks carry ledger_root="" and skip it)
        parent_led = self.ledger.get(parent) if block.header.ledger_root else None
        juror_pubs = self.recent_proposers(parent)
        parent_height = self.blocks[parent].header.height
        w, led, model = validate_block(block, self.state[parent], parent_height,
                                       parent_led, self.data_contributor,
                                       juror_pubs, self.model.get(parent),
                                       self.params)  # may raise
        self.blocks[block.hash] = block
        self.state[block.hash] = w
        self.model[block.hash] = model
        if led is not None:
            self.ledger[block.hash] = led
        else:                                                      # legacy: rewards only
            self.ledger[block.hash] = apply_ledger(
                self.ledger[parent], block, self.data_contributor, juror_pubs)
        self.cum_work[block.hash] = self.cum_work[parent] + max(1, block.header.work)
        # heaviest chain wins; ties broken by lexicographically smaller hash
        if (self.cum_work[block.hash] > self.cum_work[self.head] or
                (self.cum_work[block.hash] == self.cum_work[self.head]
                 and block.hash < self.head)):
            self.head = block.hash
            self._prune_deep()
            return True
        self._prune_deep()
        return False

    def _prune_deep(self):
        """Drop heavy per-block data (state vector, delta bodies) for blocks more
        than prune_depth below the head. Headers/txs/cum_work stay."""
        if self.prune_depth is None:
            return
        floor = self.blocks[self.head].header.height - self.prune_depth
        for bh, b in self.blocks.items():
            if bh == self.genesis.hash or b.header.height >= floor:
                continue
            self.state.pop(bh, None)
            if b.bodies:
                b.bodies = {}

    def head_state(self) -> np.ndarray:
        return self.state[self.head]

    def head_ledger(self) -> TokenLedger:
        return self.ledger[self.head]

    def head_model(self) -> ModelState | None:
        return self.model.get(self.head)

    def recent_proposers(self, tip: str) -> set[str]:
        """Proposer pubkeys of the last PROPOSER_LOOKBACK blocks ending at `tip`
        — the deterministic juror set for data-challenge votes."""
        out, cur = set(), tip
        for _ in range(PROPOSER_LOOKBACK):
            b = self.blocks.get(cur)
            if b is None or cur == self.genesis.hash:
                break
            out.add(b.header.proposer)
            cur = b.header.prev_hash
        return out

    def chain_from_genesis(self, tip: str | None = None) -> list:
        tip = tip or self.head
        out = []
        while tip != self.genesis.hash:
            b = self.blocks[tip]
            out.append(b)
            tip = b.header.prev_hash
        return list(reversed(out))

    def replay_head(self) -> np.ndarray:
        """Independently reconstruct head state from genesis + block bodies (§3.5).
        v1: replays the per-page transitions AND the ModelState fold, including
        growth activations (page appends) — bit-exact across dimension changes."""
        w = self.genesis_w.copy()
        if self.params is None:
            for b in self.chain_from_genesis():
                deltas = [b.bodies[tx.da_pointer] for tx in b.txs]
                if deltas:
                    w = w + trimmed_mean_int(deltas)
            return w
        model = ModelState.genesis(self.params.spec)
        for b in self.chain_from_genesis():
            h = b.header
            bodies = [b.bodies[tx.da_pointer] for tx in b.txs]
            claims = [tx.canonical_pages() for tx in b.txs]
            spans = [(p[0], p[1]) for p in model.pages]
            w = paged_transition(w, bodies, claims, spans)
            zero_scored = sum(1 for t in b.txs
                              if int(b.scores.get(t.txid(), 0)) == 0)
            model, activations = model_fold(model, self.params, h.height,
                                            len(b.txs), zero_scored, h.prev_hash)
            for page_id, _l, _e, trigger in activations:
                w = np.concatenate([w, page_init(trigger, page_id,
                                                 self.params.spec)])
        return w


def build_block(tree: BlockTree, parent_hash: str, accepted: list, bodies: dict,
                works: dict, proposer_key, transfers: list | None = None,
                data_txs: list | None = None,
                scores: dict | None = None,
                sketches: dict | None = None,
                attempt: int = 0) -> Block:
    """Assemble a valid block extending `parent_hash` from accepted txs, plus the
    transfer lane (rev 2), the data lane (rev 3), the proposer's VRF proof and
    sortition attempt (v1), and the proposer's committed delta scores (rev 7 —
    omitted scores default to zero, which reward-weights uniformly).

    PRODUCER/VALIDATOR SYMMETRY: this function must mirror validate_block's
    transition EXACTLY (per-page aggregation, fold, activation, roots) — a
    producer that diverges builds blocks it then rejects itself."""
    from . import lottery
    parent_w = tree.state[parent_hash]
    parent_model = tree.model.get(parent_hash)
    transfers = list(transfers or [])
    data_txs = list(data_txs or [])
    height = tree.blocks[parent_hash].header.height + 1
    blk_scores = {t.txid(): int((scores or {}).get(t.txid(), 0)) for t in accepted}
    blk_sketches = {t.txid(): [int(x) for x in
                               (sketches or {}).get(t.txid(), [0] * SKETCH_DIM)]
                    for t in accepted}
    if parent_model is not None:                       # PROTOCOL v1
        body_list = [bodies[tx.da_pointer] for tx in accepted]
        claims = [tx.canonical_pages() for tx in accepted]
        spans = [(p[0], p[1]) for p in parent_model.pages]
        w = paged_transition(parent_w, body_list, claims, spans)
        zero_scored = sum(1 for t in accepted if blk_scores[t.txid()] == 0)
        post_model, activations = model_fold(parent_model, tree.params, height,
                                             len(accepted), zero_scored,
                                             parent_hash)
        for page_id, _l, _e, trigger in activations:
            w = np.concatenate([w, page_init(trigger, page_id,
                                             tree.params.spec)])
        s_root, m_root = page_state_root(w, post_model), post_model.model_root()
        vrf_proof = lottery.vrf_prove(proposer_key, parent_hash, height, attempt)
        work = lottery.attempt_work(vrf_proof, attempt)
    else:                                              # legacy dense path
        deltas = [bodies[tx.da_pointer] for tx in accepted]
        w = parent_w + trimmed_mean_int(deltas) if deltas else parent_w.copy()
        s_root, m_root = state_root(w), ""
        vrf_proof = lottery.vrf_prove(proposer_key, parent_hash, height)
        work = lottery.vrf_work(vrf_proof)
    header = Header(
        height=height,
        prev_hash=parent_hash, state_root=s_root,
        txset_root=txset_root(accepted), n_txs=len(accepted),
        work=work, proposer=proposer_key.pub,
        vrf_proof=vrf_proof.hex(), score_root=scores_root(blk_scores),
        sketch_root=sketch_root(blk_sketches),
        model_root=m_root, vrf_attempt=attempt,
        version=expected_version(height))
    block = Block(header, accepted,
                  {t.da_pointer: bodies[t.da_pointer] for t in accepted},
                  transfers, data_txs, blk_scores, blk_sketches)
    # commit the full token transition into the header
    header.transfer_root = xfer_root(transfers)
    header.data_root = dta_root(data_txs)
    header.ledger_root = apply_ledger(tree.ledger[parent_hash], block,
                                      tree.data_contributor,
                                      tree.recent_proposers(parent_hash)).root()
    return block
