"""rev 8 — influence sketches + usage-attributed inference royalties (§8).

Delta sketches are committed per block (sketch_root), accrue onto the named
corpora's ledger entries (pure integer arithmetic, exactly recomputable from DA
bodies), and the fee's data slice pays corpora by positive alignment with the
receipt's answer sketch — falling back to the pool when unsketched.
"""

import numpy as np
import pytest

from rig.blockchain import (SKETCH_DIM, Block, BlockTree, Header,
                            ValidationError, apply_ledger, build_block)
from rig.chain import quantize
from rig.crypto import BackpropTx, Key, delta_hash
from rig.sketch import dot, sketch_dense, sketch_sparse
from rig.token import (FEE_SHARE_DATA, InferenceReceiptTx, TokenLedger,
                       address)

DIM = 40
FOUNDER = address(Key.generate(b"founder".ljust(32, b"0")).pub)


def _key(tag):
    return Key.generate(tag.encode().ljust(32, b"0"))


def _tree():
    return BlockTree(quantize(np.zeros(DIM)), data_contributor=FOUNDER)


def _tx(k, rng, shard):
    body = quantize(rng.standard_normal(DIM) * 0.1)
    ptr = f"da://s{shard}"
    tx = BackpropTx(miner=k.pub, base_height=0,
                    delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                    data_refs=["genesis"]).signed(k)
    return tx, ptr, body


def test_sketch_projection_exact_and_deterministic():
    v = [0] * 100
    v[3], v[17], v[41] = 5, -2, 7
    s1 = sketch_dense(v)
    s2 = sketch_sparse([3, 17, 41], [5, -2, 7])
    assert s1 == s2 and len(s1) == SKETCH_DIM
    assert dot(s1, s1) > 0                          # self-alignment positive
    # a disjoint-support vector gives an independent sketch (not identical)
    w = [0] * 100
    w[60] = 5
    assert sketch_dense(w) != s1


def test_sketched_block_validates_and_accrues_to_registry():
    a = _key("minerA")
    tree = _tree()
    rng = np.random.default_rng(21)
    ta, pa, ba = _tx(a, rng, 0)
    sk = sketch_dense(list(ba))
    blk = build_block(tree, tree.head, [ta], {pa: ba}, {}, a,
                      sketches={ta.txid(): sk})
    assert tree.add_block(blk)
    led = tree.ledger[blk.hash]
    entry = led.registry["genesis"]
    # accrual = sketch × 10_000 // 1 named corpus
    assert entry["sketch"] == [x * 10_000 for x in sk]


def test_tampered_sketch_rejected():
    a = _key("minerA")
    tree = _tree()
    rng = np.random.default_rng(22)
    ta, pa, ba = _tx(a, rng, 0)
    blk = build_block(tree, tree.head, [ta], {pa: ba}, {}, a,
                      sketches={ta.txid(): sketch_dense(list(ba))})
    blk.sketches[ta.txid()] = [1] * SKETCH_DIM      # tamper after commit
    with pytest.raises(ValidationError, match="sketch_root"):
        tree.add_block(blk)
    blk2 = build_block(tree, tree.head, [ta], {pa: ba}, {}, a)
    blk2.sketches[ta.txid()] = [1] * (SKETCH_DIM - 1)  # wrong dimension
    with pytest.raises(ValidationError, match="sketch malformed"):
        tree.add_block(blk2)


def _ledger_with_sketched_corpus(owner, sketch):
    led = TokenLedger()
    led.registry["C"] = {
        "owner": address(owner.pub), "data_hash": "C", "size": 0,
        "media_type": "text", "stake": 0, "weight": 1, "status": "active",
        "sketch": sketch}
    return led


def test_aligned_answer_pays_owner_directly():
    payer, server, owner = _key("payer"), _key("server"), _key("owner")
    corpus_sketch = [3] * SKETCH_DIM
    led = _ledger_with_sketched_corpus(owner, corpus_sketch)
    led.apply_reward(1, [payer.pub], "genesis", [])  # fund payer
    fee = 10_000
    ans = [1] * SKETCH_DIM                          # positively aligned
    tx = InferenceReceiptTx(payer_pub=payer.pub, server_addr=address(server.pub),
                            fee=fee, output_hash="ab" * 32, head_root="cd" * 32,
                            nonce=0, answer_sketch=ans).signed(payer)
    assert led.apply_data_tx(tx, 2, set())
    data_cut = fee * FEE_SHARE_DATA // 10_000
    assert led.balance(address(owner.pub)) == data_cut  # full slice, direct
    assert led.fee_data_pool == 0                        # nothing pooled


def test_antialigned_or_unsketched_falls_back_to_pool():
    payer, server, owner = _key("payer"), _key("server"), _key("owner")
    led = _ledger_with_sketched_corpus(owner, [3] * SKETCH_DIM)
    led.apply_reward(1, [payer.pub], "genesis", [])
    fee = 10_000
    data_cut = fee * FEE_SHARE_DATA // 10_000
    # anti-aligned answer: negative dot → owner unpaid, slice pools
    tx = InferenceReceiptTx(payer_pub=payer.pub, server_addr=address(server.pub),
                            fee=fee, output_hash="ab" * 32, head_root="cd" * 32,
                            nonce=0, answer_sketch=[-1] * SKETCH_DIM).signed(payer)
    assert led.apply_data_tx(tx, 2, set())
    assert led.balance(address(owner.pub)) == 0
    assert led.fee_data_pool == data_cut
    # unsketched receipt: also pools
    tx2 = InferenceReceiptTx(payer_pub=payer.pub, server_addr=address(server.pub),
                             fee=fee, output_hash="ab" * 32, head_root="cd" * 32,
                             nonce=1).signed(payer)
    assert led.apply_data_tx(tx2, 2, set())
    assert led.fee_data_pool == 2 * data_cut
