"""Delta scoring (rev 7): committed held-out-loss scores weight rewards.

Scores are BLOCK DATA (committed under header.score_root), so consensus stays
deterministic across GPUs; validation enforces structure/bounds/commitment, and
the ledger splits the miner pool and data credits proportionally. All-zero
scores fall back to uniform weighting.
"""

import numpy as np
import pytest

from rig.blockchain import (SCORE_CAP, Block, BlockTree, Header, ValidationError,
                            apply_ledger, build_block, effective_scores,
                            scores_root)
from rig.chain import quantize
from rig.crypto import BackpropTx, Key, delta_hash
from rig.token import SHARE_MINERS, TokenLedger, address, emission

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


def test_scored_block_builds_validates_and_weights_miner_split():
    a, b = _key("minerA"), _key("minerB")
    tree = _tree()
    rng = np.random.default_rng(7)
    ta, pa, ba = _tx(a, rng, 0)
    tb, pb, bb = _tx(b, rng, 1)
    # a scored 3x b -> a earns 3x b's miner share
    blk = build_block(tree, tree.head, [ta, tb], {pa: ba, pb: bb},
                      {}, a, scores={ta.txid(): 300_000, tb.txid(): 100_000})
    assert tree.add_block(blk)                       # full validation passes
    led = tree.ledger[blk.hash]
    pool = emission(1) * SHARE_MINERS // 10_000
    assert led.balance(address(a.pub)) - led.balance(address(b.pub)) >= 0
    assert led.balance(address(b.pub)) == pool * 100_000 // 400_000
    # a also proposed (10%), so subtract the proposer cut for the miner check
    assert led.balance(address(a.pub)) == pool * 300_000 // 400_000 + emission(1) // 10


def test_all_zero_scores_fall_back_to_uniform():
    a, b = _key("minerA"), _key("minerB")
    tree = _tree()
    rng = np.random.default_rng(8)
    ta, pa, ba = _tx(a, rng, 0)
    tb, pb, bb = _tx(b, rng, 1)
    blk = build_block(tree, tree.head, [ta, tb], {pa: ba, pb: bb}, {}, a)  # no scores
    assert tree.add_block(blk)
    led = tree.ledger[blk.hash]
    pool = emission(1) * SHARE_MINERS // 10_000
    assert led.balance(address(b.pub)) == pool // 2   # equal split


def test_tampered_or_missing_scores_rejected():
    a = _key("minerA")
    tree = _tree()
    rng = np.random.default_rng(9)
    ta, pa, ba = _tx(a, rng, 0)
    blk = build_block(tree, tree.head, [ta], {pa: ba}, {}, a,
                      scores={ta.txid(): 5})
    # tamper the score after the header committed it
    blk.scores[ta.txid()] = 50
    with pytest.raises(ValidationError, match="score_root"):
        tree.add_block(blk)
    # score for a foreign txid
    blk2 = build_block(tree, tree.head, [ta], {pa: ba}, {}, a)
    blk2.scores["ff" * 32] = 1
    with pytest.raises(ValidationError, match="scores must cover"):
        tree.add_block(blk2)


def test_score_cap_enforced():
    a = _key("minerA")
    tree = _tree()
    rng = np.random.default_rng(10)
    ta, pa, ba = _tx(a, rng, 0)
    blk = build_block(tree, tree.head, [ta], {pa: ba}, {}, a,
                      scores={ta.txid(): SCORE_CAP + 1})
    with pytest.raises(ValidationError, match="score out of range"):
        tree.add_block(blk)


def test_effective_scores_uniform_fallback_helper():
    a = _key("m")
    rng = np.random.default_rng(11)
    ta, _, _ = _tx(a, rng, 0)
    assert effective_scores([ta], {}) == {ta.txid(): 1}
    assert effective_scores([ta], {ta.txid(): 9}) == {ta.txid(): 9}
    assert effective_scores([], {}) == {}
