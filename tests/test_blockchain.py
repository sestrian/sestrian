"""Bitcoin-style blocks: validation, replay, fork choice, tamper rejection (§3, §5)."""

import numpy as np
import pytest

from rig.blockchain import (BlockTree, ValidationError, build_block, txset_root)
from rig.chain import quantize, state_root
from rig.crypto import BackpropTx, Key, delta_hash
from rig.token import address

DIM = 40

# rev 5: a founding data contributor seeds the always-active "genesis" corpus,
# so deltas can satisfy the provenance rule by naming it. A helper wraps tree
# creation so every test's tree has that entry.
FOUNDER = address(Key.generate(b"founder".ljust(32, b"0")).pub)


def _tree(w):
    return BlockTree(w, data_contributor=FOUNDER)


def _miners(n=3):
    return [Key.generate(f"m{i}".encode().ljust(32, b"0")) for i in range(n)]


def _block(tree, parent, height, miners, rng, work=1.0, tag=""):
    txs, bodies, works = [], {}, {}
    for i, k in enumerate(miners):
        body = quantize(rng.standard_normal(DIM) * 0.1)
        ptr = f"da://{tag}{height}/{i}"
        tx = BackpropTx(miner=k.pub, base_height=height - 1,
                        delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                        data_refs=["genesis"]).signed(k)
        txs.append(tx); bodies[ptr] = body; works[tx.txid()] = work
    return build_block(tree, parent, txs, bodies, works, miners[0])


def _grow(tree, n, miners, seed=1, work=1.0, tag=""):
    rng = np.random.default_rng(seed)
    head = tree.head
    for h in range(1, n + 1):
        b = _block(tree, head, tree.blocks[head].header.height + 1, miners, rng, work, tag)
        tree.add_block(b)
        head = b.hash
    return head


def test_chain_builds_and_replays_bit_exact():
    tree = _tree(quantize(np.zeros(DIM)))
    _grow(tree, 6, _miners())
    assert tree.blocks[tree.head].header.height == 6
    assert state_root(tree.replay_head()) == tree.blocks[tree.head].header.state_root


def test_heaviest_valid_chain_wins():
    # Fork choice is heaviest-cumulative-VRF-work (rev 4: header.work =
    # vrf_work(proof), no longer free-form). Build two competing chains and
    # assert the head is whichever tip carries the most cumulative work.
    m = _miners()
    tree = _tree(quantize(np.zeros(DIM)))
    _grow(tree, 5, m, seed=1, tag="A")
    tip_a = tree.head
    fork_parent = tree.chain_from_genesis()[1].hash   # from block 1
    rng = np.random.default_rng(9)
    fh = fork_parent
    for _ in range(6):                                # a longer competing fork
        height = tree.blocks[fh].header.height + 1
        b = _block(tree, fh, height, m, rng, tag="B")
        tree.add_block(b); fh = b.hash
    tip_b = fh
    assert tip_a != tip_b
    assert tree.head in (tip_a, tip_b)
    # the selected head is at least as heavy as either competing tip
    assert tree.cum_work[tree.head] == max(tree.cum_work[tip_a], tree.cum_work[tip_b])


def test_forged_signature_block_rejected():
    m = _miners()
    tree = _tree(quantize(np.zeros(DIM)))
    rng = np.random.default_rng(3)
    body = quantize(rng.standard_normal(DIM) * 0.1)
    ptr = "da://x"
    tx = BackpropTx(miner=m[0].pub, base_height=0,
                    delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                    data_refs=["genesis"])
    tx.sig = m[1].sign(tx.signing_bytes())            # wrong signer
    bad = build_block(tree, tree.head, [tx], {ptr: body}, {tx.txid(): 1.0}, m[0])
    with pytest.raises(ValidationError):
        tree.add_block(bad)


def test_withheld_or_forged_body_rejected():
    m = _miners()
    tree = _tree(quantize(np.zeros(DIM)))
    rng = np.random.default_rng(4)
    body = quantize(rng.standard_normal(DIM) * 0.1)
    ptr = "da://y"
    tx = BackpropTx(miner=m[0].pub, base_height=0,
                    delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                    data_refs=["genesis"]).signed(m[0])
    wrong = quantize(rng.standard_normal(DIM) * 0.1)  # body doesn't match its hash
    bad = build_block(tree, tree.head, [tx], {ptr: wrong}, {tx.txid(): 1.0}, m[0])
    with pytest.raises(ValidationError):
        tree.add_block(bad)


def test_orphan_rejected():
    m = _miners()
    # a block built on tree1 at height 2 (parent = tree1's block 1)
    tree1 = _tree(quantize(np.zeros(DIM)))
    _grow(tree1, 2, m, seed=5)
    orphan = tree1.blocks[tree1.head]                 # its parent is tree1's block 1
    # a fresh node that only has genesis has never seen that parent
    tree2 = _tree(quantize(np.zeros(DIM)))
    with pytest.raises(ValidationError):
        tree2.add_block(orphan)
