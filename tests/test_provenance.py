"""Provenance & data-share payout (protocol rev 5).

Exercises the deterministic Stage-1 rule directly through apply_ledger, without
the block-builder helpers (which have an unrelated pre-existing issue): a delta
must name staked, active data, and the block's data share is routed to the
owners of the corpora the block's deltas actually named — not to every
registered entry, and not to a single configured contributor.
"""

import numpy as np
import pytest

from rig.blockchain import Block, Header, ValidationError, apply_ledger
from rig.crypto import BackpropTx, Key, delta_hash
from rig.token import SHARE_DATA, TokenLedger, address, emission


def _key(seed: bytes) -> Key:
    return Key.generate(seed.ljust(32, b"0")[:32])


def _delta_tx(miner: Key, height: int, shard: int, data_refs) -> BackpropTx:
    d = np.zeros(8, dtype=np.int64)
    return BackpropTx(miner=miner.pub, base_height=height,
                      delta_hash=delta_hash(d.tobytes()),
                      da_pointer=f"da://{shard}", data_refs=list(data_refs)).signed(miner)


def _block(height: int, proposer: Key, txs) -> Block:
    h = Header(height=height, prev_hash="ab" * 32, state_root="cd" * 32,
               txset_root="ef" * 32, n_txs=len(txs), work=1, proposer=proposer.pub)
    return Block(header=h, txs=txs, bodies={})


def _register(led: TokenLedger, owner: Key, data_hash: str, weight: int = 1):
    """Directly insert an active registry entry (a staked, available corpus)."""
    led.registry[data_hash] = {
        "owner": address(owner.pub), "data_hash": data_hash, "size": 0,
        "media_type": "text", "stake": 0, "weight": weight, "status": "active"}


def _staked_ledger():
    """A ledger where alice has staked a corpus 'C' (active in the registry)."""
    miner = _key(b"miner")
    alice = _key(b"alice")
    led = TokenLedger()
    _register(led, alice, "C")
    return led, miner, alice


def test_delta_without_provenance_is_rejected():
    led, miner, _ = _staked_ledger()
    blk = _block(1, miner, [_delta_tx(miner, 0, 0, data_refs=[])])   # names nothing
    with pytest.raises(ValidationError, match="provenance required"):
        apply_ledger(led, blk, data_contributor=None)


def test_delta_naming_unstaked_data_is_rejected():
    led, miner, _ = _staked_ledger()
    blk = _block(1, miner, [_delta_tx(miner, 0, 0, data_refs=["not-staked"])])
    with pytest.raises(ValidationError, match="provenance required"):
        apply_ledger(led, blk, data_contributor=None)


def test_data_share_goes_to_the_named_owner():
    led, miner, alice = _staked_ledger()
    blk = _block(1, miner, [_delta_tx(miner, 0, 0, data_refs=["C"])])
    out = apply_ledger(led, blk, data_contributor=None)
    # alice (owner of corpus C) receives exactly the data share of the emission
    expected_data = emission(1) * SHARE_DATA // 10_000
    assert out.balance(address(alice.pub)) == expected_data
    # and the miner got the miner share (non-zero), proving the split ran
    assert out.balance(address(miner.pub)) > 0


def test_share_splits_across_named_corpora_by_weight():
    # two corpora, equal weight, each named once → data share splits 50/50
    miner = _key(b"miner")
    a = _key(b"owner-a")
    b = _key(b"owner-b")
    led = TokenLedger()
    _register(led, a, "A", weight=1)
    _register(led, b, "B", weight=1)
    blk = _block(1, miner, [_delta_tx(miner, 0, 0, ["A"]),
                            _delta_tx(miner, 0, 1, ["B"])])
    out = apply_ledger(led, blk, data_contributor=None)
    ba, bb = out.balance(address(a.pub)), out.balance(address(b.pub))
    assert ba == bb and ba > 0                         # equal split, both paid


def test_unnamed_registered_corpus_earns_nothing():
    # a corpus that exists in the registry but is NOT named this block gets $0 —
    # the rev-3 "pay everyone every block" behaviour is gone.
    miner = _key(b"miner")
    named = _key(b"named")
    idle = _key(b"idle")
    led = TokenLedger()
    _register(led, named, "NAMED")
    _register(led, idle, "IDLE")
    blk = _block(1, miner, [_delta_tx(miner, 0, 0, ["NAMED"])])
    out = apply_ledger(led, blk, data_contributor=None)
    assert out.balance(address(named.pub)) > 0
    assert out.balance(address(idle.pub)) == 0          # named nowhere ⇒ unpaid
