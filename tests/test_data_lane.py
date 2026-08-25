"""Data lane (rev 3): staked submission earns weighted royalties, challenges
settle stake-vs-stake with proposer-gated votes, and everything commits through
the ledger root. Drives RealCore directly (toy model, no network)."""

import pytest

import client.gossip as g
from rig.crypto import Key
from rig.token import (
    CHALLENGE_WINDOW, GRAIN, DataChallengeTx, DataSubmitTx, DataVoteTx,
    TokenLedger, address, emission,
)

FOUNDER_KEY = Key.generate(b"founder-wallet-test-seed-000000!")


@pytest.fixture()
def core():
    g.DATA_CONTRIBUTOR = address(FOUNDER_KEY.pub)
    c = g.RealCore(0)
    yield c
    g.DATA_CONTRIBUTOR = None


def _mine(core, n=1):
    for _ in range(n):
        hh, delta, _ = core.train_delta()
        outbox = []
        core.submit_delta(hh, delta, outbox)
        core.propose(outbox)
    return core.tree.blocks[core.tree.head].header.height


def test_genesis_registry_and_data_share(core):
    led = core.head_ledger()
    assert "genesis" in led.registry                    # founding corpus entry
    assert led.registry["genesis"]["owner"] == g.DATA_CONTRIBUTOR
    _mine(core)
    # the whole data share flows to the sole (genesis) registry entry
    assert core.head_ledger().balance(g.DATA_CONTRIBUTOR) == \
        emission(1) * 2_000 // 10_000


def test_staked_submission_earns_weighted_share(core):
    _mine(core)                                         # fund the miner
    miner = core.key
    stake = core.head_ledger().balance(address(miner.pub)) // 2
    tx = DataSubmitTx(owner_pub=miner.pub, data_hash="ab" * 32,
                      size_bytes=1234, media_type="csv", stake=stake,
                      nonce=0).signed(miner)
    outbox = []
    core.recv_data_tx(tx, outbox)
    _mine(core)                                         # block 2 includes it
    led = core.head_ledger()
    entry = led.registry[tx.txid()]
    assert entry["status"] == "active" and entry["stake"] == stake
    assert entry["media_type"] == "csv"                 # bytes are bytes — any modality
    founder_before = led.balance(g.DATA_CONTRIBUTOR)
    miner_before = led.balance(address(miner.pub))
    _mine(core)                                         # block 3: split across named
    led = core.head_ledger()
    d_founder = led.balance(g.DATA_CONTRIBUTOR) - founder_before
    d_pool = emission(3) * 2_000 // 10_000
    # rev 7: the data share splits by DELTA SCORE, not registry weight — the
    # round's single delta names both active corpora, so each takes an equal
    # slice of its score and the pool splits 50/50 (registry weight now only
    # gates active-set membership; loss scores carry the economic weighting).
    assert d_founder == d_pool // 2
    assert led.balance(address(miner.pub)) > miner_before   # miner earns data share too


def test_challenge_upheld_revokes_and_pays_challenger():
    """Ledger-level: since #93 (disinterested-juror rule + CHALLENGE_QUORUM=3)
    a single-proposer sim can never reach quorum — the owner is the only recent
    proposer and may not vote on their own entry. So the upheld path is
    exercised directly on the ledger with three disinterested jurors, the same
    shape as the consensus golden scenario."""
    led = TokenLedger()
    owner = Key.generate(b"upheld-owner-test-seed-00000000!")
    challenger = Key.generate(b"challenger-test-seed-0000000000!")
    jurors = [Key.generate(f"juror-{i}-test-seed-000000000000!".encode()[:32])
              for i in range(3)]
    led.apply_reward(1, [owner.pub], "genesis", [])         # fund the owner
    led.apply_reward(2, [challenger.pub], "genesis", [])    # fund the challenger
    sub = DataSubmitTx(owner_pub=owner.pub, data_hash="cd" * 32, size_bytes=9,
                       media_type="text", stake=1 * GRAIN, nonce=0).signed(owner)
    assert led.apply_data_tx(sub, 2, set())
    ch = DataChallengeTx(challenger_pub=challenger.pub, data_id=sub.txid(),
                         stake=GRAIN // 2, reason="validity",
                         nonce=0).signed(challenger)
    assert led.apply_data_tx(ch, 3, set())
    assert ch.txid() in led.challenges
    recent = {j.pub for j in jurors}                        # disinterested quorum
    for j in jurors:
        vote = DataVoteTx(voter_pub=j.pub, challenge_id=ch.txid(),
                          support=True, nonce=0).signed(j)
        assert led.apply_data_tx(vote, 4, recent)
    assert len(led.challenges[ch.txid()]["votes_for"]) == 3
    ch_bal_before = led.balance(address(challenger.pub))
    entry_stake = led.registry[sub.txid()]["stake"]
    led.resolve_expired_challenges(3 + CHALLENGE_WINDOW + 1)
    assert led.registry[sub.txid()]["status"] == "revoked"
    assert led.registry[sub.txid()]["stake"] == 0
    # challenger got the entry's stake + their own back
    assert led.balance(address(challenger.pub)) == \
        ch_bal_before + entry_stake + ch.stake
    # revoked entry is no longer payable: only active corpora can be credited
    active = {e["data_hash"] for e in led.registry.values()
              if e["status"] == "active"}
    assert "cd" * 32 not in active


def test_challenge_rejected_pays_owner(core):
    _mine(core)
    miner = core.key
    sub = DataSubmitTx(owner_pub=miner.pub, data_hash="ee" * 32, size_bytes=9,
                       media_type="text",
                       stake=core.head_ledger().balance(address(miner.pub)) // 4,
                       nonce=0).signed(miner)
    outbox = []
    core.recv_data_tx(sub, outbox)
    _mine(core)
    challenger = Key.generate(b"challenger-test-seed-0000000000!")
    from rig.token import TransferTx
    core.recv_transfer(TransferTx(from_pub=miner.pub,
                                  to_addr=address(challenger.pub),
                                  amount=2 * GRAIN, nonce=1).signed(miner), outbox)
    _mine(core)
    ch = DataChallengeTx(challenger_pub=challenger.pub, data_id=sub.txid(),
                         stake=1 * GRAIN, reason="ownership",
                         nonce=0).signed(challenger)
    core.recv_data_tx(ch, outbox)
    _mine(core)
    owner_before = core.head_ledger().balance(address(miner.pub))
    # NO votes -> challenge fails at expiry; challenger's stake -> owner
    heights_mined = _mine(core, CHALLENGE_WINDOW + 1)
    led = core.head_ledger()
    assert led.registry[sub.txid()]["status"] == "active"    # survives
    assert ch.txid() not in led.challenges
    assert led.balance(address(miner.pub)) > owner_before + ch.stake - 1  # got the stake (+mining)


def test_vote_gated_to_recent_proposers(core):
    _mine(core)
    outsider = Key.generate(b"outsider-never-proposed-0000000!")
    led = TokenLedger()
    led.seed_genesis_data(g.DATA_CONTRIBUTOR)
    vote = DataVoteTx(voter_pub=outsider.pub, challenge_id="ff" * 32,
                      support=True, nonce=0).signed(outsider)
    # even with a (fake) open challenge, a non-proposer's vote is invalid
    led.challenges["ff" * 32] = {"data_id": "x", "challenger": "y", "stake": 1,
                                 "reason": "validity", "expiry": 99,
                                 "votes_for": [], "votes_against": []}
    assert not led.apply_data_tx(vote, 5, recent_proposers={core.key.pub})
