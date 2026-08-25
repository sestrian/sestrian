"""The native token — balances as chain state, emissions as block rewards (§9).

Reference implementation (the SPEC for the Rust node). Design commitments:

  * The ledger is CHAIN STATE, exactly like the weights: deterministic integer
    arithmetic, a canonical root committed per block, replayable from genesis.
    No separate contract, no bridge, no trusted mint — the same consensus that
    agrees on the model agrees on who owns what.

  * FAIR LAUNCH (§9.8): the genesis ledger is EMPTY. No premine, no pre-sale.
    Every token in existence is minted by a block reward for verifiable work:
    training deltas, block proposal, or admitted data. The founder's wallet is
    the genesis corpus's data contributor and earns the data share under the
    same rules as any later contributor — a published address, no special path.

  * Emissions halve on a fixed schedule and STOP at the sunset height — the
    §9.3 non-amendable cap. (Mainnet ties the sunset to revenue milestones;
    the reference uses heights so the schedule is testable today.)

Units: integer "grains" (10^9 grains = 1 SESTRIAN; no floats, ever).
"""

import hashlib
import json
from dataclasses import dataclass, field

from .crypto import Key, frame, verify

GRAIN = 10**9                      # grains per whole token

# Emission schedule (reference constants; mainnet sets these at genesis ceremony).
# rev 6 (§9.3 adapted): Bitcoin-shaped halvings, but with a TAIL EMISSION instead
# of a hard sunset-to-zero. The chain's miners do useful work (training) that must
# continue forever, so — like Monero, unlike Bitcoin — the reward floors at the
# final epoch's value permanently: a guaranteed perpetual training wage, with
# inflation asymptoting to ~0%/yr. Schedule supply ≈ 99.9M tokens over the 10
# halving epochs (~19 years at 60s blocks), then ~51k tokens/yr tail (~0.05%/yr,
# declining forever).
BASE_REWARD = 50 * GRAIN           # per block at height 1
HALVING_BLOCKS = 1_000_000         # reward halves every N blocks (~2 yrs at 60s)
TAIL_EPOCH = 9                     # after this many halvings the reward stops
TAIL_REWARD = BASE_REWARD >> TAIL_EPOCH  # ≈0.0977 token/block, forever

# Block-reward split, in basis points (must sum to 10_000)
SHARE_MINERS = 7_000               # split equally among the block's delta miners
SHARE_PROPOSER = 1_000             # the block proposer
SHARE_DATA = 2_000                 # the data contributors whose corpus trained it

# Inference-fee split, in basis points (must sum to 10_000). rev 6: usage revenue
# funds ALL THREE public goods, not just serving — this is what makes "train
# forever" solvent after emission tapers. The server is paid instantly (it bore
# the serving compute, and absorbs division dust so the split is supply-exact);
# the data + training slices accumulate in on-chain pool balances, drained every
# block to that block's provenance-named data owners and delta miners. When
# sketch-based usage attribution lands (§8), only the data pool's distribution
# rule changes — the flows are already live.
FEE_SHARE_SERVER = 6_000           # the serving node, paid instantly
FEE_SHARE_DATA = 2_000             # → fee_data_pool, drained to named data owners
FEE_SHARE_TRAIN = 2_000            # → fee_train_pool, drained to delta miners

# Data lane (rev 3): staked submission + challenge market (§7.2, §9A)
CHALLENGE_WINDOW = 20              # blocks a challenge stays open for votes
PROPOSER_LOOKBACK = 32             # only recent block proposers may vote
GENESIS_DATA_WEIGHT = 1_000_000    # royalty weight of the genesis corpus entry
CHALLENGE_QUORUM = 3               # min affirmative juror votes to uphold a
                                   # challenge — one juror must never be able to
                                   # seize an owner's stake; below quorum the
                                   # challenge is rejected (safe default)
BOND_WINDOW = 20                   # blocks a delta's admission bond stays locked
                                   # (slashable for fraud) before it is returned


def address(pub_hex: str) -> str:
    """A wallet address: sha256 of the raw pubkey bytes, first 20 bytes, hex."""
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:40]


def emission(height: int) -> int:
    """Deterministic block reward at a height. Halves every HALVING_BLOCKS,
    then floors at TAIL_REWARD forever (tail emission — never zero for h>=1)."""
    if height < 1:
        return 0
    return max(BASE_REWARD >> min((height - 1) // HALVING_BLOCKS, 62), TAIL_REWARD)


@dataclass
class TransferTx:
    """A signed balance transfer. Nonce = sender's transfer count (replay-proof)."""
    from_pub: str                  # sender PUBKEY hex (address derives from it)
    to_addr: str                   # recipient address
    amount: int                    # grains
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"transfer", self.from_pub.encode(), self.to_addr.encode(),
                     str(self.amount).encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "TransferTx":
        assert key.pub == self.from_pub, "signer must match from_pub"
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.from_pub, self.signing_bytes(), self.sig)


@dataclass
class DataSubmitTx:
    """Staked data registration (§7.2): the owner wallet stakes behind a
    content-addressed corpus contribution. data_id = txid. The stake is escrowed
    in the registry entry; the entry earns the block data share weighted by its
    stake (v1 proxy — attribution-weighted royalties replace stake-weighting at
    the TRAK integration milestone) and is what a successful challenge takes."""
    owner_pub: str
    data_hash: str                 # sha256 of the corpus bytes (content address)
    size_bytes: int
    media_type: str                # "text" | "csv" | "image" | … (bytes are bytes)
    stake: int                     # grains escrowed behind this submission
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"data_submit", self.owner_pub.encode(), self.data_hash.encode(),
                     str(self.size_bytes).encode(), self.media_type.encode(),
                     str(self.stake).encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "DataSubmitTx":
        assert key.pub == self.owner_pub
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.owner_pub, self.signing_bytes(), self.sig)


@dataclass
class DataChallengeTx:
    """Stake-vs-stake challenge against a registry entry's validity or
    ownership (§7.2 challenge market). Opens a voting window; upheld → the
    entry is revoked and its stake goes to the challenger; rejected → the
    challenger's stake goes to the entry's owner. Either way, lying costs."""
    challenger_pub: str
    data_id: str
    stake: int
    reason: str                    # "validity" | "ownership" | "availability"
                                   # (rev 5/7: availability = the staked bytes are
                                   # no longer retrievable from the DA layer; an
                                   # upheld challenge slashes the stake and revokes
                                   # the entry, so vanished data stops being
                                   # namable by deltas and stops earning)
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"data_challenge", self.challenger_pub.encode(), self.data_id.encode(),
                     str(self.stake).encode(), self.reason.encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "DataChallengeTx":
        assert key.pub == self.challenger_pub
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.challenger_pub, self.signing_bytes(), self.sig)


@dataclass
class DataVoteTx:
    """A vote on an open challenge. Gated: only wallets that PROPOSED one of the
    last PROPOSER_LOOKBACK blocks may vote — juror seats are earned by verifiable
    work, not bought."""
    voter_pub: str
    challenge_id: str
    support: bool                  # True = uphold the challenge
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"data_vote", self.voter_pub.encode(), self.challenge_id.encode(),
                     str(int(self.support)).encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "DataVoteTx":
        assert key.pub == self.voter_pub
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.voter_pub, self.signing_bytes(), self.sig)


@dataclass
class InferenceReceiptTx:
    """A verified fee-bearing inference (§4.2/§8, forward-prop). The PAYER signs
    a fee to the serving node, committing to the attested output and the head
    state root it was served against — the on-chain payment + receipt. The
    server's attestation over the output is off-chain; a bad attestation is
    disputed via the challenge market. This is the revenue lane: usage fees, not
    inflation, fund the network."""
    payer_pub: str
    server_addr: str               # who is paid
    fee: int                       # grains
    output_hash: str               # sha256 of the served output bytes
    head_root: str                 # the weights-state root it was served against
    nonce: int
    # rev 8: the ANSWER SKETCH — the emitted answer's loss-gradient projected
    # through the shared seeded matrix (attribution §8), quantized. The data
    # slice of the fee splits across corpora by positive alignment with their
    # accumulated ledger sketches. Empty = unsketched → the slice pools as
    # before. Committed in the payer's signature, recomputable from the output
    # + the head_root model — a challengeable server claim.
    answer_sketch: list = None
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        sk = [int(x) for x in (self.answer_sketch or [])]
        import json as _json
        return frame(b"inference", self.payer_pub.encode(), self.server_addr.encode(),
                     str(self.fee).encode(), self.output_hash.encode(),
                     self.head_root.encode(), str(self.nonce).encode(),
                     _json.dumps(sk, separators=(",", ":")).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "InferenceReceiptTx":
        assert key.pub == self.payer_pub
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.payer_pub, self.signing_bytes(), self.sig)


class TokenLedger:
    """Balances + nonces + the data registry + open challenges — the full token
    state. Every mutation is deterministic integer math."""

    def __init__(self):
        self.balances: dict[str, int] = {}     # address -> grains
        self.nonces: dict[str, int] = {}       # address -> next expected nonce
        # data_id -> {owner, data_hash, size, media_type, stake, weight, status}
        self.registry: dict[str, dict] = {}
        # challenge_id -> {data_id, challenger, stake, reason, expiry,
        #                  votes_for: [addr…], votes_against: [addr…]}
        self.challenges: dict[str, dict] = {}
        # delta_txid -> {miner, amount, expiry} — stake bonds locked behind
        # included deltas (the admission cost; slashable for proven fraud,
        # otherwise returned at maturity). §4.1
        self.bonds: dict[str, dict] = {}
        # rev 6: inference-fee slices awaiting distribution — the data slice
        # drains to the next block's provenance-named data owners, the training
        # slice to its delta miners. Consensus state (in root + supply).
        self.fee_data_pool: int = 0
        self.fee_train_pool: int = 0

    def copy(self) -> "TokenLedger":
        led = TokenLedger()
        led.balances = dict(self.balances)
        led.nonces = dict(self.nonces)
        led.registry = {k: dict(v) for k, v in self.registry.items()}
        led.challenges = {k: {**v, "votes_for": list(v["votes_for"]),
                              "votes_against": list(v["votes_against"])}
                          for k, v in self.challenges.items()}
        led.bonds = {k: dict(v) for k, v in self.bonds.items()}
        led.fee_data_pool = self.fee_data_pool
        led.fee_train_pool = self.fee_train_pool
        return led

    def seed_genesis_data(self, owner_addr: str, data_hash: str = "genesis"):
        """The founding corpus as registry entry zero — owned by the founder's
        wallet, earning the data share under the same rules as any entry (its
        weight is a published genesis parameter; stake 0 because no tokens exist
        before block 1 — fair launch has nothing to stake with)."""
        self.registry["genesis"] = {
            "owner": owner_addr, "data_hash": data_hash, "size": 0,
            "media_type": "text", "stake": 0,
            "weight": GENESIS_DATA_WEIGHT, "status": "active"}

    def balance(self, addr: str) -> int:
        return self.balances.get(addr, 0)

    def _credit(self, addr: str, amount: int):
        if amount > 0:
            self.balances[addr] = self.balances.get(addr, 0) + amount

    # ---- block reward ----------------------------------------------------
    def apply_reward(self, height: int, miner_pubs: list[str],
                     proposer_pub: str, data_addrs: list[str] = (),
                     *, data_credits: dict[str, int] = None,
                     miner_weights: dict[str, int] = None):
        """Mint the block's emission and split it. Integer division truncates;
        the remainder (dust) is deliberately burned — supply never exceeds the
        schedule. Deterministic given identical inputs on every node.

        PROVENANCE PAYOUT (rev 5): the data share goes to the owners of the data
        THIS block's deltas actually named, weighted by `data_credits` — a
        {data_hash: weight} map the caller derives from the block's deltas (each
        named corpus's contribution weight; interim weight = registry stake-weight
        until loss-scoring supplies the real per-delta score). Only entries that
        are `active` in the registry and carry positive weight are paid; an
        unbacked hash pays nobody. This replaces the rev-3 behaviour of paying
        every registered entry every block. `data_addrs` is a legacy fallback
        (pre-rev-3 chains with no provenance)."""
        total = emission(height)
        if total == 0 and not self.fee_train_pool and not self.fee_data_pool:
            return
        miners_pool = total * SHARE_MINERS // 10_000
        proposer_cut = total * SHARE_PROPOSER // 10_000
        data_pool = total * SHARE_DATA // 10_000
        # rev 6: drain the fee pools into this block's payouts when they have
        # recipients (a block without miners / named data carries them forward).
        # Division dust is burned, same doctrine as emission dust.
        if miner_pubs and self.fee_train_pool:
            miners_pool += self.fee_train_pool
            self.fee_train_pool = 0
        if miner_pubs:
            # rev 7: split ∝ committed delta score when weights are given (and
            # nonzero); equal split otherwise. Dust burned either way.
            weights = {p: (miner_weights or {}).get(p, 0) for p in miner_pubs}
            wsum = sum(weights.values())
            if wsum > 0:
                for pub in sorted(set(miner_pubs)):        # canonical order
                    self._credit(address(pub), miners_pool * weights[pub] // wsum)
            else:
                each = miners_pool // len(miner_pubs)
                for pub in sorted(miner_pubs):
                    self._credit(address(pub), each)
        if proposer_pub and proposer_pub != "genesis":
            self._credit(address(proposer_pub), proposer_cut)
        # data share → owners named by this block's deltas, ∝ contribution weight.
        # Resolve each named data_hash to its active registry entry (the on-chain
        # availability proxy); unknown/inactive hashes are dropped.
        hash_to_owner = {e["data_hash"]: e["owner"]
                         for e in self.registry.values() if e["status"] == "active"}
        paid = {h: w for h, w in (data_credits or {}).items()
                if w > 0 and h in hash_to_owner}
        if paid and self.fee_data_pool:
            data_pool += self.fee_data_pool
            self.fee_data_pool = 0
        if paid:
            wsum = sum(paid.values())
            for h in sorted(paid):                         # ∝ weight, dust burned
                self._credit(hash_to_owner[h], data_pool * paid[h] // wsum)
        elif data_addrs:                                   # legacy fallback
            each = data_pool // len(data_addrs)
            for addr in sorted(data_addrs):
                self._credit(addr, each)

    # ---- delta admission bonds (rev 4) ----------------------------------
    def resolve_expired_bonds(self, height: int):
        """Return every stake bond whose lock window has closed, in sorted txid
        order (deterministic). A bond slashed by a proven-fraud challenge is
        removed before maturity elsewhere; the rest come back to the miner."""
        for tid in sorted(self.bonds):
            b = self.bonds[tid]
            if b["expiry"] <= height:
                self._credit(b["miner"], b["amount"])
                del self.bonds[tid]

    def lock_bond(self, delta_txid: str, miner_addr: str, amount: int, height: int) -> bool:
        """Lock a delta's admission bond from the miner's balance — the Bitcoin
        analog of paying to participate, but recoverable. Returns False (block
        invalid) if the miner can't afford it. A zero bond is a no-op, so the
        fair-launch bootstrap (miners with no balance yet) still works."""
        if amount <= 0:
            return True
        if self.balances.get(miner_addr, 0) < amount:
            return False
        self.balances[miner_addr] -= amount
        self.bonds[delta_txid] = {"miner": miner_addr, "amount": amount,
                                  "expiry": height + BOND_WINDOW}
        return True

    # ---- data lane (rev 3) ----------------------------------------------
    def resolve_expired_challenges(self, height: int):
        """Deterministically settle every challenge whose window has closed
        (processed FIRST in each block, in sorted challenge_id order).
        Upheld (more support than opposition, at least one vote): the entry is
        revoked, its escrowed stake goes to the challenger, the challenger's
        stake returns. Rejected (ties, no votes, or opposition wins): the
        challenger's stake goes to the entry's owner."""
        for cid in sorted(self.challenges):
            ch = self.challenges[cid]
            if ch["expiry"] > height:
                continue
            entry = self.registry.get(ch["data_id"])
            # QUORUM: a challenge is upheld only with a strict majority AND at
            # least CHALLENGE_QUORUM affirmative juror votes. Below quorum (too
            # few disinterested jurors showed up) it is rejected — the challenger
            # cannot seize stake on a thin or single vote.
            upheld = (len(ch["votes_for"]) >= CHALLENGE_QUORUM
                      and len(ch["votes_for"]) > len(ch["votes_against"]))
            if upheld and entry is not None:
                entry["status"] = "revoked"
                self._credit(ch["challenger"], entry["stake"] + ch["stake"])
                entry["stake"] = 0
            elif entry is not None:
                self._credit(entry["owner"], ch["stake"])
            del self.challenges[cid]

    def apply_data_tx(self, tx, height: int, recent_proposers: set[str]) -> bool:
        """Validate + apply one data-lane tx. False = invalid (block invalid)."""
        if not tx.verify():
            return False
        src = address(tx.owner_pub if isinstance(tx, DataSubmitTx)
                      else tx.challenger_pub if isinstance(tx, DataChallengeTx)
                      else tx.payer_pub if isinstance(tx, InferenceReceiptTx)
                      else tx.voter_pub)
        if tx.nonce != self.nonces.get(src, 0):
            return False
        if isinstance(tx, DataSubmitTx):
            if tx.stake <= 0 or self.balances.get(src, 0) < tx.stake:
                return False
            if tx.txid() in self.registry:
                return False
            self.balances[src] -= tx.stake                 # escrowed in the entry
            self.registry[tx.txid()] = {
                "owner": src, "data_hash": tx.data_hash, "size": tx.size_bytes,
                "media_type": tx.media_type, "stake": tx.stake,
                "weight": tx.stake, "status": "active"}
        elif isinstance(tx, DataChallengeTx):
            entry = self.registry.get(tx.data_id)
            if (entry is None or entry["status"] != "active" or tx.stake <= 0
                    or self.balances.get(src, 0) < tx.stake
                    or any(c["data_id"] == tx.data_id for c in self.challenges.values())):
                return False
            self.balances[src] -= tx.stake
            self.challenges[tx.txid()] = {
                "data_id": tx.data_id, "challenger": src, "stake": tx.stake,
                "reason": tx.reason, "expiry": height + CHALLENGE_WINDOW,
                "votes_for": [], "votes_against": []}
        elif isinstance(tx, DataVoteTx):
            ch = self.challenges.get(tx.challenge_id)
            if (ch is None or tx.voter_pub not in recent_proposers
                    or src in ch["votes_for"] or src in ch["votes_against"]):
                return False
            # DISINTERESTED JURORS ONLY: neither the challenger nor the data
            # owner may vote on their own challenge — both have a direct stake
            # in the outcome. Jurors are disinterested recent proposers.
            if src == ch["challenger"]:
                return False
            entry = self.registry.get(ch["data_id"])
            if entry is not None and src == entry["owner"]:
                return False
            (ch["votes_for"] if tx.support else ch["votes_against"]).append(src)
            ch["votes_for"].sort(); ch["votes_against"].sort()   # canonical
        elif isinstance(tx, InferenceReceiptTx):
            # a signed usage fee: the payer pays for an attested inference. rev 6:
            # the fee splits three ways — the serving node is paid instantly (and
            # absorbs division dust, keeping the split supply-exact); the data and
            # training slices accumulate in the fee pools, drained by the next
            # block's reward to its provenance-named data owners + delta miners.
            # This is what funds training + data from USAGE once emission tapers.
            if tx.fee <= 0 or self.balances.get(src, 0) < tx.fee:
                return False
            self.balances[src] -= tx.fee
            data_cut = tx.fee * FEE_SHARE_DATA // 10_000
            train_cut = tx.fee * FEE_SHARE_TRAIN // 10_000
            self._credit(tx.server_addr, tx.fee - data_cut - train_cut)
            self.fee_train_pool += train_cut
            # rev 8 USAGE ATTRIBUTION: if the receipt carries an answer sketch,
            # the data slice pays the corpora whose accumulated ledger sketches
            # POSITIVELY align with it (∝ dot product — data that pushed against
            # the answer earns nothing), directly, this receipt. Unsketched
            # receipts / no positive alignment → the slice pools as before.
            # Deterministic integer arithmetic; big-int dots (Rust: i128).
            paid_direct = False
            ans = [int(x) for x in (tx.answer_sketch or [])]
            if any(ans):
                aligns = {}
                for e in self.registry.values():
                    if e["status"] != "active":
                        continue
                    sk = e.get("sketch")
                    if not sk:
                        continue
                    d = sum(a * b for a, b in zip(sk, ans))
                    if d > 0:
                        aligns[e["owner"]] = aligns.get(e["owner"], 0) + d
                total = sum(aligns.values())
                if total > 0:
                    for owner in sorted(aligns):           # canonical, dust burned
                        self._credit(owner, data_cut * aligns[owner] // total)
                    paid_direct = True
            if not paid_direct:
                self.fee_data_pool += data_cut
        else:
            return False
        self.nonces[src] = tx.nonce + 1
        return True

    # ---- transfers -------------------------------------------------------
    def apply_transfer(self, tx: TransferTx) -> bool:
        """Validate and apply. Returns False (no-op) on any invalid condition —
        a block containing an invalid transfer is itself invalid upstream."""
        if not tx.verify() or tx.amount <= 0:
            return False
        src = address(tx.from_pub)
        if tx.nonce != self.nonces.get(src, 0):
            return False
        if self.balances.get(src, 0) < tx.amount:
            return False
        self.balances[src] -= tx.amount
        self._credit(tx.to_addr, tx.amount)
        self.nonces[src] = tx.nonce + 1
        return True

    # ---- commitment ------------------------------------------------------
    def apply_transfers(self, transfers: list["TransferTx"]) -> bool:
        """Apply a block's transfers in CANONICAL order. All must apply cleanly;
        returns False (ledger unchanged is NOT guaranteed — copy first) if any
        fails. Validators call this on a copy of the parent ledger."""
        for tx in canonical_transfers(transfers):
            if not self.apply_transfer(tx):
                return False
        return True

    def root(self) -> str:
        """Canonical ledger root: sorted compact JSON over the FULL token state
        (balances, nonces, data registry, open challenges). sort_keys sorts every
        nested dict; vote lists are kept sorted at mutation time."""
        blob = json.dumps({"balances": self.balances, "bonds": self.bonds,
                           "challenges": self.challenges,
                           "fee_data_pool": self.fee_data_pool,
                           "fee_train_pool": self.fee_train_pool,
                           "nonces": self.nonces, "registry": self.registry},
                          sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def supply(self) -> int:
        # pool balances are minted/paid tokens in flight, so they count
        return sum(self.balances.values()) + self.fee_data_pool + self.fee_train_pool


def _tx_sender(tx) -> str:
    if isinstance(tx, TransferTx):
        return address(tx.from_pub)
    if isinstance(tx, DataSubmitTx):
        return address(tx.owner_pub)
    if isinstance(tx, DataChallengeTx):
        return address(tx.challenger_pub)
    if isinstance(tx, InferenceReceiptTx):
        return address(tx.payer_pub)
    return address(tx.voter_pub)


def canonical_transfers(transfers: list[TransferTx]) -> list[TransferTx]:
    """The consensus ordering of a block's transfers: by (sender address, nonce,
    txid). Sender-then-nonce guarantees a sender's nonce sequence applies in
    order; txid breaks any remaining tie deterministically."""
    return sorted(transfers, key=lambda t: (address(t.from_pub), t.nonce, t.txid()))


def canonical_account_txs(data_txs: list, transfers: list[TransferTx]) -> list:
    """The consensus ordering of ALL account transactions in a block (data lane
    + transfer lane merged): (sender address, nonce, txid). One nonce sequence
    per account totally orders everything a wallet does."""
    return sorted(list(data_txs) + list(transfers),
                  key=lambda t: (_tx_sender(t), t.nonce, t.txid()))


def data_root(data_txs: list) -> str:
    """Order-independent commitment to a block's data-lane tx set."""
    joined = "|".join(sorted(t.txid() for t in data_txs))
    return hashlib.sha256(joined.encode()).hexdigest()


def transfer_root(transfers: list[TransferTx]) -> str:
    """Order-independent commitment to a block's transfer set (mirror of
    blockchain.txset_root)."""
    joined = "|".join(sorted(t.txid() for t in transfers))
    return hashlib.sha256(joined.encode()).hexdigest()
