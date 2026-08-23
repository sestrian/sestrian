"""Verifiable proposer sortition (§7.4, protocol v1) — the SPEC for the Rust node.

Fixed round-robin rotation can't work on an open network (who is `id` 2 of n?)
and a single scheduled proposer being offline stalls the slot. Stake-weighted,
verifiable eligibility replaces it — now ENFORCED in validation (v1), with a
deterministic liveness fallback so a tiny fleet can never stall the chain:

  * The per-height seed binds to (parent hash, height, ATTEMPT). A proposer's
    VRF proof is a DETERMINISTIC Ed25519 signature over that seed — only the
    key holder can produce it, anyone can verify it, and it is unique per
    (key, height, attempt). Each attempt is an honest re-roll.
  * Eligibility: vrf_output < 2^256 · TARGET_PROPOSERS · stake/total_stake ·
    2^attempt. The threshold doubles per attempt, so if no proposer qualifies
    at attempt 0 the eligible set widens deterministically instead of the
    height stalling. Two absolute rules, both deterministic:
      - total_stake == 0 (cold start; the genesis ledger is empty by fair
        launch): any verifying proof is eligible at attempt 0.
      - attempt == ATTEMPT_MAX: any verifying proof is eligible — the
        liveness floor for a 2-miner devnet.
  * Fork-choice weight: header.work MUST equal max(1, vrf_work(proof) >>
    attempt) — committed and validated, so a low-attempt (prompt) proposer
    strictly tends to dominate reorgs over a high-attempt straggler.

Deterministic and self-verifying: given the header's VRF proof and attempt,
every node recomputes the same eligibility decision. (The threshold-BLS beacon
in rig/beacon.py remains the unbiasable upgrade that also removes the
proposer's ability to grind the parent hash.)
"""

import hashlib

from .crypto import verify

TARGET_PROPOSERS = 2          # expected eligible proposers per height, attempt 0
ATTEMPT_MAX = 16              # unconditional-eligibility liveness floor
_TWO256 = 1 << 256


def seed(prev_hash: str, height: int, attempt: int = 0) -> bytes:
    return hashlib.sha256(
        f"sestrian-lottery|{prev_hash}|{height}|{attempt}".encode()).digest()


def vrf_prove(key, prev_hash: str, height: int, attempt: int = 0) -> bytes:
    """The proposer's VRF proof: a deterministic signature over the seed."""
    return key.sign(seed(prev_hash, height, attempt))


def vrf_output(proof: bytes) -> int:
    """A uniform value in [0, 2^256) that only the key holder could have produced."""
    return int.from_bytes(hashlib.sha256(proof).digest(), "big")


def vrf_work(proof: bytes) -> int:
    """Raw fork-choice weight from the VRF output: leading zero bits + 1 (>= 1).
    A luckier (smaller) output yields more work, and work is NON-FORGEABLE (one
    VRF per proposer per height per attempt)."""
    out = vrf_output(proof)
    lz = 256 - out.bit_length() if out > 0 else 256
    return lz + 1


def attempt_work(proof: bytes, attempt: int) -> int:
    """The committed header.work: the raw VRF work discounted by the attempt
    (>> attempt), floored at 1 — validation requires exact equality."""
    return max(1, vrf_work(proof) >> attempt)


def threshold(stake: int, total_stake: int, attempt: int = 0) -> int:
    """Eligible iff vrf_output < 2^256 · TARGET · stake/total_stake · 2^attempt.
    total_stake == 0 -> cold-start: threshold is the full range (see eligible)."""
    if total_stake <= 0:
        return _TWO256
    if stake <= 0:
        return 0
    return min(_TWO256,
               (_TWO256 * TARGET_PROPOSERS * stake << attempt) // total_stake)


def eligible(pub: str, proof: bytes, prev_hash: str, height: int, attempt: int,
             stake: int, total_stake: int) -> bool:
    """True iff `proof` is a valid VRF proof by `pub` for (height, attempt) AND
    it clears the stake-weighted, attempt-widened threshold (or a rule applies:
    cold start at attempt 0, or the ATTEMPT_MAX liveness floor)."""
    if not 0 <= attempt <= ATTEMPT_MAX:
        return False
    if not verify(pub, seed(prev_hash, height, attempt), proof):
        return False
    if attempt == ATTEMPT_MAX:
        return True                       # absolute liveness floor
    if total_stake <= 0:
        return attempt == 0               # cold start: everyone, first try
    return vrf_output(proof) < threshold(stake, total_stake, attempt)
