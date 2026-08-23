"""Minimal model-chain: weights as chain state, deltas as transactions.

Implements the determinism boundary of WHITEPAPER §6.3: miners' inner loops
may use arbitrary float math, but everything consensus touches — quantization,
robust aggregation, the state transition — is int64 fixed-point arithmetic in
a canonical order, so any node replaying the chain reproduces bit-identical
state (§3.4, §3.5).
"""

import hashlib

import numpy as np

SCALE = 1 << 16  # fixed-point scale for weights and deltas


def quantize(delta_f: np.ndarray) -> np.ndarray:
    """Float delta -> int64 fixed-point. This is the miner's last float op."""
    return np.round(np.asarray(delta_f, dtype=np.float64) * SCALE).astype(np.int64)


def dequantize(x_int: np.ndarray) -> np.ndarray:
    return x_int.astype(np.float64) / SCALE


def state_root(w_int: np.ndarray) -> str:
    """Merkle root stand-in: hash of the canonical weight bytes (§3.1)."""
    return hashlib.sha256(w_int.tobytes()).hexdigest()


def beacon(height: int, tag: str, salt: int = 0) -> np.random.Generator:
    """Unbiasable-randomness stand-in (§7.4): deterministic per (height, tag)."""
    digest = hashlib.sha256(f"sestrian|{height}|{tag}|{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def trimmed_mean_int(deltas_int: list[np.ndarray], trim: float = 0.2) -> np.ndarray:
    """Byzantine-robust aggregation (§3.4) in deterministic integer arithmetic.

    Elementwise sort, drop the top/bottom `trim` fraction, integer-mean the
    core with floor division. Bounds the influence of any minority of
    adversarial deltas that scored their way into the block (§5.5).
    """
    arr = np.stack(deltas_int).astype(np.int64)
    arr = np.sort(arr, axis=0)
    k = arr.shape[0]
    lo = int(np.floor(k * trim))
    # Byzantine robustness at LOW miner counts: with the plain 0.2 fraction, k in
    # {3,4} trims nothing and a single adversarial delta is averaged straight in.
    # Once there are >= 3 deltas, always drop >= 1 from each end so a lone
    # outlier can never dominate. (k < 3 cannot be made robust — you need >= 3
    # participants to outvote one; that regime is mitigated by launching
    # invite-only and treating < 3-miner blocks as untrusted.)
    if k >= 3:
        lo = max(1, lo)
    hi = k - lo
    core = arr[lo:hi] if hi > lo else arr
    return np.floor_divide(core.sum(axis=0, dtype=np.int64), core.shape[0])


def paged_transition(parent_w_int: np.ndarray, bodies: list[np.ndarray],
                     claims: list[list[int]],
                     spans: list[tuple[int, int]]) -> np.ndarray:
    """Protocol v1 state transition: per-page trimmed mean over each page's
    ACTUAL claimants. For page p, the contributor set is the bodies whose claim
    set includes p, sliced to p's span; pages nobody claimed are unchanged.
    When every body claims every page this reduces exactly to the global
    trimmed-mean transition. Claimant order is irrelevant (elementwise sort
    inside trimmed_mean_int), so this is deterministic from block content.
    """
    w = parent_w_int.copy()
    claim_sets = [set(c) for c in claims]
    for page_id, (start, end) in enumerate(spans):
        contributors = [b[start:end] for b, c in zip(bodies, claim_sets)
                        if page_id in c]
        if contributors:
            w[start:end] = w[start:end] + trimmed_mean_int(contributors)
    return w


class Block:
    def __init__(self, height: int, deltas_int: list[np.ndarray],
                 miner_ids: list[int], root: str):
        self.height = height
        self.deltas_int = [d.copy() for d in deltas_int]  # DA-layer bodies (§3.3)
        self.miner_ids = list(miner_ids)
        self.root = root  # weights_state_root of the post-transition state


class Chain:
    """The ledger. State = int64 weights; transition = OuterStep(RobustAggregate)."""

    def __init__(self, w0_int: np.ndarray):
        self.genesis_int = w0_int.copy()
        self.w_int = w0_int.copy()
        self.blocks: list[Block] = []

    @property
    def height(self) -> int:
        return len(self.blocks)

    def weights(self) -> np.ndarray:
        return dequantize(self.w_int)

    def apply_block(self, deltas_int: list[np.ndarray], miner_ids: list[int]) -> Block:
        if deltas_int:
            self.w_int = self.w_int + trimmed_mean_int(deltas_int)
        block = Block(self.height + 1, deltas_int, miner_ids, state_root(self.w_int))
        self.blocks.append(block)
        return block

    def replay(self, exclude_miner_ids: set[int] | None = None) -> np.ndarray:
        """Reconstruct state from genesis by re-applying recorded deltas (§3.5).

        With `exclude_miner_ids`, performs the §7.2/§10.4 excision: re-execute
        history with the offending contributions removed.
        """
        exclude = exclude_miner_ids or set()
        w_int = self.genesis_int.copy()
        for block in self.blocks:
            kept = [d for d, m in zip(block.deltas_int, block.miner_ids)
                    if m not in exclude]
            if kept:
                w_int = w_int + trimmed_mean_int(kept)
        return w_int
