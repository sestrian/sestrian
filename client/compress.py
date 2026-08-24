"""Delta compression — top-k sparse + quantized (WHITEPAPER §6).

A dense pseudo-gradient over an 86M model is ~688 MB as int64 — far too much to
gossip over a volunteer's uplink each round. DiLoCo/DisTrO showed you don't need
all of it: the pseudo-gradient is compressible by 100×+ with little convergence
loss. We take the simplest robust version — transmit only the top-k
largest-magnitude quantised components as (index, value) pairs — which is small
on the wire yet **densifies back to an exact int64 vector**, so the chain's
deterministic trimmed-mean aggregation (and its Byzantine robustness) is
unchanged: compression is a transmission concern, not a consensus one.

  compress(delta, keep_frac) -> a small serialisable payload
  decompress(payload)        -> the dense int64 delta the chain aggregates
"""

import numpy as np

from rig.chain import quantize


def compress(delta: np.ndarray, keep_frac: float = 0.02, min_keep: int = 0,
             max_keep: int = 0):
    """Quantise a float delta and keep only its top-k components by magnitude.

    Returns a payload dict with uint32 indices + int32 values (the quantised
    delta clamps comfortably into int32). Ties broken by index → deterministic.
    `min_keep` (protocol v1): the consensus WORK QUOTA is a floor on a delta's
    nonzero coordinates, so the node passes its required_nnz here and the
    compressor keeps at least that many components regardless of keep_frac.
    `max_keep` (protocol v2): the delta ENVELOPE is a hard consensus CEILING on
    nonzeros — the payload never scales with quota. 0 = uncapped.
    """
    q = quantize(delta)                                   # int64, deterministic
    n = q.size
    k = max(1, int(n * keep_frac), int(min_keep))
    if max_keep:
        k = min(k, int(max_keep))
    if k >= n:
        idx = np.nonzero(q)[0].astype(np.uint32)
    else:
        # indices of the k largest |q|; stable by index for determinism
        part = np.argpartition(np.abs(q), n - k)[n - k:]
        idx = np.sort(part).astype(np.uint32)
    vals = q[idx]
    if np.any(np.abs(vals) > np.iinfo(np.int32).max):
        raise ValueError("delta component exceeds int32 — lower the quant scale")
    return {"n": int(n), "idx": idx.tobytes(), "val": vals.astype(np.int32).tobytes()}


def decompress(payload) -> np.ndarray:
    """Reconstruct the dense int64 delta (zeros except the kept components)."""
    out = np.zeros(payload["n"], dtype=np.int64)
    idx = np.frombuffer(payload["idx"], dtype=np.uint32)
    val = np.frombuffer(payload["val"], dtype=np.int32).astype(np.int64)
    out[idx] = val
    return out


class Compressor:
    """Top-k compression with ERROR FEEDBACK — the components dropped this round
    are carried in a residual and sent later, so aggressive compression loses no
    signal in the limit (the DisTrO/DeMo trick). One per miner (stateful)."""

    def __init__(self, keep_frac: float = 0.02):
        self.keep_frac = keep_frac
        self.residual = None            # float, same shape as the delta

    def compress(self, delta: np.ndarray, min_keep: int = 0, max_keep: int = 0):
        if self.residual is None or self.residual.shape != delta.shape:
            # first round, or the model GREW (protocol v1): restart the residual
            self.residual = np.zeros_like(delta)
        full = delta + self.residual                 # this round + what we owe
        payload = compress(full, self.keep_frac, min_keep, max_keep)
        sent = decompress(payload).astype(np.float64) / (1 << 16)  # what actually went
        self.residual = full - sent                  # carry the remainder forward
        return payload


def payload_bytes(payload) -> int:
    return len(payload["idx"]) + len(payload["val"]) + 8


def ratio(delta: np.ndarray, payload) -> float:
    """Compression ratio vs the raw int64 delta."""
    return (delta.size * 8) / payload_bytes(payload)
