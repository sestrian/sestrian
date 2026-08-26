"""The DiLoCo bridge — where GPU training meets deterministic consensus (§6).

A miner trains the real model locally for H inner steps on its own GPU (float,
non-deterministic, unconstrained — §6.3), producing a *pseudo-gradient*: the
weight change over those H steps. That delta is flattened and quantised to
int64, and from there the chain takes over: it aggregates quantised deltas with
the same deterministic fixed-point arithmetic no matter which GPU produced them
(rig/chain.py), so consensus is bit-exact across a 2080 Ti, an M3, and a CPU
alike. Applying the aggregated delta is one DiLoCo outer step; the block cadence
is the outer-sync period.

flat_params / set_flat_params move between the torch model and the flat vector
the chain speaks. Only learnable parameters cross — architecture buffers (masks)
are fixed everywhere.
"""

import numpy as np
import torch

from rig.chain import dequantize, quantize, trimmed_mean_int


def flat_params(model) -> np.ndarray:
    """Flatten learnable parameters to one float64 vector, in module order.
    (Move to CPU first — MPS has no float64.)"""
    return np.concatenate([p.detach().cpu().double().numpy().ravel()
                           for p in model.parameters()])


def set_flat_params(model, vec: np.ndarray):
    """Load a flat float vector back into the model's learnable parameters."""
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = torch.from_numpy(vec[i:i + n].reshape(p.shape)).to(p.dtype).to(p.device)
            p.copy_(chunk)
            i += n


class DiLoCoMiner:
    """Trains the real model and emits quantised pseudo-gradient deltas."""

    def __init__(self, model, data, device, inner_lr=3e-4):
        self.model = model
        self.data = data
        self.device = device
        self.inner_lr = inner_lr

    def sync(self, base_int: np.ndarray):
        """Adopt the chain's current weights (int64) into the local model."""
        set_flat_params(self.model, dequantize(base_int))

    def inner_train(self, steps: int, batch_size: int = 32, seed: int = 0, shard=None,
                    between_steps=None):
        """Run H local steps on the (beacon-)assigned data shard; return the
        quantised pseudo-gradient delta (int64).

        `between_steps`, if given, is called after each optimizer step. It exists
        so a bridge can answer a chat request without waiting out a whole round:
        a round is ~24 steps of ~2s, so the worst-case wait drops from the full
        round to a single step. It is called at the ONE point where the model is
        in a clean state — gradients applied, nothing half-written — which is
        why this is a callback rather than a thread. Concurrency here would read
        torn weights and flip the module out of train mode mid-round, and a
        delta computed that way is one no validator can reproduce.
        """
        base = flat_params(self.model)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.inner_lr)
        gen = torch.Generator().manual_seed(seed)
        last = 0.0
        for _ in range(steps):
            x, y = self.data.get_batch("train", batch_size, generator=gen, shard=shard)
            _, loss = self.model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item()
            if between_steps is not None:
                between_steps()
        delta = flat_params(self.model) - base
        return quantize(delta), last


def outer_apply(base_int: np.ndarray, deltas_int: list) -> np.ndarray:
    """One DiLoCo outer step = the chain's block transition: base + robust-mean(Δ)."""
    if not deltas_int:
        return base_int.copy()
    return base_int + trimmed_mean_int(deltas_int)
