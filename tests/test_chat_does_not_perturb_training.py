"""Serving chat between training steps must not change the delta.

A miner's delta has to be reproducible: a validator re-runs the same seeded
round and compares, and a delta nobody can reproduce is slashable. Once the
bridge answers chat BETWEEN inner steps, anything generation touches becomes
part of the training round's environment — so sampling has to draw from its own
Generator rather than the global stream.

Today the hazard is not reachable: `inner_train` takes batches from an isolated
Generator and the shipped configs run dropout=0.0, so training consumes no
global RNG at all. That is exactly why this test forces dropout ON. Without it
the control passes for the wrong reason, and the day someone sets dropout in a
config the breakage would be silent, remote, and expensive.
"""

import numpy as np
import torch

from client.moe import MoEGPT, MoEGPTConfig
from client.trainer import DiLoCoMiner

# Small enough to be fast, dropout ON so the training step actually draws from
# the global RNG and can therefore be perturbed.
CFG = MoEGPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=64,
                   n_experts=4, e_max=8, top_k=2, dropout=0.1)


class _Data:
    """Batches come from the caller's generator, mirroring the real loader."""

    def get_batch(self, split, bs, generator=None, shard=None):
        x = torch.randint(0, 256, (bs, 16), generator=generator)
        return x, x.clone()


def _round(chat: bool, isolated: bool):
    torch.manual_seed(999)                       # identical global RNG start
    model = MoEGPT(CFG)
    miner = DiLoCoMiner(model, _Data(), "cpu")
    gen = torch.Generator()
    gen.manual_seed(7)

    def between_steps():
        if not chat:
            return
        idx = torch.tensor([[104, 105]], dtype=torch.long)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model.generate(idx, 4, temperature=0.85,
                           generator=(gen if isolated else None))
        model.train(was_training)

    delta, _ = miner.inner_train(4, 4, seed=42, between_steps=between_steps)
    return np.asarray(delta)


def test_interleaved_chat_leaves_the_delta_identical():
    baseline = _round(chat=False, isolated=True)
    with_chat = _round(chat=True, isolated=True)
    assert np.array_equal(baseline, with_chat), (
        "answering chat between inner steps changed the delta — a validator "
        "replaying this round would disagree with the miner")


def test_sampling_from_the_global_rng_would_break_it():
    """The guard rail's own guard rail.

    If this ever passes, the isolation above has stopped being load-bearing and
    the test above proves nothing.
    """
    baseline = _round(chat=False, isolated=True)
    leaky = _round(chat=True, isolated=False)
    assert not np.array_equal(baseline, leaky), (
        "generation drawing from the GLOBAL rng no longer perturbs training, so "
        "this pair of tests can no longer detect the failure it guards")
