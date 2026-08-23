"""DiPaCo (client/dipaco.py): a worker holds only its path and runs the whole
forward/backward LOCALLY (no other module needed — hence no pipeline), workers
on different domains jointly train one model via module-DiLoCo, and the composed
model (route each domain to its path) beats misrouting because paths specialize.

Hermetic: tiny per-domain repeating-byte tasks (distinct periods), tiny model."""

import numpy as np
import torch

from client.dipaco import (
    DiPaCoConfig, DiPaCoGPT, PathMap, build_dipaco, coarse_route, make_path,
)
from client.trainer import flat_params, set_flat_params
from rig.chain import dequantize, paged_transition, quantize


def _runs(mask: np.ndarray):
    """Contiguous (start, end) runs of a boolean mask — span-ifies a PathMap
    holding so the v1 per-page consensus aggregation can be applied to it."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    brk = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk + 1, [idx.size]])
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


def _v1_aggregate(base_int, deltas, masks):
    """The protocol-v1 replacement for the old shard_aggregate: express each
    holder's mask as page spans + a claim set and run the CHAIN's per-page
    transition (trimmed mean over actual claimants — k=1 applies in full,
    k=2 plain mean), so these sims exercise the real consensus rule."""
    spans = sorted({r for m in masks for r in _runs(m)})
    claims = [[i for i, (s, e) in enumerate(spans) if m[s:e].all()]
              for m in masks]
    return paged_transition(base_int, deltas, claims, spans)


CFG = DiPaCoConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, n_modules=4)
N_DOMAINS = 4
DISJOINT = [[p] * CFG.n_layer for p in range(N_DOMAINS)]   # path p = module p everywhere


def _domain_buf(d, n=4096):
    """Domain d is a repeating stream of a domain-specific period — a genuinely
    different next-byte function per domain, so specialization is measurable."""
    period = 5 + d
    return (np.arange(n) % period).astype(np.int64)


def _batch(buf, bs, T, gen):
    ix = torch.randint(0, len(buf) - T - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(buf[i:i + T]) for i in ix])
    y = torch.stack([torch.from_numpy(buf[i + 1:i + 1 + T]) for i in ix])
    return x, y


def _train_path(model, buf, path, base_vec, steps=40, seed=0):
    set_flat_params(model, base_vec)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x, y = _batch(buf, 16, CFG.block_size, gen)
        _, loss = model(x, y, path=path)          # trains only this path's modules
        opt.zero_grad(); loss.backward(); opt.step()
    return quantize(flat_params(model) - base_vec)


def _loss(model, buf, path, vec):
    set_flat_params(model, vec)
    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        x, y = _batch(buf, 64, CFG.block_size, gen)
        _, loss = model(x, y, path=path)
    return loss.item()


def test_forward_touches_only_its_path_modules():
    """A worker's forward/backward uses ONLY its path's modules — every other
    module gets no gradient, so it is genuinely not needed. This is why DiPaCo
    needs no pipeline: the slice is self-contained."""
    torch.manual_seed(0)
    model = DiPaCoGPT(CFG)
    path = [0, 1]                                  # module 0 at level 0, module 1 at level 1
    x = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    _, loss = model(x, x, path=path)
    model.zero_grad(); loss.backward()
    for l, block in enumerate(model.blocks):
        for m, mod in enumerate(block.mods):
            g = mod.fc.weight.grad
            if m == path[l]:
                assert g is not None and g.abs().sum() > 0     # the path's module trained
            else:
                assert g is None or g.abs().sum() == 0         # others untouched


def test_worker_holds_only_a_subset():
    model = DiPaCoGPT(CFG)
    pm = PathMap(model)
    frac = pm.hold_fraction([0, 0])               # backbone + one module per level
    assert frac < 1.0                             # not the whole model
    # holding one path omits the other (M-1) modules per level
    mask = pm.mask([0, 0])
    assert not mask.all()


def test_disjoint_domain_workers_compose_and_beat_misrouting():
    """Four workers, each holds one path and trains on one domain; their masked
    deltas aggregate (module-DiLoCo). The composed model routed correctly beats
    the same model misrouted — the paths specialized to their domains."""
    model, _ = build_dipaco(CFG, device="cpu", seed=11)
    pm = PathMap(model)
    base = flat_params(model)
    base_int = quantize(base)

    deltas, masks = [], []
    for d in range(N_DOMAINS):
        path = DISJOINT[coarse_route(d, N_DOMAINS)]
        deltas.append(_train_path(model, _domain_buf(d), path, base, seed=d))
        masks.append(pm.mask(path, include_backbone=True))
    composed = dequantize(_v1_aggregate(base_int, deltas, masks))

    correct = np.mean([_loss(model, _domain_buf(d), DISJOINT[d], composed)
                       for d in range(N_DOMAINS)])
    genesis = np.mean([_loss(model, _domain_buf(d), DISJOINT[d], base)
                       for d in range(N_DOMAINS)])
    # misroute: send each domain to the NEXT path instead of its own
    misrouted = np.mean([_loss(model, _domain_buf(d), DISJOINT[(d + 1) % N_DOMAINS], composed)
                         for d in range(N_DOMAINS)])

    assert correct < genesis - 0.05               # training through paths helped
    assert correct < misrouted - 0.05             # correct routing >> wrong: paths specialized


def test_overlapping_paths_share_modules_and_average():
    """Paths that pick the SAME (level, module) share that page. A shared module
    is averaged over both worker-holders; a module only one path uses comes from
    its owner in full — exactly the v1 per-page claimant rule (module-DiLoCo)."""
    model = DiPaCoGPT(CFG)
    pm = PathMap(model)
    pA = [0, 1]                                         # module 0 @ level 0, module 1 @ level 1
    pB = [0, 2]                                         # module 0 @ level 0, module 2 @ level 1
    assert pA[0] == pB[0] and pA[1] != pB[1]           # share page (0,0); differ at level 1
    n = pm.n
    base_int = np.zeros(n, dtype=np.int64)
    mA, mB = pm.mask(pA, include_backbone=False), pm.mask(pB, include_backbone=False)
    dA = np.full(n, 1000, dtype=np.int64)
    dB = np.full(n, 3000, dtype=np.int64)
    out = _v1_aggregate(base_int, [dA, dB], [mA, mB])
    s, e = pm.mod_span[(0, 0)]                          # shared page → averaged over 2 holders
    assert np.all(out[s:e] == 2000)                    # (1000+3000)/2
    s, e = pm.mod_span[(1, 2)]                          # only pB holds this → full value
    assert np.all(out[s:e] == 3000)                    # singly held, not halved
    s, e = pm.mod_span[(1, 1)]                          # only pA holds this → full value
    assert np.all(out[s:e] == 1000)
