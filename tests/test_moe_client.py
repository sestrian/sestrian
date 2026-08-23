"""Client MoE model (protocol v1): the torch↔chain permutation round-trips,
the client's page table matches the CONSENSUS page table bit-for-bit, growth
(add_expert) loads the chain's deterministic page init exactly, the RoPE
invariant holds (no learned positions), and sparse serving equals training."""

import numpy as np
import torch

from client.moe import ChainLayout, MoEGPT, MoEGPTConfig, build_moe, load_fraction
from client.trainer import flat_params
from rig.chain import SCALE, dequantize, quantize
from rig.model_state import ModelSpec, ModelState, page_init

CFG = MoEGPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16,
                   n_experts=3, e_max=6, top_k=2)


def _model(seed=7, epl=None):
    m, _ = build_moe(CFG, device="cpu", seed=seed, experts_per_layer=epl)
    return m


def _spec_for(model: MoEGPT, layout: ChainLayout) -> ModelSpec:
    return ModelSpec(n_layers=model.cfg.n_layer, d_model=model.cfg.n_embd,
                     d_ff=model.cfg.d_ff,
                     n_experts_initial=model.cfg.n_experts,
                     e_max=model.cfg.e_max,
                     backbone_params=layout.backbone_params)


def test_permutation_round_trips_and_is_total():
    model = _model()
    layout = ChainLayout(model)
    tv = flat_params(model)
    cv = layout.chain_of(tv)
    assert cv.shape == tv.shape
    assert np.array_equal(layout.torch_of(cv), tv)          # exact inverse
    assert np.array_equal(np.sort(layout.to_chain), np.arange(layout.n))


def test_client_page_table_matches_consensus():
    """The load-bearing check: ChainLayout's spans must equal the rig's
    ModelState.genesis page table for the same spec — the client and consensus
    must agree on every byte boundary or nothing else matters."""
    model = _model()
    layout = ChainLayout(model)
    spec = _spec_for(model, layout)
    st = ModelState.genesis(spec)
    assert layout.n_pages() == len(st.pages)
    for pid in range(layout.n_pages()):
        assert layout.page_span(pid) == st.page_span(pid)
    assert layout.expert_page_len == spec.expert_page_len
    assert layout.experts == [(l, e) for l in range(CFG.n_layer)
                              for e in range(CFG.n_experts)]


def test_add_expert_loads_deterministic_page_init():
    model = _model()
    layout = ChainLayout(model)
    spec = _spec_for(model, layout)
    page_id = layout.n_pages()                       # the appended page's id
    init_int = page_init("ab" * 32, page_id, spec)
    e_idx = model.add_expert(0, dequantize(init_int))
    assert model.experts_per_layer == [4, 3]
    # rebuild the layout; the new expert's chain page must round-trip the init
    layout2 = ChainLayout(model)
    cv = layout2.chain_of(flat_params(model))
    s, e = layout2.page_span(layout2.page_of_expert(0, e_idx))
    assert np.array_equal(quantize(cv[s:e]), init_int)
    # forward still runs with ragged expert counts
    x = torch.randint(0, 256, (2, CFG.block_size))
    logits, loss = model(x, x)
    assert torch.isfinite(loss)


def test_claim_set_helpers():
    model = _model()
    layout = ChainLayout(model)
    delta = np.zeros(layout.n, dtype=np.int64)
    s, e = layout.page_span(2)
    delta[s] = 5
    bs, _be = layout.page_span(0)
    delta[bs + 3] = -2
    assert layout.pages_touched(delta) == [0, 2]
    cleaned = layout.zero_outside(delta, [2])
    assert layout.pages_touched(cleaned) == [2]
    assert cleaned[s] == 5 and cleaned[bs + 3] == 0


def test_rope_invariant_no_learned_positions():
    model = _model()
    names = [n for n, _ in model.named_parameters()]
    assert not any("pos" in n for n in names), "no learned position table (RoPE)"
    # router preallocated to e_max regardless of instantiated experts
    assert model.blocks[0].moe.router.out_features == CFG.e_max
    # context beyond training length still runs (rotary, not table-bounded)
    x = torch.randint(0, 256, (1, CFG.block_size))
    logits, _ = model(x)
    assert logits.shape == (1, CFG.block_size, 256)


def test_sparse_serve_matches_training_forward():
    model = _model()
    model.eval()
    x = torch.randint(0, 256, (2, CFG.block_size))
    h = model.drop(model.tok(x))
    blk = model.blocks[0]
    a = h + blk.attn(blk.ln1(h))
    dense = blk.moe(blk.ln2(a))
    sparse, touched = blk.moe.serve(blk.ln2(a))
    assert torch.allclose(dense, sparse, atol=1e-6)
    assert 0 < len(touched) <= CFG.n_experts


def test_generate_and_load_fraction():
    model = _model()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 5)
    assert out.shape == (1, 6)
    layout = ChainLayout(model)
    frac = load_fraction(layout, [0, 1, 2])          # backbone + 2 experts
    assert 0 < frac < 1
