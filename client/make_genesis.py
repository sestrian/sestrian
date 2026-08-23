"""Deterministically build the genesis model and emit its raw chain bytes.

Usage:

  python -m client.make_genesis --model small-moe --seed 20260822 \
      --out genesis.bin [--expect <state_root>]

Protocol v1 (MoE presets): the emitted flat i64-LE vector is in CHAIN order —
backbone page first, then one page per (layer, expert) — and the printed
genesis_state_root is the PAGE-MERKLE root over that page table (the network
identity the node bakes in). The printed genesis_model_root commits the
ModelState (page table + capacity fold) at block 0.

Legacy dense presets emit torch order with the flat sha256 root (devnet-1).

`--expect` turns the printed root into a hard check: exit non-zero on any
mismatch, so ceremony scripts verify instead of eyeballing. Every node joining
the network reproduces this file bit-for-bit from (preset, seed) — genesis is
verified, never trusted.
"""

import argparse
import sys

from rig.chain import quantize, state_root
from .gossip import MODEL_PRESETS, build_preset
from .moe import ChainLayout, MoEGPTConfig
from .trainer import flat_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small-moe", choices=list(MODEL_PRESETS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="genesis.bin")
    ap.add_argument("--expect", default=None,
                    help="fail unless genesis_state_root equals this hex root")
    a = ap.parse_args()
    cfg = MODEL_PRESETS[a.model]
    model, _ = build_preset(a.model, device="cpu", seed=a.seed)

    if isinstance(cfg, MoEGPTConfig):
        from rig.model_state import ModelSpec, ModelState, page_state_root
        layout = ChainLayout(model)
        spec = ModelSpec(n_layers=cfg.n_layer, d_model=cfg.n_embd,
                         d_ff=cfg.d_ff, n_experts_initial=cfg.n_experts,
                         e_max=cfg.e_max, backbone_params=layout.backbone_params)
        st0 = ModelState.genesis(spec)
        # the client's layout and the consensus page table must agree exactly
        assert layout.n_pages() == len(st0.pages)
        for pid in range(layout.n_pages()):
            assert layout.page_span(pid) == st0.page_span(pid), \
                f"page {pid}: client/consensus span mismatch"
        w = quantize(flat_params(model))[layout.to_chain]
        root = page_state_root(w, st0)
        print(f"genesis: {w.size/1e6:.1f}M params ({len(st0.pages)} pages, "
              f"backbone {layout.backbone_params/1e6:.2f}M, "
              f"expert page {spec.expert_page_len/1e6:.2f}M) -> {a.out}")
        print(f"genesis_state_root: {root}")
        print(f"genesis_model_root: {st0.model_root()}")
        print(f"genesis_backbone_params: {layout.backbone_params}")
    else:
        w = quantize(flat_params(model))
        root = state_root(w)
        print(f"genesis: {w.size/1e6:.1f}M params -> {a.out}")
        print(f"genesis_state_root: {root}")

    with open(a.out, "wb") as f:
        f.write(w.tobytes())                       # i64 little-endian
    if a.expect and a.expect.strip().lower() != root:
        print(f"FATAL: genesis_state_root != --expect ({a.expect})",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
