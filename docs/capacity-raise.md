# Raising the expert cap (Sharding Road, Phase 5.1)

The model grows experts up to `e_max` per layer (the router preallocates that
many columns at genesis: `e_max=16` × `n_layers=6` = **96 experts / ~208M
params** on devnet-genesis-3). Growth events append experts until that cap.
Two paths raise it; which one applies depends on whether the cap is binding.

## Status: not binding

At 63 experts of 96, the cap has real headroom. Nothing needs to change until
the model approaches 96 — this document specifies the mechanism for when it
does, the same way v5 (lanes) was specified and gated before its activation
height.

## Path A — re-genesis with a higher e_max (trivial)

`e_max` is a `ModelSpec` field, a pure genesis parameter. A future re-genesis
(the ceremony already exists: `docs/genesis-ceremony.md`) sets it higher and
the whole page table, backbone size, and root follow deterministically. Cost:
a coordinated relaunch. This is the right path when the cap is raised as part
of a planned re-genesis anyway.

## Path B — mid-chain router extension (scheduled fork, no relaunch)

To raise the cap on a LIVE chain without a relaunch, extend the router rather
than the backbone (the backbone stays global and its offsets must not shift):

- **Append, never reshape.** A router-extension event appends a new page of
  kind `"router"`, shape `(n_embd × extra_columns)` per layer, placed AFTER all
  existing pages — so no existing coordinate offset moves (the append-only
  ratchet the whole page model already relies on).
- **Deterministic init.** New router columns initialize to a hash-stream, the
  same recipe as `page_init` for experts (golden-vectored, byte-identical in
  rig and Rust). Zero-ish init means a new column is never selected until it is
  trained — new experts behind it stay dormant until a miner routes to them.
- **Client forward pass concatenates.** `client/moe.py::MoEExpertLayer.route`
  reads the backbone router logits and, when router-extension pages exist for a
  layer, concatenates their columns before the top-k selection — so the
  effective `e_max` for that layer rises by the extension width. This is the
  one MODEL-ARCHITECTURE change the raise requires; it lands with the fork.
- **Consensus side is a growth variant.** The controller, on reaching the cap
  under sustained saturation, schedules a router-extension exactly like an
  expert-growth event (announce lead, window boundary, deterministic init,
  committed via `model_root`). Golden family `router_extension` pins the init
  bytes; `chain_replay` crosses the activation like every other fork.

Path B is model-architecture work coordinated across chain + client + trainer,
scheduled by version height like v3/v4/v5 — it is specified here and built when
the cap becomes binding, not before.

## Which, when

- Approaching 96 with a re-genesis already planned → **Path A**.
- Approaching 96 with a healthy live chain we do not want to relaunch → **Path
  B** (the router-extension fork).

Either way the backbone stays universally validated (Sharding Road invariant):
Path A rebuilds it at genesis; Path B never touches its existing columns, only
appends new router pages that every node still validates.
