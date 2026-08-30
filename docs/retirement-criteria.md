# Retiring the Training Wheels — criteria (Sharding Road, Phase 6)

The founders' FULL validators (the anchors) are the safety net under the whole
sharding transition: while any full validator runs, a paged validator that
trusted a foreign page's committed leaf is backstopped, because the full node
would reject the fraudulent block on its own recomputed root and emit a proof.
Retirement is the moment that net comes down — when the model has genuinely
outgrown any single machine and the sampled-plus-disputed regime carries the
chain alone. It is the ONLY irreversible step on the road, so it is gated on
evidence, not a date.

## Hard preconditions (all required)

1. **Mixed operation, proven.** ≥ one quarter (90 days) of continuous
   production with full + paged validators, during which every paged
   validator's SETTLED view (`settled_height` state) matched the full
   validators' truth at every settled height — zero divergences.

2. **A real fault caught by the mechanisms alone.** At least one *organic*
   (not drilled) fault — a byzantine block, a withholding attempt, a custody
   desertion — detected and resolved by fraud proofs / availability sampling /
   slashing WITHOUT a full validator being the thing that caught it. Until the
   mechanisms have caught a real fault unaided, the net has never been tested
   in anger.

3. **Redundancy floor.** ≥ 3 independent paged validators holding each page,
   and ≥ 3 independent full-or-archive validators still willing to serve deep
   history for joiners — so retiring the founders' nodes removes no unique
   custody.

4. **The model actually needs it.** The full state exceeds the RAM of the
   smallest machine we expect a validator to run (the real trigger — retiring
   the net while one machine can still hold everything trades safety for
   nothing).

5. **The backbone stays global.** The shared backbone (a few MB) is
   universally validated by every node forever — a trustless core at any
   scale. Retirement removes full EXPERT-PAGE validation, never backbone
   validation.

## The switch

A scheduled version bump, same ceremony as every fork on this road: rig spec →
golden → core → net → adversarial testnet → the fleet, never deploying across
the activation boundary. After it, `--mode full` is still permitted (anyone
paranoid may hold everything), but the network's SECURITY no longer depends on
one existing.

## Published record

A post-mortem accompanying the switch, showing:
- the 90-day divergence log (must be empty at settled heights),
- the organic fault(s) and how the mechanisms resolved them,
- the custody-redundancy census at the activation height,
- an explicit statement that the safety net was never load-bearing when it
  came down (proven by preconditions 1–2).

## What retirement does NOT change

- Backbone validation (global, forever).
- The right of any operator to run a full node.
- Finality semantics (settled = head − FRAUD_WINDOW) — unchanged; retirement
  changes WHO enforces it for expert pages, not WHEN a block settles.

Beyond this line the model can grow without any machine holding all of it —
wall #1 is genuinely behind us. The other mountain, post-training (turning a
learned model into an assistant), is a different summit on the same chain; it
gets its own map when this road is walked.
