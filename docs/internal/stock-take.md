# Stock-take: what we built vs the original proposition

*As of the seventh Phase-0 pass. 117 tests, ~5,600 lines, 33 modules, one
private repo, verified across two machines.*

## The original proposition

The founding idea, in the owner's words:

> A blockchain-based system for training a model with public weights, where the
> compute on the chain either helps **train** the weights or **serves** the API
> calls of the chain. The money generated internally by the chain pays the actors
> in the chain, so the chain is self-serving: it keeps getting better to generate
> more revenue to grow further. And to poison the training you'd need to control
> 50%+1 of the nodes once the community is large enough, which prevents most bad
> actors.

Six load-bearing claims sit inside that: (1) a blockchain whose content is a
model with public weights; (2) compute that trains **or** serves; (3) internal
revenue that pays contributors; (4) a self-improving flywheel; (5) majority-based
poisoning resistance; (6) it can actually be built and run.

## The audit

| # | Original claim | Status | Evidence in the rig |
|---|---|---|---|
| 1 | Blockchain **is** a public-weights model | **Built** | The model is the chain state; blocks are optimizer steps; replay reconstructs the model bit-exact (`chain.py`, `blockchain.py`; determinism tests) |
| 2 | Compute **trains or serves** | **Built** | Training: miners produce scored deltas (`node.py`, `p2p.py`, `integrated.py`). Serving: sparse, Merkle-attested inference with a partial-recompute verifier (`moe.py`, `moe_transformer.py`) |
| 3 | Internal revenue **pays actors** | **Modelled** | Fee split (55/10/25/10), reward distribution, stake & slashing (`e2e.py`, `economics.py`). The *loop closing* — inference revenue exceeding training cost — is a market outcome, not a code property (Phase 3 goal) |
| 4 | **Self-improving flywheel** | **Built (sim)** | The full loop runs end-to-end: train → score → apply → attested serve → fee → reward (`e2e.py`); the model climbs 0.13 → 1.0 through the chain |
| 5 | **50%+1 poisoning resistance** | **Corrected** | See below — the honest deviation |
| 6 | **Can be built and run** | **Built** | 117 passing tests; runs coordinator-free across the Mac + the GPU server over Tailscale; a live browser viewer |

Plus everything the proposition implied but didn't name, now real: an unbiasable
threshold-BLS beacon with DKG (`beacon.py`, `dkg.py`), a peer-to-peer gossip
network with fork choice and partition-healing (`p2p.py`, `gossip_net.py`), an
erasure-coded data-availability layer with sampling (`da.py`), signed
transactions (`crypto.py`), and a difficulty-style write-price homeostat
(`economics.py`).

## The one honest deviation — claim 5

The original intuition was that 50%+1 honest nodes prevents poisoning. The
Phase-0 red-team (`redteam.py`) tested this directly and it does **not** hold as
stated. Majority-honest compute secures *gradient integrity and liveness* — that
part is real. But **poisoning is a different attack surface**: a stealthy
backdoor keyed to a secret trigger is invisible to blind detection, and a
slow-drip coalition can implant one from deltas each less conspicuous than honest
work — regardless of how honest the majority is.

What actually defends against poisoning is not majority voting but **(a) staked
data-admission that raises the cost of getting poison in, and (b) replay-excision
that removes a backdoor once it's discovered** (driven to zero effect in the
rig). So the corrected claim is: *majority secures the compute; poisoning is made
costly and reversible, not impossible.* We found this before it could mislead a
fundraise — which is exactly what a Phase-0 rig is for.

## Verdict

The original proposition is **substantially realized in a working system**, with
one security claim corrected from "prevented" to "made reversible." Nothing in
the vision was found to be infeasible; the parts that remain unproven are
market-facing (does a paying vertical exist, does the revenue loop close at
scale), not architectural.

---

# Remaining timeline to completion

Phase 0 (the rig) is essentially complete. What remains is turning proven
mechanisms into a funded, running network. Durations are engineering estimates
assuming a small funded team; they are sequential where dependencies force it and
parallel where they don't.

### Phase 1 — Devnet at real scale (≈ months 0–6 post-funding)
- Swap the toy models for a real **7B model**; move training onto GPUs with a
  real distributed-training stack (DiLoCo/DisTrO-class), keeping the chain's
  deterministic aggregation and replay.
- Harden the network layer: NAT traversal, peer scoring/eviction, DoS resistance.
- Wire the real beacon and DA into the async socket node (the integrated loop
  exists in-process; production folds it onto `gossip_net.py`).
- Leader election / proposer lottery to replace round-robin.
- **Exit criterion:** a real-scale model trains across ≥10 independent machines,
  replays bit-exact, with beacon-driven selection and DA-validated bodies.

### Phase 2 — Incentivized testnet + a chosen vertical (≈ months 4–12, overlaps)
- Pick the vertical with community-owned data (the survival decision, §11.2);
  bring that community on.
- Points-not-token incentives; first external paying inference on the attested
  tier; the anti-theatre decentralization dashboard live from day one.
- Adversarial hardening at scale: does the §12.3 residual behave as the toy
  predicts on a real model; does verification overhead stay under the 25% line.
- **Exit criterion:** best-in-vertical model on public evals; paying pilot users.

### Phase 3 — Mainnet & the crossover (≈ months 10–24)
- Token genesis under the fairness constraints (no pre-sale of emission rights,
  milestone-vested team allocation, monotone non-amendable emission sunset).
- Emission-funded scale-up; revenue milestones ticking the sunset down.
- **The single public goal:** the crossover block — the first block where
  fee-funded training rewards exceed emissions. No network has reached it; it is
  the existence proof the whole thesis rests on.

### Phase 4 — Scale (≈ months 24+)
- Grow the one model with the bandwidth frontier; deepen the vertical before
  widening; resist multi-model sprawl (the single-model discipline is a
  structural advantage).

### The critical path
Funding → Phase 1 devnet → a signed vertical partner → Phase 2 revenue signal →
Phase 3 crossover. The gating risks, in order: (1) does DiLoCo-class training hold
at 7B+ across untrusted machines; (2) is there a vertical whose users pay more
than its training costs; (3) does the §12.3 residual stay manageable at scale.
The rig has retired the architectural and distributed-systems risks; these three
are what the money buys down.
