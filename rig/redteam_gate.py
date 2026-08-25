"""Red-team: the v3 LEARNING GATE against a Byzantine proposer (WHITEPAPER §9.4a).

The v3 gate (rig/model_state.fold) decides whether the model may GROW by asking
whether the network is measurably learning:

    staleness_4dp = 0 if win_score_sum > 0 else 10_000      # rev >= 3

where win_score_sum is the sum of proposer-COMMITTED delta scores accumulated
across the whole retarget window. Growth (a permanent enlargement every node
must adopt and serve) then follows once the quota is pinned and staleness sits
under the ceiling for k_sustain windows.

Committed scores are validated only for RANGE ([0, SCORE_CAP]); their ACCURACY
is not checked on-chain (that is the multi-evaluator committee, a testnet item).
So a proposer may commit any score it likes. This module measures what that buys
an attacker, honestly, the way rig/redteam.py does for poisoning.

The asymmetry it finds:
  * FORCE-GROWTH is 1-of-N. win_score_sum is a SUM over the window, so ONE
    Byzantine proposer committing a single positive micro-nat, on an otherwise
    plateaued (genuinely-not-learning) network, flips the gate open for the
    entire window. Growth then proceeds on the attacker's say-so.
  * SUPPRESS-GROWTH is N-of-N. To hold the gate CLOSED on a genuinely-learning
    network, EVERY proposer in the window must commit zero; one honest positive
    score defeats it.

Force-growth — the expensive, permanent, everyone-pays direction — is the cheap
one. That is backwards, and it means the gate is a proposer-policy heuristic,
not a Byzantine-robust control. This experiment proves it against the live
devnet retarget constants.
"""

from __future__ import annotations

from . import model_state as ms
from .capacity import (K_SUSTAIN, QUOTA_MAX_4DP, STALE_CEILING_4DP,  # noqa: F401
                       TARGET_DELTAS)

# Devnet spec + retarget constants (node/net/src/main.rs DEVNET). v3_height=0
# so the learning gate is active from the first block of the experiment.
SPEC = ms.ModelSpec(n_layers=6, d_model=512, d_ff=2048,
                    n_experts_initial=8, e_max=16, backbone_params=6_628_352)
PARAMS = ms.GenesisParams(spec=SPEC, v3_height=0)
W = PARAMS.retarget_window               # 16


def _run(byzantine_positive_per_window: bool, windows: int = 12):
    """Fold `windows` retarget windows of a SATURATED but PLATEAUED network:
    every block is full (accepted deltas == target) and every delta is honestly
    scored ZERO — the model has stopped improving, so it must NOT grow.

    If byzantine_positive_per_window, ONE block per window instead commits a
    single positive micro-nat (score_sum=1) — the whole Byzantine act.
    Returns (growth_events, quota_path).
    """
    s = ms.ModelState.genesis(SPEC)
    # deltas per block so the window total hits the target the surplus test wants
    per_block = max(1, TARGET_DELTAS // W) + 1
    height = 0
    quota_path = []
    for w in range(windows):
        for b in range(W):
            height += 1
            # honest plateau: every delta scored zero (no measured improvement)
            zero_scored = per_block
            score_sum = 0
            # the attacker's single positive micro-nat, once per window
            if byzantine_positive_per_window and b == 0:
                score_sum = 1
                zero_scored = per_block - 1
            s, _ = ms.fold(s, PARAMS, height, n_txs=per_block,
                           zero_scored=zero_scored,
                           prev_hash=f"{height:064x}", score_sum=score_sum)
        quota_path.append(s.quota_4dp)
    return s.events_total, quota_path


def run_gate_redteam(verbose: bool = True):
    honest_growth, _ = _run(byzantine_positive_per_window=False)
    attack_growth, _ = _run(byzantine_positive_per_window=True)
    if verbose:
        print("v3 learning-gate red-team (saturated, genuinely-plateaued net)")
        print(f"  honest (all deltas scored 0): growth events = {honest_growth}"
              f"  → gate holds, model does not grow ✓")
        print(f"  1 Byzantine proposer/window (one +1 micro-nat): "
              f"growth events = {attack_growth}")
        one_of_n = attack_growth > 0 and honest_growth == 0
        print(f"  VERDICT: single-proposer force-growth "
              f"{'CONFIRMED' if one_of_n else 'not reproduced'} "
              f"({'1-of-%d proposers' % W if one_of_n else '—'})")
    return {"honest_growth": honest_growth, "attack_growth": attack_growth}


if __name__ == "__main__":
    run_gate_redteam()
