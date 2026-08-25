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
PARAMS = ms.GenesisParams(spec=SPEC, v3_height=0, v4_height=10**9)
# v4 (quorum gate) active from block 0, quorum sized as devnet ships it.
PARAMS_V4 = ms.GenesisParams(spec=SPEC, v3_height=0, v4_height=0,
                             growth_quorum=3)
W = PARAMS.retarget_window               # 16


def _run(byzantine_positive_per_window: bool, windows: int = 12,
         params=None, n_attackers: int = 1, honest_scorers: int = 0):
    """Fold `windows` retarget windows of a SATURATED but PLATEAUED network:
    every block is full (accepted deltas == target) and every delta is honestly
    scored ZERO — the model has stopped improving, so it must NOT grow.

    If byzantine_positive_per_window, ONE block per window instead commits a
    single positive micro-nat (score_sum=1) — the whole Byzantine act.
    Returns (growth_events, quota_path).
    """
    params = params or PARAMS
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
            proposer = f"honest{b % 4}"
            # the attacker(s): one positive micro-nat each, once per window.
            # n_attackers distinct keys = the cost of beating a quorum of that
            # size, which is what v4 prices.
            if byzantine_positive_per_window and b < n_attackers:
                score_sum = 1
                zero_scored = per_block - 1
                proposer = f"attacker{b}"
            # genuinely-learning honest proposers, when modelling a real signal
            elif honest_scorers and b >= n_attackers \
                    and b < n_attackers + honest_scorers:
                score_sum = 50
                zero_scored = per_block - 1
                proposer = f"honest{b}"
            s, _ = ms.fold(s, params, height, n_txs=per_block,
                           zero_scored=zero_scored,
                           prev_hash=f"{height:064x}", score_sum=score_sum,
                           proposer=proposer)
        quota_path.append(s.quota_4dp)
    return s.events_total, quota_path


def run_gate_redteam(verbose: bool = True):
    honest_growth, _ = _run(byzantine_positive_per_window=False)
    attack_growth, _ = _run(byzantine_positive_per_window=True)
    # --- v4 quorum gate: the same attack, and the honest path that must survive
    v4_1 = _run(True, params=PARAMS_V4, n_attackers=1)[0]
    v4_2 = _run(True, params=PARAMS_V4, n_attackers=2)[0]
    v4_3 = _run(True, params=PARAMS_V4, n_attackers=3)[0]
    v4_honest = _run(False, params=PARAMS_V4, honest_scorers=3)[0]
    v4_plateau = _run(False, params=PARAMS_V4)[0]
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
        print(f"v4 QUORUM gate (growth_quorum={PARAMS_V4.growth_quorum}), "
              f"same plateaued network:")
        print(f"  1 Byzantine proposer : growth = {v4_1}  "
              f"{'BLOCKED ✓' if v4_1 == 0 else 'STILL FORCED ✗'}")
        print(f"  2 Byzantine proposers: growth = {v4_2}  "
              f"{'BLOCKED ✓' if v4_2 == 0 else 'STILL FORCED ✗'}")
        print(f"  3 Byzantine proposers: growth = {v4_3}  "
              f"(= quorum: forcing now costs winning {PARAMS_V4.growth_quorum} "
              f"blocks with {PARAMS_V4.growth_quorum} keys — priced, not free)")
        print(f"  honest plateau       : growth = {v4_plateau}  "
              f"{'✓ still refuses to grow' if v4_plateau == 0 else '✗'}")
        print(f"  honest LEARNING net  : growth = {v4_honest}  "
              f"{'✓ real growth survives' if v4_honest > 0 else '✗ REGRESSION'}")
    return {"honest_growth": honest_growth, "attack_growth": attack_growth,
            "v4_attack_1": v4_1, "v4_attack_2": v4_2, "v4_attack_quorum": v4_3,
            "v4_honest_learning": v4_honest, "v4_honest_plateau": v4_plateau}


if __name__ == "__main__":
    run_gate_redteam()
