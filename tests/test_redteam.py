"""§12.3 red-team: the honest findings must hold, or the security story is wrong."""

from rig.redteam import (TH_CLEAN, TH_DELTA_Z, TH_ORACLE, TH_RANDOM, detected_by,
                         experiment_A, experiment_B_drip)


def test_stealthy_backdoor_beats_blind_input_probes():
    """A stealthy OOD-triggered backdoor is invisible to clean-loss and
    in-distribution probing, while the oracle (known trigger) always catches it."""
    A = experiment_A(seed=0)
    for a in A["results"]:
        assert a.backdoor_success > 0.8                 # the backdoor works
        assert a.p_oracle > TH_ORACLE                   # known trigger -> caught
        if a.strategy in ("stealthy", "minimal"):
            assert a.p_clean < TH_CLEAN                  # clean-loss probe misses
            assert a.p_random < TH_RANDOM                # in-distribution probe misses


def test_naive_backdoor_is_caught():
    """The crude attack that a real defender would obviously catch, is caught."""
    A = experiment_A(seed=0)
    naive = next(a for a in A["results"] if a.strategy == "naive")
    d = detected_by(naive)
    assert d["delta"] or d["oracle"]


def test_slow_drip_evades_anomaly_and_accumulates():
    """Drip coalition: each delta is no more conspicuous than honest work, yet
    the backdoor accumulates across blocks."""
    B = experiment_B_drip(seed=0)
    assert B["poisoned_backdoor"] > 0.5                  # backdoor accumulated
    assert B["max_coal_z"] < TH_DELTA_Z                  # per-delta anomaly misses it
    assert B["curve"][-1] > B["curve"][0]                # it grew over blocks


def test_excision_recovers_from_poisoning():
    """The design's durable, detection-independent guarantee: replay-excision
    removes a discovered backdoor while preserving clean accuracy (§10.4)."""
    B = experiment_B_drip(seed=0)
    assert B["excised_backdoor"] < 0.1
    assert B["excised_clean"] > B["poisoned_clean"] - 0.15


# --- v3 learning-gate red-team (rig/redteam_gate.py) ---------------------------

from rig.redteam_gate import run_gate_redteam        # noqa: E402


def test_v3_gate_holds_on_an_honest_plateau():
    """A saturated but genuinely-not-learning network (every delta honestly
    scored zero) must NOT grow: the gate does its job against honest inputs."""
    r = run_gate_redteam(verbose=False)
    assert r["honest_growth"] == 0


def test_v3_gate_is_not_byzantine_robust_force_growth_is_1_of_n():
    """The honest finding: a SINGLE Byzantine proposer per window, committing
    one positive micro-nat on a plateaued network, forces the model to grow —
    win_score_sum is a window-wide SUM, so one liar flips the gate open. The
    gate is a proposer-policy heuristic, not a Byzantine-robust control; real
    robustness gates on the multi-evaluator committee (testnet). If this ever
    stops holding, the gate was hardened — update the threat model."""
    r = run_gate_redteam(verbose=False)
    assert r["attack_growth"] > 0
    assert r["attack_growth"] > r["honest_growth"]


def test_v4_quorum_gate_blocks_the_single_byzantine_proposer():
    """The v4 fix: counting DISTINCT positive-scoring proposers instead of
    summing forgeable scores defeats the 1-of-N force-growth attack. Two
    colluding proposers are still short of a quorum of 3."""
    r = run_gate_redteam(verbose=False)
    assert r["v4_attack_1"] == 0
    assert r["v4_attack_2"] == 0


def test_v4_quorum_is_a_price_not_a_proof():
    """Honest limit, stated as a test so it cannot be quietly forgotten: a
    coalition that actually wins `growth_quorum` blocks with that many keys
    still forces growth. v4 prices the attack (stake-weighted sortition); it
    does not make the gate trustless — only the multi-evaluator committee
    does. If this ever starts passing as 0, the committee landed: update the
    threat model."""
    r = run_gate_redteam(verbose=False)
    assert r["v4_attack_quorum"] > 0


def test_v4_does_not_regress_the_honest_paths():
    """A genuinely-learning network must still grow, and a genuine plateau
    must still refuse to — the gate's whole purpose, preserved across the fix."""
    r = run_gate_redteam(verbose=False)
    assert r["v4_honest_learning"] > 0
    assert r["v4_honest_plateau"] == 0
