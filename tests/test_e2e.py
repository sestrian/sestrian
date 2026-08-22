"""End-to-end flywheel invariants, plus the falsifier suite runs green."""

import pytest

from rig.e2e import run_chain


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_flywheel_end_state(seed):
    r = run_chain(seed=seed, blocks=40, verbose=False)
    assert r["final_acc"] > 0.9          # model learned
    assert r["replay_ok"]                # model-as-chain-state holds
    assert not r["fake_ever_verified"]   # fake serving node never passed attest
    assert r["all_honest_ok"]            # honest receipts always verify


def test_shorter_run_is_reproducible():
    a = run_chain(seed=7, blocks=10, verbose=False)
    b = run_chain(seed=7, blocks=10, verbose=False)
    assert a["rows"][-1]["root"] == b["rows"][-1]["root"]


def test_falsifier_suite_verdicts_all_pass():
    # Import here so a numpy-less collector still imports the module lazily.
    from rig.consensus_sim import run_consensus_suite
    from rig.econ_sim import run_econ_suite
    from rig.training_sim import run_all_training

    t = run_all_training()
    assert t["A"].replay_ok and t["B"].replay_ok and t["C"].replay_ok
    assert t["B"].poison_passed_loss > 20            # stealthy poison passes loss
    # This previously asserted `C.atk_rate < 0.5 * B.atk_rate` — "probes collapse
    # the backdoor". That was a SEED ARTIFACT: measured across 8 seeds it holds on
    # 1. Absolute attack rates are tiny (0.00–0.13) so the ratio is mostly noise;
    # it only ever passed because the old beacon domain string happened to make
    # seed 7 favourable, and renaming that string re-rolled the dice and exposed
    # it. Our own red-team write-up already says blind probes CANNOT catch a
    # stealthy attacker-triggered backdoor, so the assertion contradicted the
    # documented position. Assert the honest property instead: probes must not
    # make things worse.
    assert t["C"].atk_rate <= t["B"].atk_rate + 0.02  # probes never backfire
    # Excision is the guarantee we actually claim, and it is robust: the attack
    # rate returns to 0.0000 on 8/8 seeds measured.
    assert t["D_excised"]["atk_rate"] <= t["A"].atk_rate + 0.02  # excision works

    k = run_consensus_suite()
    assert all(v.ev_attacker < v.ev_honest for v in k["freeriders"] if v.param >= 0.01)
    assert all(v.ev_attacker <= v.ev_honest + 1e-9
               for v in k["colluders"] if v.param < 0.5)

    e = run_econ_suite()
    assert all(r.crossover_month is not None and not r.cap_hit
               for r in e if r.scenario in ("base", "aggressive"))
    assert all(r.crossover_month is None
               for r in e if r.scenario == "conservative")
