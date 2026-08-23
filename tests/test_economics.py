"""Write-price homeostat (§9.4) + stake/slash ledger (§7, §9)."""

import numpy as np

from rig.crypto import BackpropTx, Key, delta_hash
from rig.economics import (StakeLedger, WritePriceController, simulate_homeostat,
                           slash_on_fraudulent_score, slash_on_invalid_tx)


def test_homeostat_drives_load_to_target():
    r = simulate_homeostat(target=8, blocks=120, seed=0)
    settled = np.mean(r["loads"][-20:])
    assert abs(settled - 8) <= 2                     # holds load near target
    assert r["loads"][0] > 20                         # started far above target
    assert r["final_price"] > 1.0                     # price rose to throttle


def test_homeostat_prices_out_spam():
    r = simulate_homeostat(target=8, blocks=120, n_spam=40, seed=1)
    assert r["spam_admitted"][0] > 0                  # spam got in at the floor
    assert r["final_spam"] == 0                        # …and was priced out


def test_homeostat_relaxes_when_idle():
    ctrl = WritePriceController(target_rate=8, window=5)
    for _ in range(30):                                # persistent under-load
        ctrl.observe(1)
        ctrl.maybe_retarget()
    assert ctrl.price == ctrl.min_price                # falls back to the floor


def test_stake_reward_and_slash_flow():
    led = StakeLedger(bounty_share=0.5)
    led.stake("a", 100)
    led.stake("m", 100)
    led.reward("a", 10)
    bounty = led.slash("m", "fault", "watcher")
    assert led.staked["m"] == 0.0                      # bond slashed
    assert bounty == 50 and led.rewards["watcher"] == 50
    assert led.burned == 50                            # the rest burns


def test_slash_on_invalid_signature():
    led = StakeLedger()
    k = Key.generate(b"m".ljust(32, b"0"))
    attacker_signed = Key.generate(b"x".ljust(32, b"0"))
    led.stake(k.pub, 100)
    tx = BackpropTx(miner=k.pub, base_height=0,
                    delta_hash=delta_hash(b"BODY"), da_pointer="da://z")
    tx.sig = attacker_signed.sign(tx.signing_bytes())  # forged
    assert slash_on_invalid_tx(led, tx, None, "watcher")
    assert led.staked[k.pub] == 0.0


def test_slash_on_fraudulent_score():
    led = StakeLedger()
    led.stake("validator", 100)
    # claimed a score that recomputation contradicts
    assert slash_on_fraudulent_score(led, "validator", claimed=0.9,
                                     recomputed=0.1, challenger="w")
    assert led.staked["validator"] == 0.0
    # an honest score is not slashed
    led.stake("honest", 100)
    assert not slash_on_fraudulent_score(led, "honest", claimed=0.5,
                                        recomputed=0.5, challenger="w")
    assert led.staked["honest"] == 100
