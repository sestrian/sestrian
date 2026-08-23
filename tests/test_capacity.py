"""Capacity retarget (§9.4a), INTEGER SPEC: the fast knob holds the delta rate
as compute grows, growth fires only on SUSTAINED surplus (never a transient
spike), the total ratchets monotonically while the active set breathes, thaw
takes priority over new growth, and the whole controller is deterministic
integer arithmetic (fleet traces in 1e-2 units: 100 == 1.0 fleet unit)."""

from rig.capacity import (QUOTA_MAX_4DP, QUOTA_MIN_4DP, QUOTA_ONE_4DP,
                          CapacityRetarget, retarget_decide, simulate)


def test_fast_knob_absorbs_fleet_growth():
    c = CapacityRetarget()
    # fleet doubles: quota should rise (harder deltas), not the model (yet)
    simulate(c, [100] * 6 + [200] * 6)
    assert c.log[-1]["quota_4dp"] > c.log[5]["quota_4dp"]
    assert c.total_modules == 4                     # no growth without saturation


def test_growth_only_on_sustained_surplus():
    c = CapacityRetarget()
    # transient one-window spike: must NOT grow
    simulate(c, [100] * 5 + [5000] + [100] * 5)
    assert c.total_modules == 4
    # sustained large fleet: quota pins at ceiling -> growth event fires,
    # bounded and after the announcement lead
    c2 = CapacityRetarget()
    log = simulate(c2, [100] * 3 + [5000] * 20)
    assert c2.total_modules > 4
    grew_windows = [e["window"] for e in log if e["grew"]]
    sched_windows = [e["window"] for e in log if e["scheduled"]]
    assert grew_windows, "sustained surplus must grow the model"
    assert sched_windows, "growth must be scheduled before it activates"
    # the announcement lead separates scheduling from activation
    assert grew_windows[0] - sched_windows[0] == c2.announce_lead
    # bounded: each event adds at most growth_bound
    assert all(e["grew"] <= c2.growth_bound for e in log)


def test_ratchet_and_elastic_active_set():
    c = CapacityRetarget()
    simulate(c, [5000] * 25)                        # grow under a big fleet
    grown_total = c.total_modules
    assert grown_total > 4
    grown_after_pending = grown_total + len(c.pending_growth) * c.growth_bound
    simulate(c, [5] * 30)                           # fleet collapses
    # TOTAL is monotone (ratchet): never shrinks — it may still tick up once
    # from a growth event announced during the boom (announcement lead)
    assert grown_total <= c.total_modules <= grown_after_pending
    assert c.active_modules < c.total_modules       # ACTIVE froze instead
    assert c.active_modules >= c.min_active
    frozen_active = c.active_modules
    simulate(c, [5000] * 10)                        # fleet returns
    assert c.active_modules > frozen_active         # thaw before new growth


def test_thaw_has_priority_over_growth():
    c = CapacityRetarget()
    simulate(c, [5000] * 25)                        # grow
    simulate(c, [5] * 30)                           # collapse -> freezes
    assert c.active_modules < c.total_modules
    total_before = c.total_modules + len(c.pending_growth) * c.growth_bound
    log = simulate(c, [5000] * 6)                   # surplus returns
    # while frozen modules remain, surplus windows thaw; nothing new schedules
    for e in log:
        if e["thawed"]:
            assert not e["scheduled"]
    assert c.active_modules > c.min_active
    assert c.total_modules <= total_before + c.growth_bound


def test_quota_bounds_and_integer_math():
    # quota never leaves [min, max]; decide is pure and integer-typed
    q, p, s = QUOTA_ONE_4DP, 0, 0
    for accepted, stale in [(0, 0), (100, 0), (0, 10_000), (64, 0), (2, 5000)] * 10:
        d = retarget_decide(q, p, s, accepted, stale)
        q, p, s = d["quota_4dp"], d["pinned_streak"], d["slack_streak"]
        assert isinstance(q, int) and isinstance(p, int) and isinstance(s, int)
        assert QUOTA_MIN_4DP <= q <= QUOTA_MAX_4DP


def test_deterministic():
    trace = [100] * 4 + [3000] * 12 + [20] * 8 + [1000] * 6
    a = CapacityRetarget()
    b = CapacityRetarget()
    la, lb = simulate(a, trace), simulate(b, trace)
    assert la == lb                                 # same trace -> same decisions
    assert (a.total_modules, a.active_modules, a.quota_4dp) == \
        (b.total_modules, b.active_modules, b.quota_4dp)
