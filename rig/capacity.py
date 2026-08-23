"""The capacity retarget (§9.4a) — model size as difficulty. INTEGER SPEC.

Protocol v1 rewrite: the controller is consensus state (folded per block into
`rig/model_state.py::ModelState` and committed via `model_root`), so every
quantity is scaled-integer arithmetic — floats never touch consensus.

  * FAST knob: `quota_4dp`, the per-delta work quota in 1e-4 units
    (10_000 = 1.0). Damped toward holding accepted deltas per retarget window
    at target — the Bitcoin-difficulty analog, continuous and reversible.
  * SLOW knob: growth events. Only when the fast knob has been pinned at its
    ceiling for `k_sustain` consecutive windows (a SUSTAINED surplus) is a
    growth event SCHEDULED, `announce_lead` windows ahead; activation (the
    actual page append) is applied by the ModelState fold. Bounded modules per
    event, ratcheted (total capacity never shrinks). When compute leaves,
    ACTIVE modules freeze (stop training, keep serving) instead of the model
    shrinking — MoE sparsity's graceful degradation. Freeze order is LIFO over
    grown modules; genesis modules never freeze (min_active).

`retarget_decide` is THE decision function — one source of truth shared by the
standalone controller below (tests, simulation, golden vectors) and the
consensus fold in model_state.py. All divisions are floor division on ints
(Rust: `div_euclid` / `i128` intermediates).
"""

from dataclasses import dataclass, field

# 4-decimal fixed point: 10_000 == a quota of 1.0
QUOTA_ONE_4DP = 10_000
QUOTA_MIN_4DP = 2_500          # 0.25x
QUOTA_MAX_4DP = 80_000         # 8.0x
TARGET_DELTAS = 8              # accepted deltas per window we steer toward
DAMP_DIV = 4                   # apply 1/4 of the raw correction per window
STALE_CEILING_4DP = 2_000      # above 20% staleness, never count as surplus
K_SUSTAIN = 3                  # pinned windows before growth is scheduled
GROWTH_BOUND = 1               # max modules scheduled per event
ANNOUNCE_LEAD = 2              # windows between scheduling and activation


def retarget_decide(quota_4dp: int, pinned_streak: int, slack_streak: int,
                    accepted: int, staleness_4dp: int,
                    *, quota_min: int = QUOTA_MIN_4DP,
                    quota_max: int = QUOTA_MAX_4DP,
                    target_deltas: int = TARGET_DELTAS,
                    damp_div: int = DAMP_DIV,
                    stale_ceiling: int = STALE_CEILING_4DP,
                    k_sustain: int = K_SUSTAIN) -> dict:
    """One window's pure retarget decision. Integer-only; deterministic.

    Returns {quota_4dp, pinned_streak, slack_streak, schedule, freeze,
    thaw_ok}. Streaks are the INCREMENTED values — the caller resets the
    relevant streak only when it actually applies the action (a schedule that
    is skipped because an event is already pending, or a freeze with nothing
    freezable, must not consume the streak differently on different nodes:
    both caller implementations reset iff they acted, in the same order).
    Action precedence at the caller: thaw (recovery) BEFORE any new growth.
    """
    # FAST: damped multiplicative correction toward the target delta rate.
    if accepted > 0:
        raw = quota_4dp * accepted // target_deltas
    else:
        raw = quota_min
    quota_4dp = quota_4dp + (raw - quota_4dp) // damp_div
    quota_4dp = min(quota_max, max(quota_min, quota_4dp))

    # SLOW: surplus = ceiling-pinned AND healthy staleness AND target met.
    # Band tolerances are wide (5%) because the damped quota asymptotes to its
    # bounds without ever exactly reaching them.
    surplus = (quota_4dp >= quota_max * 95 // 100
               and staleness_4dp <= stale_ceiling
               and accepted >= target_deltas)
    deficit = quota_4dp <= quota_min * 105 // 100 and accepted < target_deltas
    pinned_streak = pinned_streak + 1 if surplus else 0
    slack_streak = slack_streak + 1 if deficit else 0

    return {"quota_4dp": quota_4dp,
            "pinned_streak": pinned_streak,
            "slack_streak": slack_streak,
            "schedule": pinned_streak >= k_sustain,
            "freeze": slack_streak >= k_sustain,
            "thaw_ok": surplus}


@dataclass
class CapacityRetarget:
    """Standalone integer controller (tests / simulation / golden vectors).

    Mirrors the consensus fold's decision math exactly (both call
    `retarget_decide`); module bookkeeping here uses abstract counters where
    the fold uses real page statuses.
    """
    quota_4dp: int = QUOTA_ONE_4DP
    quota_min: int = QUOTA_MIN_4DP
    quota_max: int = QUOTA_MAX_4DP
    target_deltas: int = TARGET_DELTAS
    damp_div: int = DAMP_DIV
    stale_ceiling: int = STALE_CEILING_4DP
    k_sustain: int = K_SUSTAIN
    growth_bound: int = GROWTH_BOUND
    announce_lead: int = ANNOUNCE_LEAD
    total_modules: int = 4
    active_modules: int = 4
    min_active: int = 4            # genesis modules; never freezable
    pinned_streak: int = 0
    slack_streak: int = 0
    pending_growth: list = field(default_factory=list)   # activation window ids
    window_id: int = 0
    log: list = field(default_factory=list)

    def observe_window(self, accepted: int, staleness_4dp: int) -> dict:
        """Feed one retarget window's chain-observable signals; returns the
        decisions taken this window (all deterministic, all integer)."""
        self.window_id += 1
        events = {"window": self.window_id, "grew": 0, "froze": 0, "thawed": 0,
                  "scheduled": 0}

        # activate any growth event whose announcement lead has elapsed
        due = [w for w in self.pending_growth if w <= self.window_id]
        for _ in due:
            self.total_modules += self.growth_bound
            self.active_modules += self.growth_bound
            events["grew"] += self.growth_bound
        self.pending_growth = [w for w in self.pending_growth if w > self.window_id]

        d = retarget_decide(self.quota_4dp, self.pinned_streak, self.slack_streak,
                            accepted, staleness_4dp,
                            quota_min=self.quota_min, quota_max=self.quota_max,
                            target_deltas=self.target_deltas,
                            damp_div=self.damp_div,
                            stale_ceiling=self.stale_ceiling,
                            k_sustain=self.k_sustain)
        self.quota_4dp = d["quota_4dp"]
        self.pinned_streak = d["pinned_streak"]
        self.slack_streak = d["slack_streak"]

        # recovery FIRST: thaw frozen modules before any new growth is considered
        if d["thaw_ok"] and self.active_modules < self.total_modules:
            self.active_modules += 1
            events["thawed"] = 1
            self.pinned_streak = 0        # thawing consumes the surplus signal
        elif d["schedule"] and not self.pending_growth:
            self.pending_growth.append(self.window_id + self.announce_lead)
            events["scheduled"] = 1
            self.pinned_streak = 0
            # growth resets the fast knob to mid-band: the bigger model absorbs
            # the surplus the quota ceiling could not
            self.quota_4dp = (self.quota_min + self.quota_max) // 2

        # decline: freeze active modules (total NEVER shrinks — the ratchet)
        if d["freeze"] and self.active_modules > self.min_active:
            self.active_modules -= 1
            events["froze"] = 1
            self.slack_streak = 0

        events.update(quota_4dp=self.quota_4dp, total=self.total_modules,
                      active=self.active_modules)
        self.log.append(events)
        return events


def simulate(controller: CapacityRetarget, fleet_trace: list[int],
             per_unit: int = 8) -> list[dict]:
    """Drive the controller with a synthetic fleet, integer-only: fleet values
    are in 1e-2 units (100 == 1.0 fleet unit). Each window the fleet produces
    ~ capacity/quota deltas, degraded by staleness when the ACTIVE model
    outweighs the fleet. Deterministic — no randomness, no floats."""
    out = []
    modules_per_fleet_unit = 4          # a fleet of 1.0 comfortably trains 4 modules
    for fleet_2dp in fleet_trace:
        capacity_4dp = fleet_2dp * per_unit * 100        # -> 4dp work units
        accepted = capacity_4dp // max(controller.quota_4dp, 1)
        # if the active model outweighs the fleet, deltas arrive late
        load_4dp = (controller.active_modules * 10_000 * 100
                    // (modules_per_fleet_unit * max(fleet_2dp, 1)))
        staleness_4dp = max(0, min(10_000, load_4dp - 10_000))
        accepted = accepted * (10_000 - staleness_4dp) // 10_000
        out.append(controller.observe_window(accepted, staleness_4dp))
    return out
