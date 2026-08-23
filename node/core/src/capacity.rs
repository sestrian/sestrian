//! The capacity retarget (§9.4a) — model size as difficulty. INTEGER SPEC.
//!
//! Protocol v1 rewrite mirroring `rig/capacity.py`: the controller is consensus
//! state (folded per block into `model_state::ModelState` and committed via
//! `model_root`), so every quantity is scaled-integer arithmetic — floats never
//! touch consensus.
//!
//!   * FAST knob: `quota_4dp`, the per-delta work quota in 1e-4 units
//!     (10_000 = 1.0), damped toward holding accepted deltas per window at
//!     target — the Bitcoin-difficulty analog, continuous and reversible.
//!   * SLOW knob: growth events. Only a ceiling-pinned surplus sustained for
//!     `k_sustain` windows schedules growth, `announce_lead` windows ahead.
//!     Decline freezes grown modules LIFO (total never shrinks — the ratchet).
//!
//! `retarget_decide` is THE decision function — one source of truth shared by
//! the standalone controller below (tests / golden vectors) and the consensus
//! fold in `model_state.rs`. All divisions are floor division (`div_euclid`,
//! matching Python `//`), with i128 intermediates so quota math cannot overflow.

// 4-decimal fixed point: 10_000 == a quota of 1.0
pub const QUOTA_ONE_4DP: i64 = 10_000;
pub const QUOTA_MIN_4DP: i64 = 2_500; // 0.25x
pub const QUOTA_MAX_4DP: i64 = 80_000; // 8.0x
pub const TARGET_DELTAS: i64 = 8; // accepted deltas per window we steer toward
pub const DAMP_DIV: i64 = 4; // apply 1/4 of the raw correction per window
pub const STALE_CEILING_4DP: i64 = 2_000; // above 20% staleness, never a surplus
pub const K_SUSTAIN: i64 = 3; // pinned windows before growth is scheduled
pub const GROWTH_BOUND: i64 = 1; // max modules scheduled per event
pub const ANNOUNCE_LEAD: u64 = 2; // windows between scheduling and activation

/// One window's pure retarget decision. Streaks are the INCREMENTED values —
/// the caller resets the relevant streak only when it actually applies the
/// action, in the same order on every node (thaw before any new growth).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub quota_4dp: i64,
    pub pinned_streak: i64,
    pub slack_streak: i64,
    pub schedule: bool,
    pub freeze: bool,
    pub thaw_ok: bool,
}

/// Line-for-line mirror of `rig/capacity.py::retarget_decide`. Integer-only.
#[allow(clippy::too_many_arguments)]
pub fn retarget_decide(
    quota_4dp: i64,
    pinned_streak: i64,
    slack_streak: i64,
    accepted: i64,
    staleness_4dp: i64,
    quota_min: i64,
    quota_max: i64,
    target_deltas: i64,
    damp_div: i64,
    stale_ceiling: i64,
    k_sustain: i64,
) -> Decision {
    // FAST: damped multiplicative correction toward the target delta rate.
    // i128 intermediates: quota * accepted can exceed i64 for large fleets.
    let raw: i128 = if accepted > 0 {
        (quota_4dp as i128 * accepted as i128).div_euclid(target_deltas as i128)
    } else {
        quota_min as i128
    };
    let q = quota_4dp as i128 + (raw - quota_4dp as i128).div_euclid(damp_div as i128);
    let quota_4dp = q.clamp(quota_min as i128, quota_max as i128) as i64;

    // SLOW: surplus = ceiling-pinned AND healthy staleness AND target met.
    // Band tolerances are wide (5%) because the damped quota asymptotes to its
    // bounds without ever exactly reaching them.
    let surplus = quota_4dp >= (quota_max * 95).div_euclid(100)
        && staleness_4dp <= stale_ceiling
        && accepted >= target_deltas;
    let deficit = quota_4dp <= (quota_min * 105).div_euclid(100) && accepted < target_deltas;
    let pinned_streak = if surplus { pinned_streak + 1 } else { 0 };
    let slack_streak = if deficit { slack_streak + 1 } else { 0 };

    Decision {
        quota_4dp,
        pinned_streak,
        slack_streak,
        schedule: pinned_streak >= k_sustain,
        freeze: slack_streak >= k_sustain,
        thaw_ok: surplus,
    }
}

/// The decisions taken for one retarget window (mirrors the Python `events`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowEvents {
    pub window: u64,
    pub grew: i64,
    pub froze: i64,
    pub thawed: i64,
    pub scheduled: i64,
    pub quota_4dp: i64,
    pub total: i64,
    pub active: i64,
}

/// Standalone integer controller (tests / simulation / golden vectors).
/// Mirrors the consensus fold's decision math exactly (both call
/// `retarget_decide`); module bookkeeping here uses abstract counters where
/// the fold uses real page statuses.
pub struct CapacityRetarget {
    pub quota_4dp: i64,
    pub quota_min: i64,
    pub quota_max: i64,
    pub target_deltas: i64,
    pub damp_div: i64,
    pub stale_ceiling: i64,
    pub k_sustain: i64,
    pub growth_bound: i64,
    pub announce_lead: u64,
    pub total_modules: i64,
    pub active_modules: i64,
    pub min_active: i64, // genesis modules; never freezable
    pub pinned_streak: i64,
    pub slack_streak: i64,
    pub pending_growth: Vec<u64>, // activation window ids
    pub window_id: u64,
}

impl Default for CapacityRetarget {
    fn default() -> Self {
        CapacityRetarget {
            quota_4dp: QUOTA_ONE_4DP,
            quota_min: QUOTA_MIN_4DP,
            quota_max: QUOTA_MAX_4DP,
            target_deltas: TARGET_DELTAS,
            damp_div: DAMP_DIV,
            stale_ceiling: STALE_CEILING_4DP,
            k_sustain: K_SUSTAIN,
            growth_bound: GROWTH_BOUND,
            announce_lead: ANNOUNCE_LEAD,
            total_modules: 4,
            active_modules: 4,
            min_active: 4,
            pinned_streak: 0,
            slack_streak: 0,
            pending_growth: Vec::new(),
            window_id: 0,
        }
    }
}

impl CapacityRetarget {
    /// Feed one retarget window's chain-observable signals; returns the
    /// decisions taken this window (all deterministic, all integer).
    /// Line-for-line mirror of the reference `observe_window`.
    pub fn observe_window(&mut self, accepted: i64, staleness_4dp: i64) -> WindowEvents {
        self.window_id += 1;
        let (mut grew, mut froze, mut thawed, mut scheduled) = (0i64, 0i64, 0i64, 0i64);

        // activate any growth event whose announcement lead has elapsed
        let due = self.pending_growth.iter().filter(|&&w| w <= self.window_id).count();
        for _ in 0..due {
            self.total_modules += self.growth_bound;
            self.active_modules += self.growth_bound;
            grew += self.growth_bound;
        }
        self.pending_growth.retain(|&w| w > self.window_id);

        let d = retarget_decide(
            self.quota_4dp,
            self.pinned_streak,
            self.slack_streak,
            accepted,
            staleness_4dp,
            self.quota_min,
            self.quota_max,
            self.target_deltas,
            self.damp_div,
            self.stale_ceiling,
            self.k_sustain,
        );
        self.quota_4dp = d.quota_4dp;
        self.pinned_streak = d.pinned_streak;
        self.slack_streak = d.slack_streak;

        // recovery FIRST: thaw frozen modules before any new growth is considered
        if d.thaw_ok && self.active_modules < self.total_modules {
            self.active_modules += 1;
            thawed = 1;
            self.pinned_streak = 0; // thawing consumes the surplus signal
        } else if d.schedule && self.pending_growth.is_empty() {
            self.pending_growth.push(self.window_id + self.announce_lead);
            scheduled = 1;
            self.pinned_streak = 0;
            // growth resets the fast knob to mid-band: the bigger model absorbs
            // the surplus the quota ceiling could not
            self.quota_4dp = (self.quota_min + self.quota_max).div_euclid(2);
        }

        // decline: freeze active modules (total NEVER shrinks — the ratchet)
        if d.freeze && self.active_modules > self.min_active {
            self.active_modules -= 1;
            froze = 1;
            self.slack_streak = 0;
        }

        WindowEvents {
            window: self.window_id,
            grew,
            froze,
            thawed,
            scheduled,
            quota_4dp: self.quota_4dp,
            total: self.total_modules,
            active: self.active_modules,
        }
    }
}

/// Drive the controller with a synthetic fleet, integer-only: fleet values are
/// in 1e-2 units (100 == 1.0 fleet unit). Each window the fleet produces
/// ~ capacity/quota deltas, degraded by staleness when the ACTIVE model
/// outweighs the fleet. Deterministic — mirrors `rig/capacity.py::simulate`.
pub fn simulate(
    controller: &mut CapacityRetarget,
    fleet_trace: &[i64],
    per_unit: i64,
) -> Vec<WindowEvents> {
    let mut out = Vec::with_capacity(fleet_trace.len());
    let modules_per_fleet_unit: i64 = 4; // a fleet of 1.0 comfortably trains 4 modules
    for &fleet_2dp in fleet_trace {
        let capacity_4dp = fleet_2dp as i128 * per_unit as i128 * 100; // -> 4dp work units
        let mut accepted = capacity_4dp.div_euclid(controller.quota_4dp.max(1) as i128);
        // if the active model outweighs the fleet, deltas arrive late
        let load_4dp = (controller.active_modules as i128 * 10_000 * 100)
            .div_euclid(modules_per_fleet_unit as i128 * fleet_2dp.max(1) as i128);
        let staleness_4dp = (load_4dp - 10_000).clamp(0, 10_000);
        accepted = (accepted * (10_000 - staleness_4dp)).div_euclid(10_000);
        out.push(controller.observe_window(accepted as i64, staleness_4dp as i64));
    }
    out
}
