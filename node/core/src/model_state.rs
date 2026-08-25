//! ModelState — the model's shape as consensus state (protocol v1, §3.1/§9.4a).
//! Mirrors `rig/model_state.py` bit-exactly (pinned by golden vectors).
//!
//! The chain state is still ONE flat int64 vector, but its layout is governed
//! by a PAGE TABLE: page 0 is the backbone, followed by one page per
//! (layer, expert) in creation order. `state_root` commits the page-Merkle root
//! over page bytes (`merkle.rs`); growth appends a leaf, so page ids and
//! existing proofs are stable forever.
//!
//! ModelState carries the page table plus the capacity controller's fold state,
//! is advanced deterministically by every block (like the TokenLedger), and is
//! committed in the header as `model_root = sha256(canonical_json)` —
//! recomputed AND committed, so any divergence in the fold is a loud validation
//! error instead of a silent fork.

use crate::capacity::{
    retarget_decide, ANNOUNCE_LEAD, DAMP_DIV, GROWTH_BOUND, K_SUSTAIN, QUOTA_MAX_4DP,
    QUOTA_MIN_4DP, QUOTA_ONE_4DP, STALE_CEILING_4DP, TARGET_DELTAS,
};
use crate::{int64_bytes, merkle};
use sha2::{Digest, Sha256};

// Page-init distribution: uniform ±0.02 in fixed point (±0.02 * 65536 ≈ ±1311).
const INIT_SPAN: u64 = 2623; // (u % 2623) - 1311  ->  [-1311, 1311]
const INIT_HALF: i64 = 1311;

pub const ACTIVE: &str = "A";
pub const FROZEN: &str = "F";

/// The consensus-frozen shape parameters (a GENESIS constant, identical on
/// every node; the client derives its torch architecture + permutation from
/// this — consensus only ever needs spans and page lengths).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    pub n_layers: u64,
    pub d_model: u64,
    pub d_ff: u64,
    pub n_experts_initial: u64,
    pub e_max: u64,           // router columns preallocated per layer
    pub backbone_params: u64, // total backbone span length (client-derived, frozen)
}

impl ModelSpec {
    /// W1 (d*d_ff) + b1 (d_ff) + W2 (d_ff*d) + b2 (d), row-major, in this order.
    pub fn expert_page_len(&self) -> u64 {
        self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model
    }
}

/// Retarget constants — genesis parameters, NOT part of per-block state.
#[derive(Clone, Debug)]
pub struct GenesisParams {
    pub spec: ModelSpec,
    pub retarget_window: u64, // blocks per window
    pub target_deltas: i64,
    pub quota_min_4dp: i64,
    pub quota_max_4dp: i64,
    pub stale_ceiling_4dp: i64,
    pub k_sustain: i64,
    pub growth_bound: i64,
    pub announce_lead: u64,
    /// PROTOCOL v2 — the delta ENVELOPE: a body may never carry more nonzero
    /// coordinates than this (~8MB raw sparse). The payload never scales with
    /// quota: a rising quota narrows the claimable span (specialization)
    /// instead of fattening the wire. Bitcoin's block-size lesson.
    pub delta_max_nnz: u64,
    /// PROTOCOL v3 (the learning gate): activation height for the scheduled
    /// upgrade. From here the fold tracks the window's summed committed
    /// scores and growth requires win_score_sum > 0 instead of the per-delta
    /// staleness ceiling (which conflated junk with signal below the eval
    /// noise floor — found live).
    pub v3_height: u64,
    /// PROTOCOL v4 (the QUORUM gate): activation height for the upgrade that
    /// makes the learning gate resistant to a lying proposer. v3 gated growth
    /// on win_score_sum > 0 — a SUM of proposer-COMMITTED scores, whose
    /// accuracy consensus cannot check — so ONE Byzantine proposer forced
    /// growth on a plateaued network (proven in rig/redteam_gate.py). From v4
    /// the gate counts DISTINCT positive-scoring proposers and requires
    /// `growth_quorum` of them. This prices the attack (win that many blocks
    /// with that many keys); it does not make the gate trustless — only the
    /// multi-evaluator committee does.
    pub v4_height: u64,
    /// Distinct positive-scoring proposers a window needs before growth may be
    /// scheduled. 3 matches the >=3-honest-miners regime the trimmed mean
    /// already assumes; a smaller fleet ships a smaller value and raises it.
    pub growth_quorum: usize,
}

impl GenesisParams {
    pub fn new(spec: ModelSpec) -> Self {
        GenesisParams {
            spec,
            retarget_window: 16,
            target_deltas: TARGET_DELTAS,
            quota_min_4dp: QUOTA_MIN_4DP,
            quota_max_4dp: QUOTA_MAX_4DP,
            stale_ceiling_4dp: STALE_CEILING_4DP,
            k_sustain: K_SUSTAIN,
            growth_bound: GROWTH_BOUND,
            announce_lead: ANNOUNCE_LEAD,
            delta_max_nnz: 1_000_000,
            v3_height: 288,
            v4_height: 608,
            growth_quorum: 3,
        }
    }
}

/// One page-table entry: `[start, end, kind, layer, expert, status]` in the
/// canonical JSON; the backbone uses layer = expert = -1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    pub start: u64,
    pub end: u64,
    pub kind: String, // "backbone" | "expert"
    pub layer: i64,
    pub expert: i64,
    pub status: String, // "A" | "F"
}

/// An expert page appended by a block's fold: (page_id, layer, expert, trigger).
/// The caller extends the weight vector with `page_init` for each, AFTER
/// aggregating the block's deltas over the OLD page set and BEFORE the root.
pub type Activation = (u64, i64, i64, String);

/// The page table plus the controller fold fields. Canonical JSON is the
/// committed form; keep field handling in exact sync with the Python spec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelState {
    pub pages: Vec<Page>,
    pub quota_4dp: i64,
    pub pinned_streak: i64,
    pub slack_streak: i64,
    pub pending_growth: Vec<(u64, i64, String)>, // [activation_window, layer, trigger]
    pub window_id: u64,
    pub win_accepted: u64,
    pub win_zero_scored: u64,
    /// v3: the window's summed committed scores (the learning gate) and the
    /// state's protocol rev. Both enter the canonical JSON only from rev 3,
    /// keeping every pre-activation model_root byte-identical.
    pub win_score_sum: u64,
    pub rev: u64,
    /// v4: DISTINCT proposers that committed a positive score this window,
    /// capped at growth_quorum entries. Enters the canonical JSON only from
    /// rev 4, keeping every pre-activation model_root byte-identical.
    pub win_scorers: Vec<String>,
    pub events_total: u64,
}

impl ModelState {
    // ---- construction ----------------------------------------------------
    pub fn genesis(spec: &ModelSpec) -> ModelState {
        let mut pages = vec![Page {
            start: 0,
            end: spec.backbone_params,
            kind: "backbone".into(),
            layer: -1,
            expert: -1,
            status: ACTIVE.into(),
        }];
        let mut off = spec.backbone_params;
        for l in 0..spec.n_layers {
            for e in 0..spec.n_experts_initial {
                pages.push(Page {
                    start: off,
                    end: off + spec.expert_page_len(),
                    kind: "expert".into(),
                    layer: l as i64,
                    expert: e as i64,
                    status: ACTIVE.into(),
                });
                off += spec.expert_page_len();
            }
        }
        ModelState {
            pages,
            quota_4dp: QUOTA_ONE_4DP,
            pinned_streak: 0,
            slack_streak: 0,
            pending_growth: Vec::new(),
            window_id: 0,
            win_accepted: 0,
            win_zero_scored: 0,
            events_total: 0,
            win_score_sum: 0,
            win_scorers: Vec::new(),
            rev: 2,
        }
    }

    // ---- commitments -----------------------------------------------------
    /// Byte-identical to the Python reference's
    /// `json.dumps({...}, sort_keys=True, separators=(",", ":"))` — keys in
    /// alphabetical order, compact separators, page entries as JSON arrays.
    pub fn canonical_json(&self) -> String {
        let pages: Vec<String> = self
            .pages
            .iter()
            .map(|p| {
                format!(
                    "[{},{},\"{}\",{},{},\"{}\"]",
                    p.start, p.end, p.kind, p.layer, p.expert, p.status
                )
            })
            .collect();
        let pending: Vec<String> = self
            .pending_growth
            .iter()
            .map(|(w, l, t)| format!("[{},{},\"{}\"]", w, l, t))
            .collect();
        if self.rev >= 4 {
            let scorers: Vec<String> =
                self.win_scorers.iter().map(|x| format!("\"{}\"", x)).collect();
            format!(
                "{{\"events_total\":{},\"pages\":[{}],\"pending_growth\":[{}],\"pinned_streak\":{},\"quota_4dp\":{},\"rev\":{},\"slack_streak\":{},\"win_accepted\":{},\"win_score_sum\":{},\"win_scorers\":[{}],\"win_zero_scored\":{},\"window_id\":{}}}",
                self.events_total,
                pages.join(","),
                pending.join(","),
                self.pinned_streak,
                self.quota_4dp,
                self.rev,
                self.slack_streak,
                self.win_accepted,
                self.win_score_sum,
                scorers.join(","),
                self.win_zero_scored,
                self.window_id
            )
        } else if self.rev >= 3 {
            format!(
                "{{\"events_total\":{},\"pages\":[{}],\"pending_growth\":[{}],\"pinned_streak\":{},\"quota_4dp\":{},\"rev\":{},\"slack_streak\":{},\"win_accepted\":{},\"win_score_sum\":{},\"win_zero_scored\":{},\"window_id\":{}}}",
                self.events_total,
                pages.join(","),
                pending.join(","),
                self.pinned_streak,
                self.quota_4dp,
                self.rev,
                self.slack_streak,
                self.win_accepted,
                self.win_score_sum,
                self.win_zero_scored,
                self.window_id
            )
        } else {
            format!(
                "{{\"events_total\":{},\"pages\":[{}],\"pending_growth\":[{}],\"pinned_streak\":{},\"quota_4dp\":{},\"slack_streak\":{},\"win_accepted\":{},\"win_zero_scored\":{},\"window_id\":{}}}",
                self.events_total,
                pages.join(","),
                pending.join(","),
                self.pinned_streak,
                self.quota_4dp,
                self.slack_streak,
                self.win_accepted,
                self.win_zero_scored,
                self.window_id
            )
        }
    }

    pub fn model_root(&self) -> String {
        hex::encode(Sha256::digest(self.canonical_json().as_bytes()))
    }

    /// Parse a ModelState back from its canonical JSON form (snapshots, tests).
    pub fn from_json_value(v: &serde_json::Value) -> Option<ModelState> {
        let pages = v
            .get("pages")?
            .as_array()?
            .iter()
            .map(|p| {
                let a = p.as_array()?;
                Some(Page {
                    start: a.first()?.as_u64()?,
                    end: a.get(1)?.as_u64()?,
                    kind: a.get(2)?.as_str()?.to_string(),
                    layer: a.get(3)?.as_i64()?,
                    expert: a.get(4)?.as_i64()?,
                    status: a.get(5)?.as_str()?.to_string(),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let pending = v
            .get("pending_growth")?
            .as_array()?
            .iter()
            .map(|e| {
                let a = e.as_array()?;
                Some((a.first()?.as_u64()?, a.get(1)?.as_i64()?, a.get(2)?.as_str()?.to_string()))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(ModelState {
            pages,
            quota_4dp: v.get("quota_4dp")?.as_i64()?,
            pinned_streak: v.get("pinned_streak")?.as_i64()?,
            slack_streak: v.get("slack_streak")?.as_i64()?,
            pending_growth: pending,
            window_id: v.get("window_id")?.as_u64()?,
            win_accepted: v.get("win_accepted")?.as_u64()?,
            win_zero_scored: v.get("win_zero_scored")?.as_u64()?,
            events_total: v.get("events_total")?.as_u64()?,
            win_score_sum: v.get("win_score_sum").and_then(|x| x.as_u64())
                .unwrap_or(0),
            rev: v.get("rev").and_then(|x| x.as_u64()).unwrap_or(2),
            win_scorers: v.get("win_scorers").and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|e| e.as_str().map(String::from))
                     .collect())
                .unwrap_or_default(),
        })
    }

    // ---- queries ---------------------------------------------------------
    pub fn dim(&self) -> u64 {
        self.pages.last().expect("page table never empty").end
    }

    pub fn page_span(&self, page_id: usize) -> (u64, u64) {
        let p = &self.pages[page_id];
        (p.start, p.end)
    }

    pub fn is_active(&self, page_id: usize) -> bool {
        page_id < self.pages.len() && self.pages[page_id].status == ACTIVE
    }

    pub fn genesis_page_count(spec: &ModelSpec) -> usize {
        (1 + spec.n_layers * spec.n_experts_initial) as usize
    }

    pub fn n_expert_pages(&self) -> usize {
        self.pages.iter().filter(|p| p.kind == "expert").count()
    }

    pub fn n_active_expert_pages(&self) -> usize {
        self.pages.iter().filter(|p| p.kind == "expert" && p.status == ACTIVE).count()
    }

    pub fn claimed_params(&self, page_ids: &[u32]) -> u64 {
        page_ids.iter().map(|&p| {
            let pg = &self.pages[p as usize];
            pg.end - pg.start
        }).sum()
    }

    /// The work quota: a delta claiming these pages must have at least this
    /// many nonzero coordinates. quota 1.0 (10_000) => 1% density.
    pub fn required_nnz(&self, page_ids: &[u32]) -> u64 {
        (self.claimed_params(page_ids) as i128 * self.quota_4dp as i128)
            .div_euclid(1_000_000) as u64
    }
}

/// The v1 state commitment: page-Merkle root over page bytes, page-id order.
pub fn page_state_root(w: &[i64], state: &ModelState) -> String {
    let page_bytes: Vec<Vec<u8>> = state
        .pages
        .iter()
        .map(|p| int64_bytes(&w[p.start as usize..p.end as usize]))
        .collect();
    let leaves: Vec<&[u8]> = page_bytes.iter().map(|b| b.as_slice()).collect();
    hex::encode(merkle::root(&leaves))
}

/// Deterministic new-expert init: a SHA-256 hash-stream (four big-endian u64
/// lanes per digest), byte-identical to `rig/model_state.py::page_init` — no
/// platform RNG. Weight ranges draw uniform ±0.02 in fixed point; bias ranges
/// are zero.
pub fn page_init(trigger_hex: &str, page_id: u64, spec: &ModelSpec) -> Vec<i64> {
    let n = spec.expert_page_len() as usize;
    let (d, f) = (spec.d_model as usize, spec.d_ff as usize);
    let w1_end = d * f;
    let b1_end = w1_end + f;
    let w2_end = b1_end + f * d;
    let mut out = vec![0i64; n];
    let prefix = format!("sestrian-page-init|v1|{trigger_hex}|{page_id}|");
    for blk in 0..n.div_ceil(4) {
        let digest: [u8; 32] =
            Sha256::digest(format!("{prefix}{blk}").as_bytes()).into();
        for lane in 0..4 {
            let j = blk * 4 + lane;
            if j >= n {
                break;
            }
            if j < w1_end || (b1_end..w2_end).contains(&j) {
                // weight coordinate
                let u = u64::from_be_bytes(digest[8 * lane..8 * lane + 8].try_into().unwrap());
                out[j] = (u % INIT_SPAN) as i64 - INIT_HALF;
            }
            // bias coordinates stay 0
        }
    }
    out
}

/// The deterministic per-block ModelState transition — a line-by-line mirror of
/// `rig/model_state.py::fold`. Returns (post_state, activations); the caller
/// extends the weight vector with `page_init` for each activation, AFTER
/// aggregating this block's deltas over the OLD page set and BEFORE computing
/// state_root.
///
/// Restart-equivalence invariant: folding blocks one at a time from any prefix
/// state must equal folding them all from genesis.
pub fn fold(
    state: &ModelState,
    params: &GenesisParams,
    height: u64,
    n_txs: u64,
    zero_scored: u64,
    prev_hash: &str,
    score_sum: u64,
    proposer: &str,
) -> (ModelState, Vec<Activation>) {
    let mut s = state.clone();
    // v3 activates at its scheduled height (GenesisParams::v3_height)
    if height >= params.v3_height {
        s.rev = 3;
    }
    if height >= params.v4_height {
        s.rev = 4;
    }
    s.win_accepted += n_txs;
    s.win_zero_scored += zero_scored;
    if s.rev >= 3 {
        s.win_score_sum += score_sum;
    }
    // v4: record this block's proposer as a DISTINCT positive scorer, capped
    // at the quorum (past it the answer cannot change). An empty proposer
    // never counts — the gate must not be openable by an unattributable block.
    if s.rev >= 4
        && score_sum > 0
        && !proposer.is_empty()
        && !s.win_scorers.iter().any(|x| x == proposer)
        && s.win_scorers.len() < params.growth_quorum
    {
        s.win_scorers.push(proposer.to_string());
    }
    let mut activations: Vec<Activation> = Vec::new();

    let w = params.retarget_window;
    if height > 0 && height % w == 0 {
        s.window_id += 1;
        // 1. activate any growth event whose announcement lead has elapsed
        let (due, still): (Vec<_>, Vec<_>) =
            s.pending_growth.iter().cloned().partition(|e| e.0 <= s.window_id);
        s.pending_growth = still;
        for (_w, layer, trigger) in due {
            let spec = &params.spec;
            let start = s.dim();
            let expert_idx = s
                .pages
                .iter()
                .filter(|p| p.kind == "expert" && p.layer == layer)
                .map(|p| p.expert)
                .max()
                .unwrap_or(-1)
                + 1;
            let page_id = s.pages.len() as u64;
            s.pages.push(Page {
                start,
                end: start + spec.expert_page_len(),
                kind: "expert".into(),
                layer,
                expert: expert_idx,
                status: ACTIVE.into(),
            });
            activations.push((page_id, layer, expert_idx, trigger));
        }

        // 2. the window decision (shared math with capacity.rs)
        let mut staleness_4dp = (s.win_zero_scored as i128 * 10_000)
            .div_euclid(1.max(s.win_accepted as i128)) as i64;
        // v3 LEARNING GATE: growth requires the window to show the network
        // improving at all (summed committed scores > 0); per-delta staleness
        // conflated junk with signal below the eval noise floor (found live).
        if s.rev >= 3 {
            staleness_4dp = if s.win_score_sum > 0 { 0 } else { 10_000 };
        }
        if s.rev >= 4 {
            // v4 QUORUM gate: distinct proposers, not a forgeable sum.
            staleness_4dp = if s.win_scorers.len() >= params.growth_quorum {
                0
            } else {
                10_000
            };
        }
        let d = retarget_decide(
            s.quota_4dp,
            s.pinned_streak,
            s.slack_streak,
            s.win_accepted as i64,
            staleness_4dp,
            params.quota_min_4dp,
            params.quota_max_4dp,
            params.target_deltas,
            DAMP_DIV,
            params.stale_ceiling_4dp,
            params.k_sustain,
        );
        s.quota_4dp = d.quota_4dp;
        s.pinned_streak = d.pinned_streak;
        s.slack_streak = d.slack_streak;

        let genesis_pages = ModelState::genesis_page_count(&params.spec);
        let frozen_grown: Vec<usize> = (genesis_pages..s.pages.len())
            .filter(|&i| s.pages[i].status == FROZEN)
            .collect();
        let active_grown: Vec<usize> = (genesis_pages..s.pages.len())
            .filter(|&i| s.pages[i].status == ACTIVE)
            .collect();

        // recovery FIRST: thaw frozen pages before any new growth is considered
        // (reverse of the LIFO freeze order = lowest frozen id thaws first)
        if d.thaw_ok && !frozen_grown.is_empty() {
            s.pages[frozen_grown[0]].status = ACTIVE.into();
            s.pinned_streak = 0; // thawing consumes the surplus signal
        } else if d.schedule && s.pending_growth.is_empty() {
            let layer = (s.events_total % params.spec.n_layers) as i64;
            s.pending_growth
                .push((s.window_id + params.announce_lead, layer, prev_hash.to_string()));
            s.events_total += 1;
            s.pinned_streak = 0;
            // growth resets the fast knob to mid-band
            s.quota_4dp = (params.quota_min_4dp + params.quota_max_4dp).div_euclid(2);
        }

        // decline: freeze grown pages LIFO (newest first); genesis never freezes
        if d.freeze && !active_grown.is_empty() {
            s.pages[*active_grown.last().unwrap()].status = FROZEN.into();
            s.slack_streak = 0;
        }

        // 3. window accumulators reset
        s.win_accepted = 0;
        s.win_zero_scored = 0;
        s.win_score_sum = 0;
        s.win_scorers.clear();
    }

    (s, activations)
}
