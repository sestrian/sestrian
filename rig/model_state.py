"""ModelState — the model's shape as consensus state (protocol v1, §3.1/§9.4a).

The chain state is still ONE flat int64 vector, but its layout is now governed
by a PAGE TABLE: page 0 is the backbone (one contiguous span), followed by one
page per (layer, expert) in creation order. `state_root` commits the page-Merkle
root over page bytes (rig/merkle.py); growth appends a leaf, so page ids and
existing proofs are stable forever.

ModelState carries the page table plus the capacity controller's fold state,
is advanced deterministically by every block (like the TokenLedger), and is
committed in the header as `model_root = sha256(canonical_json(ModelState))` —
recomputed AND committed, so any divergence in the fold is a loud validation
error instead of a silent fork.

Growth event lifecycle (all at retarget-window boundaries, h % W == 0):
  scheduled  — the controller pins at its ceiling for k_sustain windows; the
               event is recorded as [activation_window, layer, trigger_hash]
               with trigger = the scheduling block's prev_hash (committed and
               available during the fold; init grinding is profitless — the
               init distribution is symmetric and worthless at birth).
  activated  — at the boundary block of the activation window: this block's
               deltas aggregate over the OLD page set, then the new expert page
               is appended with a deterministic hash-stream init, then
               state_root commits the EXTENDED page set. The new page is
               claimable from the next block.
  frozen     — sustained deficit freezes grown pages LIFO (newest first);
               genesis pages never freeze. Frozen pages reject deltas but keep
               serving. Total page count never shrinks — the ratchet.

Everything here is integer arithmetic in a fixed order (Rust mirror:
node/core/src/model_state.rs, pinned by golden vectors).
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .capacity import (ANNOUNCE_LEAD, GROWTH_BOUND, K_SUSTAIN, QUOTA_MAX_4DP,
                       QUOTA_MIN_4DP, QUOTA_ONE_4DP, STALE_CEILING_4DP,
                       TARGET_DELTAS, retarget_decide)
from . import merkle

# Page-init distribution: uniform ±0.02 in fixed point (±0.02 * 65536 ≈ ±1311).
_INIT_SPAN = 2623            # (u % 2623) - 1311  ->  [-1311, 1311]
_INIT_HALF = 1311


@dataclass(frozen=True)
class ModelSpec:
    """The consensus-frozen shape parameters (a GENESIS constant, identical on
    every node; the client derives its torch architecture + permutation from
    this — consensus only ever needs spans and page lengths)."""
    n_layers: int
    d_model: int
    d_ff: int
    n_experts_initial: int
    e_max: int                  # router columns preallocated per layer
    backbone_params: int        # total backbone span length (client-derived, frozen)

    @property
    def expert_page_len(self) -> int:
        # W1 (d*d_ff) + b1 (d_ff) + W2 (d_ff*d) + b2 (d), row-major, in this order
        return self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model


@dataclass(frozen=True)
class GenesisParams:
    """Retarget constants — genesis parameters, NOT part of per-block state."""
    spec: ModelSpec
    retarget_window: int = 16          # blocks per window
    target_deltas: int = TARGET_DELTAS
    quota_min_4dp: int = QUOTA_MIN_4DP
    quota_max_4dp: int = QUOTA_MAX_4DP
    stale_ceiling_4dp: int = STALE_CEILING_4DP
    k_sustain: int = K_SUSTAIN
    growth_bound: int = GROWTH_BOUND
    announce_lead: int = ANNOUNCE_LEAD
    # PROTOCOL v2 — the delta ENVELOPE (Bitcoin's block-size lesson): a delta
    # body may never carry more than this many nonzero coordinates (~8MB raw
    # sparse). Consensus must never scale the payload: a rising quota therefore
    # forces miners to claim FEWER pages and train them DENSER (specialization)
    # instead of uploading more bytes. At quota q the claimable span is bounded
    # by delta_max_nnz * 1e6 / q_4dp params. Growth events relieve sustained
    # saturation by widening the model — bytes per block stay bounded forever.
    delta_max_nnz: int = 1_000_000
    # PROTOCOL v3 (page committees, stage 1): the LEARNING GATE. From this
    # height the fold tracks the window's summed committed scores and growth
    # requires the network to be measurably learning (win_score_sum > 0)
    # instead of the per-delta staleness ceiling — which conflated "junk work"
    # with "signal below the eval noise floor" (found live). Scheduled via the
    # version schedule: a coordinated upgrade, not a re-genesis.
    v3_height: int = 288
    # PROTOCOL v4 (the QUORUM gate): activation height for the scheduled
    # upgrade that makes the learning gate resistant to a lying proposer.
    # v3 gated growth on win_score_sum > 0 — a SUM over the window of
    # proposer-COMMITTED scores, whose accuracy consensus cannot check. A
    # single Byzantine proposer committing one positive micro-nat therefore
    # forced growth on a plateaued network (1-of-N; proven in
    # rig/redteam_gate.py). From v4 the gate counts DISTINCT proposers who
    # committed a positive score in the window and requires `growth_quorum`
    # of them, so forcing growth costs winning that many blocks with that
    # many keys — priced by stake, like everything else here. Honest limit:
    # this raises the price, it does not make the gate trustless. Only the
    # multi-evaluator committee does that.
    v4_height: int = 608
    # Distinct positive-scoring proposers a window needs before growth may be
    # scheduled. 3 matches the >=3-honest-miners regime the trimmed-mean
    # aggregation already assumes; a network with fewer miners than this sets
    # it to its fleet size (see NetworkParams) and raises it as miners join.
    growth_quorum: int = 3


ACTIVE, FROZEN = "A", "F"


class ModelState:
    """pages: list of [start, end, kind, layer, expert, status]; backbone uses
    layer = expert = -1. Plus the controller fold fields. Canonical JSON is the
    committed form; keep field handling in exact sync with the Rust mirror."""

    def __init__(self, pages, quota_4dp=QUOTA_ONE_4DP, pinned_streak=0,
                 slack_streak=0, pending_growth=None, window_id=0,
                 win_accepted=0, win_zero_scored=0, events_total=0,
                 win_score_sum=0, rev=2, win_scorers=None):
        self.pages = [list(p) for p in pages]
        self.quota_4dp = int(quota_4dp)
        self.pinned_streak = int(pinned_streak)
        self.slack_streak = int(slack_streak)
        self.pending_growth = [list(e) for e in (pending_growth or [])]
        self.window_id = int(window_id)
        self.win_accepted = int(win_accepted)
        self.win_zero_scored = int(win_zero_scored)
        # v3: the window's summed committed scores (the learning gate) and the
        # state's protocol rev. BOTH enter the canonical JSON only from rev 3,
        # so every pre-activation model_root is byte-identical to v2.
        self.win_score_sum = int(win_score_sum)
        self.rev = int(rev)
        # v4: the DISTINCT proposers that committed a positive score this
        # window, capped at growth_quorum entries (once the quorum is met no
        # more are needed, so the state stays tiny). Enters the canonical JSON
        # only from rev 4, keeping every pre-activation model_root identical.
        self.win_scorers = [str(x) for x in (win_scorers or [])]
        self.events_total = int(events_total)

    # ---- construction ----------------------------------------------------
    @staticmethod
    def genesis(spec: ModelSpec) -> "ModelState":
        pages = [[0, spec.backbone_params, "backbone", -1, -1, ACTIVE]]
        off = spec.backbone_params
        for l in range(spec.n_layers):
            for e in range(spec.n_experts_initial):
                pages.append([off, off + spec.expert_page_len, "expert", l, e, ACTIVE])
                off += spec.expert_page_len
        return ModelState(pages)

    def copy(self) -> "ModelState":
        return ModelState(self.pages, self.quota_4dp, self.pinned_streak,
                          self.slack_streak, self.pending_growth, self.window_id,
                          self.win_accepted, self.win_zero_scored,
                          self.events_total, self.win_score_sum, self.rev,
                          self.win_scorers)

    # ---- commitments -----------------------------------------------------
    def canonical_json(self) -> str:
        d = {
            "events_total": self.events_total,
            "pages": self.pages,
            "pending_growth": self.pending_growth,
            "pinned_streak": self.pinned_streak,
            "quota_4dp": self.quota_4dp,
            "slack_streak": self.slack_streak,
            "win_accepted": self.win_accepted,
            "win_zero_scored": self.win_zero_scored,
            "window_id": self.window_id,
        }
        if self.rev >= 3:
            d["rev"] = self.rev
            d["win_score_sum"] = self.win_score_sum
        if self.rev >= 4:
            d["win_scorers"] = self.win_scorers
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def model_root(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    # ---- queries ---------------------------------------------------------
    def dim(self) -> int:
        return self.pages[-1][1]

    def page_span(self, page_id: int) -> tuple[int, int]:
        p = self.pages[page_id]
        return p[0], p[1]

    def is_active(self, page_id: int) -> bool:
        return 0 <= page_id < len(self.pages) and self.pages[page_id][5] == ACTIVE

    def genesis_page_count(self, spec: ModelSpec) -> int:
        return 1 + spec.n_layers * spec.n_experts_initial

    def n_expert_pages(self) -> int:
        return sum(1 for p in self.pages if p[2] == "expert")

    def n_active_expert_pages(self) -> int:
        return sum(1 for p in self.pages if p[2] == "expert" and p[5] == ACTIVE)

    def claimed_params(self, page_ids: list[int]) -> int:
        return sum(self.pages[p][1] - self.pages[p][0] for p in page_ids)

    def required_nnz(self, page_ids: list[int]) -> int:
        """The work quota: a delta claiming these pages must have at least this
        many nonzero coordinates. quota 1.0 (10_000) => 1% density."""
        return self.claimed_params(page_ids) * self.quota_4dp // 1_000_000


def page_state_root(w_int: np.ndarray, state: ModelState) -> str:
    """The v1 state commitment: page-Merkle root over page bytes, page-id order."""
    leaves = [w_int[p[0]:p[1]].tobytes() for p in state.pages]
    return merkle.root(leaves).hex()


def page_init(trigger_hex: str, page_id: int, spec: ModelSpec) -> np.ndarray:
    """Deterministic new-expert init: a SHA-256 hash-stream (four big-endian u64
    lanes per digest), byte-identical in Python and Rust — no platform RNG.
    Weight ranges draw uniform ±0.02 in fixed point; bias ranges are zero."""
    n = spec.expert_page_len
    d, f = spec.d_model, spec.d_ff
    w1_end = d * f
    b1_end = w1_end + f
    w2_end = b1_end + f * d
    out = np.zeros(n, dtype=np.int64)
    prefix = f"sestrian-page-init|v1|{trigger_hex}|{page_id}|".encode()
    for blk in range((n + 3) // 4):
        digest = hashlib.sha256(prefix + str(blk).encode()).digest()
        for lane in range(4):
            j = blk * 4 + lane
            if j >= n:
                break
            if j < w1_end or (b1_end <= j < w2_end):        # weight coordinate
                u = int.from_bytes(digest[8 * lane:8 * lane + 8], "big")
                out[j] = (u % _INIT_SPAN) - _INIT_HALF
            # bias coordinates stay 0
    return out


def fold(state: ModelState, params: GenesisParams, height: int, n_txs: int,
         zero_scored: int, prev_hash: str,
         score_sum: int = 0, proposer: str = "") -> tuple[ModelState, list]:
    """The deterministic per-block ModelState transition. Returns
    (post_state, activations) where activations = [(page_id, layer, expert_idx,
    trigger_hex), ...] for expert pages appended by THIS block — the caller
    extends the weight vector with `page_init` for each, AFTER aggregating this
    block's deltas over the OLD page set and BEFORE computing state_root.

    Restart-equivalence invariant: folding blocks one at a time from any prefix
    state must equal folding them all from genesis (tests/test_model_state.py).
    """
    s = state.copy()
    # v3 activates at its scheduled height: the state carries rev + the
    # window's score sum from then on (both enter the canonical JSON, so the
    # first v3 block's model_root is the coordinated-upgrade boundary).
    if height >= params.v3_height:
        s.rev = 3
    if height >= params.v4_height:
        s.rev = 4
    s.win_accepted += n_txs
    s.win_zero_scored += zero_scored
    if s.rev >= 3:
        s.win_score_sum += int(score_sum)
    # v4: record this block's proposer as a DISTINCT positive scorer. Capped
    # at the quorum — past it the answer cannot change, so the list never
    # grows beyond a handful of entries. An empty proposer never counts (the
    # gate must not be openable by an unattributable block).
    if s.rev >= 4 and int(score_sum) > 0 and proposer \
            and proposer not in s.win_scorers \
            and len(s.win_scorers) < params.growth_quorum:
        s.win_scorers.append(proposer)
    activations: list = []

    W = params.retarget_window
    if height > 0 and height % W == 0:
        s.window_id += 1
        # 1. activate any growth event whose announcement lead has elapsed
        due = [e for e in s.pending_growth if e[0] <= s.window_id]
        s.pending_growth = [e for e in s.pending_growth if e[0] > s.window_id]
        for _w, layer, trigger in due:
            spec = params.spec
            start = s.dim()
            expert_idx = max((p[4] for p in s.pages
                              if p[2] == "expert" and p[3] == layer), default=-1) + 1
            page_id = len(s.pages)
            s.pages.append([start, start + spec.expert_page_len, "expert",
                            layer, expert_idx, ACTIVE])
            activations.append((page_id, layer, expert_idx, trigger))

        # 2. the window decision (shared math with rig/capacity.py)
        staleness_4dp = s.win_zero_scored * 10_000 // max(1, s.win_accepted)
        # v3 LEARNING GATE: the window must show the network improving at all
        # (summed committed scores > 0) — per-delta staleness conflated junk
        # with signal-below-noise-floor. Encoded as a staleness override so
        # retarget_decide's surplus math is shared verbatim across revs.
        if s.rev >= 4:
            # v4 QUORUM gate: distinct proposers, not a forgeable sum.
            staleness_4dp = (0 if len(s.win_scorers) >= params.growth_quorum
                             else 10_000)
        elif s.rev >= 3:
            staleness_4dp = 0 if s.win_score_sum > 0 else 10_000
        d = retarget_decide(s.quota_4dp, s.pinned_streak, s.slack_streak,
                            s.win_accepted, staleness_4dp,
                            quota_min=params.quota_min_4dp,
                            quota_max=params.quota_max_4dp,
                            target_deltas=params.target_deltas,
                            stale_ceiling=params.stale_ceiling_4dp,
                            k_sustain=params.k_sustain)
        s.quota_4dp = d["quota_4dp"]
        s.pinned_streak = d["pinned_streak"]
        s.slack_streak = d["slack_streak"]

        genesis_pages = 1 + params.spec.n_layers * params.spec.n_experts_initial
        frozen_grown = [i for i in range(genesis_pages, len(s.pages))
                        if s.pages[i][5] == FROZEN]
        active_grown = [i for i in range(genesis_pages, len(s.pages))
                        if s.pages[i][5] == ACTIVE]

        # recovery FIRST: thaw frozen pages before any new growth is considered
        # (reverse of the LIFO freeze order = lowest frozen id thaws first)
        if d["thaw_ok"] and frozen_grown:
            s.pages[min(frozen_grown)][5] = ACTIVE
            s.pinned_streak = 0            # thawing consumes the surplus signal
        elif d["schedule"] and not s.pending_growth:
            layer = s.events_total % params.spec.n_layers
            s.pending_growth.append([s.window_id + params.announce_lead,
                                     layer, prev_hash])
            s.events_total += 1
            s.pinned_streak = 0
            # growth resets the fast knob to mid-band
            s.quota_4dp = (params.quota_min_4dp + params.quota_max_4dp) // 2

        # decline: freeze grown pages LIFO (newest first); genesis never freezes
        if d["freeze"] and active_grown:
            s.pages[max(active_grown)][5] = FROZEN
            s.slack_streak = 0

        # 3. window accumulators reset
        s.win_accepted = 0
        s.win_zero_scored = 0
        s.win_score_sum = 0
        s.win_scorers = []

    return s, activations
