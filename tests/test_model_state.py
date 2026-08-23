"""ModelState (protocol v1): the page table + controller fold that governs the
model's shape as consensus state. Covers: genesis layout, canonical commitment,
fold determinism + restart equivalence (fold(prefix)+fold(suffix) == fold(all)),
window boundaries, growth scheduling/activation timing, deterministic page
init bytes, the ratchet, LIFO freeze / reverse thaw, and the work quota."""

import numpy as np

from rig import merkle
from rig.model_state import (ACTIVE, FROZEN, GenesisParams, ModelSpec,
                             ModelState, fold, page_init, page_state_root)

# a toy spec: tiny numbers, same code paths
SPEC = ModelSpec(n_layers=2, d_model=4, d_ff=8, n_experts_initial=2, e_max=4,
                 backbone_params=100)
# short windows so tests cross many boundaries quickly
PARAMS = GenesisParams(spec=SPEC, retarget_window=4, target_deltas=8,
                       k_sustain=3, announce_lead=2)

EXPERT_LEN = SPEC.expert_page_len          # 4*8 + 8 + 8*4 + 4 = 76


def genesis():
    return ModelState.genesis(SPEC)


def drive(state, heights, n_txs, zero_scored=0, prev_hash="ab" * 32):
    """Fold a run of blocks with constant signals; returns (state, activations)."""
    acts = []
    for h in heights:
        state, a = fold(state, PARAMS, h, n_txs, zero_scored, prev_hash)
        acts.extend(a)
    return state, acts


def test_genesis_layout():
    s = genesis()
    assert s.pages[0] == [0, 100, "backbone", -1, -1, ACTIVE]
    assert len(s.pages) == 1 + 2 * 2
    # contiguous, ordered spans
    for a, b in zip(s.pages, s.pages[1:]):
        assert a[1] == b[0]
    assert s.dim() == 100 + 4 * EXPERT_LEN
    assert s.n_expert_pages() == 4 and s.n_active_expert_pages() == 4


def test_canonical_commitment_deterministic_and_sensitive():
    a, b = genesis(), genesis()
    assert a.canonical_json() == b.canonical_json()
    assert a.model_root() == b.model_root()
    b.quota_4dp += 1
    assert a.model_root() != b.model_root()


def test_page_state_root_is_page_merkle():
    s = genesis()
    w = np.arange(s.dim(), dtype=np.int64)
    leaves = [w[p[0]:p[1]].tobytes() for p in s.pages]
    assert page_state_root(w, s) == merkle.root(leaves).hex()
    # perturbing one page changes the root
    w2 = w.copy()
    w2[0] += 1
    assert page_state_root(w2, s) != page_state_root(w, s)


def test_fold_restart_equivalence():
    """Folding block-by-block from any prefix must equal folding from genesis —
    the invariant that makes snapshots/fast-boot safe."""
    heights = list(range(1, 41))
    signals = [(h, (h * 7) % 12, (h * 3) % 4) for h in heights]
    full = genesis()
    for h, n, z in signals:
        full, _ = fold(full, PARAMS, h, n, z, f"{h:064x}")
    for cut in (1, 7, 16, 33):
        prefix = genesis()
        for h, n, z in signals[:cut]:
            prefix, _ = fold(prefix, PARAMS, h, n, z, f"{h:064x}")
        resumed = prefix
        for h, n, z in signals[cut:]:
            resumed, _ = fold(resumed, PARAMS, h, n, z, f"{h:064x}")
        assert resumed.model_root() == full.model_root()


def test_window_boundary_only_at_multiples():
    s = genesis()
    s1, _ = fold(s, PARAMS, 1, 5, 0, "00" * 32)
    assert s1.window_id == 0 and s1.win_accepted == 5
    s2, _ = fold(s1, PARAMS, 4, 5, 0, "00" * 32)     # boundary (W=4)
    assert s2.window_id == 1 and s2.win_accepted == 0


def test_growth_schedules_then_activates_after_lead():
    s = genesis()
    trigger = "cd" * 32
    h, acts_all = 0, []
    sched_window = None
    # saturating signal: many accepted deltas, no staleness
    while h < 200 and not acts_all:
        h += 1
        s, acts = fold(s, PARAMS, h, 64, 0, trigger)
        if s.pending_growth and sched_window is None:
            sched_window = s.window_id
            assert s.pending_growth[0][0] == sched_window + PARAMS.announce_lead
            assert s.pending_growth[0][2] == trigger
        acts_all.extend(acts)
    assert acts_all, "sustained surplus must activate growth"
    page_id, layer, expert_idx, trig = acts_all[0]
    # activation happened exactly announce_lead windows after scheduling
    assert s.window_id == sched_window + PARAMS.announce_lead
    assert h % PARAMS.retarget_window == 0
    assert trig == trigger
    # the appended page extends the table contiguously with the next expert slot
    assert page_id == len(s.pages) - 1
    assert s.pages[page_id][0] == s.pages[page_id - 1][1]
    assert s.pages[page_id][1] - s.pages[page_id][0] == EXPERT_LEN
    assert layer == 0 and expert_idx == 2            # first event: layer 0, slot 2
    assert s.events_total == 1


def test_ratchet_freeze_lifo_and_thaw_reverse():
    s = genesis()
    # grow twice
    h = 0
    grown = []
    while len(grown) < 2 and h < 400:
        h += 1
        s, acts = fold(s, PARAMS, h, 64, 0, "ee" * 32)
        grown.extend(a[0] for a in acts)
    assert len(grown) == 2
    total_pages = len(s.pages)
    # collapse: zero accepted -> deficit -> freeze, newest grown page first
    while s.pages[grown[1]][5] == ACTIVE and h < 800:
        h += 1
        s, _ = fold(s, PARAMS, h, 0, 0, "ee" * 32)
    assert s.pages[grown[1]][5] == FROZEN            # LIFO: newest froze first
    while s.pages[grown[0]][5] == ACTIVE and h < 1200:
        h += 1
        s, _ = fold(s, PARAMS, h, 0, 0, "ee" * 32)
    assert s.pages[grown[0]][5] == FROZEN
    # genesis pages never freeze; total never shrinks
    assert all(p[5] == ACTIVE for p in s.pages[:1 + 4])
    assert len(s.pages) >= total_pages
    # recovery thaws in reverse (lowest frozen id first) before any new growth
    while s.pages[grown[0]][5] == FROZEN and h < 1600:
        h += 1
        s, acts = fold(s, PARAMS, h, 64, 0, "ee" * 32)
        assert not acts or s.pages[grown[1]][5] == ACTIVE, \
            "no new growth while frozen pages remain"
    assert s.pages[grown[0]][5] == ACTIVE


def test_page_init_deterministic_and_shaped():
    a = page_init("ab" * 32, 7, SPEC)
    b = page_init("ab" * 32, 7, SPEC)
    assert np.array_equal(a, b)
    assert a.dtype == np.int64 and a.shape[0] == EXPERT_LEN
    # different trigger / page id -> different bytes
    assert not np.array_equal(a, page_init("ba" * 32, 7, SPEC))
    assert not np.array_equal(a, page_init("ab" * 32, 8, SPEC))
    # weight ranges bounded ±1311; bias ranges exactly zero
    d, f = SPEC.d_model, SPEC.d_ff
    w1, b1 = a[:d * f], a[d * f:d * f + f]
    w2, b2 = a[d * f + f:d * f + f + f * d], a[-d:]
    assert int(np.abs(w1).max()) <= 1311 and int(np.abs(w2).max()) <= 1311
    assert w1.any() and w2.any()                     # weights are seeded, not dead
    assert not b1.any() and not b2.any()             # biases start at zero


def test_page_proof_verifies_against_committed_state_root():
    """The serving-attestation primitive (§8): a node holding ONLY one page can
    prove it belongs to the model committed by header.state_root — same Merkle
    construction, same partition, so partial-recompute inference verification
    anchors directly to consensus state."""
    s = genesis()
    w = np.arange(s.dim(), dtype=np.int64) * 3
    root = bytes.fromhex(page_state_root(w, s))
    leaves = [w[p[0]:p[1]].tobytes() for p in s.pages]
    levels = merkle.build(leaves)
    for pid in (0, 2, len(s.pages) - 1):
        pf = merkle.proof(levels, pid)
        assert merkle.verify(leaves[pid], pid, pf, root)
        assert not merkle.verify(b"tampered" + leaves[pid][8:], pid, pf, root)


def test_required_nnz_quota():
    s = genesis()
    all_pages = list(range(len(s.pages)))
    # quota 1.0 => 1% of claimed params
    assert s.required_nnz(all_pages) == s.dim() * 10_000 // 1_000_000
    s.quota_4dp = 80_000                              # ceiling: 8%
    assert s.required_nnz(all_pages) == s.dim() * 80_000 // 1_000_000
    # single-page claim scales with the page, not the model
    assert s.required_nnz([1]) == EXPERT_LEN * 80_000 // 1_000_000
