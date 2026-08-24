# Protocol v3 — Page Committees: how consensus stays O(pages) while miners scale

**The invariant** (the envelope's lesson, extended from bytes to everything):

> Consensus cost per block is O(pages touched), never O(miners).

Everything below follows from holding that line at 20, 200, 2,000 and 20,000
miners. Staging is explicit: each stage ships only when the network can
actually exercise it — consensus rules no test can reach do not ship.

## Where v2 breaks, decade by decade

| Miners | Binding failure |
|---|---|
| ~20 | INCLUDE_K caps inclusion at 8/block — most honest work is discarded; the quota controller can only demand more work *per delta*, not throttle arrival. Redundant gradients (shared data) drive marginal per-delta value toward zero. |
| ~200 | Proposer cannot eval N candidates per round; N×8MB cannot ride a block regardless of caps; 1-page-per-event growth cannot open claim-space fast enough. |
| ~20,000 | Every per-miner on-chain object is impossible. Bitcoin patched this era with off-protocol mining pools; our work is divisible *by structure* — the model is made of pages — so the protocol can do it natively. |

## The four mechanisms

### 1. Pages as committees (v4 — activates with real contention)
Each expert page has a bounded claimant set per round (k in [3, 16]),
assigned by stake-weighted VRF (the beacon's original purpose). A miner joins
a page's committee, not "the block". The delta bond becomes a **bid**;
oversubscribed pages admit the highest bonds — the fee market, but for
training slots. Degenerates gracefully: below contention, every miner may
claim any active page (exactly v2), so the rule can ship dormant.

### 2. Optimistic per-page aggregation (v4 — needs testnet operators)
At scale the block carries ONE aggregated delta per touched page plus a
Merkle commitment to member contributions. Member deltas live on the DA layer
(erasure-coded, sampled — existing machinery). The aggregate is a bonded
claim; the fraud proof is cheap by construction: re-run one page's trimmed
mean from published members, O(one page). Block bytes = pages_touched ×
envelope forever, independent of miner count. (The rig's `dipaco.py` sketched
this shape — composed training paths — before we had the vocabulary.)

### 3. Growth retargets on contention + a learning gate (v3 — NOW)
v2's growth gate (staleness = zero-scored fraction) conflated "junk work"
with "per-delta signal below the measurement noise floor" — found live: with
honest multi-batch eval, true per-delta improvement at a shared-corpus
plateau is ~0-400 µnats, so zeros are *accurate* and growth never fires.
v3 replaces the gate: **grow when capacity is saturated AND the window shows
the network learning at all** — the window's summed committed scores must be
positive (`win_score_sum > 0`). Per-delta scores keep weighting rewards;
the gate asks the right question ("is the network improving?"), robust to
tiny individual contributions. Later (v4), the saturation signal itself
becomes committee contention (median bidders per seat), making model size
track the number of independent contributors — the founding thesis, precise.
Growth events may then append multiple pages per event: inits are
deterministic hash-streams, so growth is O(1) on-chain bytes at any width.

### 4. Leave-one-out committee scoring (proposer policy — NOW)
Score a member by the aggregate-with-them minus aggregate-without-them,
evaluated on tokens routed through the claimed pages. Same eval cost as
today (1 + k forwards), far better signal (the aggregate carries k× one
delta's effect), and it **prices redundancy at zero automatically**: a miner
duplicating the committee's gradient earns nothing, so the rational strategy
is bringing *different data*. The data economy stops being a parallel
feature and becomes the profit-maximizing move. (First observed live with
2 miners on one corpus scoring each other ~0 — that force, working.)

## Staging & migration

- **v3 (this rev, scheduled hard fork via VERSION_SCHEDULE — no re-genesis):**
  - ModelState gains `win_score_sum` (canonical JSON includes it only from
    the activation height, so pre-activation roots are untouched).
  - Growth surplus condition: quota pinned AND accepted ≥ target AND
    `win_score_sum > 0` (replaces the staleness ceiling).
  - Leave-one-out scoring ships as proposer policy alongside (no fork).
  - Activation: height chosen at rollout; every node carries both rule sets;
    un-upgraded nodes fail loudly at the boundary ("upgrade your node").
- **v4 (testnet, needs >K independent miners to exercise):** committee
  assignment + bond bidding + optimistic aggregation + sampled jury
  verification. Ships dormant-degenerate where possible; fraud-proof paths
  golden-vectored before activation.

## What each stage needs from the rig
- v3: fold changes + activation-boundary vectors (root flips exactly at the
  boundary), gate goldens, replay across the boundary bit-exact.
- v4: committee assignment vectors (VRF -> seats), aggregation fraud-proof
  vectors (aggregate vs recomputed member mean), bid-admission orderings.
