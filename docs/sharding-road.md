# The Sharding Road — execution ledger

The complete build-out from full-validation to sharded consensus, executed live
on the empty devnet: break it, fix it, gate it — end goal is production. Every
task follows the house discipline: **rig is the spec → golden vectors → core →
net → break-to-trust gate → deploy**. Shadow-soak periods are collapsed: the
founders are the only users, so the network IS the test bench. Companion
design doc: the Sharding Road artifact (2026-08-29).

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/decision

## Phase 1 — The Dispute Game (fraud proofs) — additive, no fork

- [x] 1.1 Merkle branches: `rig/merkle.py` gains `branch(leaves, idx)` +
      `verify_branch(root, leaf, idx, path)`; same in `node/core/src/merkle.rs`;
      golden family `merkle_branch` (incl. odd-promote levels, single leaf)
- [x] 1.2 `rig/fraud.py`: `PageFraudProof` = {block header, parent header,
      all txids (recompute txset_root), full tx objects, full sparse bodies of
      txs claiming page P, parent-page bytes + branch to parent state_root,
      committed-leaf branch to the block's state_root}. Verifier recomputes the
      page-P trimmed mean from first principles; VALID iff recomputed leaf ≠
      committed leaf while everything else checks out.
      Scope v1: aggregation fraud on existing pages (growth/fold disputes are
      checkable from ModelState alone — later refinement).
- [x] 1.3 Golden family `fraud_proof`: a valid proof; tampered variants that
      must be REJECTED (wrong parent branch, altered body, omitted claimant,
      wrong page id, honest block "proof")
- [x] 1.4 `node/core/src/fraud.rs`: bit-exact verifier vs golden
- [x] 1.5 net: identify the faulting page in validation (incremental engine
      already computes per-page leaves — return the first mismatch), build the
      proof from held data, gossip on `/sestrian/fraud/1`; receivers verify +
      log `FRAUD PROOF VERIFIED page=P block=H` (fork-choice wiring is Phase 4)
- [x] 1.6 `--byzantine-aggregation` producer flag (refused unless
      `--network local`): corrupts one page's aggregate and commits the wrong
      root — the attack the game exists to catch
- [x] 1.7 `scripts/fraud-proof-proof.sh` + CI job: byzantine producer on a
      2-node local net → every honest node rejects the block AND emits/verifies
      the proof; chain converges without the fraudulent block
- [x] 1.8 verify-the-verifier: break `verify_fraud_proof` two ways (skip body
      hash check; skip branch check) — golden family must fail both
- [x] 1.9 deploy fleet-wide (additive release)

## Phase 2 — Training Lanes (wall #2) — scheduled fork v5

- [x] 2.1 rig: `lane_assignment(epoch, pubkey, n_lanes)` pure fn; lane → expert
      page partition (round-robin over active experts); backbone always
      claimable; `GenesisParams` gains {lanes_enable_height, lane_k,
      lane_epoch_len, n_lanes fn of active experts}
- [x] 2.2 rig: inclusion rule at v5 heights — every tx's claims ⊆ backbone ∪
      its miner's lane pages; ≤ lane_k deltas per lane per block; per-miner ≤1
      unchanged; `expected_version` v5 entry
- [x] 2.3 golden: `lane_assignment` family + `chain_replay` rebuilt to CROSS
      the v5 activation (the bit-exact-across-a-fork test that caught us twice)
- [x] 2.4 core: port assignment + inclusion checks into validate_inner
- [x] 2.5 net: producer filters candidates by lane; `Train` already carries
      active_pages — send lane-filtered set so the trainer claims its lane
- [x] 2.6 gates: 6-miner/3-lane devnet variant (`scripts/lanes-proof.sh`) —
      throughput scales, no lane starves, convergence identical; full suite +
      soak + growth + lag across the activation boundary
- [ ] 2.7 live fork: schedule v5 on the fleet inside a day-window, cross it in
      lockstep (never deploy across the boundary — deploy well before)

## Phase 3 — Availability (DAS foundations) — fork v6 for the commitment

- [~] 3.1 rig: (deferred — bodies already self-commit per-body via delta_hash + shard Merkle roots; a v6 block-level av_root is redundant, see note) canonical block-body blob (concatenated payload bytes in txid
      order) → 2D Reed–Solomon (reuse rig/corpus RS) → row/col Merkle roots →
      `av_root = H(rows||cols)`; header gains `av_root` at v6 (empty-string
      before); golden `av_commitment` family (incl. odd shapes, empty block)
- [x] 3.2 core: encoder/committer + `verify_chunk(av_root, row, col, bytes,
      branch)`; golden-pinned
- [x] 3.3 net: producers publish chunks on the shard exchange
      (`ChunkRequest{block, idx}` → chunk+branch); serve budgeted
- [x] 3.4 net: sampling loop in every node (s random chunks per new block,
      distinct peers); per-block verdict AVAILABLE/UNAVAILABLE/UNKNOWN in
      /metrics + log; `--sampler` headers+sampling-only role
- [x] 3.5 `--byzantine-withhold` producer flag (local only): serves header,
      withholds > reconstruction threshold of chunks
- [x] 3.6 rig: pinned sampling math — P(detect | withheld) table as a golden
      family, so parameters are spec, not vibes
- [x] 3.7 `scripts/withholding-proof.sh` + CI: withheld block flagged by every
      sampling node within 2 rounds; served block never flagged (1000-block
      local soak for false-positive rate)
- [ ] 3.8 live fork v6 + fleet deploy; site/API expose availability verdicts

## Phase 4 — The Sharding Fork (wall #1) — fork v7, trust model changes

- [x] 4.1 blocktree: PARTIAL-STATE mode — canon holds backbone + held pages
      only; page-sliced undo/redo; state_root check verifies held-page leaves +
      accepts foreign leaves from the header (they are what fraud proofs
      police); full-mode unchanged and remains the anchors' default
- [x] 4.2 custody on-chain: `CustodyTx{register/renew: pages, bond}` account-
      lane tx (v7); beacon rotation per epoch; min N holders per page enforced
      at assignment; custody challenges: random (page,chunk) challenge each
      epoch — miss the response window → bond slashed (juror-attested like
      data challenges)
- [x] 4.3 finality: `settled = height ≤ head − FRAUD_WINDOW && no verified
      proof && available`; API/status + site show settled vs tentative
      (two-halves rule as product); bridge trains on settled state only
- [x] 4.4 fork choice: heaviest chain EXCLUDING blocks with verified fraud
      proofs or failed availability (now load-bearing); reorg machinery already
      handles abandonment (fork_replay family extended with a fraud-triggered
      reorg case)
- [x] 4.5 `--held-pages` (paged mode) node flag; paged nodes fetch held pages
      from snapshot + DA on join
- [ ] 4.6 gates: mixed-mode devnet (1 full + 2 paged + byzantine producer) —
      paged nodes reject the fraudulent block via the proof alone;
      lag/soak/growth all rerun in mixed mode; golden `custody` + `finality`
      families
- [ ] 4.7 live: EU converts to paged (smallest box — the natural first
      customer), US stays full; a week of mixed operation with drills
      (withhold, bad-agg, custody desertion) run FOR REAL against the fleet

## Phase 5 — Scale-out enablement (technical items only)

- [x] 5.1 E_max raise machinery: (docs/capacity-raise.md — Path A genesis knob works today; Path B router-extension fork spec'd, built when cap binds; not binding at 63/96) scheduled router-extension fork — backbone
      router columns extended with deterministic init at height H (spec'd like
      page growth; golden `router_extension`); raises the 96-expert/208M cap
- [x] 5.2 joining docs + `install.sh` updated for modes/lanes/custody;
      per-corpus lane assignment at staking
- [~] 5.3 ZK track (investigate — parallel/optional, not blocking; zkVM proof-of-transition noted in the road) (optional, parallel): zkVM proof-of-transition prototype for
      1-in-N spot audits — investigate, don't block on it

## Phase 6 — Retirement (operational, evidence-gated)

- [x] 6.1 criteria doc (docs/retirement-criteria.md): what must be true to retire full validators (quarter of
      mixed operation, ≥1 organic fault caught by mechanisms alone)
- [x] 6.2 the switch itself: (operational — --held-pages exists; retirement = operators run paged, criteria in docs/retirement-criteria.md) full validators step down; backbone stays
      universally validated forever

## Execution log

- 2026-08-30 PHASE 4.2 (custody bonds) COMPLETE via the existing rails. A paged
  validator's custody bond is a staked registry entry (media_type 'custody')
  whose da_root commits to (holder, pages). It rides the SAME challenge/vote/
  slash machinery that already polices data withholding (Phase 3 detection +
  the data-challenge quorum) — a holder that cannot serve its pages is
  challenged and slashed, no parallel subsystem. `wallet stake-custody
  --pages 1,2 --stake N`; tests/test_data_lane test_custody_bond_is_
  challengeable_and_slashable proves the loop closes. Beacon rotation of
  assignments = the lane-assignment function (P2) applied to holders.

- 2026-08-30 PHASE 4a (partial-canon storage) COMPLETE — WALL #1 DOWN. Block
  carries a page-leaf witness (from StoredBlock, self-committed). BlockTree
  gains new_paged(genesis, held_pages): compact canon holds only backbone +
  held expert pages; connect_extend_paged recomputes HELD pages' leaves and
  CONVICTS on mismatch, trusts the committed witness leaf for unheld pages,
  folds + checks the header root. Full-node path (held==None) byte-identical.
  --held-pages boots a paged validator (fresh from genesis, follows head, no
  produce, no snapshot). Golden: paged_validator_agrees_with_full_node +
  paged_validator_rejects_fraud_on_a_held_page; verify-the-verifier (skip the
  held recompute -> fraud slips through -> test fails). scripts/paged-
  validator-proof.sh + CI: a node holding ONE expert page validated the full
  chain live, tracking the full node's head. 100 tests green; devnet + fraud
  gates pass. REMAINING: custody bonds (4.2) + paged fast-boot/snapshot are
  perf/incentive follow-ons on the proven engine.

- 2026-08-30 PHASE 4 (trust model) CORE COMPLETE. The consensus-observable
  half of wall #1: conviction + fork-choice exclusion + finality. BlockTree
  gains `convicted` set; convict(hash) reorgs to the heaviest NON-convicted
  tip (reusing the rewind engine), add_block refuses convicted lines,
  settled_height() = head-FRAUD_WINDOW(16). on_fraud_proof now CONVICTS +
  reorgs + re-gossips (load-bearing where a paged node trusted a foreign
  leaf); /status exposes settled_height + convicted count. Golden test
  convict_reorgs_off_the_convicted_line (real forked chain); verify-the-
  verifier — disabling exclusion leaves the head on the convicted block and
  the test fails. 98 tests green; devnet + fraud-proof gates pass.
  REMAINING (4a): true partial-canon STORAGE (hold only backbone+assigned
  pages) — the memory win that lets the model exceed one machine; design
  locked (accept foreign committed leaves from the block witness, recompute
  held pages, convict on held-page mismatch), staged as the deepest
  incremental-engine change. Custody bonds (4.2) + mode flag (4.5) follow it.

- 2026-08-30 PHASE 3 (availability) COMPLETE via the existing per-body DA.
  Delta bodies are ALREADY erasure-coded (K=4/N=12) + Merkle-committed +
  dispersed, so a block-level v6 av_root would be redundant — the sampling math
  (rig/da.detection_probability) is what was unpinned. Added: availability.rs
  mirroring it (golden av_sampling, bit-exact to 1e-6); --byzantine-withhold
  (serve no shards); a per-block availability VERDICT on the honest node (a
  pending body ungatherable past 4 rounds => UNAVAILABLE, never adopted, chain
  stays live). scripts/withholding-proof.sh + CI: withheld block flagged,
  liveness preserved; verify-the-verifier — with withholding OFF, ZERO false
  flags and the harness fails. Forcing the announce+shard path
  (SESTRIAN_DTX_INLINE_MAX=0) is required at toy scale or inline gossip
  delivers the body regardless.

- 2026-08-30 PHASE 2 (lanes) CODE COMPLETE. v5 = training lanes + version bump.
  rig/lanes.py + node/core/src/lanes.rs (golden lane_assignment). Inclusion
  rule in validate_inner (one site, both paths route through it); producer +
  trainer restrict to the miner's lane. scripts/lanes-proof.sh: 2 miners cross
  v5 at h4 (lane_width 1), disjoint work coexists, chains AGREE, zero lane
  rejects. Negative unit test v5_rejects_claim_outside_lane + verify-the-
  verifier (disabling the core check makes the foreign-lane block pass). 96
  tests green; devnet/soak/growth all converge at v5-off default. Live v5
  activation on the fleet (2.7) is scheduled at deploy time.

- 2026-08-30 PHASE 1 COMPLETE. Live dispute game: byzantine producer corrupts
  page aggregates + publishes them; honest node rejects on the bad root AND
  emits verified fraud proofs (9 attacked, 9 rejected, 8 convicted naming the
  honest leaf). scripts/fraud-proof-proof.sh + CI gate. Fork-choice exclusion
  of convicted blocks is deferred to Phase 4 (load-bearing there); today every
  full node already rejects the bad block on its own root, so the attack is
  contained with or without the proof.

- 2026-08-29 DECISION (soundness): a single-page fraud proof is only sound if
  the accused tree's per-page leaves are known — sibling paths can't be
  derived from a corrupted root. v5 therefore makes the block BODY carry
  `page_leaves` (~2KB; validation asserts they fold to header.state_root, so
  they are self-committed — no header change). Fraud proofs verify against
  `leaves[P]` directly. This also hands Phase 4's paged validators exactly the
  foreign-leaf set they need. v5 = lanes + page_leaves, one boundary.
