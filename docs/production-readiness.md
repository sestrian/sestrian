# Sestrian production readiness

The go/no-go tracker for the phased launch. Phase 0 = the rig (done). Phase 1 =
a small, **monitored open devnet** run by known operators (the network is
permissionless (see [joining.md](joining.md)), so "small" means few people
bother running nodes yet, not a cryptographic gate). Phase 2 = testnet.
Phase 3 = open mainnet. This maps every hardening task to its state and gates
each phase.

## Legend
- ✅ implemented + tested in the shipping node
- 🧪 core/primitive implemented + golden-tested; live integration/validation
  pending the testnet
- 📐 designed (rig / whitepaper); not yet in the node
- ☐ operational item; manifest/script written, applied per-environment

## Consensus safety: ✅ COMPLETE (blocks Phase 1)
- ✅ Block height linkage (90) · delta-length guard (91) · n_txs/height-0 (95)
- ✅ Challenge quorum + disinterested jurors (93)
- ✅ Snapshot ledger validation (94)
- ✅ Length-prefixed signing preimages (96)
- ✅ Overflow-safe arithmetic, numpy parity, dust documented (97)
- ✅ **Non-forgeable work**: header.work = vrf_work(VRF proof), verified in
  validate_block; VRF proposer sortition wired in, fixed-rotation SPOF removed (92, 113)
- ✅ **Byzantine-robust aggregation** at low miner counts (always trim ≥1 at k≥3) (110)
- Golden vectors: 28 families (protocol v1: +page_root, page_init, model_state,
  quota, controller_fold; chain_replay rebuilt with an on-chain growth event)
  incl. negative, overflow, VRF-attempt, and low-count-robustness cases;
  Rust == Python. 67 Rust tests; devnet + soak (kill/restart) converge IN CI.

## Runtime & DoS hardening: ✅ COMPLETE (blocks Phase 1)
- ✅ Bounded mempools/caches + admission gating (98/99)
- ✅ Admin-token-gated mutating API; balance-before-write upload (100)
- ✅ Byte-budgeted, continuous sync (101)
- ✅ Durable fsync persistence; fatal-on-write-fail; torn-line self-heal (102/103)
- ✅ Single-writer data-dir lock (104)
- ✅ Gossip peer scoring + tight sync limits (105)
- ✅ Keys off argv/git, zeroized (106)
- ✅ Trainer watchdog + clock guard (107)
- ✅ SIGTERM graceful shutdown → final snapshot (131)

## Protocol v2: the delta envelope (devnet-genesis-3)
- ✅ `delta_max_nnz` consensus cap (1M coords ≈ 8MB): the payload never scales
  with quota; capacity pressure produces specialization (miners claim fewer
  pages, train them denser; claim budget = cap × 1e6 / quota) and sustained
  saturation still triggers on-chain growth. Bitcoin's block-size lesson,
  applied after the quota-fork incident proved the inverse design fails.
  Enforced in rig + core validation, mempool admission, producer filter;
  golden-vectored (quota family envelope rows); trainer plans claims by
  gradient mass under the budget. Proven live locally: devnet convergence with
  a forced tight envelope + growth-proof both green under v2.

## Trust model: 🧪/📐 (blocks Phase 3; a small monitored devnet mitigates Phase 1/2)
- 🧪 DA layer primitive: erasure coding + Merkle sampling (`core::da`) (111)
  - ☐ node routing: disperse on submit, sample on validate, reconstruct on
    replay (112); integration + testnet validation
- ✅ Proposer sortition ENFORCED (protocol v1, devnet-genesis-2): stake-weighted
  VRF eligibility gates every block in `validate_block`, with a deterministic
  attempt-widening liveness ladder (seed binds (prev, height, attempt);
  `header.vrf_attempt` committed; work = attempt-discounted; ATTEMPT_MAX floor
  keeps a 2-miner fleet live; cold-start rule covers the empty genesis ledger).
  Golden-vectored (lottery family) + negative-tested. Devnet rotation deleted.
  - 📐 threshold-BLS beacon for unbiasability (`rig/beacon.py`)
- ✅ Capacity retarget ENFORCED (protocol v1): the integer controller folds into
  consensus `ModelState` per block (committed as `header.model_root`), the WORK
  QUOTA (nnz floor over the claimed pages) is validated per delta, and GROWTH
  events activate on-chain: a new expert page appended with a deterministic
  hash-stream init, replay bit-exact across the dimension change. Proven live:
  `scripts/growth-proof.sh` (2 nodes + 2 torch trainers grow the model and stay
  converged). Frozen pages reject deltas; genesis pages never freeze (117 →
  DONE; the multi-operator delta-score committee remains testnet-phase).
- ✅ Protocol VERSION field + `VERSION_SCHEDULE` (the previously-missing upgrade
  affordance): headers commit `version`, validation pins it per height, unknown
  versions fail with "upgrade your node". Pre-v1 wire/disk artifacts fail to
  parse (no serde defaults on v1 fields) instead of half-loading.
- ✅ Stale-transport healing after churn (found live by the v1 soak, on CI):
  a SIGKILLed peer's QUIC connection lingers looking healthy under the SAME
  PeerId as its restarted successor, black-holing both gossip and sync pulls;
  the partition looked permanent. Two-layer heal shipped: (1) MESH-BLINDNESS
  PULL: if peers are connected but no foreign Head has been heard for 2+
  rounds, sync directly over request-response from every connected peer;
  (2) CONNECTION RECYCLING: at 3 silent rounds, drop all connections and
  redial the configured peers (guarded: inbound-only anchors never recycle).
  Liveness-only; fork choice reconciles on fresh transport. The soak asserts
  settled-prefix convergence through a mid-run SIGKILL.
- ✅ Sync-window catch-up deadlock (found live: the first payload-heavy WAN
  fresh-join; the second anchor wedged at height 3 forever): requests anchor at
  head−2 for reorg safety, but the server packs oldest-first under a 48MB byte
  budget, and with ~20MB delta payloads a batch is EXACTLY the already-known
  overlap, zero progress per round-trip. Fix: a per-peer request cursor jumps
  past a batch that taught nothing while the peer is ahead (any batch containing
  a new block clears it, so the reorg margin holds exactly when it matters), and
  a node still behind re-requests immediately instead of once per Head gossip.
  CI never saw it: toy-moe payloads are tiny. Regression: devnet convergence ✓.
- ✅ OOM during WAN catch-up (found live, same join): the ~860MB genesis state
  was pinned in RAM forever "to serve joiners", but sync already refuses to
  serve a genesis that big, so the pin only burned a hard-won 8GB margin and
  the OOM killer took the anchor down mid-sync. Fix: prune an oversized genesis
  state like any block below the prune floor (still re-derivable from
  genesis.bin); provision-seed.sh now also adds a 4GB swap backstop.
- ✅ QUIC stream-gap connection kill (found live, same join): quinn axes the
  whole connection with INTERNAL_ERROR "too many gaps in stream buffer" when a
  multi-MB sync response arrives out-of-order faster than the event loop reads
  it (block validation pauses reads for seconds). Fixes: 2MB QUIC stream window
  (bounds reassembly fragmentation; still ~20MB/s at 100ms RTT), sync batch
  budget 48→16MB, the connect-time opportunistic pull now respects the
  in-flight throttle + cursor (it used to fire a duplicate ~50MB transfer on
  every reconnect), and request-response failures / connection closes are now
  logged with their cause instead of being silently swallowed.
- ✅ BLOCK RATE is TRAINING-BOUND BY DESIGN, not a defect (investigated after
  observing ~360s against a 180s interval). `build_candidate` returns None when
  no includable delta is held, so a node never mints an EMPTY block — confirmed
  on-chain: every one of the last 16 blocks carried exactly 1 delta, none
  carried 0. The interval is therefore a CEILING on the rate, not a cadence:
  the chain advances only as fast as miners produce fresh in-window deltas,
  and a round whose deltas went stale simply produces nothing. That is the
  right behaviour for a chain whose blocks are training steps — an empty block
  would mint a reward for no work — but it means block rate tracks trainer
  latency (~110-150s/round here plus staleness losses), and if every trainer
  stalls the chain halts rather than minting empties. fleet-health's liveness
  window follows the OBSERVED rate for this reason.
- ✅ SERVE BACKPRESSURE (second attempt, after the first was reverted). Keyed
  on the inbound REQUEST ID and released on the single path every reply takes,
  so the leak that broke the first version — releasing only on swarm events
  that never fire when the reply pump drops a channel — is structurally
  impossible. Over-cap requests now get an explicit `busy: true` reply instead
  of an empty body list: BUSY is not ABSENT, and conflating them is exactly
  what stopped delta bodies flowing and left a routine head tie unhealable.
- ✅ ISOLATION GATE + additive `--peers`. `--peers` used to REPLACE the baked-in
  bootstrap (the adjacent comment claimed otherwise), so giving the mac miner a
  single LAN peer silently dropped both anchors; that address turned out to be
  UDP-filtered, leaving the node with ZERO peers. Being a producer, it then
  minted its own chain for an hour — the second mac fork of the day, and caused
  by the fix for the first. `--peers` is now a union with the bootstrap, and a
  producer with no connected peers (when peers were configured) refuses to
  extend the chain: halting is recoverable, a silent fork costs a full resync.
- ✅ PEER DISCOVERY (peer exchange over `/sestrian/peerx/1`). A node asks a
  connected peer who else it knows and dials a bounded number of them, so a
  star through the anchors closes itself into a mesh without a DHT. Addresses
  are SELF-DECLARED in the request and verified against the sender's peer id:
  libp2p identify advertises only CONFIRMED external addresses, which on a
  private/NAT'd network is nothing — measured, identify returned zero
  addresses and peer exchange had nothing to hand out, which is why the first
  implementation silently did nothing. Bounded by TARGET_PEERS (8) and a
  24-address share cap; loopback is never shared. Proven by
  `scripts/peerx-proof.sh`: two nodes configured with only a shared hub end up
  connected to each other. Was: ☐ NO PEER DISCOVERY — bounds how many operators can safely join. The
  behaviour set is gossipsub + identify + autonat/dcutr/relay + ping: there
  is no Kademlia, no gossipsub peer exchange, no mDNS. A node therefore knows
  only the baked-in bootstrap anchors plus whatever `--peers` it was given,
  so it never learns about any other node. Three consequences: (1) every
  joiner's traffic routes through the two anchors, which become a bandwidth
  bottleneck and a gossip SPOF as N grows; (2) miners never link directly —
  the mac and cuda miners sit on the SAME LAN and still had no direct
  connection, routing everything through remote anchors; (3) that is what let
  the two miners fork on 2026-08-25 with nothing to reconcile them once
  anchor connectivity churned. Interim mitigation applied: explicit `--peers`
  between the two miners. Real fix before an open invite: peer exchange or a
  DHT, so the topology is a mesh rather than a star through two hosts.
- ✅ PROPOSAL FAIRNESS (two interacting bugs; found only because v4 made it
  consensus-relevant). (a) The per-miner phase offset was a fixed function of
  the pubkey, so the lowest-hashing miner proposed first every round; now
  bound to (key, height) so the order rotates. (b) THE BINDING CONSTRAINT:
  `t0` defaulted to each node's OWN boot time, giving every node a private
  round metronome — two miners started at different moments had permanently
  offset boundaries and the earlier one proposed first forever. Live effect:
  cuda last proposed at h372 and mac took the next 60+ blocks unbroken while
  cuda was provably eligible throughout. Harmless under v3; FATAL under v4,
  whose quorum needs `growth_quorum` distinct proposers per window — the gate
  meant to make growth honest would have made it impossible. Epoch now
  anchored at unix time. Diagnosis required adding once-per-round eligibility
  logging: two plausible theories (phase bias, stake feedback) were both
  wrong, and only the log settled it.
- ✅ Corpora STAKED with real §7.2a availability commitments: the founding
  corpus (18,087,897,989 B, da_root a3d4cb2a…, 4313 chunks) verified to hash
  85aa06fb…e3ae — byte-identical to the genesis-ceremony record, the first
  independent check since it was built — plus the 1.8GB mac subset. The
  registry was previously a single hollow `genesis` chainparam with no hash,
  size or commitment; deltas now carry real data_refs.
- ✅ REGRESSION COVER for the recovery paths (added after four live
  catch-up/restart failures in one day, every one of which passed the whole
  suite on the way to production). The gap was structural: devnet and soak
  only ever exercised nodes that AGREE, so the decision logic a node uses
  when it is behind or forked was untested. Now: `catchup_decision` is a
  pure, total function with unit tests AND a convergence simulation (a
  lagging node must reach the head in bounded rounds and never re-request a
  window — the livelock showed up as non-convergence, not a wrong value);
  the restart-wedge invariant (snapshot must checkpoint the head's PARENT)
  is asserted against the real forked chain the golden replay builds; and
  `scripts/fork-catchup-proof.sh` (isolate → fork → reconnect → restart →
  converge) gates every PR. Each test was validated by REINTRODUCING its bug
  and confirming it fails: cursor reset-on-learn, compounding walkback, and
  snapshot-at-head all caught.
- ✅ PROTOCOL v4 — the QUORUM gate (scheduled, devnet height 416): growth is
  gated on `growth_quorum` DISTINCT positive-scoring proposers per window
  instead of v3's window-wide score SUM, which one lying proposer could open
  (found by our own red team, `rig/redteam_gate.py`, the day after v3 shipped).
  Second scheduled upgrade via VERSION_SCHEDULE, no re-genesis; win_scorers
  enters the canonical JSON only at rev 4 so every pre-608 model_root is
  byte-identical (golden diff purely additive). Golden family `v4_fold` pins
  Rust to the rig; growth-proof.sh now runs WITH v4 active and still grows.
  Devnet quorum is 2 (fleet-sized); raise as miners join.
- ✅ Serve memory bound: at most 2 outstanding big responses per peer
  (decremented on delivery or failure; over the cap a request gets an empty
  response the catch-up loop treats as a no-op). Closes the US-anchor OOM
  (~250 stalled ~31MB responses pinned while solo-serving a join).
- ✅ Trainer architecture from the chain: the node names its model preset in
  the bridge state message; miner_bridge builds from it and --model becomes
  a local-net override (the old default silently built the wrong toy model
  against devnet).
- ✅ SNAPSHOT LAG (the durable close-out of the restart-wedge): checkpoints
  are now written at the head's PARENT, so a live proposal tie at the head
  can never sit on fast-boot's state floor. Second live occurrence (US
  anchor, OOM-killed while solo-serving a 290-block join, rebooted onto a
  276 tie) confirmed the class before the fix landed. Follow-ups still
  open: serve-path memory bound (the OOM), catch-up request-supersede
  churn, divergence deeper than the prune window stays a resync.
- ✅ Restart-wedge at a head tie (found live: the EU anchor, restarted for a
  deploy during a live 252 tie, could never rejoin): fast-boot replay walked
  one branch linearly and shed the stored rival tie + children, while the
  serving cache kept them, and install() treated "in the cache" as "have it"
  so the exact block the node needed was discarded on every arrival, across
  every restart. Fixes: install() gates on tree membership (the cache is a
  cache), and fast-boot runs a reconnect fixpoint over stored side blocks.
  Note: the two prior sync-path attempts that night (cursor semantics,
  current=false handling) treated symptoms of this; the cursor-march fix
  was kept on its merits, the current=false change reverted by both agents.
- ✅ Catch-up cursor livelock (found live: the EU anchor pinned 7+ blocks
  behind at v2 payload sizes): the taught-nothing cursor advanced correctly,
  but any batch that taught something CLEARED it, so the next head announce
  re-anchored at head-2 and re-downloaded the same ~31MB overlap; each full
  climb learned 2 blocks while the fleet minted one per interval. Net
  progress ~zero, plus mutual walkback churn (peers exploring the laggard's
  side branch) starving its shard body fetches. Fix: while still behind, a
  useful batch continues the cursor (from + served); the cursor clears (and
  the reorg margin returns) only once caught up. Regression: devnet + soak
  (mid-run SIGKILL catch-up) both converge.
- ✅ THE ACTOR SPLIT + INCREMENTAL STATE ENGINE (the structural close-out of
  every transport incident above): the swarm loop now owns transport ONLY
  (its worst-case pause is microseconds) while a chain actor owns all state
  behind typed channels. And validation itself became O(envelope): one
  resident weight vector mutated in place, sparse undo/redo per block,
  cached page leaves (only touched pages rehash), sparse bodies end to end
  with a streaming delta hash. Committed-root checks make the incremental
  math self-verifying: divergence is loud rejection, never a fork. Proven:
  golden chain replay (fork + growth) bit-exact, devnet + growth-proof +
  forced-announce + kill/restart soak all green on the new engine.
- ✅ Delta scoring (rev 7): held-out-shard loss scores COMMITTED per block
  (header.score_root), enforced structure/bounds/commitment in validation;
  miner pool + data credits split ∝ score, uniform fallback; the trainer bridge
  evaluates candidates on a seeded held-out batch (108). Scores are bonded,
  challengeable proposer claims; the commit-reveal committee (multi-evaluator
  score verification + slashing automation) remains the testnet upgrade.
- ✅ Provenance (rev 5): deltas must name staked, active corpora (data_refs in
  the signing preimage); the data share pays the named owners (139/140).
  "availability" is a documented challenge reason: vanished bytes → slash +
  revoke → unnamable. Deep byte-audit sampling is the testnet extension.
- ✅ Economics (rev 6): tail emission (never zero), 1M-block epochs, 60/20/20
  inference fee split with on-chain fee pools drained to named data owners +
  miners (see economics-lifecycle.md).
- ✅ Delta stake bond (admission cost): lock/return done + golden-tested (109);
  slashing on proven fraud couples to scoring (testnet)
- ✅ Byzantine-robust aggregation at low miner counts (110): trim ≥1 at k≥3
- 🧪 Dtx cross-inclusion (anti-censorship) (114): per-proposer omission
  monitoring live in /metrics + /miners; the consensus-level inclusion
  challenge is testnet-phase (a validator can only expect deltas it saw)
- ✅ Fee-bearing inference receipts (116): on-chain fee payer→server + receipt
  done + golden-tested; off-chain output attestation is the challenge-market
  extension (testnet)

**Phase-1 mitigation:** the network is open, so instead of gating *who* joins,
launch **small and monitored** on a low-value model, run with people you can
watch, treat rewards as testnet play, and watch for bad deltas. An attacker
gains little from a near-worthless early model, and delta scoring (108) closes
the gap before the model is worth attacking. Keep mutating API endpoints
token-gated/disabled.

## Operations: ☐ manifests/scripts ready; apply per environment
- ☐ Persistent-volume StatefulSet (118): written; the live network runs the
  bare-VPS anchor model instead (provision-seed.sh), so this applies when a
  k8s environment returns
- ✅ Prebuilt image + CI push (120): images job green on main
- ✅ Prometheus /metrics endpoint + alert rules (121)
- ✅ Backup/restore script (122) APPLIED: nightly cron on both anchors
  (deploy/backup-cron, keep-7 rotation); restore drill documented in the script
- ◐ TLS termination (123): Caddy read-only HTTPS facade (deploy/Caddyfile.api)
  installed + validated on contabo-us-1, GET allow-list + CORS for the site's
  live panel; goes live (auto Let's Encrypt) once the api.sestrian.com A record
  points at it. Operator APIs stay plain-http loopback/LAN.
- ✅ Anchors dial each other (both units carry --peers), so either recycles
  stale transport after churn; DNS-named anchors with IP floor shipped (6ac8aac)
- ✅ Second bootstrap/DA anchor (119): contabo-us-1 (13.140.32.27) live on a
  separate continent; regenerated the genesis root independently (fourth
  platform), synced the chain over WAN (shaking out the three catch-up bugs
  above), holds lockstep with contabo-eu-1, and a fresh joiner syncs through it
  ALONE (bootstrap SPOF closed). Baked into the shipped bootstrap pair.

## Process
- ☐ npm 0.4.0 publish: the founder has no npm account yet; the published 0.3.2
  package works against devnet-genesis-2 with the documented
  `SESTRIAN_GENESIS_TAG=devnet-genesis-2` override (joining.md), and a stale
  install fails loudly, never silently. Create the account, `npm publish` from
  npm/, then drop the override from joining.md.

- ✅ CI: warning-clean build + tests + golden parity; image build (124)
- 🧪 node/net tests: store lock/torn-line, mempool window, API auth (125);
  expand alongside integrations
- 📐 adversarial/chaos suite (126) · cross-machine e2e + soak (128)
- ✅ Python reference suite pinned + green + BLOCKING in CI (127): protocol v1;
  devnet-convergence job on every PR, soak on main + nightly
- ✅ Threat model (132) · this readiness doc (133)

## Open design question: the growth gate vs specialization (v2, live)
Growth requires staleness <= 20% (zero-scored deltas are "junk" by design).
Live observation at max quota: ~half of committed scores are ZERO with a
systematic shape: the proposer's held-out eval scores its OWN delta positive
and the rival's specialized claim zero (a delta training experts the
evaluator's held-out batch barely routes to shows ~no improvement alone).
Consequence: the quality gate blocks organic growth exactly when sustained
saturation says the model needs capacity. Candidate fixes to DISCUSS (all
consensus-adjacent): score deltas against held-out slices weighted to their
CLAIMED pages; score the joint application; or exempt the staleness gate when
every miner's aggregate across the window scored positive at least once.
RESOLVED DIAGNOSIS (after claim-aware masks + 4-batch noise reduction shipped
as proposer policy): with the noise floor lowered, true per-delta improvement
at the plateau measures ~0-400 u-nats; the zeros are honest. Two miners on
one shared corpus produce near-redundant gradients, so per-delta marginal
value is genuinely tiny. Recommended for the next protocol rev: gate growth
on the WINDOW'S cumulative held-out trend ("is the network learning?") rather
than per-delta zeros; per-delta scores stay for reward weighting. The deep
fix is data diversity across miners (the corpus economy), which is also
what makes specialization real. Decide before the next protocol rev; do not
patch ad hoc.

## Remaining testnet-phase extensions (need a multi-party network)
The single-operator devnet can't validate these; the testnet is their gate.
- DEEP-REORG LIMIT (found live, quota-fork incident): a node cannot reorg onto
  a rival fork whose divergence point is below its state prune window; the
  fork-point state is gone, so the rival chain can never validate locally.
  Same property as Bitcoin pruned nodes; the resolution is a re-sync from
  genesis (or, later, from a checkpoint). The heal machinery (walk-back,
  budgeted shard fetch, age-based pending) handles any fork WITHIN the prune
  window automatically; beyond it, operator re-sync. prune_depth is therefore
  a consensus-adjacent availability knob, not just a memory knob.
- checkpoint sync: joining today replays every block from genesis, which is why
  ARCHIVE nodes (--da-retain-blocks 0, the anchors) must retain every body's
  shards forever. A verified state-snapshot join would let the whole network
  prune deep history. Until then: anchors archive, home nodes prune
  (--da-retain-blocks N deletes shard sets beyond the window; shard growth
  filled a founder disk within a day of the first live quota rise).
- gossip topic is `sestrian/v1` for every chain: a node on a DIFFERENT genesis
  (observed live: a devnet-1 straggler still mining the old chain) lands in the
  same mesh and its Head announcements trigger wasted sync pulls; validation
  rejects its blocks, so it's noise, not risk. At the next coordinated protocol
  bump, namespace the topic by genesis id (a change today would orphan the
  published v0.4.0 binaries for a cosmetic win).
- 108 committee upgrade: multi-evaluator commit-reveal score verification +
  automated slashing (the committed-scores mechanism itself is ✅ live; what
  remains is removing trust in the lone proposer's evaluation)
- 114 consensus-level cross-inclusion challenge (omission MONITORING is ✅ live)
- 141 sketch-based usage attribution for the fee data pool (§8): the pool +
  pro-rata drain are ✅ live; the sketch commitment/verification pipeline rides
  on the same off-chain-eval infrastructure as the 108 committee
- large-corpus DA ingestion: staked corpora register hash+stake on-chain and
  are challengeable ("availability"), but the DA byte-store path (built for
  ~13MB delta bodies; 64MB API cap) cannot ingest multi-GB corpora yet; a
  chunked content-addressed manifest format is needed before third-party data
  at scale. Devnet corpora (18GB/1.8GB) are served from the operators'
  always-on machines in the interim.
- ✅ 109 delta stake bond DONE (see above); slashing automation gates on the
  108 committee (testnet)
- ✅ 111/112 DA routing DONE at the storage layer: bodies are erasure-coded into
  Merkle-committed shards on write, and replay/sync reconstruct from any K shards
  instead of hard-stopping on a missing body (tested: recover from K, fail below
  K; devnet converges with live dispersal). The multi-node piece, distributing
  shards across peers + availability-sampling over gossip, is the testnet extension.
- 114 Dtx cross-inclusion: inherently a network property (a validator can only
  expect a delta it *saw* gossiped); an anti-censorship challenge, not a hard
  rule, so it validates on the testnet
- ✅ 115 chunked sparse aggregation DONE: Payload::dense_range + chunked_aggregate
  (bit-identical to dense trimmed_mean, golden-proven) wired into the producer,
  halving its delta memory. The validator-side lazy-body refactor (so verifiers
  also skip dense materialization) is the scale follow-on.
- ✅ 116 fee-bearing inference DONE on-chain (see above)

## Phase gates
- **Phase 1 (small open devnet): ✅ READY**: consensus safety complete (incl.
  non-forgeable work + robust aggregation), runtime hardening complete,
  multi-node DA + stake bonds + inference fees done, ops manifests written, repo
  public, genesis fetchable+verifiable from a peer. Soak (kill/restart) and the
  genesis-bootstrap test pass. Publish a bootstrap address + genesis id and
  point testers at [joining.md](joining.md).
- **Phase 2 (testnet):** multi-node DA availability-sampling under churn; a
  second anchor (✅ documented, provision it); load/soak on real hosts.
- **Phase 3 (open mainnet):** delta scoring (108) + slashing enforced;
  cross-inclusion (114); threshold-BLS beacon for unbiasable sortition;
  **external audit sign-off**; ✅ repo public + reproducible checksummed builds.
