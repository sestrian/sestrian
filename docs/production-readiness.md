# Sestrian — Production Readiness

The go/no-go tracker for the phased launch. Phase 0 = the rig (done). Phase 1 =
a small, **monitored open devnet** run by known operators (the network is
permissionless — see [joining.md](joining.md) — so "small" means few people
bother running nodes yet, not a cryptographic gate). Phase 2 = testnet.
Phase 3 = open mainnet. This maps every hardening task to its state and gates
each phase.

## Legend
- ✅ implemented + tested in the shipping node
- 🧪 core/primitive implemented + golden-tested; live integration/validation
  pending the testnet
- 📐 designed (rig / whitepaper); not yet in the node
- ☐ operational item; manifest/script written, applied per-environment

## Consensus safety — ✅ COMPLETE (blocks Phase 1)
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

## Runtime & DoS hardening — ✅ COMPLETE (blocks Phase 1)
- ✅ Bounded mempools/caches + admission gating (98/99)
- ✅ Admin-token-gated mutating API; balance-before-write upload (100)
- ✅ Byte-budgeted, continuous sync (101)
- ✅ Durable fsync persistence; fatal-on-write-fail; torn-line self-heal (102/103)
- ✅ Single-writer data-dir lock (104)
- ✅ Gossip peer scoring + tight sync limits (105)
- ✅ Keys off argv/git, zeroized (106)
- ✅ Trainer watchdog + clock guard (107)
- ✅ SIGTERM graceful shutdown → final snapshot (131)

## Trust model — 🧪/📐 (blocks Phase 3; a small monitored devnet mitigates Phase 1/2)
- 🧪 DA layer primitive: erasure coding + Merkle sampling (`core::da`) (111)
  - ☐ node routing: disperse on submit, sample on validate, reconstruct on
    replay (112) — integration + testnet validation
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
  EVENTS activate on-chain — a new expert page appended with a deterministic
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
  PeerId as its restarted successor, black-holing both gossip and sync pulls —
  the partition looked permanent. Two-layer heal shipped: (1) MESH-BLINDNESS
  PULL — if peers are connected but no foreign Head has been heard for 2+
  rounds, sync directly over request-response from every connected peer;
  (2) CONNECTION RECYCLING — at 3 silent rounds, drop all connections and
  redial the configured peers (guarded: inbound-only anchors never recycle).
  Liveness-only; fork choice reconciles on fresh transport. The soak asserts
  settled-prefix convergence through a mid-run SIGKILL.
- ✅ Sync-window catch-up deadlock (found live: the first payload-heavy WAN
  fresh-join — the second anchor wedged at height 3 forever): requests anchor at
  head−2 for reorg safety, but the server packs oldest-first under a 48MB byte
  budget, and with ~20MB delta payloads a batch is EXACTLY the already-known
  overlap — zero progress per round-trip. Fix: a per-peer request cursor jumps
  past a batch that taught nothing while the peer is ahead (any batch containing
  a new block clears it, so the reorg margin holds exactly when it matters), and
  a node still behind re-requests immediately instead of once per Head gossip.
  CI never saw it: toy-moe payloads are tiny. Regression: devnet convergence ✓.
- ✅ OOM during WAN catch-up (found live, same join): the ~860MB genesis state
  was pinned in RAM forever "to serve joiners" — but sync already refuses to
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
  logged with their cause instead of being silently swallowed. Follow-on
  (testnet): move block validation off the swarm event loop so reads never
  stall.
- ✅ Delta scoring (rev 7) — held-out-shard loss scores COMMITTED per block
  (header.score_root), enforced structure/bounds/commitment in validation;
  miner pool + data credits split ∝ score, uniform fallback; the trainer bridge
  evaluates candidates on a seeded held-out batch (108). Scores are bonded,
  challengeable proposer claims — the commit-reveal COMMITTEE (multi-evaluator
  score verification + slashing automation) remains the testnet upgrade.
- ✅ Provenance (rev 5) — deltas must name staked, active corpora (data_refs in
  the signing preimage); the data share pays the named owners (139/140).
  "availability" is a documented challenge reason: vanished bytes → slash +
  revoke → unnamable. Deep byte-audit sampling is the testnet extension.
- ✅ Economics (rev 6) — tail emission (never zero), 1M-block epochs, 60/20/20
  inference fee split with on-chain fee pools drained to named data owners +
  miners (see economics-lifecycle.md).
- ✅ Delta stake bond (admission cost) — lock/return done + golden-tested (109);
  slashing on proven fraud couples to scoring (testnet)
- ✅ Byzantine-robust aggregation at low miner counts (110) — trim ≥1 at k≥3
- 🧪 Dtx cross-inclusion (anti-censorship) (114) — per-proposer omission
  monitoring live in /metrics + /miners; the consensus-level inclusion
  challenge is testnet-phase (a validator can only expect deltas it saw)
- ✅ Fee-bearing inference receipts (116) — on-chain fee payer→server + receipt
  done + golden-tested; off-chain output attestation is the challenge-market
  extension (testnet)

**Phase-1 mitigation:** the network is open, so instead of gating *who* joins,
launch **small and monitored** on a low-value model — run it with people you can
watch, treat rewards as testnet play, and watch for bad deltas. An attacker
gains little from a near-worthless early model, and delta scoring (108) closes
the gap before the model is worth attacking. Keep mutating API endpoints
token-gated/disabled.

## Operations — ☐ manifests/scripts ready; apply per environment
- ☐ Persistent-volume StatefulSet (118) · prebuilt image + CI push (120)
- ✅ Prometheus /metrics endpoint + alert rules (121)
- ☐ Backup/restore script (122)
- ☐ TLS termination for non-loopback API (123) — see below
- ✅ Second bootstrap/DA anchor (119): contabo-us-1 (13.140.32.27) live on a
  separate continent — regenerated the genesis root independently (fourth
  platform), synced the chain over WAN (shaking out the three catch-up bugs
  above), holds lockstep with contabo-eu-1, and a fresh joiner syncs through it
  ALONE (bootstrap SPOF closed). Baked into the shipped bootstrap pair.

## Process
- ☐ npm 0.4.0 publish — the founder has no npm account yet; the published 0.3.2
  package works against devnet-genesis-2 with the documented
  `SESTRIAN_GENESIS_TAG=devnet-genesis-2` override (joining.md), and a stale
  install fails loudly, never silently. Create the account, `npm publish` from
  npm/, then drop the override from joining.md.

- ✅ CI: warning-clean build + tests + golden parity; image build (124)
- 🧪 node/net tests: store lock/torn-line, mempool window, API auth (125) —
  expand alongside integrations
- 📐 adversarial/chaos suite (126) · cross-machine e2e + soak (128)
- ✅ Python reference suite pinned + green + BLOCKING in CI (127) — protocol v1;
  devnet-convergence job on every PR, soak on main + nightly
- ✅ Threat model (132) · this readiness doc (133)

## Remaining — testnet-phase extensions (need a multi-party network)
The single-operator devnet can't validate these; the testnet is their gate.
- gossip topic is `sestrian/v1` for every chain: a node on a DIFFERENT genesis
  (observed live: a devnet-1 straggler still mining the old chain) lands in the
  same mesh and its Head announcements trigger wasted sync pulls — validation
  rejects its blocks, so it's noise, not risk. At the next coordinated protocol
  bump, namespace the topic by genesis id (a change today would orphan the
  published v0.4.0 binaries for a cosmetic win).
- 108 committee upgrade: multi-evaluator commit-reveal score verification +
  automated slashing (the committed-scores mechanism itself is ✅ live; what
  remains is removing trust in the lone proposer's evaluation)
- 114 consensus-level cross-inclusion challenge (omission MONITORING is ✅ live)
- 141 sketch-based usage attribution for the fee data pool (§8) — the pool +
  pro-rata drain are ✅ live; the sketch commitment/verification pipeline rides
  on the same off-chain-eval infrastructure as the 108 committee
- large-corpus DA ingestion — staked corpora register hash+stake on-chain and
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
  K; devnet converges with live dispersal). The multi-node piece — distributing
  shards across peers + availability-sampling over gossip — is the testnet extension.
- 114 Dtx cross-inclusion — inherently a network property (a validator can only
  expect a delta it *saw* gossiped); an anti-censorship challenge, not a hard
  rule, so it validates on the testnet
- ✅ 115 chunked sparse aggregation DONE: Payload::dense_range + chunked_aggregate
  (bit-identical to dense trimmed_mean, golden-proven) wired into the producer,
  halving its delta memory. The validator-side lazy-body refactor (so verifiers
  also skip dense materialization) is the scale follow-on.
- ✅ 116 fee-bearing inference DONE on-chain (see above)

## Phase gates
- **Phase 1 (small open devnet): ✅ READY** — consensus safety complete (incl.
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
