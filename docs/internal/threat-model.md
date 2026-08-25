# Sestrian threat model & security posture

Status as of the production-hardening pass. This is the brief for the external
audit (a genesis-ceremony precondition) and the map from each attack to its
mitigation in the shipping node. It is deliberately honest about what is
enforced in code today versus what is designed but awaits the testnet phase.

## Assets

1. **Weight integrity**: the chain's state IS the model; corrupting it corrupts
   the product.
2. **Token supply**: the emission schedule (halving + sunset) must be the only
   source of new tokens.
3. **Ledger correctness**: balances, nonces, staked registry, challenges.
4. **Liveness**: the chain must keep advancing and converge to one head.
5. **Data availability**: every accepted delta body must remain retrievable so
   new nodes can validate from genesis.

## Actors

- **Malicious miner**: submits deltas (any keypair, no permission).
- **Malicious proposer**: builds blocks (open proposing on mainnet).
- **Malicious peer**: sends gossip / sync traffic.
- **Malicious uploader / API caller**: hits the node's HTTP API.
- **Malicious trainer**: the PyTorch process the node trusts over the local
  bridge.
- **Colluding coalition**: multiple of the above under one controller.

## Attack → mitigation

### Consensus safety (enforced in code, golden-tested)

| Attack | Mitigation | Where |
|---|---|---|
| Pin a low block height to mint max reward forever | height must equal parent+1 | `blocktree.rs` validate_block (task 90) |
| Crash every validator with mismatched-length deltas | delta body length must equal model dim, checked before aggregation | `blocktree.rs` (task 91) |
| Height-0 / n_txs header lies | height-0 rejected; `n_txs` must equal tx count | `blocktree.rs` (task 95) |
| Seize a staked data entry with one juror vote | challenge quorum (≥3 affirmative) + a strict majority | `token.rs` resolve_expired_challenges (task 93) |
| Vote on your own challenge (challenger or owner) | disinterested-juror rule | `token.rs` apply_data_tx (task 93) |
| txid collision via delimiter injection in signed fields | length-prefixed signing preimages | `lib.rs`/`token.rs` `frame` (task 96) |
| Integer overflow → panic or wrong ledger | numpy-parity `wrapping_add` for state math; `saturating_add` + u128 intermediates for the ledger | `lib.rs`, `token.rs` (task 97) |
| Malformed fast-boot snapshot panics/​corrupts a node | snapshot ledger fully validated on load; reject → full replay | `token.rs` from_value, `store.rs` (task 94) |

### Runtime / DoS (enforced in code)

| Attack | Mitigation | Where |
|---|---|---|
| Unbounded mempool/cache growth → OOM | every pool size-capped with eviction; deltas admitted only within a near-head window; account txs gated by ledger nonce; `seen` is a bounded ring | `node.rs` (tasks 98/99) |
| Fill the disk via `/upload` | balance checked before writing; endpoint gated by admin token | `node.rs`/`api.rs` (task 100) |
| Spend the operator's wallet / monopolize the trainer via the API | `/upload` + `/chat` require `Authorization: Bearer` (SESTRIAN_API_TOKEN); disabled if unset | `api.rs` (task 100) |
| Force a 512MB allocation via a sync response | request read capped 64KB, response 96MB; serve is byte-budgeted | `node.rs` (tasks 101/105) |
| Flood max-size gossip messages | gossipsub peer scoring + graylisting | `node.rs` behaviour (task 105) |
| Strand a lagging node (old 2-block/90s cap) | byte-budgeted serve + continuous re-request | `node.rs` (task 101) |
| Two processes corrupt one data-dir | exclusive advisory flock on the data-dir | `store.rs` (task 104) |
| Disk-full silently truncates the chain | append/​payload fsync; block-persist failure is fatal (halt, don't advance); torn-line self-heal, mid-file corruption stops replay loudly | `store.rs`/`node.rs` (tasks 102/103) |
| Private key visible in `ps`/`/proc` or committed to git | key from `--key-file`/env only, zeroized; k8s Secret, not inline | `main.rs`, deploy (task 106) |
| Hung trainer silences the node | training-round watchdog resets in-flight | `node.rs` (task 107) |
| Reschedule wipes the sole seed's chain | StatefulSet + persistent volume | `deploy/seed-node.yaml` (task 118) |

### Trust model (designed; core primitives built + golden-tested; enforcement is testnet-phase)

These are the properties that make "the chain's state is a *trustworthy* model"
true. The deterministic primitives are implemented and pinned by golden vectors;
wiring them into block validation + production, and validating their economic
equilibrium, requires the multi-node testnet (Phase 2).

| Property | Status |
|---|---|
| **Delta verification**: a delta must be a real, loss-reducing gradient, scored on a held-out shard via commit-reveal committee, with audit + slash for score fraud | REV 7: held-out-shard scoring ENFORCED at the committed-scores level: the proposer's trainer evaluates each delta on a seeded held-out batch, commits the scores in the block (header.score_root), and validation enforces structure/bounds/commitment; rewards split ∝ score. What remains for the testnet: the multi-evaluator commit-reveal COMMITTEE (removing trust in the lone proposer's evaluation) + automated slashing. Until then a proposer can inflate its own scores, bounded by SCORE_CAP, its bond, the challenge market, and `trimmed_mean` (robust for ≥3 honest miners), so keep the devnet **small, monitored, low-value**. (tasks 108/109/110) |
| **Data provenance** (rev 5): every delta names the staked, DA-available corpora it trained on; the data share pays the named owners; "upload the hash then delete the data" is impossible because unnamable data earns nothing and vanished data is challengeable ("availability" reason → slash + revoke) | ENFORCED in validation + ledger, golden-tested. Deep byte-audit availability *sampling* is the testnet extension. |
| **Data availability**: erasure-coded shards + Merkle availability sampling so a body is provably retrievable and survives some holders vanishing | PRIMITIVE built + golden-tested (`core::da`); node routing (disperse on submit, sample on validate, reconstruct on replay) is the integration (tasks 111/112) |
| **Proposer sortition**: verifiable, stake-weighted per-height eligibility instead of fixed rotation | PRIMITIVE built + golden-tested (`core::lottery`, deterministic-Ed25519 VRF); the threshold-BLS beacon (`rig/beacon.py`) is the unbiasable upgrade; wiring into validate_block + produce is the integration (tasks 113/92) |
| **Capacity retarget**: model size as difficulty | ENFORCED (protocol v1): work quota validated per delta (nnz floor over the signed page-claim set); controller state committed as `header.model_root`; growth events append deterministically-initialized expert pages on-chain (proven live: `scripts/growth-proof.sh`); frozen pages reject deltas. New consensus surface to watch: the growth trigger hash is the scheduling block's prev_hash: grindable in principle, worthless in practice (symmetric init distribution), closed for good by the threshold-BLS beacon |
| **Verified fee-bearing inference** | DESIGN only (task 116) |

## Cross-hardware determinism (holds)

Training float nondeterminism (MPS vs CUDA) occurs BEFORE the consensus
boundary: each miner quantizes to int64 and commits its own delta (hash-pinned);
consensus math is pure integer arithmetic (`wrapping_add`, `div_euclid`,
sorted maps), so two honest nodes reach identical roots regardless of GPU. Pinned
by 17 golden-vector families, including an overflow case.

## Known weaknesses (red-teamed, honest)

- **The v3 learning gate was NOT Byzantine-robust in the force-growth
  direction — FIXED by the v4 quorum gate (devnet height 608).** The v4 rule
  counts DISTINCT positive-scoring proposers in the window and requires
  `growth_quorum` of them, so forcing growth now costs winning that many
  blocks with that many keys (priced by stake-weighted sortition) instead of
  one. Red-teamed both ways in `rig/redteam_gate.py`: 1 and 2 attackers are
  blocked at quorum 3, a full-quorum coalition still succeeds, and both
  honest paths (a learning network grows, a plateau does not) are preserved.
  **Honest limit: v4 raises the price, it does not make the gate trustless.**
  Committed-score accuracy is still unverified by consensus; only the
  multi-evaluator committee closes that. The original finding, for the
  record:

- **[HISTORICAL, pre-height-608] The v3 learning gate is NOT Byzantine-robust
  in the force-growth direction** (`rig/redteam_gate.py`, found 2026-08-25). Growth is gated on
  `win_score_sum > 0`, a SUM of proposer-committed scores over the whole
  retarget window, and committed scores are validated only for range, not
  accuracy (that is the multi-evaluator committee, a testnet item). So a
  SINGLE Byzantine proposer committing one positive micro-nat, on a
  genuinely-plateaued network, flips the gate open for the entire window and
  drives unjustified growth — a 1-of-N griefing / resource-exhaustion vector
  (the model everyone stores and serves grows on one liar's say-so). It is
  strictly weaker than pre-v3 per-delta staleness, which bounded a proposer to
  its own block. Suppression (holding the gate closed on a learning network)
  is by contrast N-of-N — one honest positive score defeats it. Harm is
  bounded (1 page per ~5 windows, random-init experts, no theft/safety break)
  and mitigated by the small monitored devnet; the durable fix is to gate on a
  quorum of DISTINCT proposers (matching the ≥3-honest robustness the
  aggregation already assumes) or on the committee itself — a version-scheduled
  consensus change, not a hotfix. Until then the gate is a proposer-policy
  heuristic against an honest plateau, not a control against a Byzantine
  proposer.

## Residual risk / do-not-do

- Do **not** expose an unauthenticated node's mutating endpoints to the open
  internet; keep `/upload` + `/chat` token-gated (default: disabled).
- Do **not** launch a high-value mainnet before delta scoring is enforced and
  the external audit is complete. The network is open, so mitigate by keeping the
  early devnet **small, monitored, and low-value** (Phase 1/2); an attacker
  gains little from a near-worthless early model.
- The interim VRF sortition is grindable via the parent hash by the proposer;
  the threshold-BLS beacon closes this and is required before an open, high-value
  network.
