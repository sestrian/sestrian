# The Genesis Ceremony

How the real Sestrian network is born. Everything here is decided, published,
and verifiable **before** block 1 exists — credibility is set at launch and
cannot be retrofitted (WHITEPAPER §9.8). This document is the checklist and the
script; each item names the mechanism that already implements it.

## Ceremony 3 — devnet-genesis-3 (protocol v2, 2026-08-24)

**Why.** The first live capacity retarget exposed a design inversion: the work
quota scaled the *payload*, so rising capacity made every delta bigger until no
transport path could carry one (~92MB at 4.7x) and the network forked. Protocol
v2 applies Bitcoin's block-size lesson: **consensus never scales the payload.**
A delta may never exceed `delta_max_nnz = 1,000,000` nonzero coordinates
(~8MB). A rising quota now narrows the claimable span — miners specialize on
the experts their data teaches best — and sustained saturation still grows the
model on-chain. Bytes per block are bounded forever (~3Mbit/s worst case).

**Ceremony-3 deltas from the table below:** `genesis_seed` **20260824**;
`genesis_state_root` **`91bdcc281c0dbbd7b3bea3d38003e4c61565bcaa5fd8e7bfca296e6a4994ddb1`**
(model spec unchanged — small-moe, model_root identical to ceremony 2);
`protocol_version` **2** (`VERSION_SCHEDULE` [(0, 2)]); `delta_max_nnz`
1,000,000 joins the retarget row as a consensus parameter. Anchors unchanged.
devnet-2 record: forked at height 64 by the quota rise (the incident this
ceremony fixes); final unified branch height ~165; both miner identities were
the founders'. Backups retained on every operator machine.

## Ceremony 2 — devnet-genesis-2 (protocol v1, 2026-08-22)

**Why a re-genesis.** Protocol v1 changed the chain's consensus surface in ways
this document's own rule (§1: "any change to this table after ceremony is a
hard fork by definition") makes a hard fork: the state commitment moved from a
flat sha256 to a **page-Merkle root** over a consensus page table; delta
transactions gained a signed **page-claim set** and lost `shard_id`; the header
gained `model_root`, `vrf_attempt`, and a **protocol `version`** field (the
upgrade affordance the chain previously lacked); proposer **eligibility** (VRF
sortition with a deterministic attempt-widening liveness ladder) and the
**capacity work quota** are now enforced in validation; and the network model
became a **growable MoE** — the §9.4a capacity retarget can now append expert
pages on-chain, with deterministically-derived weights. Rather than defer any
of this into a future flag-day against strangers, the devnet was relaunched
clean while its operators were only the founders. devnet-1 rewards were always
documented as testnet play; wallets and keys carry over, balances do not.

**Ceremony-2 parameters** (the §1 table's deltas):

| Parameter | devnet-genesis-2 value |
|---|---|
| `model_config` | `small-moe`: 6 layers · 8 heads · d_model 512 · d_ff 2048 (=4d, frozen) · 8 experts/layer at genesis · `E_max` 16 router columns/layer · top-2 routing · RoPE (no position table) · byte vocab 256 |
| `params` | 107,414,528 total (backbone 6,628,352 + 48 expert pages × 2,099,712 = 859,316,224 raw bytes); ~32M active per token |
| `genesis_seed` | **20260822** (the ceremony date). A devnet re-genesis needs no grind-resistance — the weights are worthless at t=0 and the init distribution is seed-independent; the drand derivation below remains the **mainnet** procedure. |
| `genesis_state_root` | **`a597316003dbf12122b7cc6f39226ce7c8f7a871e58e7ddf364e56b08102527b`** — the PAGE-MERKLE root printed by `make_genesis`. Verified by regeneration on the MPS founder host against the final code (`--expect` pass; generation is CPU+numpy, byte-stable across platforms, previously verified MPS/CUDA/CPU cross-machine). The CUDA founder host's verification is STRUCTURALLY ENFORCED at its first join: the node checks any genesis against this baked root at startup (`--check` / boot verification), so a divergent regeneration cannot join silently — it fails loudly and would trigger a re-ceremony. `scripts/release-genesis.sh` refuses to publish on mismatch. |
| `retarget` | window 16 blocks · target 8 deltas/window · quota 0.25×–8× (4dp fixed point) · k_sustain 3 · announce lead 2 windows · growth bound 1 expert page/event · genesis experts never freeze |
| `protocol_version` | 1 (header-committed; `VERSION_SCHEDULE` is the upgrade mechanism) |
| `bootstrap_peers` | `/ip4/169.58.211.248/udp/9800/quic-v1` (contabo-eu-1) + `/ip4/13.140.32.27/udp/9800/quic-v1` (contabo-us-1, provisioned at relaunch — independently regenerated + verified the genesis root as a fourth platform) — QUIC is canonical |
| devnet-1 record | **final height 151, head `9896723303f2a3f9`, supply 7,549,999,999,998 grains** (recorded at shutdown in `/root/devnet1-final-status.json` on the seed; 2 miner identities in the whole history — both founders'). The chain store backup is retained (deploy/backup-restore.sh), the `devnet-genesis-1` release assets stay published forever, and the pre-v1 code is tagged `devnet-1-final`. |

Everything else in the §1 table (fair launch, zero balances, emission,
reward split, founding corpus `85aa06fb…e3ae`, data_contributor) is unchanged.

## 0. Principles (all already protocol invariants)

- **From scratch, on-chain.** Genesis weights are a deterministic random
  initialization from a *public seed* — no pretrained artifact anyone must
  trust. Every parameter of the model is thereafter explainable as a sum of
  signed, attributed, replayable deltas (§3.1).
- **Fair launch.** The genesis ledger has **zero balances**. No premine, no
  pre-sale, no allocation. Every grain is minted by a block reward for
  verifiable work (`rig/token.py`, mirrored bit-exact in `node/core`).
- **Bytes forever; RoPE positions.** Vocabulary = 256; no tokenizer; no learned
  position table — context is a runtime/market choice (§3.1).
- **The founding corpus is a contribution, not a gift.** It enters as registry
  entry zero owned by the founder's wallet at a published weight, earning the
  data share under exactly the rules any later contributor faces — including
  challengeability (§7.2 challenge market).

## 1. Published launch parameters (the genesis file)

A single JSON document, hashed and pinned, containing:

| Parameter | Value at ceremony | Where enforced |
|---|---|---|
| `model_config` | layers / heads / width / training block size | `client/gpt.py` GPTConfig |
| `genesis_seed` | derived from a public randomness beacon (below) | `apply_genesis` (numpy, version-independent) |
| `genesis_state_root` | sha256 of the quantized genesis vector | `rig/chain.state_root` — anyone recomputes |
| `emission` | BASE_REWARD, HALVING_BLOCKS, SUNSET_HEIGHT | `rig/token.py` / `node/core/token.rs` |
| `reward_split` | 7000/1000/2000 bps miners/proposer/data | same |
| `challenge_params` | CHALLENGE_WINDOW, PROPOSER_LOOKBACK | same |
| `data_contributor` | founder wallet address | genesis registry entry zero |
| `genesis_data_weight` | GENESIS_DATA_WEIGHT | same |
| `founding_corpus_hash` | **`85aa06fba4ef397b19bc5bc8e62d394bdb067b5eddde418ef5f4680ce1aae3ae`** (18,087,897,989 bytes · 48,284 documents · built 2026-08-20) | registry entry `data_hash`; corpus pinned on the CAS/DA layer |
| `founding_corpus` | **decided: public-domain only** — ~48k English Project Gutenberg books (~21 GB), built + hashed by `scripts/build_founding_corpus.py` with a per-shard manifest. No web crawl, no share-alike, no gated sources: the founding entry earns the founder's share and is challengeable by design, so its provenance is bulletproof. Code, Wikipedia, and web-scale text enter later through OTHER contributors' staked submissions and the §10.2 campaign track — the data economy working as intended. | `founding_manifest.json` |
| `block_interval` | seconds per round | node config |
| `bootstrap_peers` | seed-node multiaddrs (first public seed: `/ip4/169.58.211.248/udp/9800/quic-v1`) | node config |

**Rule: any change to this table after ceremony is a hard fork by definition.**

## 2. Seed derivation — nobody chooses the genesis weights

`genesis_seed = sha256("sestrian-genesis" || drand_round_R_signature)` where
`R` is a **pre-announced future round** of the drand public randomness beacon
(the League of Entropy). Because `R` is announced before its value exists,
neither the team nor anyone else can grind the initialization. Anyone can
verify: fetch round `R`, hash, compare. (`apply_genesis` then expands the seed
with numpy's byte-stable RNG, so the same seed yields bit-identical weights on
every platform — already verified cross-machine MPS/CUDA/CPU.)

## 3. The founding wallet

- Generated **fresh, offline**, on a machine the founder trusts
  (`python -m client.wallet new` on an air-gapped box; the dev wallet used
  during testnet is retired). Mnemonic backup written down; encrypted wallet
  file backed up separately (see wallet hardening).
- Only the **address** enters the genesis file. The key never touches a server.

## 4. The founding data transaction

The corpus enters through the standard admission path, visible in block 1's
lineage: registry entry zero (`seed_genesis_data`) carries the founder's
address, the corpus content hash, and the published weight. The corpus bytes
are pinned content-addressed (CAS/Bitswap — `client/cas.py`; the DA layer at
scale) so any node can fetch and hash-check exactly what the model eats.
It is **challengeable like any entry** (validity or ownership, §7.2) — the
founder holds no special immunity.

## 5. Ceremony procedure (the runbook)

1. **T−7 days**: publish the genesis file *minus* `genesis_seed` /
   `genesis_state_root`, naming drand round `R`. Publish repo tag, binary
   checksums, seed-node addresses.
2. **T−0**: drand round `R` lands. Anyone (including us) computes
   `genesis_seed`, runs `apply_genesis`, publishes `genesis_state_root`.
   Independent parties confirm the root.
3. **Launch**: seed nodes + founder nodes start with the complete genesis file.
   Block 1 is mined by whoever gets there first — emissions begin, the founding
   registry entry starts earning its weighted data share.
4. **T+window**: the founding corpus sits in its public challenge window like
   any submission.

## 6. Preconditions before the ceremony can run

- [x] Token legal posture — **founder's decision: no counsel engaged at launch.**
      The network fair-launches with no sale, no premine, and no profit
      promises; the founder's data-contributor share is publicly disclosed in
      the genesis parameters. Revisit before any exchange listing or any
      conversion of founder holdings — those are the events that change the
      legal character of the token, not its existence.
- [ ] External audit of consensus + ledger (rig is the spec; `node/core` golden-vector-pinned)
- [x] NAT traversal live — AutoNAT/DCUtR/relay-v2 shipped in the Rust node; the
      first PUBLIC seed+relay is up (`/ip4/169.58.211.248/udp/9800/quic-v1`),
      and a fresh node dialing only that multiaddr connects and agrees on genesis.
- [x] Wallet hardening shipped (encrypted files, BIP39 mnemonics, checksummed pal1… addresses)
- [x] Real corpus decision + license posture — public-domain-only Gutenberg
      (composition table above); pipeline + manifest in
      `scripts/build_founding_corpus.py`; hash lands in this file when the
      ceremony build runs.
- [ ] Repo public; binaries reproducibly built and checksummed

## 7. What the testnet already rehearsed

Every mechanism above is running today on the internal testnet: from-scratch
genesis with a published seed (1337), the founder wallet earning the data share
per block, transfers and data-lane txs settling through `ledger_root`, the
seed node on the cluster, and the Rust devnet converging byte-identically.
The ceremony is those same steps with a beacon-derived seed, a fresh wallet,
the real corpus, and the world watching.
