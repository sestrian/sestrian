# Sestrian — a chain whose state is a mind

*Working codename: Sestrian (a surface written over many times, whose history stays recoverable). Final name TBD.*
*Master design document. Each section tracks a task in the project task list; sections are drafted in order and drilled down individually.*

> **Implementation status.** This is the design. For what is actually enforced
> in the shipping node today versus what is designed or has only its core
> primitive built, see [`docs/production-readiness.md`](docs/production-readiness.md)
> and [`docs/internal/threat-model.md`](docs/internal/threat-model.md). In short:
> all consensus-safety and runtime-hardening properties are implemented and
> golden-tested; the DA layer, proposer sortition, and capacity-retarget
> **primitives** are built and golden-tested but not yet wired into block
> validation; delta scoring, stake/slashing, the threshold-BLS beacon, and
> fee-bearing inference remain design + simulation. The launch is phased
> (invite-only → testnet → open mainnet) accordingly. The genesis is trained
> **from scratch** on-chain (the deterministic-from-seed `client/make_genesis`
> path); `client/convert_ckpt.py` exists but is **not** used by the ceremony.

---

## 1. Overview

**Thesis.** Sestrian is a blockchain whose state is the weights of a single public neural network. Transactions are the model's own computations: **backprops** — gradient deltas that transition the state and earn rewards — and **forward-props** — inference calls that pay fees and emit cryptographically attested outputs. A block is one aggregation step of distributed training; the block header commits a weights-state root; replaying the chain from genesis reconstructs the model bit-for-bit. The chain does not *record* a model. The chain *is* a model.

**The flywheel.** Compute joins the network and does one of two jobs: improve the weights, or serve them. Inference revenue — real usage fees, not token inflation — pays both jobs. A better model attracts more usage; more usage funds more training; more training makes a better model. The network is self-funding by construction, and the design's central economic discipline is that rewards are anchored to revenue the chain actually earns, with bootstrap emissions carrying a hard, milestone-tied sunset. (§9)

**Security in one sentence.** Majority-honest compute secures gradient integrity (replicated recompute plus Byzantine-robust aggregation — the 51% argument, applied where it actually holds); staked challenge games secure the two surfaces majority-compute cannot see: data admission and serving honesty. (§7)

**Six claims that position this design.** Each is either proven by someone else (we adopt it) or uniquely ours (we defend it):

1. **The model is the ledger** — no existing network makes the weight-update itself the transaction such that replaying blocks reconstructs the weights; every surveyed competitor (Ambient, Templar, Psyche, Gensyn, 0G, Pluralis, Flock, Bagel) keeps coordination on-chain and the model in someone's bucket. *Uniquely ours; verified Aug 2026.* (§2, §3)
2. **The block interval is the DiLoCo outer step** — miners run local inner steps between blocks; block production is the outer synchronization that makes internet-scale training feasible (~500× communication reduction). The blockchain's cadence and distributed training's cadence are the same rhythm. *Uniquely ours as a unification; DiLoCo proven at 100B scale.* (§6)
3. **Consensus over improvement, not just ordering** — the mempool is a scoring arena: a delta is included only if sampled validators measure that it helps, Gauntlet-style. This is the novel consensus problem and the section that must be airtight, because it is the part Bittensor demonstrably got wrong. *Scoring proven on Templar; the consensus formulation is ours.* (§5)
4. **Attestation is free** — "this output came from weights-hash X at block N" is just the block header. The verified-inference product our nearest competitor engineered and sells is, in this architecture, a native property. (§8)
5. **Mirrors are distribution, not leakage** — weights are public per block, deliberately. On-chain serving nodes earn fees *plus* protocol rewards for the same silicon; off-chain mirrors earn only margin, so rational GPU owners serve on-chain and the network structurally underprices its own copies. Public weights become the funnel; the chain keeps the drain. (§8, §9)
6. **The model has no landlord** — competitors' "decentralized" models live in Cloudflare R2 and on HuggingFace; their incentive layers are decentralized but the artifact is hostage. Here the model inherits the ledger's durability: as long as the chain exists, the model exists. (§3)

**Strategy in one sentence.** No network on earth currently pays training rewards from inference income; that crown is unclaimed, and it does not require a frontier model — we close the loop first at specialized-model scale (7–70B, inside today's proven permissionless envelope, in a vertical with community-owned data), own the existence proof, and scale as the bandwidth ceiling rises. (§11)

**What this document is.** The master design document for the project. Sections 2–12 drill each mechanism to whitepaper depth; the section map below is the table of contents and the build order.

---

## 2. Background & prior art

### 2.1 What the field has proven

Three results, from three different teams, establish that every component of this design is individually feasible as of mid-2026:

**Permissionless training works.** Templar (Bittensor subnet 3) completed **Covenant-72B** in March 2026: 72.7B parameters, ~1.1T tokens, trained by 70+ anonymous nodes over commodity internet with free join/leave, reaching Llama-2-70B-class quality (MMLU 67.1). Nous Research's **Psyche** network pretrained Consilience-40B to the 20T-token mark coordinated via Solana, and Nous now trains its own commercial Hermes models on the network. **DiLoCoX** pre-trained a 107B model over ordinary 1Gbps links with negligible convergence loss. The feasibility envelope today is roughly the 100B class; frontier scale (10²⁶ FLOP) remains bandwidth-bound.

**Incentivized quality-scoring works.** Templar's **Gauntlet** system evaluates every miner's submitted update by its measured effect on loss, pays good updates, and slashes bad ones — contribution scoring that survived contact with anonymous, self-interested participants at 72B scale.

**Verifiable delegation works.** Gensyn's mainnet (April 2026) ships **Verde**: refereed delegation that bisects a disputed training computation to the first divergent step and re-executes only that operation, made possible by **RepOps** — bitwise-reproducible ML primitives across heterogeneous hardware. Deterministic, disputable ML computation is production infrastructure, not a research wish.

### 2.2 What the field has *not* done

**Nobody has closed the revenue loop.** No network pays training rewards out of inference income. Bittensor — the largest decentralized-AI economy — booked ~$43M in Q1 2026 external revenue across 128 subnets, but token emissions still dominate incentives at an estimated 22–40:1 subsidy ratio. Ambient, whose thesis is closest to ours, remains pre-mainnet; its shipped product is a conventional verified-inference API over third-party models.

**Nobody puts the model on the chain.** The architectural survey (verified August 2026):

| Project | On-chain | Model lives |
|---|---|---|
| Ambient | Logit-hash fingerprints, job auctions, checkpoint-hash commitments | Sharded across miners |
| Templar (Bittensor) | Miner scores, rankings, TAO emissions | Cloudflare R2 buckets |
| Psyche (Nous) | Coordinator contract: assignments, randomness, attestations, payouts | P2P mesh; checkpoints on HuggingFace |
| Gensyn | Task metadata, proof registrations, disputes, payments | Solver hardware; S3/IPFS pointers |
| 0G | Verification records | 0G Storage/DA service |
| Pluralis | Coordination | Deliberately never materialized anywhere |
| Flock / Bagel | Aggregation records / ZK proofs | Off-chain |

The universal pattern: **chain = ledger of who did what and who gets paid; model = external artifact held together with the ledger by incentives.** Academic proposals (PoGO, BlockTrain, BlockFUL) put gradient *Merkle commitments* on-chain — pointers, not the thing itself. Note the practical consequence: every "decentralized" model above has a landlord. Templar's canonical gradients sit in Cloudflare buckets; Psyche's checkpoints sit on HuggingFace. The incentive layers are decentralized; the artifacts are hostage.

### 2.3 The cautionary tale

Bittensor is the field's largest natural experiment in incentivized ML, and its documented pathologies are design requirements for us: **weight-copying** (validators out-earn honest evaluation by copying the consensus median — a strategy our commit-reveal scoring in §5 is built to kill), **validator–miner collusion**, the **emissions treadmill** (rewards denominated in inflation, revenue an afterthought), and a governance credibility collapse — Covenant AI, the team behind the network's flagship training result, exited in April 2026 calling it "decentralization theatre," erasing ~$900M of market value in days. Each of these has a named countermeasure in this document (§5 scoring, §9 emissions sunset, §10 anti-theatre tests).

### 2.4 The uniqueness claim, stated precisely

Ambient markets "the model as network state" philosophically, and commits checkpoint hashes on-chain. So our claim is made at the mechanism level, where it is uncontested: **no existing network makes the weight-update itself the transaction, such that replaying the chain's blocks reconstructs the model's weights.** Sestrian's blocks *are* optimizer steps; its state root *is* a commitment to the weights; its history *is* the training run. Everything we adopt from the field (Gauntlet-style scoring, Verde-style disputes, DiLoCo-style communication, DisTrO-style compression, TEE attestation) is proven; everything we add (model-as-state, block-as-outer-step, scored-mempool consensus, revenue-anchored economics) is unclaimed.

## 3. Architecture — the model as chain state

### 3.1 State

The chain's state at height N is **W_N**: the full parameter set of the network's single model, plus the outer-optimizer state (momentum buffers) and a small parameter store (fee rates, emission schedule position, stake table, admitted-data registry root). Weights are partitioned into fixed-size **pages** (e.g., 16MB tensor-aligned chunks); the **weights-state root** is the Merkle root over page hashes. Two properties follow immediately:

- Any node can prove it holds the correct page i of W_N with a Merkle path — partial verification without holding the full model.
- "Which model produced this output?" has a canonical, chain-native answer: the state root in the block header (§8).

**Invariant: the chain's interface is bytes.** The model consumes and emits raw bytes (vocabulary = 256, defined by physics, forever); there is no tokenizer. This is a decided, permanent commitment, not a placeholder. Rationale: (1) a tokenizer would be consensus-critical infrastructure — a version mismatch is a silent fork, and its vocabulary a frozen governance artifact whose replacement is a hard fork plus a full retrain, since weights do not migrate across vocabularies; (2) BPE vocabularies embed the linguistic bias of their training corpus into the *payment layer* — the same sentence costs 2–4× more tokens in most non-English languages, structurally underpaying their data contributors, which a per-byte meter avoids; (3) bytes are universal — code, any language, binary formats, genomic data — so nothing the data economy (§9A) admits is ever out-of-vocabulary. The known ~4× compute penalty of naive byte-level modeling is a *model-architecture* cost, not a data-format law: dynamic byte-patching architectures (MegaByte, Byte Latent Transformer) recover BPE-class efficiency inside the model — and on this chain the model is the one component that is upgradeable by construction, while the data format is the one that must never change. BPE fixes efficiency in the data format; patching architectures fix it in the model; only the model is upgradeable on-chain.

**Invariant: positions are rotary (RoPE) — context length is a market, not a model constant.** The model carries no learned position table, so its context window is not a weight-shape commitment: how long a context a node trains or serves is a *runtime choice bounded by its own hardware*. Miners train on whatever window fits their GPU (short-window deltas aggregate with long-window deltas — mixed-length training is standard practice and the reward already scales with compute done); serving nodes and verifiers **advertise their max context in their capacity registration**, and the API router matches each request to nodes whose declared context covers it, with fees scaling with context (attention cost is quadratic). Small GPUs earn on the fat head of short requests; long-context capacity earns a premium; growing the network's effective context requires no model surgery and no governance event — just nodes with bigger cards showing up.

**Genesis is from scratch, on-chain.** The genesis state is a deterministic random initialization from a published seed — no pretrained artifact anyone must trust. Every parameter of the model is therefore explainable, from block zero, as a sum of signed, attributed, replayable deltas: the entire model has on-chain provenance. The founding corpus itself enters through the data-admission path (§7.2, §9A) as the network's first data transaction — contributed by the founder's wallet and earning attribution and royalties under exactly the same rules as any later contributor (no special genesis privilege, no exemption either; the address and its terms are published in the genesis parameters).

### 3.2 Block anatomy

```
Header:  prev_block_hash | height N | weights_state_root(W_N)
         | delta_set_root | inference_receipt_root | data_registry_root
         | randomness_beacon | validator_attestation_aggregate | timestamp
Body:    accepted backprop tx commitments   (each ~1KB: miner sig, base-height ref,
           assigned data-shard id, delta hash, DA pointer, quorum score)
         inference receipt batch commitments (§8)
         stake/governance/fee txs            (conventional)
```

Blocks are small — kilobytes to low megabytes — because **delta bodies never enter blocks**. A DisTrO-class compressed delta for a 40B model runs ~100MB–1GB; blocks carry only commitments.

### 3.3 The data-availability layer

Delta bodies (and inference audit data) are erasure-coded and dispersed across the node set, with **data-availability sampling**: light verification that the committed data actually exists without any node downloading all of it — the same pattern Ethereum reached with blobs and Celestia built as a product. Retention is bounded: a delta body must remain available from its inclusion until the next finalized **checkpoint** plus the challenge window (§5.4); after that it may be pruned by non-archival nodes. A commitment whose body fails availability sampling is invalid — a delta the network cannot re-check is a delta the network never accepted.

### 3.4 State transition

The block-level transition is the outer optimizer step (§6):

```
W_{N+1} = OuterStep( W_N,  RobustAggregate({Δ_i : accepted in block N+1}) )
```

Both `RobustAggregate` (trimmed-mean/median over deltas, fixed-point accumulation) and `OuterStep` (Nesterov momentum per DiLoCo) are **bit-deterministic** (§6.3). This is the load-bearing engineering constraint of the whole design: given W_N and the accepted delta bodies, every node computes an identical W_{N+1}, hence an identical state root, hence consensus over the model is consensus in the ordinary blockchain sense.

### 3.5 Checkpoints and replay

Every K blocks (K sized so checkpoint bandwidth ≪ delta bandwidth, e.g., daily), the network finalizes a **full checkpoint**: W_N erasure-coded across the node set and its root enshrined. The replay guarantee has two tiers:

- **Full replay**: genesis weights + every accepted delta body ⇒ recompute the entire training history. Archival nodes maintain this; it is what makes influence auditing and backdoor excision possible (§7.2).
- **Fast sync**: latest finalized checkpoint + subsequent deltas ⇒ current state in hours, not weeks. This is how new nodes, serving nodes, and mirrors stay hot.

Checkpoints are also the **public-weights release mechanism**: checkpoint roots and bodies are public by construction (claim 5, §8.4). Publishing weights is not a policy decision layered on the chain; it is what the chain physically is.

### 3.6 Durability — the no-landlord property

Because the model is chain state, it inherits the ledger's survival properties: no storage vendor to deplatform it, no foundation server whose disappearance orphans the artifact, no single jurisdiction that can seize it. The model exists as long as a quorum of nodes anywhere keeps the chain — and any party holding one checkpoint plus the delta stream can independently reconstruct and verify it against the enshrined roots. Contrast §2.2: every competitor's model has a landlord; ours has a genesis block.

### 3.7 Concrete envelope (v1 targets)

| Quantity | v1 target |
|---|---|
| Model | 7–40B params (≈14–80GB fp16) |
| Page size / page count | 16MB / ~1k–5k pages |
| Block interval | 5–15 min (= DiLoCo outer period, §6) |
| Compressed delta | ~50MB–500MB (DisTrO/DeMo-class, 100–1000× compression) |
| Deltas per block | 10–100 (scored inclusion, §5) |
| Block body | ≤ ~1MB commitments; bodies on DA layer |
| Checkpoint interval K | ~100–300 blocks (≈ daily) |

All figures sit inside envelopes already demonstrated by Covenant-72B, Psyche, and DiLoCoX (§2.1); nothing in v1 requires a distributed-systems result that does not exist.

## 4. Transaction model

Two first-class transaction types express the network's two jobs. Backprops change the state and consume rewards; forward-props leave the state untouched and generate the fees that fund those rewards. The flywheel is not a diagram in this design — it is the relationship between the two halves of the ledger.

### 4.1 Backprop transactions (state transitions)

A miner assigned data shard s at height N (assignment from the randomness beacon — miners never choose their data, §7.2) trains locally for the inner-step window and submits:

```
BackpropTx {
  miner_pubkey, signature, stake_ref
  base_height: N                      // weights the delta was computed against
  shard_id: s                         // beacon-assigned; mismatch ⇒ invalid
  delta_hash, da_pointer              // body on DA layer (§3.3)
  self_reported_metrics               // advisory only; never trusted
}
```

**Lifecycle**: submit → availability-sampled → scored by the validator committee (§5) → included or dropped → if included, applied in the block transition and credited reward share proportional to quorum score. **Staleness rule**: a delta based on height N may be included up to height N+g (small grace g, DiLoCo tolerates modest staleness) at a discounted score; beyond that it is invalid. Submission requires a stake bond and a floating admission fee (the write-price homeostat, §9.4); slashing conditions are score fraud and DA withholding, not "being wrong" — an honest low-quality delta merely earns nothing and forfeits its admission fee.

### 4.2 Forward-prop transactions (fee-bearing inference)

An API call is economically a transaction even though it never touches state:

```
ForwardPropTx (receipt) {
  request_hash                        // prompt commitment (privacy-preserving)
  output_hash
  weights_state_root, height N        // exactly which model answered (§8)
  serving_node_pubkey, signature
  fee, fee_split_version
  decode_params_hash                  // greedy/temperature settings, for auditability
  logit_commitment (optional)         // pre-sampling commitment for sampled decoding (§8.2)
}
```

Receipts are batched by serving nodes and anchored per block via the `inference_receipt_root`; individual API latency is *not* bound to block cadence — payment channels and receipt batching give millisecond serving with per-block settlement (§8.3). The **fee split** (parameters governed per §10, initial targets in §9.2): serving node share, validator/verification share, **training reward pool** share, burn. The training-pool share is the sentence "inference funds training" enforced by the protocol rather than promised by a foundation.

### 4.3 Feedback transactions (the loop's third wire)

A forward-prop may carry an optional feedback flag (thumbs, task-success signal, structured evaluation) — logged as a **candidate** training signal only. Feedback enters the actual corpus solely through the data-admission pipeline (§7.2, §10.2): usage data is a firehose of value and the single easiest poisoning vector, so the transaction model records it freely and trusts it never. Customers who contribute admitted feedback earn a rebate stream (their usage measurably improved the asset they rent — the CAPITALISM principle: pay for measured marginal information, priced per channel).

### 4.4 Conventional transactions

Stake/unstake, governance votes, fee-parameter updates, data-registry operations, and transfers — standard machinery, standard nonce/replay protection. They matter here only insofar as §9 and §10 define them.

## 5. Consensus — the scored mempool

### 5.1 The novel problem

In Bitcoin, transaction validity is *syntactic*: a signature verifies or it doesn't, and consensus is only over ordering. Here, a backprop transaction's validity is *empirical* — "does this delta improve the model?" — and consensus must be reached over a measurement. This is the part of the design with no clean precedent, the part Bittensor's economy demonstrably failed at, and therefore the section engineered with the most redundancy. Three properties make it tractable: scoring is **deterministic** (given a delta, a base state, and an evaluation set, the score is a reproducible number — RepOps arithmetic, §6.3), evaluation sets are **unpredictable** (beacon-drawn after deltas are committed), and every claim is **exactly recomputable** by anyone later (replayable chain, §3.5).

### 5.2 Mechanism

Per block, from global stake, the randomness beacon samples a **validator committee** C (size ~30–100). Then:

1. **Delta commitment closes.** Candidate BackpropTxs for height N+1 are fixed (availability-sampled, fee-paid). No delta may be altered after this point.
2. **Evaluation draw.** The beacon draws held-out evaluation shards *after* the commitment closes — from a reserved, rotating holdout pool never used for training (§7.2). Miners cannot train toward an evaluation set they cannot predict.
3. **Commit.** Each validator computes, for each candidate delta, the deterministic loss-impact score on the drawn shards (plus canary/trigger probes, §7.2) and publishes a **hash** of its score vector.
4. **Reveal.** After a quorum of commitments, validators reveal. The protocol score per delta is the **median** of revealed scores.
5. **Inclusion.** The proposer assembles the block: top-scoring deltas above the floating quality threshold, up to the per-block delta budget. Committee members attest that the assembled set matches the revealed scores; the block finalizes on quorum attestation.
6. **Settlement.** Included deltas earn reward share ∝ median score (§9.2). Validator payment for the block is released on schedule, subject to the audit rule below.

### 5.3 Why each Bittensor pathology dies here

- **Weight-copying** (copy the consensus median, skip the work): impossible to copy what doesn't exist yet — scores are committed *before* any reveal. A copier must commit blind; a blind commitment that deviates from the deterministic true score is *provable* fraud, because…
- **Lazy or fraudulent validation**: any party can later recompute any validator's exact claimed score (determinism + replayability) and submit a fraud proof; a validator whose revealed score deviates from the recomputable true value beyond arithmetic tolerance is slashed. Honest work is the *only* strategy that survives audit, and audit is cheap because it targets single (validator, delta) pairs — Verde-style narrow re-execution, not global recompute.
- **Validator–miner collusion**: committees are beacon-sampled per block from global stake — capturing a committee requires capturing global stake (§5.5); a colluding minority inflating one delta's score is pulled to the median, and their outlier scores are fraud-provable as above.
- **Grinding/self-dealing**: miners can't choose their data shards (beacon), can't predict evaluation shards (beacon, post-commitment), and can't resubmit tweaked deltas free (admission fee, §9.4).

### 5.4 Optimistic backstop — the challenge window

Scored inclusion is the fast path; a **challenge window** (until the next checkpoint finalizes) is the safety net. Any staked party may challenge an included delta or a validator's score with a counter-evaluation; the dispute resolves by deterministic re-execution (bisection to the disputed operation where needed, per Verde), slashing whichever side the arithmetic contradicts. A successful challenger earns the slashed stake — the same verify/produce economics that secure optimistic rollups. A state transition that survives its window is final; checkpoints (§3.5) are the finality horizon.

### 5.5 What 51% actually buys, and what less than 51% still buys

- **Liveness** (blocks keep being produced, deltas keep being scored): requires an honest majority of committee stake — the classic 51% condition, and where the founding intuition applies directly.
- **State safety** (no harmful delta becomes permanent): degrades far more gracefully. Because every acceptance is deterministically recomputable and challengeable, a *single* honest, staked watcher within the challenge window suffices to evict a fraudulently-scored delta — even against a majority-captured committee, provided data availability holds (which is why DA sampling in §3.3 is a consensus-critical primitive, not a storage optimization). Majority captures liveness; corrupting *history* requires suppressing every honest challenger and the DA layer simultaneously.
- **Statistical robustness** (many small adversaries rather than one big one): the aggregation step (trimmed mean/median over accepted deltas, §3.4) bounds the influence of any minority mass of adversarial deltas that scored honestly-but-marginally.

The residual risk this machinery does *not* close — an update that genuinely improves measured loss while carrying a stealthy backdoor — is a data problem, not a consensus problem, and is treated as such in §7.2 and honestly in §12.

### 5.6 Block production and the fork rule

Proposer selection is stake-weighted from the committee; fork choice is heaviest-attested chain from the last finalized checkpoint. Nothing exotic: the novelty budget of this design is spent on scored inclusion, deliberately not on fork-choice research.

## 6. Training protocol — block interval as DiLoCo outer step

### 6.1 The unification

DiLoCo-family training — the reason internet-scale training exists — has workers train *locally* for H inner steps (hundreds), then synchronize once in an outer aggregation step, cutting communication ~100–500×. Sestrian does not bolt a blockchain onto this loop; it observes that the loop *already is* a blockchain cadence:

| DiLoCo | Sestrian |
|---|---|
| Outer synchronization round | Block |
| Worker's local inner run (H steps) | Miner's work between blocks |
| Pseudo-gradient (weight delta after H steps) | BackpropTx body |
| Outer optimizer (Nesterov momentum) | Block state transition (§3.4) |
| Synchronization barrier | Block finalization |
| Global step count | Block height |

The block interval (5–15 min, §3.7) is chosen as a *training* parameter — the outer-sync period that balances convergence against WAN bandwidth — and the chain inherits it. One rhythm, two literatures.

### 6.2 The miner's round

1. At block N finalization, sync W_N (delta-stream from N−1, or fast-sync, §3.5).
2. Read the beacon: shard assignment s ← Beacon(N, miner) from the admitted-corpus registry (§7.2). No self-selected data, ever.
3. Run H local inner steps (any hardware, any precision, any kernel — deliberately unconstrained, §6.3) on shard s.
4. Compress the resulting pseudo-gradient (DisTrO/DeMo-class compression + quantization), post the body to the DA layer, submit the BackpropTx.
5. Scored per §5; if included in block N+1, apply the finalized transition and go again.

**Join/leave is free** (Covenant-72B proved this works at 72B): a joining miner fast-syncs and requests assignment; a leaving miner simply stops — its assigned shard rotates back into the pool, and an unsubmitted round costs only that miner's own electricity. **Stragglers** are handled by the staleness discount (§4.1): modestly late deltas earn less; very late deltas are invalid. DiLoCo's tolerance of heterogeneous, unreliable workers is precisely why the permissionless setting is survivable.

### 6.3 The determinism boundary — where bit-exactness is and isn't required

The design's key relaxation, worth stating precisely because it decides feasibility:

- **NOT deterministic — the miner's inner loop.** How a miner computes its delta is its own business: any GPU, fused kernels, fp8, exotic optimizers, even better *algorithms*. The chain never re-executes inner loops; it only ever evaluates the *submitted delta*. This keeps the permissionless door open to heterogeneous hardware and lets miners compete on training efficiency — that competition is where the network's per-FLOP handicap (§12) gets clawed back.
- **Deterministic — everything consensus touches**: delta decompression, scoring forward-passes on evaluation shards (RepOps-class bitwise-reproducible kernels), robust aggregation and the outer step (fixed-point/integer accumulation in a canonical order). These are the operations for which "recompute and compare" must yield bit-identical answers on any conformant node (§5.3), and they are a small, auditable fraction of total compute.

Verification cost stays sub-linear in training cost because scoring is *forward passes on small evaluation sets* — orders of magnitude cheaper than the H inner steps that produced the delta — plus rare, narrow dispute re-executions.

### 6.4 Continuous post-training as the steady state

Pretraining from scratch happens once, at bootstrap (§11). The network's steady state is **continuous post-training**: RL runs, preference optimization, and fine-tuning on newly admitted data — the regime the field is converging on anyway (INTELLECT-3's lesson), the friendliest to parallel, loosely-coupled workers, and the engine of the freshness moat (§8.4): the chain's model is always the newest, because improving it never stops. RL episode generation is itself forward-prop work, blurring productively into serving capacity (§8.3): the same fleet that serves customers generates rollouts.

### 6.5 Feasibility envelope

Everything above is inside demonstrated bounds (§2.1): pseudo-gradient outer-sync at 107B over 1Gbps (DiLoCoX), permissionless incentivized workers at 72B (Covenant), 20T tokens coordinated on-chain (Psyche), reproducible-op dispute games in production (Gensyn). The v1 target of a 7–40B specialized model (§11) sits comfortably in the middle of the envelope rather than at its edge — deliberately: the novel risk in this project is economic and consensus-layer, so the training layer takes zero moonshot risk.

## 7. Security model — three attack surfaces, three locks

The founding intuition — "poisoning requires 50%+1 once the community is large" — is correct, but it protects less than Bitcoin's 51% does, because a model-chain has **three separate attack surfaces** and majority-compute covers only the first. This section states each surface, its lock, and its residual risk.

### 7.1 Lock 1 — wrong compute (gradient integrity)

**Threat**: a miner submits a delta that is garbage, adversarially crafted, or simply not what its assigned shard produces.

**Lock**: the scored mempool *is* the verification (§5) — a delta earns inclusion only by measurably improving loss on unpredictable held-out shards, scored deterministically by a beacon-sampled committee under commit-reveal, with every score exactly recomputable and challengeable. On top: robust aggregation (trimmed mean/median, §3.4) bounds the joint influence of adversarial deltas that scored marginally, and the staleness rule prevents replaying old honest deltas. **Where 51% applies**: committee liveness needs honest majority; state safety needs only one honest challenger plus data availability (§5.5).

**Residual**: a delta that genuinely improves measured loss while encoding something harmful is, by definition, not a compute attack — it is a data or objective attack, handled below and confessed in §12.

### 7.2 Lock 2 — poisoned data (the real frontier)

**Threat**: backdoor research shows a trigger can be implanted with well under 1% of training data. Majority recompute is *blind* to this: honest nodes would faithfully verify the correct gradient *of poisoned data*. This is the surface where naive 51% intuition fails, so it gets defense-in-depth:

1. **No self-selected data.** Miners train only beacon-assigned shards from the on-chain **admitted-corpus registry** (§4.1, §6.2). Poisoning therefore requires getting bad data *admitted* — the attack is funneled into one auditable gate.
2. **Staked admission with a challenge window.** Data submitters stake per shard batch; admitted shards sit in a public quarantine window before entering the training pool. Challenges (provenance fraud, trigger patterns, license violations, duplication) resolve by evidence; successful challengers earn the submitter's stake (§10.2 governs large-corpus campaigns).

   *Implemented (protocol rev 3) as the on-chain **challenge market**.* A data submission is a signed, staked `DataSubmitTx`; the entry lives in the ledger's **data registry** (owner wallet, content hash, media type, escrowed stake) and earns the block data share in proportion to weight (v1: stake-weighted, with the genesis corpus at a published weight; attribution-weighted royalties replace weights at the TRAK milestone). Anyone may file a staked `DataChallengeTx` against an entry's *validity* or *ownership*, opening a fixed voting window. **Jurors are the recent block proposers** (the last `PROPOSER_LOOKBACK` blocks) — seats earned by verifiable work, not bought — voting via `DataVoteTx`. At expiry, deterministically: upheld → the entry is revoked, its escrowed stake goes to the challenger; rejected (including no-quorum) → the challenger's stake goes to the entry's owner. Either way, lying costs and honesty pays; the registry, challenges, and votes are all part of the ledger root, so every node enforces identical outcomes. Media type is a registry field, not a protocol constraint — bytes are bytes (§3.1), so spreadsheets, images, and any future modality enter through the same gate.
3. **Canary and trigger probing as validator duty.** Committee scoring (§5.2) includes probes beyond loss: behavioral canaries, regression suites on protected capabilities, and sweeps over *known* trigger families. This catches crude poisoning and any trigger the defender can generate — but the red-team (Phase 0, `rig/redteam.py`) is unambiguous about its limit: a stealthy backdoor keyed to an attacker-chosen, out-of-distribution trigger is **invisible to blind probing**, because you cannot probe a trigger you don't know. Probing is a cost-raiser, not a guarantee. The guarantees live in mechanisms 2 (admission cost), 4 (vested clawback), and 5 (excision) — not here.
4. **Vested data rewards with clawback.** Data rewards vest over many checkpoints; a shard later proven poisoned slashes its submitter's remaining vest and bounties the prover — extending the challenge economics deep into the past.
5. **The immune system: influence audit and excision.** Unique to this architecture: because the chain replays (§3.5), a discovered backdoor can be *traced* (which shards, which deltas, which blocks — influence analysis over the recorded history) and *excised*: replay the chain from the last clean checkpoint with the offending shards' deltas removed, under a governance-declared emergency procedure (§10.4). Every off-chain competitor can only apologize and retrain from scratch; a model-chain can perform verifiable surgical unlearning. Poisoning Sestrian is not impossible — it is *evidence-generating and reversible*, which changes the attacker's calculus entirely.

**Residual**: a stealthy backdoor that passes probes, survives the window, and is never detected is not excised. The Phase 0 red-team confirms this is the real state of affairs, not a hedge: blind input-space detection fails, and a slow-drip coalition can accumulate a strong backdoor from deltas each *less* conspicuous than honest work. What the red-team also confirms is that this residual is **recoverable, not fatal** — once a trigger is disclosed (by bounty, audit, or the attacker exercising it), the known-trigger probe catches it trivially and replay-excision removes it (backdoor 0.94 → 0.00 at modest clean cost). Detection quality is the open research frontier (§12.3); reversibility is the guarantee we actually hold.

### 7.3 Lock 3 — fake serving (inference honesty)

**Threat**: a serving node runs a cheaper model (quantized, distilled, stale) while charging for the canonical one, or tampers with outputs.

**Lock**: every receipt binds output to the weights-state root (§4.2). Enforcement: staked serving with **random spot-check re-queries** — greedy-decode probes recomputed exactly (RepOps) for bit-match, sampled-decode probes verified against the receipt's pre-sampling **logit commitment** (the proof-of-logits pattern); mismatch is a fraud proof, slash on-chain. Enterprise SLA tier adds **TEE attestation** (production-grade on current NVIDIA hardware) for per-call hardware-rooted proof. Economics complete the lock: serving stake is large relative to the margin from serving a cheap model between audits, making fraud negative-EV at the audit rate (rate and stake sizes are protocol parameters, §9).

**Residual**: TEE trust reduces to the hardware vendor; sampling gives probabilistic rather than universal coverage — both priced and tiered rather than hidden (§8.2).

### 7.4 Chain-level attacks (conventional surface)

Data withholding (defeated by DA sampling as a validity condition, §3.3 — consensus-critical, since challenges need the data); long-range rewrites (bounded by finalized checkpoints, §5.4); eclipse/network partitions and beacon manipulation (standard mitigations; the beacon must be unbiasable — e.g., threshold-VRF — because shard assignment, committee sampling, and evaluation draws all hang from it: **the beacon is the root of the security tree** and is specified accordingly).

### 7.5 Summary table

| Surface | Lock | Majority needed | Residual |
|---|---|---|---|
| Wrong compute | Scored mempool + commit-reveal + challenges + robust aggregation | 51% for liveness; 1 honest challenger for safety | none material |
| Poisoned data | Admission gate + probes + vested clawback + **replay excision** | governance-honest majority for excision | undetected stealthy backdoors (§12.3) |
| Fake serving | Receipts + spot-checks/logit commitments + TEE tier + stake sizing | none (per-node economics) | TEE vendor trust; sampling coverage |
| Ledger itself | Checkpoints, DA sampling, unbiasable beacon | standard BFT assumptions | standard |

## 8. Verified inference & serving

### 8.1 Attestation is a property, not a product

Our closest competitor's shipped, revenue-generating product is "open models plus cryptographic proof of which model ran." In this architecture that feature costs zero additional machinery: the weights-state root lives in every block header (§3.2), and every receipt (§4.2) binds an output to it. *"Output o came from weights W_N at block N, served by staked node k"* is a chain-native, third-party-checkable statement. For any buyer who cares about auditability — compliance regimes requiring provenance of AI outputs, builders who need to know they weren't silently downgraded, agents paying other agents for verifiable work — the mirror of our weights was never a substitute for our API (§8.4).

### 8.2 Verification tiers

| Tier | Mechanism | Guarantee | Cost |
|---|---|---|---|
| Base | Staked receipts + random spot-check re-queries (§7.3): greedy probes bit-matched via RepOps; sampled decoding audited against pre-sampling logit commitments | Fraud is negative-EV; probabilistic detection with slashing | ~zero marginal |
| Attested | TEE serving (NVIDIA confidential computing), attestation in receipt | Per-call hardware-rooted proof | small perf/premium |
| Replayable | Full deterministic decode, receipt sufficient for exact third-party re-execution | Absolute, court-grade | full recompute on audit |

Customers pick the guarantee they'll pay for; the fee split (§9.2) prices each tier. Verification spend is itself a fee-funded line item — watchers earn from the verification share plus slash bounties, so audit coverage scales with usage rather than with charity.

### 8.3 The serving layer

Serving nodes stake, sync per block via the delta stream (§3.5 — staying hot is cheap: one compressed aggregate delta per block), register capacity, and serve standard OpenAI-compatible endpoints. **Latency is decoupled from block cadence**: requests stream over payment channels in milliseconds; receipts batch-anchor per block for settlement (§4.2). Fiat/stablecoin gateways convert customer payments at the edge so API buyers never need token UX (§9.2) — the token is plumbing, not a checkout obstacle. Because the steady-state training regime is RL post-training (§6.4), serving capacity and rollout-generation capacity are the same fleet with one scheduler: idle serving GPUs earn training rewards, spiky demand pulls them back — the "compute either trains or serves" founding sentence, implemented as a single dispatch decision.

### 8.4 Mirrors: the leakage non-problem

Weights are public every block by construction (§3.5). Anyone may pull the delta stream and serve the model off-chain. This is deliberate, and it resolves the "open weights leak revenue" problem that the rest of the industry treats as unsolvable — by inverting who holds the structural advantage:

- **Supply-side arbitrage (the decisive one).** A mirror operator and an on-chain serving node run the same silicon at the same cost. The mirror earns inference margin only. The on-chain node earns inference fees *plus* protocol rewards (serving share, and training rewards for idle-cycle rollouts, §8.3). Rational GPU owners therefore route capacity on-chain until margins equalize — which means the chain's posted API price *undercuts* any mirror's breakeven as long as protocol rewards carry value. Mirrors cannot win a price war against the network subsidizing its own servers; this is Bitcoin's issuance-subsidized-service economics, aimed at inference.
- **Demand-side differentiation.** Only the chain issues receipts (§8.1) — a mirror serving "the same weights" is an unverifiable claim; attested and replayable tiers don't exist off-chain. Only the chain is always freshest — mirrors serve block N−lag while continuous post-training (§6.4) keeps the canonical model moving. And only on-chain usage feeds back (§4.3): mirror traffic is dead signal, chain traffic steers the asset and earns rebates.
- **Therefore: mirrors are marketing.** Every mirror grows the model's install base and funnels quality-, trust-, or freshness-sensitive demand to the canonical endpoint. The correct posture is to make mirroring *easy* — clean delta-stream APIs, reference serving stacks — because the funnel drains toward whoever holds the structural advantage, and §9 constructs that advantage explicitly.

**Honest caveat** (expanded in §12.2): the supply-side subsidy is reflexive — it works while the token has value. The demand-side moats (attestation, freshness, feedback) survive token weakness; the fee base is therefore weighted toward them.

## 9. Economics — revenue-anchored rewards, emissions sunset

### 9.1 The discipline

The field's flagship economy runs a 22–40:1 emissions-to-revenue subsidy and calls it a flywheel (§2.3). Sestrian's economic constitution is one sentence: **rewards are anchored to fees the chain actually earns; issuance exists only to bootstrap, and it sunsets on revenue milestones enshrined in the protocol.** Every design choice below serves that sentence.

### 9.2 The token and the flows

One native token with three jobs: **stake** (validators §5, serving nodes §7.3, data submitters §7.2 — all slashable), **settlement** (fees clear in it; customers pay fiat/stablecoin at gateways that market-buy the token, so demand for the API is demand for the token without crypto UX at checkout), and **reward** (the two pools below).

```
Inference fee (per ForwardPropTx, initial targets, governed per §10):
  55% serving node   | 10% verification pool (watchers, spot-checks, challenge bounties)
  25% training pool  | 10% burn

Training reward per block  =  training-pool accrual  +  Emission(N)
  → distributed to included deltas ∝ median score (§5.2)
  → minus data-royalty share to the shards' submitters (vested, clawback-liable, §7.2)
```

The 25% training share is the founding thesis — *inference funds training* — as a protocol constant rather than a promise. Data royalties implement the deeper principle (the CAPITALISM rule): pay every contributor by measured marginal value — miners per scored delta, data per influence, at rates the market tunes.

### 9.3 Emission schedule and the sunset

`Emission(N)` starts high enough to pay for bootstrap training (§9.5) and **steps down as trailing protocol revenue crosses enshrined milestones** — e.g., each time trailing-90-day fees sustain X% of current emission value, emission steps down 25%; below a revenue floor it pauses stepping but never rises. Properties: total issuance is hard-capped; the sunset is monotone (no governance vote can re-inflate — an anti-Bittensor commitment device, credible precisely because it is *not* adjustable); the crossover block — the first block where fee-funded training rewards exceed emission — is the network's public, verifiable "first profitable breath," and reaching it is the entire Phase-3 goal (§11).

### 9.4 The write-price homeostat

Backprop admission (§4.1) carries a floating fee: when candidate-delta volume exceeds what committees can evaluate well (or aggregate quality sags), the write price rises; when capacity idles, it falls — a difficulty-adjustment controller holding *evaluation load and workspace interference* at the network's capacity frontier, exactly as Bitcoin's difficulty holds block rate against hash power. (This is the design's origin story earning its keep: spam, sybil grinding, and low-effort delta floods are all the same phenomenon — interference — and all priced out by the same controller.) A matching, gentler controller floats the inclusion-quality threshold (§5.2) against the block's delta budget.

### 9.4a The capacity retarget — model size as difficulty

The fourth knob of the homeostat family, and the one that makes joining compute *compound* instead of merely crowding: **the parameter count itself retargets against the network's training capacity**, the way Bitcoin's difficulty retargets against hash power. Where Bitcoin spends surplus compute on a harder puzzle (security), Sestrian spends it on a bigger brain (capability — which *is* our security: every joined GPU is embodied in weights a challenger must reproduce, and outvoting the training set grows proportionally costlier).

Two things Bitcoin's controller never had to face, and their resolutions:

- **Measurement.** Hash power is self-evidencing; training compute is not. The retarget therefore consumes only *chain-observable, sybil-resistant* signals: the count of accepted-and-scored deltas per retarget window (junk deltas score ~zero, so flooding doesn't register), block fullness against the inclusion budget, and delta staleness (late deltas = the model already strains the fleet). No FLOP-proofs required — the same signal family the write price already consumes.

- **Irreversibility.** Difficulty adjusts down; a model cannot un-grow without destroying trained value. The controller is therefore a **ratchet with an elastic active set**, on two timescales:
  1. **Fast knob (continuous, reversible):** the per-delta work quota — required inner steps and shard size per block — adjusts damped every window, holding block cadence and per-node load steady as miners come and go. This is the literal difficulty analog.
  2. **Slow knob (discrete, ratcheted):** when the fast knob has been pinned at its ceiling for K consecutive windows — a *sustained* compute surplus — a **growth event** fires: bounded (at most one new module per level per event; §3.1 pages, DiPaCo modules as the growth unit), announced N blocks ahead so nodes pre-provision memory, with the new pages initialized **deterministically from the trigger block's hash** — every node computes the same decision at the same height with identical new weights. No vote, no coordinator; consensus-safe by construction (on-chain model growth is Phase-0-proven).
  When compute *leaves*, total parameters never shrink — the network instead **freezes modules** (they stop training but keep serving inference) until capacity returns: total capacity ratchets monotonically, *active* capacity breathes with the fleet — a graceful degradation MoE sparsity provides for free.

The governance consequence is the point: capacity growth is **algorithmic, not political**. Nobody votes on whether the model grows — governance only sets the controller's constants (window sizes, damping, growth bound), which are then as hard to move as any §10 parameter. "The network grew its brain at block N because its miners earned it" is simultaneously the mechanism, the security model, and the story.

### 9.5 The bootstrap tunnel, with honest numbers

Target: a 7–40B specialized model (§11). Order-of-magnitude arithmetic a seed deck must survive: pretraining a ~20B model on ~1–2T tokens costs low-single-digit $M in centralized compute; multiply by ~2–3× for the permissionless handicap (WAN, scoring overhead, redundancy — §12.1) → **$5–15M of emission-funded work to reach a servable v1**, then continuous post-training at a small fraction of that rate. The crossover condition is then: sustained API revenue ≥ ~25× the steady-state training burn × (1/training-share). A specialized model earning $2–5M ARR — a modest B2B business — sustains a $0.5–1.5M/yr continuous-training budget at the 25% share. These are achievable, checkable numbers, and publishing them is itself a differentiator in a field that hides its unit economics behind emissions.

### 9.6 The serving-arbitrage ledger

Making §8.4 concrete: let c = a node's cost per token served, m = market inference margin. Mirror earns m − c. On-chain node earns (fee share − c) + protocol rewards r. The protocol sets the posted price such that fee share − c ≈ m − c *minus* r — i.e., the API undercuts mirror breakeven by approximately r while nodes stay whole. r is funded by emission during bootstrap and by the burn/reward flows at maturity. The controller keeping serving capacity matched to demand adjusts r's serving component the way §9.4 adjusts the write price — one homeostat family, four knobs (write price, quality threshold, serving reward, and the capacity retarget of §9.4a).

### 9.7 Reflexivity, stated plainly

The arbitrage in §9.6 runs on token value; in a deep token bear a hyperscaler mirror could underprice the network's nodes. Mitigations, in order of importance: (1) the **fee base leans on demand-side moats** — attested/replayable tiers, freshness, feedback rebates (§8.4) — which are token-price-independent; (2) a protocol **treasury buffer** (funded from the burn share pre-crossover) smooths reward flows in fiat terms; (3) rewards are *denominated* in fee value, not token count, where possible. Reflexivity is not eliminated — it is the residual risk this design consciously carries (§12.2) in exchange for permissionless supply.

### 9.8 Launch fairness

No pre-sale of emission rights; team/treasury allocation capped and vested against the same milestones as the sunset; genesis weights, corpus registry, and all launch parameters published and replayable from block zero. The credibility of "revenue-anchored" is set at launch and cannot be retrofitted — this section is written before the token exists on purpose.

## 9A. Data as a priced input

The chain already prices one input to the model — gradient contributions, paid to miners by scored improvement. **Data is the other input to the same production function, so it is priced by the same mechanism**: a scored mempool pointed at data. This turns a data contributor from a cost to be curated into a stakeholder who is paid for provable value, and it is what makes "community-owned data" (§11.2) an economic relationship rather than a slogan. Two revenue streams, both implemented and tested (`rig/data.py`, `rig/attribution.py`, `rig/data_flywheel.py`).

### 9A.1 Stage 1 — pay for contribution (the signing bonus)

A **DataTx** is a signed submission to the on-chain data registry. Each **channel** carries a base value rate (research low, professional/proprietary high — the coarse knob, §9.2's per-channel `$/bit`). Within a channel, a shard is priced by its **marginal value**: the first-order effect of training on it on a beacon-drawn holdout — the alignment of the shard's gradient with the descent direction the holdout wants. Two properties fall out for free and are verified in the rig: data that fills a *gap the queries need* prices highest; data the model already covers has a near-zero gradient and prices ≈ 0, so **duplicates and Sybil floods earn nothing without a special rule**. The bonus is paid into a vested ledger.

### 9A.2 Stage 2 — pay for downstream usage (royalties)

The signing bonus pays once; the royalty pays again every time the data helps answer a paying query — the piece that turns data into a standing, income-producing asset. The mechanism is TRAK/TracIn influence made cheap and verifiable: each admitted shard gets an **influence sketch** — its training gradient projected through a shared random matrix (Johnson–Lindenstrauss-faithful, so a small sketch stands in for the full gradient), *summed over the chain's checkpoints* (the chain already has them — TracIn is native to a ledger that replays). For each served query, the emitted answer is sketched the same way, and the royalty slice of the fee is split across the shards whose training **supported** the answer (positive gradient alignment), paid to their owners. Because the sketches are on-chain and recomputable, the royalty split is independently **verifiable**, like everything else here. In the rig this attributes correctly ~90%+ of the time and royalties track usage: the more the world asks about a contributor's corner of knowledge, the more their data earns.

### 9A.3 The tension faced head-on — paying for influence is paying poisoners

Rewarding influential data is, by construction, a bounty on influential *backdoors* — the §12.3 dragon reappears exactly here. The resolution is that **influence gets you paid, but only durable, beneficial influence keeps you paid**: the bonus vests over a window with clawback, and the same replay-excision that removes a discovered backdoor (§10.4) forfeits its unvested reward, halts its royalties, and slashes its stake bond. The rig confirms the economics: a poisoner who earns a large bonus and early royalties still ends **net-negative** once discovery (before full vest) triggers clawback and slashing. Poisoning is not prevented — it is made unprofitable and reversible, which is the same honest posture as §7.2. The residual is the same too: a backdoor never discovered is never clawed back; fast, well-incentivised detection (bounties, challenges) is what shrinks the window.

## 10. Governance

### 10.1 Principle: govern as little as possible, verifiably

Most of the protocol is mechanical (scoring, aggregation, emission sunset, homeostats) precisely so that governance's surface is small. What remains governed is what *cannot* be mechanical: what the model learns from, what the model is, and emergencies. Every governed process must pass the **anti-theatre test** (§10.5) — the standard Bittensor failed when its flagship team quit calling centralized control by its name.

### 10.2 Data admission (the security frontier wearing a governance hat)

Two tracks, matching §7.2:

- **Permissionless track** (default): any staked submitter posts shard batches with provenance metadata → quarantine/challenge window → beacon-audited probes → admission to the registry. Governance sets the *parameters* (stake sizes, window length, probe suites) but never votes on individual shards — no committee decides what truth is; the challenge economics do.
- **Campaign track**: large corpus decisions — a vertical's foundational dataset, a licensing deal for proprietary data, deprecation of a corpus segment — are token-holder votes with a mandatory public audit period, because they move the model's identity, not just its margin. Licensed-data deals settle royalties through the same vested, clawback-liable channel as permissionless data (§9.2).

### 10.3 Model lifecycle

- **Objective campaigns**: the steady-state RL/post-training direction (§6.4) — reward specs, eval suites, capability targets — proposed with a measurable eval delta, voted, and *scored after the fact* against that eval: campaigns whose promised deltas don't materialize auto-expire. Objectives are falsifiable or they are not adopted.
- **Architecture upgrades** (scale-up, structural change): a hard fork with a defined **weight-migration procedure** (warm-start/distillation from W_N into the new architecture, executed as an on-chain campaign whose output state root is verified by replay before switchover). The model's continuity across forks is part of the state-transition rules — the chain never has two canonical minds.
- **Parameter changes**: fee splits, block interval, probe suites, stake sizes — token vote + timelock. **Non-amendable**: the emission cap and sunset monotonicity (§9.3). A constitution needs at least one clause that cannot be voted away; that is ours.

### 10.4 Emergency procedure: excision

The immune response of §7.2, pre-committed so it cannot be improvised under panic: on evidence of an admitted backdoor, a supermajority emergency vote freezes affected checkpoints, an influence audit over the replayable history identifies the offending shards and deltas, and the chain re-executes from the last clean checkpoint with those contributions excised — producing a new, verifiable state root and slashing the responsible vested stakes. Slow by design (it re-runs history), rare by intention, and unique in the industry: every other network's answer to discovered poisoning is a blog post.

### 10.5 The anti-theatre test

Published, on-chain, quarterly: validator and stake Nakamoto coefficients; committee-capture cost; the share of blocks proposed by the top-5 operators; foundation/team stake and its vesting state; the count of governance actions where a single entity's vote was pivotal. Targets are enshrined with the schedule for hitting them (e.g., no single operator >10% of committee seats by end of Phase 3). If the numbers say the network is centralized, the network is centralized, and the document that admits it is the protocol's own dashboard — not a competitor's exit letter.

## 11. Roadmap & go-to-market

### 11.1 The strategic bet, restated

We do not race funded specialists to "decentralized ChatGPT" — a general model pays the network's 2–3× per-FLOP handicap (§12.1) to compete with subsidized frontier APIs on their strongest axis. We race to the **unclaimed crown**: the first network in history where inference revenue genuinely funds training (§2.2). That crown does not require a frontier model; it requires a model whose users pay more than its continuous-training burn — a gate that opens at specialized scale (7–40B), squarely inside the proven permissionless envelope (§6.5), with the bootstrap arithmetic of §9.5. Win the existence proof, own the mechanism design, scale as the bandwidth ceiling rises.

### 11.2 Vertical selection

Criteria, weighted in order: (1) **paying demand tolerant of 7–40B quality** — specialized fine-tunes beat generalists in-domain; (2) **community-owned data** no lab can license — the moat that survives open weights; (3) **attestation premium** — buyers who need provenance receipts (§8.2), turning our free property into margin; (4) **feedback density** — usage that generates admissible training signal (§4.3), compounding the freshness moat. Candidate shortlist to be scored against these in Phase 0: security/code-audit tooling (crypto-native buyers, attestation-sensitive, dense feedback); legal/regulatory per-jurisdiction (attestation premium, licensed-corpus campaigns); biotech/chem literature+assay (community data, B2B contracts); agent-economy services (x402/A2A-native demand — agents paying for *verifiable* inference is the most naturally aligned customer in existence); high-engagement creative communities (data-rich, price-tolerant). The vertical is a Phase-0 decision made with the community we recruit, not a whiteboard guess — the data-owning community *is* the go-to-market.

### 11.3 Phases

**Phase 0 — the rig (now → +3 months).** The simulation this project was conceived demanding: scored-mempool consensus + attack suite (weight-copying agents, colluding committees, backdoor shards, lazy validators) against the §5/§7 mechanisms; economic simulation of the §9.5 tunnel under demand scenarios; DiLoCo-cadence training loop with deterministic apply on a toy model. Falsifiable exit criteria: attacks lose money in sim, the tunnel closes under realistic demand, replay reconstructs bit-exact state. *This is the founding memo's "stop citing and start seeing" doctrine applied to our own design — if the rig kills the thesis, it dies for a few GPU-days.*

> **Phase 0 status (`rig/`, results in `results/report.md`, 30-test suite via `scripts/run test`).** A toy-scale implementation is built and passing all falsifier checks. Findings that sharpened the design: (a) replay is bit-exact under fixed-point aggregation, confirming model-as-chain-state is mechanically sound; (b) loss-scoring alone is *not* a poisoning defense — stealthy triggers in task-unused capacity pass it (and the fifth-pass red-team below shows blind probes don't save it either; the durable lock is excision); (c) commit-reveal makes freeriding negative-EV above a ~0.4% audit rate, and collusion turns unprofitable exactly at 51% stake — but state safety collapses (~83% bad-delta finalization) without a funded challenge layer, promoting the §9.2 verification share from a line item to a security primitive; (d) the bootstrap tunnel closes in 16–30 months for base/aggressive demand but *never* for a $1.5M-ARR vertical, making §11.2 vertical selection the quantified survival decision.
>
> **Phase 0 (second pass): the flywheel runs on a real model, across real processes, with durable state.** The demo model is now a genuine tiny transformer (`rig/model.py`, single-layer single-head, ~2.5k params, manual backprop verified against numerical gradients to 1e-8), learning a delayed-copy task that requires attention and position. The full loop — train → score → apply → attested serve → reward — runs both in-process (`rig/e2e.py`) and across **separate miner OS processes over localhost sockets** (`rig/node.py`), and the two transports produce the byte-identical chain, so multiprocess consensus is reproducible. The chain **persists to disk** and a restarted node **fast-syncs** from the latest checkpoint to the identical state root as full replay from genesis (`rig/storage.py`). Over 40 blocks the on-chain model goes 0.13 → 1.00 accuracy, rewards spread across all miners, and a fake-serving node is caught by batched spot-check attestation and slashed every block.
>
> **Phase 0 (third pass): asynchrony, depth, and sparse serving — the scaling primitives.** (a) *Async* — the synchronous barrier is dropped (`rig/async_node.py`): heterogeneous fast/slow miners submit whenever they finish, stale deltas are scored against the current head and dropped past a grace window (§4.1), the model still trains to 1.0, and fast miners out-earn slow ones as staleness discounts late work (a real centralizing pressure, cf. §12.1). (b) *Depth* — a minimal reverse-mode autograd engine (`rig/autograd.py`, gradient-checked op by op) carries a multi-layer, multi-head transformer with RMSNorm (`rig/model2.py`, ~68k params) that converges through DiLoCo aggregation. (c) *Sparse serving* — a mixture-of-experts (`rig/moe.py`) where each query routes to top-k of E experts, so serving loads only the router + k expert pages; a Merkle root over pages (`rig/merkle.py`) lets the serving node prove those pages are the committed ones, and the verifier recomputes from only them — **inference and attestation both cost O(k), not O(E)**, the concrete answer to serving a model too large to hold in memory (§3.1, §8).
>
> **Phase 0 (fourth pass): the primitives fused — a MoE transformer, on-chain.** `rig/moe_transformer.py` replaces every FFN block of the deep transformer with a mixture of experts (top-k routing, gradient-checked to 1e-8) and trains it *through the chain* — both synchronously and through the async, staleness-handling path (`scripts/run async_node --moe`) — a 2-layer × 8-expert model reaching 1.0 accuracy via DiLoCo aggregation, replaying bit-exact. Weights are paged as a backbone plus one page per (layer, expert). Serving is attested by a **true partial-recompute verifier**: the verifier holds *only* the pages the receipt names (backbone + the query's expert union), checks their Merkle inclusion proofs against the committed root, and then recomputes the whole output from those pages alone — routing comes from the backbone, so a token routing to an unloaded expert rejects the receipt (the server under-loaded), and un-routed experts are never materialized, so the whole model is never loaded to verify. Tampered pages, wrong roots, under-loading, and forged outputs all fail. The incremental per-token cost — a **decode step** — routes to only top-k experts per layer, so per-token serving is O(top_k), not O(E): 0.2% of expert capacity at 1024 experts/layer. The suite is 65 tests (`scripts/run test`). Still toy-scale — these validate mechanisms, systems plumbing, and the sparse-serving primitive at architecture level, not model quality.

> **Phase 0 (fifth pass): the §12.3 red-team — and an honest correction.** `rig/redteam.py` finally attacks the assumption the whole security story rested on, and the result forced a correction to our own claims. Findings: (a) a stealthy backdoor keyed to a rare out-of-distribution trigger is **invisible to blind input-space probing** — clean-loss and in-distribution trigger sweeps miss it — while a known-trigger oracle catches it trivially; (b) a **slow-drip coalition** accumulates a strong backdoor (0.94 success) from deltas each *below the honest anomaly band*, defeating weight-space detection too; (c) **replay-excision recovers**, driving the discovered backdoor from 0.94 → 0.00 at modest clean-accuracy cost. The correction: earlier passes credited "canary probes" with catching poisoning — that used a *known* trigger and overclaimed. §7.2 and §12.3 are now stated precisely: poisoning is not prevented by scoring or blind probes; the real defenses are staked data-admission (raising the cost of getting poison in) and excision (recovering once a backdoor is disclosed). The residual — an undetected, never-triggered backdoor — is real and disclosed. This is exactly the kind of result Phase 0 exists to produce: a claim killed cheaply before it could mislead a fundraise. The suite is 69 tests.

> **Phase 0 (sixth pass): off the laptop, and off the coordinator.** Two more things that were faked are now real (`docs/internal/distributed-systems.md`). (a) *Across machines* — `rig/lan.py` trains across two physical machines (Mac + a 32-core Linux box over Tailscale); the cross-machine chain head is **byte-identical** to the all-local chain, so where a miner runs never changes the state. (b) *The Bitcoin-shaped distributed-systems layer* — Ed25519-signed transactions (`rig/crypto.py`), hash-linked block headers with independent first-principles validation and **Nakamoto heaviest-valid-chain fork choice** (`rig/blockchain.py`), a **peer-to-peer gossip network with no coordinator** (`rig/p2p.py`) that reaches consensus, forks under a network partition, and heals to one agreed history on reconnect, and the **difficulty-style write-price homeostat plus a stake/slash ledger** (`rig/economics.py`) that holds the admission rate at target, prices out spam, and slashes provable faults with a challenger bounty. The suite is 93 tests.

> **Phase 0 (seventh pass): the three faked primitives made real.** The distributed-systems doc's three "still faked" items are now real, tested implementations. (a) *Unbiasable randomness* — a **threshold-BLS (drand-style) beacon** (`rig/beacon.py`, on BLS12-381 pairings): a group key Shamir-shared t-of-n, per-round partial signatures Lagrange-combined into the unique group signature s·H(r); any t-subset yields the *identical* value (unbiasable), fewer than t reveal nothing (unpredictable), verified by one pairing (verifiable). (b) *Real gossip* — the p2p logic now runs over **real async sockets** (`rig/gossip_net.py`), verified coordinator-free across machines (2 nodes on the Mac + 1 on a second machine over Tailscale converged to the identical head). (c) *Real DA* — an **erasure-coded, availability-sampled** layer (`rig/da.py`): Reed-Solomon over a self-contained GF(256) makes any k of n shards reconstruct a body, Merkle-committed and sampled so a withholding attack is unrecoverable *and* detected by a few random samples. What remains is integration, not invention (distributed key generation, wiring beacon+DA into live block production, leader election) — scoped in `docs/internal/distributed-systems.md`. The suite is 109 tests.

**Phase 1 — devnet (+3 → +9 months).** Real chain, small model (~7B), invited nodes. Ship: page-Merkle state + DA layer, BackpropTx/receipt pipeline, commit-reveal scoring, delta-stream serving, replay/fast-sync. Steal shamelessly: Gauntlet's scoring economics, Verde/RepOps determinism, DisTrO compression, existing DAS designs (§2.1). Build-vs-fork decision for the base chain made here (candidate: OP-stack or Solana-fork with custom state machine — the state transition is ours either way).

**Phase 2 — incentivized testnet (+9 → +15 months).** Chosen vertical's community aboard: data-admission pipeline live with real corpus campaigns, points-not-token incentives (regulatory posture, §12.4), first external API users on the base verification tier, anti-theatre dashboard live from day one (§10.5). Exit: model demonstrably best-in-vertical on public evals, paying pilot customers.

**Phase 3 — mainnet & the crossover (+15 months →).** Token genesis under §9.8 fairness, emission-funded scale-up of the model, revenue milestones ticking the sunset. The single public goal: **the crossover block** (§9.3) — fee-funded training rewards exceeding emissions — announced in advance, verifiable by anyone, and unprecedented in the industry when it lands.

**Phase 4 — scale.** Grow the one model with the bandwidth frontier (DiLoCo-class methods keep improving; the ceiling at ~100B in 2026 will not hold); deepen the vertical before widening; resist multi-model sprawl — the single-model discipline (one artifact, one quality signal, fully-loaded GPUs) is a structural advantage borrowed from our closest competitor's correct intuition, and the temptation to become a marketplace is how the field's largest player acquired its pathologies.

### 11.4 What we take from the incumbents, and whom we hire

Adopt: Gauntlet (scoring), Verde/RepOps (determinism), DisTrO/DeMo (compression), DiLoCo (cadence), Celestia-pattern DAS (availability), TEE stacks (attested tier). Watch: Ambient (nearest thesis, pre-mainnet — their launch is our market validation, their architecture our §2.4 contrast), the ex-Covenant team ("decentralization theatre" critics, proven at 72B, currently unannounced — the best possible hires or the most credible competitors; engage early either way).

## 12. Limitations & open problems

Per the founding memo's doctrine, the limits section is first-principles and unflinching. These are the sentences a hostile reviewer should have to write, written by us first.

### 12.1 Physics and cost

- **The frontier is out of reach.** Internet-scale training is proven to ~100B; 10²⁶-FLOP frontier runs remain bandwidth-bound. Sestrian will not train the world's best general model this decade; the strategy (§11) is built *around* that fact, not in denial of it. If bandwidth research stalls, the network's ceiling stalls with it.
- **The permanent handicap.** Scoring overhead, DA redundancy, WAN sync, and commit-reveal latency cost an estimated 2–3× per useful FLOP versus a centralized cluster. The design claws some back (miner-side efficiency competition, §6.3; serving/training fleet unification, §8.3) and the economics must simply carry the rest. If the handicap turns out to be 10× rather than 3×, §9.5's tunnel arithmetic fails.

### 12.2 Economic reflexivity

The serving-arbitrage moat (§8.4, §9.6) runs on token value; a deep bear plus a determined hyperscaler mirror could underprice the network's own nodes for an extended period. Mitigations exist (§9.7) but the residual is real: **the supply-side moat is procyclical.** The demand-side moats (attestation, freshness, feedback) are the ones that must hold in winter, and they are the younger, less-proven half of the argument. Relatedly: if the chosen vertical's demand never materializes at §9.5's thresholds, the sunset never triggers, and the network dies as one more emissions machine — the exact failure we designed against, still reachable through ordinary commercial failure.

### 12.3 The Goodhart frontier (the honest hard problem)

Scored inclusion optimizes *measured* improvement. Two known gaps:

- **Stealthy backdoors**: an update or shard engineered to improve held-out loss *and* implant a trigger passes every *blind* probe we can specify — the Phase 0 red-team (`rig/redteam.py`) demonstrates this directly: clean-loss and in-distribution trigger probes miss a backdoor keyed to a rare OOD trigger, and a slow-drip coalition accumulates one from individually-inconspicuous deltas. Replay-excision makes poisoning reversible *when detected*; detection of a never-triggered backdoor is the open research problem, and we inherit it from the entire field rather than solving it. Our honest claim is strictly: *attacks here are costlier, evidence-generating, and reversible — not impossible.*
- **Loss ≠ value**: a delta can improve perplexity while degrading what customers pay for. Probe suites and objective campaigns (§10.3) patch this with explicit evals, but eval design is a treadmill, and validator-run evals are themselves gameable surface. The scored mempool is a mechanism for *agreeing on a measurement* — the measurement's alignment with value is forever curatorial work.

### 12.4 Legal and regulatory surface

A token with fee flows is a securities-law event in most jurisdictions (the §9.8/Phase-2 points-first posture is mitigation, not immunity). An unstoppable public model raises model-liability questions no regulator has settled (who recalls a model with no landlord? §10.4's excision is our partial answer, and regulators may not accept it). Conversely, the EU AI Act's provenance requirements are the attestation tier's commercial tailwind — the same regulatory wave breaks for us on one shore and against us on another.

### 12.5 Competition

Ambient shipping its mainnet validates the category and occupies the narrative high ground of "model as network state" (§2.4). The ex-Covenant team could launch a Sestrian-shaped network with a two-year credibility head start. A frontier lab releasing strong open weights in our chosen vertical compresses the quality gap our fee base rests on. And the field's protocol bodies (A2A/x402 foundations) could standardize enough agent-payment trust that parts of §8's premium become commodity. Speed to the crossover proof (§11.3) is the only durable answer.

### 12.6 Falsifiers

The thesis is dead, and should be declared dead, if any of the following holds:

1. The Phase-0 rig shows commit-reveal scored consensus gameable at <51% stake with positive EV after slashing (kills §5).
2. Deterministic apply + scoring overhead exceeds ~25% of network compute in practice (kills §6.3's sub-linearity claim and stretches §12.1's handicap past viability).
3. No candidate vertical clears: paying demand at 7–40B quality ≥ ~25× steady-state training burn (kills §9.5/§11).
4. DA costs for delta bodies dominate fee revenue at target scale (kills §3.3's economics).
5. ~~A stealthy-backdoor construction defeats probes + influence audit *reliably and cheaply* in the rig's red-team suite.~~ **Partially fired, and resolved by re-scoping the claim (Phase 0 red-team).** A stealthy backdoor *does* defeat blind probing cheaply — so §7.2 never rests on detection. It is not theater, because the surviving guarantee is reversibility: excision removes any *discovered* backdoor. The design's poisoning claim was corrected from "probes catch it" to "admission cost + excision", which is what the rig actually supports. The live version of this falsifier is now: *a backdoor that is never discovered AND causes standing harm before it is* — an argument for aggressive bounties and disclosure incentives, not a defeater.

---

## Coda

Five requirements generated the transformer; this document's claim is that five requirements generate Sestrian: a public model no one owns, compute that must be paid from value rather than inflation, training that must be verifiable by strangers, a ledger durable enough to be the model's only body, and incentives that make honesty the profitable strategy. Enumerate the ways to satisfy all five at once and — as with attention — very little survives: the model must *be* the chain, blocks must *be* optimizer steps, and revenue must *be* the reward. Whether the surviving design also survives contact with reality is what Phase 0 exists to measure. Stop citing; start seeing.
