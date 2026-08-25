# Data Provenance & Payouts (protocol rev 5)

The rule that makes Sestrian a market for *data*, not just compute. It has one
invariant and two payment moments. This is the reference the rig implements and
the Rust node mirrors bit-for-bit.

## The invariant: provenance-required deltas

**A delta (backprop) is accepted only if it names data that is staked and
retrievable on the DA layer.** Concretely, `BackpropTx` carries `data_refs`: the
sorted, de-duplicated set of `data_hash`es (content addresses) of the corpora the
gradient was trained on. A delta is valid iff every hash in `data_refs`:

1. resolves to an **active** entry in the on-chain data registry (a `DataSubmitTx`
   was accepted, a bond is locked), and
2. is **DA-available**: the corpus bytes are retrievable from the DA layer
   (erasure-coded shards + availability sampling), so anyone can fetch them.

A delta with an empty or unbacked `data_refs` is **rejected**, not merely unpaid.
There is therefore no such thing as an "unprovenanced" block, and no orphan data
share to dispose of. The consequence the design wants: **the model is only ever
trained on data the network can audit**, and "upload the hash then delete the
data" is impossible: the bytes must live on the DA layer to train at all.

Auditing is by recompute-and-challenge, like everything else on the chain: a
challenger fetches the named corpus, re-runs the (deterministic, seeded) training
step, and if the resulting gradient does not match the committed `delta_hash`, the
delta is fraudulent and its bond is slashed.

Because staked data lives on the DA layer, it is a **shared resource**: any miner
can fetch any staked corpus and train on it. Compute (miners) and data (owners)
are decoupled: a miner with a GPU and no data earns the miner share by training
on others' corpora; the corpus owner still earns the data share. Nodes need the
source bytes only to *train on* or *challenge* a corpus, never for ordinary block
validation (which needs only the delta bodies, already on DA).

## Payment moment 1: contribution, at training

When a delta lands in a block, that block's **data share** (the data slice of the
emission) is split across the owners of the corpora the delta named, in proportion
to the delta's **loss-reduction score**: how much it actually improved the model
that round (the held-out-shard score; see delta scoring). No recency decay: a
contribution is worth what it measurably taught the model, whenever it was made.

- If a delta names one corpus, that corpus's owner takes the delta's whole
  score-weight. If it names several, the weight splits equally across them (the
  miner is asserting all were used; finer intra-delta attribution is Stage 2's
  job).
- Multiple deltas in one block: the data share splits across all their
  (corpus, score) contributions, dust burned, deterministic.

This pays data *once per round it is trained on*, for as long as it keeps teaching
the model something. As the model saturates on a corpus its scores fall and this
income tapers, not by a clock, but by measured usefulness.

## Payment moment 2: usage, at recall

The block reward pays data for being *learned*. The inference fee pays it for
being *used*. When the model answers a paying query, the fee's **royalty slice**
is split across data sources by how much each one shaped *this* answer.

Exact influence of a datum on an output is nonlinear and path-dependent; we use
the standard first-order estimate (TracIn/TRAK), made cheap and verifiable by a
shared random projection:

- **Per-source sketch.** The model is `w = genesis + Σ deltas`, so each weight is
  a sum of data-attributed gradient contributions. We never store the full
  per-weight-per-source decomposition (infeasible at 10^8 params). Instead each
  data source keeps a fixed-size **influence sketch**: its accumulated training
  gradient projected through a shared, seeded ±1/√d matrix `P` (256 dims). By
  Johnson–Lindenstrauss, `⟨sketch(a), sketch(b)⟩ ≈ ⟨grad(a), grad(b)⟩`, so the
  256-float sketch stands in for the whole gradient. Sketches are committed
  on-chain and **recomputable from the DA-available corpus**: the split is
  independently checkable and challengeable.
- **Answer sketch.** For a served query, sketch the gradient of the *emitted
  answer* (its log-prob w.r.t. the weights) through the same `P`.
- **Split.** A source *supported* the answer iff `⟨sketch(source),
  sketch(answer)⟩ > 0`; the royalty slice is divided across supporting sources in
  proportion to that positive alignment, and paid to their owners.

No recency term. A latent-but-valuable corpus earns nothing while unused and earns
in full the moment queries begin leaning on the weights it shaped, next week or
in four years. **Usage is the clock, not wall-time.**

## Determinism & verifiability

Both moments are consensus state (they mint/move tokens), so both must be
deterministic across nodes and recomputable by any verifier:

- The projection `P` is fixed from a published seed; sketches are integer-quantized
  before commitment, so every node computes identical sketches (same discipline as
  the weight quantization at the consensus boundary).
- Scores and sketches are recomputable from DA-available data + the committed
  model state; a wrong score or sketch is a challengeable, slashable claim.
- The inference receipt commits to the head `state_root`, the query hash, the
  emitted-answer hash, and the answer sketch, so which model served the reply,
  and how the royalty split was computed, are both provable after the fact.

## What gates on off-chain execution

The **ledger structure** (`data_refs` on the delta, the data-share payout routed
to named owners, the royalty-pool split routed by sketch alignment) is pure,
deterministic ledger arithmetic and is enforced + golden-tested in the Rust node
now. The **inputs that require running the model**, the loss-reduction score
(Stage 1 weight) and the gradient sketches (Stage 2 alignment), ride on the same
off-chain execution + commit-reveal infrastructure as delta scoring, and are
validated in the rig and on the testnet rather than by golden vector (model
execution crosses the float-nondeterminism boundary that golden vectors sit below).
Until that infrastructure is enforced, the node runs with score/sketch inputs it
is given and the challenge market is the backstop; the payout *routing* is live and
exact regardless.
