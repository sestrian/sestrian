# Sestrian

**A blockchain whose state is the weights of a single public neural network.**

[sestrian.com](https://sestrian.com) · [live chain](http://169.58.211.248:8080/status) · [join the devnet](docs/joining.md)

Transactions are the model's own computations. A *backprop* is a signed gradient
that transitions the state and earns a reward; a *forward-prop* is an inference
that pays a fee. Replaying the chain from genesis reconstructs the model
bit-for-bit. The chain does not *record* a model — the chain **is** a model, and
it trains itself in public, paying the people who train it.

No company owns the weights. No API key gates them. You download the chain, replay
it, and you are holding the same model everyone else is — plus the ledger that
says who trained it and who paid for it.

## How it works

- **State = weights.** Genesis is a from-scratch model. Each block commits an
  aggregated gradient delta and a new `state_root`. `weights = genesis + Σ deltas`.
- **Mining = training.** Miners pull the head weights, train locally (any GPU —
  the network already trains cross-vendor, Apple MPS + NVIDIA CUDA, with no
  coordinator), and submit compressed int-quantized deltas. Included deltas earn
  the block reward.
- **Fees = inference.** Anyone can run the model. A forward-prop is a signed,
  fee-bearing receipt that settles from the payer to the server on-chain.
- **Data is staked, owned, and paid.** Training corpus lives off-chain; the chain
  holds only its hash, its owner, and a stake bond. Good data earns a share of
  every block trained on it; bad data can be challenged and slashed.
- **The token is the model's own money.** Fair launch, every grain minted by
  verifiable work, emissions halving to a hard sunset. It exists to pay trainers
  and price inference — nothing is pre-mined.

## Start in two lines

**See it run, locally, right now** — a real PyTorch model training through a
local chain, loss falling live in your browser, chat with the head:

```bash
uv run --with torch --with numpy --with pynacl python -m client.watch --demo
```

**Join the live network** — build the node, generate the genesis (it's
deterministic from a published seed, so you reproduce it rather than trust a
download), then sync:

```bash
cd node && cargo build --release && cd ..
python -m client.wallet new                      # your identity AND your balance

# reproduce the genesis — must print state_root a5973160…527b (the page-Merkle root)
uv run --with torch --with numpy --with pynacl \
    python -m client.make_genesis --model small-moe --seed 20260822 --out genesis.bin

# check you can actually contribute BEFORE committing hours to it
node/target/release/sestrian-node --check \
  --data-dir ~/.sestrian --wallet ~/.sestrian/wallet.json --genesis-file genesis.bin

# then run it
node/target/release/sestrian-node \
  --data-dir ~/.sestrian --wallet ~/.sestrian/wallet.json --genesis-file genesis.bin
```

There are no chain parameters to get right: like Bitcoin's `-testnet`, the
network's genesis id, bootstrap peer and genesis-ledger constants are **baked
into the binary** (`--network devnet`, the default). Passing a value that
contradicts the network is a startup error, not a silent fork. Running your own
chain? `--network local` and supply everything yourself. Your node serves a
dashboard + API at `http://localhost:8090`.

> **Always run `--check` first.** It verifies the peer is reachable, your genesis
> matches the network, and — if you're mining — that your GPU and block interval
> are compatible. The failure modes here are silent by nature: a trainer slower
> than the block interval produces deltas that are always too late to include, so
> it would mine forever and earn nothing.

## Three ways to participate

Pick one or all three. They share one economic loop: **you earn coins by
training, and you spend coins to submit data — which then earns you coins back
every time the model trains on it.**

| Role | You do | You earn |
|---|---|---|
| **⛏ Mine** | run the node with `--produce` and attach the trainer; it trains the head and submits deltas | block rewards for every included delta |
| **🔌 Serve** | run an inference bridge; answer `POST /inference` calls | the fee on every request, settled to your wallet |
| **📚 Give data** | stake coins behind a corpus you own | a share of every block reward trained on your data |

Every block reward splits three ways: the **trainers** whose deltas landed, the
block's **proposer**, and the **data** that trained them. That is why you must
*stake* to submit data — the stake is your skin in the game against a challenge
market, and it's paid in coins you got by mining.

### Mine (train the model, earn rewards)

Run the node in producing mode and attach the PyTorch trainer over the local
bridge. The trainer pulls head weights, trains on your corpus, and hands back
compressed deltas the node gossips:

```bash
target/release/sestrian-node ... --produce --bridge-port 7999
python -m client.miner_bridge --node-port 7999 --model small-moe --data corpus.txt --device cuda
```

A better-scoring delta earns more of the block reward. The proposer lottery is
stake-weighted VRF sortition, so more stake means more blocks you get to propose.

### Give data (stake to submit, earn the data share)

Training data stays off your machine's business and off the chain — only its
`sha256`, size, media type, owner, and stake bond go on-chain. Submit from a
funded wallet:

```bash
python -m client.wallet submit-data --file corpus.txt --stake 10 --media-type text
```

That locks a `10`-coin bond and registers you as the owner. From then on, every
block whose training touched your data pays you a slice of the data share. If
someone thinks your data is invalid or not yours, they open a challenge; a
disinterested-juror quorum votes, and a loser is slashed. Watch the registry with
`python -m client.wallet registry`.

> Prefer to push raw bytes through the node instead of just the hash? `POST
> /upload` takes the file, stakes from the node's own wallet, and stores it in the
> content-addressed DA layer. It requires `SESTRIAN_API_TOKEN` (Bearer) and a
> funded node wallet — see [docs/joining.md](docs/joining.md).

### Serve the API (answer inference, earn fees)

Every node already serves an HTTP API and a dashboard on `--api-port` (default
`8090`):

```
GET  /            dashboard (blocks, loss, chat)     GET  /status   height, peers, supply
GET  /balance     ?addr=…                            GET  /chain    recent blocks
GET  /metrics     Prometheus                          GET  /data/registry
POST /inference   signed, fee-bearing forward-prop   POST /transfer  move coins
POST /data/submit staked data                        POST /chat     (token-gated) talk to the head
```

To sell inference, run a serve-only bridge; callers pay per request with a signed
receipt that settles payer → your wallet on-chain:

```bash
python -m client.miner_bridge --node-port 7999 --model small-moe --serve-only
# callers: POST http://<you>:8090/inference  { prompt, fee, signature } → answer + on-chain receipt
```

Keep mutating endpoints (`/chat`, `/upload`) behind `SESTRIAN_API_TOKEN`, and
don't expose them to the open internet unauthenticated. See the
[threat model](docs/internal/threat-model.md).

## The map

| Path | What it is |
|---|---|
| **[WHITEPAPER.md](WHITEPAPER.md)** | the master design document (§1–12) — invariants: bytes-only interface, RoPE positions, from-scratch genesis, fair launch |
| **[docs/joining.md](docs/joining.md)** | the tester's join guide — the three public things you need and the exact run command |
| **client/** | the Python client: real PyTorch GPT trained *through the chain*, the wallet CLI, the `watch.py` web UI, DiPaCo sharding, content-addressed storage |
| **rig/** | the reference implementation — consensus, token ledger + data lane, DA, beacon, economics; the SPEC the Rust node must match |
| **node/** | the Rust node: `sestrian-core` (bit-exact consensus, pinned to the reference by golden vectors) + `sestrian-node` (libp2p GossipSub/QUIC networking) |
| **docs/** | including **[genesis-ceremony.md](docs/genesis-ceremony.md)** — how the real network launches, and **[production-readiness.md](docs/production-readiness.md)** — the go/no-go tracker |
| **deploy/** | the bootstrap seed node (Kubernetes) |

## Status — honest map

This is an active Phase-0/1 project, and the README above describes what the
shipping node actually does. What is *enforced in code today* versus *designed /
testnet-phase* is tracked precisely in
**[docs/production-readiness.md](docs/production-readiness.md)** and
**[docs/internal/threat-model.md](docs/internal/threat-model.md)**.

Enforced and pinned to the reference by golden vectors: all consensus-safety and
runtime hardening, verifiable VRF proposer sortition + non-forgeable work,
Byzantine-robust aggregation, erasure-coded multi-node data availability, the
stake-bond admission cost, and fee-bearing inference. Still testnet-phase (they
need live compute + a running network): delta loss-scoring, cross-inclusion, and
the threshold-BLS beacon.

The network is **open** — permissionless, like Bitcoin: anyone joins from one
peer address plus the published genesis id. Launch is deliberately phased —
a small, monitored, low-value devnet → testnet → open mainnet — because until
delta scoring is enforced on-chain, the aggregation defense assumes an honest
majority. Run it with people you can watch and treat early rewards as testnet
play. Reproduce the guarantees yourself:

```bash
uv run --with torch --with numpy --with pynacl --with py_ecc --with pytest \
    python -m pytest tests/ -q          # Python reference
cd node && cargo test                   # Rust node vs golden vectors
scripts/devnet.sh 30                    # three nodes gossip, validate, converge
```

## License

MIT — see [LICENSE](LICENSE).
