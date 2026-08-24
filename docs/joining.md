# Join the Sestrian devnet

```bash
SESTRIAN_GENESIS_TAG=devnet-genesis-3 npx sestrian run
```

That is the whole thing. It downloads a prebuilt node for your platform, creates
your wallet, fetches the 860MB genesis weights and joins the network. No clone,
no compiler, no Python. (The `SESTRIAN_GENESIS_TAG` override is temporary: the
published npm package predates devnet-genesis-3 and defaults to the retired
genesis — the env var points it at the current one, and the node verifies
whatever arrives against the id compiled into the binary either way. It
disappears once the 0.5.0 package is published.) Then:

```bash
npx sestrian status     # height, peers, whether you are actually earning
npx sestrian check      # preflight: can this machine contribute?
```

Everything it downloads is checked against a published sha256 before it runs,
and the genesis is checked twice: once as an archive, and again as raw weights
whose hash **is** this chain's genesis id. A tampered mirror fails loudly
instead of quietly putting you on a different chain.

> **The first run takes a while, and most of it is sync.** The downloads are a
> few minutes; catching up to the head is longer, because every block carries a
> multi-megabyte training delta — it is bandwidth-bound, not CPU-bound.

`npx sestrian run` **watches and serves**; it does not mine. Training needs
PyTorch and a GPU, which is what the installer below sets up.

## Mining

Mining means training the model and earning for the improvement you contribute,
so it needs a GPU and the Python trainer:

```bash
git clone https://github.com/sestrian/sestrian && cd sestrian
scripts/install.sh --mine     # train and earn (needs a GPU)
scripts/install.sh --service  # ...and survive reboots
```

The installer builds the node, creates your wallet, obtains the genesis and
verifies it against the network, then runs a **preflight** that refuses to leave
you in a state where you'd silently earn nothing. It is safe to re-run.

### What you need

- **To mine:** a GPU with PyTorch support — NVIDIA (CUDA) or Apple silicon (MPS).
  A laptop GPU is fine; the trainer measures your speed and sizes each round to
  fit the block interval. Plus **Rust** and **Python 3.11+** — the installer
  offers to install Rust; for Python it uses [uv](https://astral.sh/uv).
- **To just watch/serve:** nothing beyond `npx sestrian run`.

### Skip the genesis build

The installer reproduces the genesis from scratch (~2–3 min of CPU, plus a torch
install). Point it at the published copy instead:

```bash
SESTRIAN_GENESIS_URL=https://github.com/sestrian/sestrian/releases/download/devnet-genesis-3/genesis.bin.zst \
SESTRIAN_GENESIS_SHA256=<zstd sha256 from the devnet-genesis-3 release manifest> \
  scripts/install.sh
```

Convenience, **not** trust: the node verifies whatever you give it against the
state root compiled into the binary, so a tampered file fails at startup rather
than silently forking you. `npx sestrian genesis` performs the same fetch and
verification on its own if you want the file without the installer.

### Or run the container (no toolchain at all)

```bash
curl -fL -o genesis.bin.zst https://github.com/sestrian/sestrian/releases/download/devnet-genesis-3/genesis.bin.zst
zstd -d genesis.bin.zst && mkdir -p sestrian-data

docker run --rm -it \
  -v "$PWD/genesis.bin:/genesis/genesis.bin:ro" \
  -v "$PWD/sestrian-data:/data" \
  -p 8090:8090 \
  -e SESTRIAN_KEY_SEED=$(head -c32 /dev/urandom | xxd -p -c64) \
  ghcr.io/sestrian/sestrian-node \
  --data-dir /data --genesis-file /genesis/genesis.bin --api-bind 0.0.0.0
```

That image carries the node only — **watch, sync and serve**. To mine in Docker,
use the miner image, which bundles the PyTorch trainer:

```bash
docker run --rm -it --gpus all \
  -v "$PWD/genesis.bin:/genesis/genesis.bin:ro" \
  -v sestrian-data:/data \
  -p 8090:8090 \
  -e SESTRIAN_KEY_SEED=$(head -c32 /dev/urandom | xxd -p -c64) \
  ghcr.io/sestrian/sestrian-miner
```

It runs its own preflight, refuses to start if the genesis is missing or the
config can't earn, and **exits if either the node or the trainer dies** rather
than sitting there looking healthy while earning nothing.

> **Docker mining is Linux + NVIDIA only.** Docker Desktop on macOS has no GPU
> passthrough, so the Apple GPU is invisible inside a container and you would
> silently fall back to CPU — the image warns you if that happens. **On a Mac,
> run `scripts/install.sh --mine` natively** to use MPS.
>
> The published image is the **CPU** build (it works everywhere, just slowly).
> For NVIDIA, build the CUDA flavour once — it's ~3GB:
> ```bash
> docker build -f deploy/Dockerfile.miner --build-arg TORCH_FLAVOR=cu121 \
>     -t sestrian-miner:cuda .
> ```

Or with compose, which wires the volumes and ports for you:

```bash
echo "SESTRIAN_KEY_SEED=$(head -c32 /dev/urandom | xxd -p -c64)" > .env   # SAVE THIS
docker compose -f deploy/docker-compose.yml up node             # watch/serve
docker compose -f deploy/docker-compose.yml --profile miner up miner
```

## You do not configure the network

Bootstrap peer, genesis id, the genesis-ledger contributor and the block cadence
are **compiled into the binary** and selected with `--network devnet` (the
default), exactly like Bitcoin's `-testnet`. You cannot typo yourself onto a
chain that will never validate — a flag contradicting the network is a startup
error, not a silent fork.

The current values, so you can verify what your node is using:

| | value |
|---|---|
| network | `devnet` |
| bootstrap peers | `/ip4/169.58.211.248/udp/9800/quic-v1` (EU) · `/ip4/13.140.32.27/udp/9800/quic-v1` (US) — either alone is enough to join |
| genesis id (state root) | `91bdcc281c0dbbd7b3bea3d38003e4c61565bcaa5fd8e7bfca296e6a4994ddb1` — the PAGE-MERKLE root over the model's page table (protocol v2) |
| model | 107.4M-param growable MoE GPT (~32M active/token), from scratch (`--model small-moe --seed 20260824`); the chain can GROW it — see /status `model` |
| genesis-ledger contributor | `3432d48fd6878b4f2e7a1e40cc15e112c512fae7` |
| block interval | 180s |
| protocol version | 2 — the DELTA ENVELOPE: a training update may never exceed 1M nonzero coordinates (~8MB), no matter the quota. Capacity pressure makes miners specialize on fewer experts, and sustained pressure grows the model — bytes per block stay bounded forever |
| public API | http://169.58.211.248:8080/status |

Running your own chain instead: `--network local`, and supply everything yourself.

## Is it working?

```bash
curl -s localhost:8090/status
```

| field | what you want |
|---|---|
| `height` | climbing, and matching the [public API](http://169.58.211.248:8080/status) |
| `peers` | at least 1 |
| `model_attached` | `true` if you're mining |
| **`stale_deltas`** | **0** |

**`stale_deltas` is the one to watch.** A delta can only be included at the
current head, so if your training round finishes after the head moves on, your
work is dropped — you would mine forever and earn nothing. The trainer auto-fits
its step count to the block interval to prevent this, and the node logs a loud
warning naming the cause if it still happens. Non-zero and climbing means lower
`--inner` on the trainer.

Your balance, any time:

```bash
uv run --with numpy --with pynacl python -m client.wallet balance
```

## The three ways to earn

**⛏ Mine.** `scripts/install.sh --mine`, or add `--produce --data-refs genesis`
to the node and attach the trainer:

```bash
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7999 --model small
```

Your GPU is detected automatically (`--device cuda|mps|cpu` to force it), and no
`--data` is needed — the trainer fetches a public-domain corpus on first run, so
you can start immediately. Point `--data` at your own text once you've staked it.

Every delta must name the staked corpus it trained on — that's provenance, and
it's enforced. `genesis` is the always-staked founding corpus and the right
starting point. Blocks pay ∝ the held-out loss improvement your delta actually
achieved.

**📚 Supply data.** Stake coins behind a corpus you own, and earn the data share
of every block trained on it plus a cut of inference fees when the model actually
leans on it:

```bash
uv run --with numpy --with pynacl python -m client.wallet submit-data \
    --file corpus.txt --stake 5
```

It prints the `--data-refs` value to mine with afterwards, so your own data
starts paying you. You need coins first — mine for a while with `--data-refs
genesis`, then stake.

**🔌 Serve.** Run a serve-only bridge and answer paid inference:

```bash
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7999 --model small-moe --serve-only
```

Callers `POST /inference` with a signed, fee-bearing receipt that settles to your
wallet on-chain.

## Back up your wallet

`~/.sestrian/wallet.json` **is** your identity and your balance. There is no
recovery service. Copy it somewhere safe.

## Honest status

This is an **open devnet**, so treat rewards as testnet play.

Live and enforced: consensus safety, provenance (deltas must name staked,
challengeable data), committed delta scoring, influence-sketch usage royalties,
the tail-emission economics, and erasure-coded data availability — all pinned to
the Python reference by golden vectors.

Still testnet-phase, because they need *independent operators* rather than more
code: the multi-evaluator scoring committee (today the block proposer commits the
scores, bounded by its bond and the challenge market), consensus-level
cross-inclusion challenges, and sketch verification. Details in
[production-readiness.md](production-readiness.md) and
[the threat model](internal/threat-model.md).
