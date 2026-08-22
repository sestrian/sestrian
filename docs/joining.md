# Join the Sestrian devnet

One command gets you a running node. Two get you mining.

```bash
git clone https://github.com/sestrian/sestrian && cd sestrian
scripts/install.sh            # watch + sync
scripts/install.sh --mine     # train and earn (needs a GPU)
scripts/install.sh --service  # ...and survive reboots
```

The installer builds the node, creates your wallet, reproduces the genesis and
verifies it against the network, then runs a **preflight** that refuses to leave
you in a state where you'd silently earn nothing. It is safe to re-run.

> **Expect the first run to take a while.** Compiling the Rust node is ~2 min,
> and reproducing the 85.4M-parameter genesis is ~2–3 min of CPU. Syncing the
> chain then takes longer than you'd guess: every block carries a multi-megabyte
> training delta, so catching up is bandwidth-bound, not CPU-bound.

## What you need

- **Rust** and **Python 3.11+** — the installer offers to install Rust; for
  Python it uses [uv](https://astral.sh/uv) if present.
- **To mine:** a GPU with PyTorch support — NVIDIA (CUDA) or Apple silicon (MPS).
  A laptop GPU is fine; the trainer measures your speed and sizes each round to
  fit the block interval.
- **To just watch/serve:** no GPU at all.

### Skip the genesis build

The genesis is deterministic, so you can download a prebuilt copy (190MB) instead
of spending CPU on it. Convenience, **not** trust: the node verifies whatever you
give it against the state root compiled into the binary, so a tampered file fails
at startup rather than silently forking you.

```bash
SESTRIAN_GENESIS_URL=https://github.com/sestrian/sestrian/releases/download/devnet-genesis-1/genesis.bin.zst \
SESTRIAN_GENESIS_SHA256=6987fb34ebf654655cebbd1d0133f3d70d7f470d7a279dcf2b9f498a27468978 \
  scripts/install.sh
```

That removes Python and torch from the requirements entirely — you need only
Rust to build the node, or the container below to skip that too.

### Or run the container (no toolchain at all)

```bash
curl -fL -o genesis.bin.zst https://github.com/sestrian/sestrian/releases/download/devnet-genesis-1/genesis.bin.zst
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
| bootstrap peer | `/ip4/169.58.211.248/tcp/9800` |
| genesis id (state root) | `30ea20da27f1da0c94512d50a6291370a63a426b77dc425b9826ca17bd213c28` |
| model | 85.4M-param GPT, from scratch (`--model small --seed 1337`) |
| genesis-ledger contributor | `3432d48fd6878b4f2e7a1e40cc15e112c512fae7` |
| block interval | 180s |
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
    --node-port 7999 --model small --serve-only
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
