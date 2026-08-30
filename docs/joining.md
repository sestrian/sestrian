# Join the Sestrian devnet

```bash
npx sestrian run
```

That is the whole thing. It downloads a prebuilt node for your platform, creates
your wallet, fetches the genesis weights and joins the network. Every download is hash-checked before it runs, and the node
re-derives the chain's genesis id from the weights at startup and refuses to
start on a mismatch. Then:

```bash
npx sestrian status     # height, peers, whether you are earning
npx sestrian check      # preflight: can this machine contribute?
```

The first run takes a while; most of it is syncing blocks.

`npx sestrian run` watches and serves. Mining needs a GPU and the trainer:

## Mining

```bash
git clone https://github.com/sestrian/sestrian && cd sestrian
scripts/install.sh --mine     # train and earn (needs a GPU)
scripts/install.sh --service  # ...and survive reboots
```

The installer builds the node, creates your wallet, verifies the genesis and
checks your setup can actually earn before leaving you running. Safe to re-run.

### What you need

- **To mine:** a GPU with PyTorch support (NVIDIA CUDA or Apple silicon MPS;
  a laptop GPU is fine), plus Rust and Python 3.11+. The installer offers to
  install both.
- **To watch/serve:** nothing beyond `npx sestrian run`.

### Skip the genesis build

The installer rebuilds the genesis from scratch (~2–3 min). To download it
instead:

```bash
SESTRIAN_GENESIS_URL=https://github.com/sestrian/sestrian/releases/download/devnet-genesis-3/genesis.bin.zst \
SESTRIAN_GENESIS_SHA256=<sha256 from the release manifest> \
  scripts/install.sh
```

The node verifies whatever arrives against the genesis id compiled into the
binary, so a tampered file fails at startup.

### Docker

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

That image watches, syncs and serves. To mine in Docker (Linux + NVIDIA only;
on a Mac run `scripts/install.sh --mine` natively for GPU access):

```bash
docker run --rm -it --gpus all \
  -v "$PWD/genesis.bin:/genesis/genesis.bin:ro" \
  -v sestrian-data:/data \
  -p 8090:8090 \
  -e SESTRIAN_KEY_SEED=$(head -c32 /dev/urandom | xxd -p -c64) \
  ghcr.io/sestrian/sestrian-miner
```

The published miner image is the CPU build. For NVIDIA, build the CUDA flavour
once:

```bash
docker build -f deploy/Dockerfile.miner --build-arg TORCH_FLAVOR=cu121 \
    -t sestrian-miner:cuda .
```

Or with compose:

```bash
echo "SESTRIAN_KEY_SEED=$(head -c32 /dev/urandom | xxd -p -c64)" > .env   # SAVE THIS
docker compose -f deploy/docker-compose.yml up node             # watch/serve
docker compose -f deploy/docker-compose.yml --profile miner up miner
```

## Network values

The network identity is compiled into the binary; `--network devnet` is the
default, and a flag contradicting it is a startup error, not a silent fork.

| | value |
|---|---|
| network | `devnet` |
| bootstrap peers | `/ip4/169.58.211.248/udp/9800/quic-v1` (EU) · `/ip4/13.140.32.27/udp/9800/quic-v1` (US); either alone is enough |
| genesis id (state root) | `91bdcc281c0dbbd7b3bea3d38003e4c61565bcaa5fd8e7bfca296e6a4994ddb1` |
| model | 107.4M-param growable MoE (~32M active/token), defined by the genesis |
| genesis-ledger contributor | `3432d48fd6878b4f2e7a1e40cc15e112c512fae7` |
| block interval | 180s |
| protocol version | 2 (training updates capped at ~8MB each) |
| public API | http://169.58.211.248:8080/status (EU) · http://13.140.32.27:8080/status (US) — compare both; an anchor that is resyncing answers correctly but reports a lower height |

Running your own chain instead: `--network local`, and supply everything yourself.

## Is it working?

```bash
curl -s localhost:8090/status
```

| field | what you want |
|---|---|
| `height` | climbing, and matching the [public API](http://169.58.211.248:8080/status) or [the US anchor](http://13.140.32.27:8080/status) — whichever is HIGHER is the chain |
| `peers` | at least 1 |
| `model_attached` | `true` if you're mining |
| **`stale_deltas`** | **0** |

A delta only counts at the current head, so training rounds that finish late are
dropped: non-zero, climbing `stale_deltas` means you're mining without earning.
The trainer auto-fits its round to prevent this; if it still happens, lower
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
    --node-port 7999 --model small-moe
```

Your GPU is detected automatically (`--device cuda|mps|cpu` to force it). No
`--data` needed to start: the trainer fetches a public-domain corpus on first
run. Every delta names the staked corpus it trained on; `genesis` is the
always-staked founding corpus. Blocks pay in proportion to the held-out loss
improvement your delta achieved.

**📚 Supply data.** Stake coins behind a corpus you have the rights to, and earn
the data share of every block trained on it plus a cut of inference fees when
the model uses it:

```bash
uv run --with numpy --with pynacl python -m client.wallet submit-data \
    --file corpus.txt --stake 5
```

It prints the `--data-refs` value to mine with afterwards. You need coins
first: mine for a while with `--data-refs genesis`, then stake.

**🔌 Serve.** Run a serve-only bridge and answer paid inference:

```bash
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7999 --model small-moe --serve-only
```

Callers `POST /inference` with a signed, fee-bearing receipt that settles to
your wallet on-chain.

**🧩 Validate a slice (paged validator).** When the model grows past one
machine's RAM, you don't have to hold all of it. Run a **paged validator** that
keeps only the backbone + a few expert pages and trusts the rest, backstopped
by fraud proofs:

```bash
sestrian-node --held-pages 3,7,12 --peers <anchor> --data-refs genesis
```

It validates the whole chain holding a fraction of the model, recomputing its
own pages (it cannot be fooled there) and trusting the committed leaf for
others. Post a **custody bond** to commit to serving your pages and earn for
holding them:

```bash
uv run --with numpy --with pynacl python -m client.wallet stake-custody \
    --pages 3,7,12 --stake 5
```

A holder that stops serving its pages is challenged and slashed — the same
availability machinery that polices data withholding. (Today the model still
fits one machine, so this is optional; it is how the network scales past that.)

## Node roles at a glance

| Flag | Role | Holds |
|------|------|-------|
| `--produce` + trainer | miner | full model |
| (none) | full validator | full model |
| `--held-pages P,Q` | paged validator | backbone + P,Q |
| `--relay-server` | anchor/relay | full model + deep history |

Lanes: once training lanes activate (v5), the beacon assigns each miner an
expert-page lane per epoch; your trainer trains and claims only that lane, so a
thousand miners no longer contend for a handful of block seats. Nothing to
configure — assignment is a deterministic function of your key and the height.

## Back up your wallet

`~/.sestrian/wallet.json` **is** your identity and your balance. There is no
recovery service. Copy it somewhere safe.

## Status

This is an open devnet: coins are for testing, not trading, and the chain can
still reset. What is enforced today and what waits on the testnet phase is
tracked in [production-readiness.md](production-readiness.md).
