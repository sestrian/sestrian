# Running a Sestrian Node

The production node is Rust (`node/`, `sestrian-node`); training is a
PyTorch plugin that attaches locally. Consensus and networking never depend on
Python; training never touches consensus; the two meet only at the compressed,
signed delta (the consensus boundary, WHITEPAPER §6.3).

## Build

```bash
cd node && cargo build --release      # single binary: node/target/release/sestrian-node
```

## Identity

Your wallet is your miner identity; rewards mint to its address.

```bash
python -m client.wallet new           # encrypted file + 24-word mnemonic + pal1… address
sestrian-node --wallet ~/.sestrian/wallet.json …   # encrypted: set
export SESTRIAN_WALLET_PASSPHRASE=…                  # (argon2id + XSalsa20-Poly1305)
```

Infra nodes (seeds/relays) that never earn can use `--key-seed <32-byte hex>`.

## Genesis

Every node must load the network's published genesis artifact:

```bash
python -m client.make_genesis --model small-moe --seed <published> --out genesis.bin \
    --expect <published genesis id>
# verify the printed genesis_state_root against the ceremony publication
sestrian-node --genesis-file genesis.bin …
```

Once loaded it persists in the data dir; the flag is only needed on first run.

## A full mining node

```bash
# terminal 1: the node (consensus + networking + API):
sestrian-node \
  --data-dir ~/.sestrian/node \
  --wallet ~/.sestrian/wallet.json \
  --genesis-file genesis.bin \
  --port 7900 --api-port 8090 --bridge-port 7999 \
  --produce --interval 60 \
  --peers /ip4/<seed-ip>/udp/7900/quic-v1 \
  --data-contributor <published-founder-address>

# terminal 2: the trainer (your GPU; any device torch supports):
python -m client.miner_bridge --node-port 7999 --model small-moe \
    --data <corpus.txt> --inner 300 --batch 32 --device cuda
```

The node hands the trainer the head state once, then keeps it synced with
sparse per-block diffs. Each round the trainer returns a compressed quantized
delta; the node signs, gossips, and (when it proposes) settles it. Watch it:
`curl localhost:8090/status`, or point the wallet CLI at `--node
http://localhost:8090` for balances, transfers, and data-lane actions.

## An observer / API node

Omit `--produce` (and skip the bridge). The node follows the chain, serves
sync to peers, and answers the API. This is what powers explorers and wallets.

## A seed / relay node

```bash
sestrian-node --data-dir /var/sestrian --key-seed <hex> \
  --genesis-file genesis.bin --port 7900 --api-port 8090 \
  --relay-server --external-address /ip4/<public-ip>/udp/7900/quic-v1
```

`--relay-server` enables circuit-relay v2: peers behind hostile NATs reach the
network through you, and DCUtR upgrades them to direct connections when
hole-punching succeeds. Seeds should have a reachable address (public IP or a
port-forward) and be listed in the published bootstrap set.

## NAT: what to expect

The node ships AutoNAT (detects whether you're reachable), DCUtR (QUIC hole
punching), and relay-client (fallback through seeds). Home-router operators
need no configuration: dial a seed and the stack negotiates the rest. If you
*can* forward `--port` (UDP+TCP), do; direct connectivity helps the mesh.

## Persistence & recovery

Everything lives in `--data-dir`: genesis, an append-only block log, the
compressed delta payloads (the DA bodies), and periodic head-state snapshots.
On restart the node **replays its chain with full validation**: a corrupt or
truncated store degrades safely to the last valid block. Deleting the data dir
means re-syncing from peers.

## The devnet (development)

`scripts/devnet.sh [seconds]`: two nodes + two PyTorch trainers on localhost,
asserts byte-identical convergence at exit. Golden vectors
(`cd node && cargo test`) pin the consensus math to the Python reference.

## Live bootstrap peers

| Seed | Multiaddr | API |
|---|---|---|
| contabo-eu-1 (public, relay) | `/ip4/169.58.211.248/udp/9800/quic-v1` | `http://169.58.211.248:8080/status` |
| contabo-us-1 (public, relay) | `/ip4/13.140.32.27/udp/9800/quic-v1` | `http://13.140.32.27:8080/status` |
| cluster (private net) | `/ip4/10.0.1.1/udp/30980/quic-v1` | `http://10.0.1.1:30981/status` |

## Disk: pruned vs archive nodes

Delta bodies are erasure-coded into shards so peers can fetch them; kept
forever that is LINEAR disk growth. Home/miner nodes should run
`--da-retain-blocks 1500` (shard sets for deeper blocks are deleted; the node
still serves catch-up inside the window). Public anchors run the default
`0` = archive: they keep everything so a fresh joiner can always replay from
genesis. Do not prune an anchor.

## Production hardening

**Two+ anchors (no single point of bootstrap/DA).** Run at least two seeds on
separate hosts/regions, each with `--relay-server` and its own persistent
volume, and list *both* in every node's `--peers`. The StatefulSet
(`deploy/seed-node.yaml`) gives each replica its own PVC, so `replicas: 2`
yields two independent anchors; add the second to the bootstrap table above. DA
bodies should be retained on more than one anchor so losing one never loses a
body.

**API auth + exposure.** Mutating endpoints (`/upload`, `/chat`) require
`SESTRIAN_API_TOKEN` (a `Bearer` token) and are *disabled* if it is unset.
Read + signed-tx endpoints are safe to expose. Restrict the bind interface with
`--api-bind` where a public dashboard isn't wanted.

**TLS in transit.** The node serves plain HTTP; signed transactions don't need
TLS for authentication, but for confidentiality/integrity put a TLS-terminating
reverse proxy in front of any public API and point the wallet at `https://`.
Minimal Caddy:

    api.example.com {
        reverse_proxy 127.0.0.1:8080
    }

Run the node with `--api-bind 127.0.0.1` so only the proxy reaches it.

**Monitoring.** Scrape `GET /metrics` (Prometheus text). Load
`deploy/monitoring/alerts.yml` for NodeDown / ChainStalled / pool-near-cap /
sync-lag alerts.

**Backup.** `deploy/backup-restore.sh backup <out>` takes a consistent snapshot;
a restored node fast-boots.
