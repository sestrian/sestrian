# Running Sestrian across machines

`rig/lan.py` runs a coordinator and miners on **different physical machines**
over TCP (the same length-prefixed protocol the local node uses). Rounds are
synchronous — each block the coordinator ships weights + a beacon-assigned shard
to every miner and waits for all deltas — so consensus stays reproducible even
though the miners now have different clocks and hardware.

## Verified result (Mac coordinator + the GPU server miners, over Tailscale)

A 4-miner run — 2 miners local on the Mac, 2 on the GPU server (32-core Linux,
reached over Tailscale at `<tailnet-ip>`) — trained a `TinyTransformer` to
0.999 accuracy in 20 blocks and replayed bit-exact. The decisive check:

```
cross-machine head: 35669e352ea33f95
all-local    head: 35669e352ea33f95   ✓ identical
```

The chain produced by miners spread across two machines is **byte-for-byte
identical** to the all-local chain with the same seed. Determinism holds across
the network: the deterministic fixed-point aggregation and beacon-driven shard
assignment mean *where* a miner runs never changes the state.

## Workflow

Both machines need the repo and numpy. On a fresh miner box:

```bash
rsync -az --exclude '.git' --exclude '__pycache__' ./ <host>:~/sestrian/
ssh <host> 'cd ~/sestrian && python3 -m venv .venv && .venv/bin/pip install numpy'
```

On the coordinator machine (binds 0.0.0.0 so remote miners can dial in):

```bash
python3 -m rig.lan coordinator --port 9000 --miners 4 --blocks 20
```

On each miner machine (point `--host` at the coordinator's IP):

```bash
python3 -m rig.lan miner --host <coordinator-ip> --port 9000 --id 2
```

Miner ids must be distinct and cover `0..miners-1`; the coordinator waits until
all have connected before producing blocks.

## What this is and isn't

It **is** real cross-machine distributed training with reproducible consensus —
the first step off a single box. It is **not** yet a real network: the
coordinator is a single trusted party, peers are hand-assigned rather than
discovered, there is no gossip, no real block propagation, no staking or
slashing on the wire, and no data-availability layer. Those are the Bitcoin-
inspired hard parts that come next (WHITEPAPER §3, §5, §7; task: distributed
systems).
