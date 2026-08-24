#!/bin/bash
# N-miner local swarm: N producing nodes + N CPU trainers on toy-moe.
# The multi-miner testbed the 2-founder devnet cannot be — exercises inclusion
# pressure, score competition, claim contention and (v4) committee mechanics.
#
#   scripts/swarm.sh [N] [seconds]         # default 10 miners, 240s
#   SESTRIAN_LOCAL_V3_HEIGHT=6 scripts/swarm.sh 10 240   # cross the upgrade too
#
# Asserts: every node converged on the settled prefix, blocks were produced,
# and (diagnostics) prints per-height tx counts + score distribution so the
# inclusion/score dynamics at N miners are visible, not guessed.
set -e
cd "$(dirname "$0")/.."
( cd node && cargo build --release )
B=node/target/release/sestrian-node
N=${1:-10}
S=${2:-240}
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
BASE_PORT=7700
BASE_API=8500
BASE_BRIDGE=7600

rm -rf /tmp/swarm[0-9]* /tmp/swarm_*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/swarm_genesis.bin

for i in $(seq 0 $((N - 1))); do
    SEED=$(printf "%064x" $((i + 1)))
    PEERS=""
    if [ "$i" -gt 0 ]; then
        PEERS="--peers /ip4/127.0.0.1/udp/$BASE_PORT/quic-v1"
    fi
    $B --network local --data-dir /tmp/swarm$i --key-seed "$SEED" \
       --genesis-file /tmp/swarm_genesis.bin \
       --port $((BASE_PORT + i)) --api-port $((BASE_API + i)) \
       --bridge-port $((BASE_BRIDGE + i)) --produce --data-refs genesis \
       --interval 10 --seconds $S $PEERS \
       --data-contributor $FOUNDER > /tmp/swarm_n$i.log 2>&1 &
done
sleep 4
for i in $(seq 0 $((N - 1))); do
    uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
        --node-port $((BASE_BRIDGE + i)) --model toy-moe --inner 6 --batch 8 \
        --device cpu > /tmp/swarm_b$i.log 2>&1 &
done

# wait for the node processes only (trainers are killed after)
for i in $(seq 1 $N); do wait %$i || true; done
pkill -f "client.miner_bridge --node-port 76" 2>/dev/null || true

# ---- convergence: every lineage agrees on the settled prefix ----
REF=""
FAIL=0
for i in $(seq 0 $((N - 1))); do
    L=$(grep LINEAGE /tmp/swarm_n$i.log | sed 's/.*LINEAGE[: ]*//' | tr -d '[:space:]')
    if [ -z "$L" ]; then echo "node$i: EMPTY LINEAGE ✗"; FAIL=1; continue; fi
    if [ -z "$REF" ]; then REF=$L; fi
    python3 - "$REF" "$L" "$i" <<'PY' || FAIL=1
import sys
a, b, i = sys.argv[1].split(">"), sys.argv[2].split(">"), sys.argv[3]
n = min(len(a), len(b)); settle = max(0, n - 3)
if n < 4: print(f"node{i}: TOO SHORT ✗ ({n})"); sys.exit(1)
if a[:settle] != b[:settle]:
    print(f"node{i}: DIVERGED ✗"); sys.exit(1)
print(f"node{i}: ok ({settle} settled, tip {len(b)})")
PY
done

# ---- diagnostics: inclusion + score dynamics at N miners ----
python3 - "$N" <<'PY'
import json, sys, collections
N = int(sys.argv[1])
heights = collections.OrderedDict()
miners = collections.Counter()
scores = []
try:
    with open("/tmp/swarm0/blocks.jsonl") as f:
        for ln in f:
            b = json.loads(ln)
            h = b["header"]["height"]
            heights[h] = len(b.get("txs", []))
            for t in b.get("txs", []):
                miners[t["miner"][:8]] += 1
            scores.extend(b.get("scores", {}).values())
except FileNotFoundError:
    sys.exit(0)
inc = list(heights.values())
zero = sum(1 for s in scores if s == 0)
print(f"SWARM DIAGNOSTICS: {N} miners, {len(heights)} blocks, "
      f"deltas/block avg {sum(inc)/max(1,len(inc)):.1f} (cap 8), "
      f"{len(miners)} distinct miners included, "
      f"scores {zero}/{len(scores)} zero")
PY

if [ "$FAIL" = "0" ]; then echo "SWARM CONVERGED ✓ ($N miners)"; else
    echo "SWARM FAILED ✗"; exit 1; fi
