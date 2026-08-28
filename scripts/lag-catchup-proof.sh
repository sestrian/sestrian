#!/usr/bin/env bash
# THE WEDGE REPRODUCTION: a node that falls further behind than the body
# retention window must still catch up.
#
#   scripts/lag-catchup-proof.sh [mine_seconds]
#
# This is exactly how the EU anchor died twice on 2026-08-27/28: it fell more
# than BODY_WINDOW blocks behind (one OOM restart was enough), whole bodies
# for the gap existed nowhere (pruned to per-node shard zones), the catch-up
# state machine counted parked blocks as progress, and sync re-shipped bodies
# it already had at one block per round-trip — so it recovered slower than the
# chain grew, forever.
#
# Here node0 is killed while node1 mines PAST a deliberately tiny body window
# (SESTRIAN_BODY_WINDOW=4, transport knob), so node0's gap is mostly
# shard-only history — then node0 must converge to node1's tip anyway, via
# headers-first sync + the shard pump. PASSES only if node0 reaches node1's
# settled chain.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
( cd node && cargo build --release )
B=node/target/release/sestrian-node
MINE=${1:-70}
export SESTRIAN_BODY_WINDOW=4
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
A=$(printf 'a%.0s' {1..64} | head -c 64)
Bk=$(printf 'b%.0s' {1..64} | head -c 64)
rm -rf /tmp/lag0 /tmp/lag1 /tmp/lag*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/lag_genesis.bin >/dev/null 2>&1

start_n0() {
  $B --network local --data-dir /tmp/lag0 --key-seed "$A" \
     --genesis-file /tmp/lag_genesis.bin --port 7920 --api-port 8120 \
     --bridge-port 7979 --produce --interval 5 \
     --peers /ip4/127.0.0.1/udp/7921/quic-v1 \
     --data-refs genesis --seconds "$1" --data-contributor "$FOUNDER" \
     >> /tmp/lag0.log 2>&1 &
  echo $!
}
: > /tmp/lag0.log; : > /tmp/lag1.log
N0=$(start_n0 25)
$B --network local --data-dir /tmp/lag1 --key-seed "$Bk" \
   --genesis-file /tmp/lag_genesis.bin --port 7921 --api-port 8121 \
   --bridge-port 7978 --produce --interval 5 \
   --data-refs genesis --seconds $((MINE + 90)) \
   --peers /ip4/127.0.0.1/udp/7920/quic-v1 \
   --data-contributor "$FOUNDER" >> /tmp/lag1.log 2>&1 &
N1=$!
sleep 3
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7979 --model toy-moe --inner 8 --batch 16 --device cpu \
    > /tmp/lagb0.log 2>&1 &
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7978 --model toy-moe --inner 8 --batch 16 --device cpu \
    > /tmp/lagb1.log 2>&1 &

# node0 dies early; node1 mines ALONE far past the 4-block body window
sleep 20
echo ">>> killing node0; node1 mines alone for ${MINE}s (window is 4 blocks)"
kill -9 "$N0" 2>/dev/null || true
sleep "$MINE"

H1=$(curl -s -m 5 localhost:8121/status | python3 -c \
    "import json,sys;print(json.load(sys.stdin)['height'])" 2>/dev/null || echo 0)
echo ">>> node1 at h$H1; restarting node0 — it must cross a shard-only gap"
N0=$(start_n0 80)

# node0 must reach node1's tip (minus a settling margin) within 75s
DEADLINE=$((SECONDS + 75)); OK=0
while [ $SECONDS -lt $DEADLINE ]; do
  H0=$(curl -s -m 5 localhost:8120/status | python3 -c \
      "import json,sys;print(json.load(sys.stdin)['height'])" 2>/dev/null || echo 0)
  HT=$(curl -s -m 5 localhost:8121/status | python3 -c \
      "import json,sys;print(json.load(sys.stdin)['height'])" 2>/dev/null || echo "$H1")
  if [ "${H0:-0}" -ge $((HT - 2)) ] && [ "${H0:-0}" -gt "$H1" ] 2>/dev/null; then
    OK=1; break
  fi
  sleep 4
done
pkill -f "data-dir /tmp/lag" 2>/dev/null || true
pkill -f "node-port 797[89]" 2>/dev/null || true
echo "node0 h${H0:-?} vs node1 h${HT:-?} (was behind since h<${H1})"
if [ "$OK" = 1 ]; then
  echo "LAG-CATCHUP PROOF ✓ — recovered across a beyond-window gap"
else
  echo "LAG-CATCHUP PROOF ✗ — still behind (the wedge)"
  tail -5 /tmp/lag0.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-140
  exit 1
fi
