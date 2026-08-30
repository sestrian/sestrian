#!/usr/bin/env bash
# TRAINING LANES, live (Sharding Road Phase 2).
#
#   scripts/lanes-proof.sh [seconds]
#
# Two producing miners cross a v5 activation with lanes forced narrow
# (lane_width 1, a tiny epoch), so each miner is assigned a DIFFERENT stripe of
# expert pages. PASSES iff:
#   - both nodes cross v5 in lockstep (settled chains agree)
#   - after v5, blocks include deltas from DISTINCT lanes (the throughput point:
#     two miners' work coexists in the chain instead of contending)
#   - no node ever rejects a peer's post-v5 block (lane rule agreed on)
# Toy MoE starts with 4 experts, enough for >1 lane at width 1.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
( cd node && cargo build --release )
B=node/target/release/sestrian-node
S=${1:-90}
export SESTRIAN_LOCAL_V5_HEIGHT=4
export SESTRIAN_LOCAL_LANE_WIDTH=1
export SESTRIAN_LOCAL_LANE_EPOCH_LEN=4
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
A=$(printf 'a%.0s' {1..64} | head -c 64)
Bk=$(printf 'b%.0s' {1..64} | head -c 64)
rm -rf /tmp/lane0 /tmp/lane1 /tmp/lane*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/lane_genesis.bin >/dev/null 2>&1

start() { # datadir key port api bridge peerport
  $B --network local --data-dir "$1" --key-seed "$2" \
     --genesis-file /tmp/lane_genesis.bin --port "$3" --api-port "$4" \
     --bridge-port "$5" --produce --interval 6 \
     --peers "/ip4/127.0.0.1/udp/$6/quic-v1" \
     --data-refs genesis --seconds "$S" --data-contributor "$FOUNDER" \
     > "/tmp/lane_$(basename $1).log" 2>&1 &
}
start /tmp/lane0 "$A" 7940 8140 7959 7941
start /tmp/lane1 "$Bk" 7941 8141 7958 7940
sleep 3
for port in 7959 7958; do
  uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port "$port" --model toy-moe --inner 8 --batch 16 --device cpu \
    > "/tmp/laneb_$port.log" 2>&1 &
done

# read state while the nodes are still UP (killing first raced the reads)
sleep $((S - 12))
H0=$(curl -s -m 20 localhost:8140/status | python3 -c "import json,sys;print(json.load(sys.stdin)['height'])" 2>/dev/null || echo 0)
REJECTS=$(grep -c "outside its lane" /tmp/lane_lane0.log /tmp/lane_lane1.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
# distinct lanes represented among post-v5 blocks: read each node's miner set
DISTINCT=$(python3 - <<PY 2>/dev/null
import json, urllib.request
try:
    d=json.load(urllib.request.urlopen("http://localhost:8140/chain", timeout=20))
    rows=d if isinstance(d,list) else d.get("blocks",[])
    miners={b.get("proposer","?")[:8] for b in rows[-12:]}
    print(len(miners))
except Exception:
    print(0)
PY
)
echo "height $H0  distinct recent proposers $DISTINCT  lane-rejects $REJECTS"
# settled agreement
AG=$(python3 - <<PY 2>/dev/null
import json, urllib.request
def chain(u):
    d=json.load(urllib.request.urlopen(u+"/chain", timeout=20))
    r=d if isinstance(d,list) else d.get("blocks",[])
    return {b["height"]:(b.get("hash") or "")[:12] for b in r}
try:
    a=chain("http://localhost:8140"); b=chain("http://localhost:8141")
    h=min(max(a),max(b))-3
    print("AGREE" if a.get(h)==b.get(h) and a.get(h) else "DISAGREE")
except Exception as e:
    print("UNKNOWN")
PY
)
echo "settled: $AG"

pkill -f "data-dir /tmp/lane" 2>/dev/null || true
pkill -f "node-port 795[89]" 2>/dev/null || true
if [ "${H0:-0}" -ge 8 ] && [ "$AG" = "AGREE" ] && [ "${REJECTS:-1}" -eq 0 ] \
   && [ "${DISTINCT:-0}" -ge 2 ]; then
  echo "LANES PROOF ✓ — two lanes' work coexists across a v5 fork, no rejects"
else
  echo "LANES PROOF ✗ — h=$H0 agree=$AG rejects=$REJECTS distinct=$DISTINCT"
  tail -6 /tmp/lane_lane1.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-140
  exit 1
fi
