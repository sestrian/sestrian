#!/bin/bash
# Proves PEER EXCHANGE closes a star topology into a mesh.
#
#   scripts/peerx-proof.sh [seconds]
#
# Topology under test — exactly the one the live fleet had on 2026-08-25:
#
#     A (hub)          B and C are each configured with ONLY A.
#    /     \           They have no way to learn about each other except
#   B       C          by asking A who else it is connected to.
#
# Before peer exchange, B and C never linked and could fork with nothing to
# reconcile them (which is what happened to the two miners). PASSES only if B
# ends up connected to 2 peers — i.e. it dialled C, a node it was never
# configured with.
set -e
cd "$(dirname "$0")/.."
( cd node && cargo build --release )
B=node/target/release/sestrian-node
SECS=${1:-90}

rm -rf /tmp/px_a /tmp/px_b /tmp/px_c /tmp/px_*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/px_genesis.bin >/dev/null 2>&1

start() {  # name port api extra...
    local n=$1 port=$2 api=$3; shift 3
    $B --network local --data-dir /tmp/px_$n --key-seed "$(printf '%064x' $port)" \
       --genesis-file /tmp/px_genesis.bin --port "$port" --api-port "$api" \
       --bridge-port $((port + 100)) --interval 10 --seconds "$SECS" "$@" \
       > /tmp/px_$n.log 2>&1 &
}

start a 7810 8710                                            # hub, no peers
sleep 3
start b 7811 8711 --peers /ip4/127.0.0.1/udp/7810/quic-v1    # knows only A
start c 7812 8712 --peers /ip4/127.0.0.1/udp/7810/quic-v1    # knows only A

# give the mesh time to form (peer exchange runs once per round)
sleep $((SECS - 20))

peers_of() { curl -s -m 5 "localhost:$1/status" \
    | python3 -c "import json,sys;print(json.load(sys.stdin).get('peers',-1))" 2>/dev/null || echo -1; }
PB=$(peers_of 8711); PC=$(peers_of 8712)
echo "B peers=$PB  C peers=$PC  (configured with 1 each)"
grep -h "peer exchange: dialing" /tmp/px_b.log /tmp/px_c.log 2>/dev/null | head -3 \
    | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-140

wait 2>/dev/null || true
if [ "${PB:-0}" -ge 2 ] || [ "${PC:-0}" -ge 2 ]; then
    echo "PEERX PROOF ✓ — a node dialled a peer it was never configured with"
else
    echo "PEERX PROOF ✗ — the star never closed into a mesh (B=$PB C=$PC)"
    exit 1
fi
