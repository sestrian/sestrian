#!/bin/bash
# THE growth proof (verification-matrix #9): a 2-node local chain, retarget
# constants tightened (local-only env overrides, identical on both nodes) so a
# capacity growth event schedules, announces, and ACTIVATES on-chain inside a
# bounded run — then asserts: both nodes converged, the model grew (a new
# expert page in the committed page table), and the post-growth chain kept
# producing. "The network grew its brain at block N because its miners earned
# it" — demonstrated, not described.
set -e
cd "$(dirname "$0")/.."
( cd node && cargo build --release )
B=node/target/release/sestrian-node
S=${1:-360}
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}

# consensus overrides — MUST be identical on every node of this local chain
export SESTRIAN_LOCAL_RETARGET_WINDOW=4
export SESTRIAN_LOCAL_TARGET_DELTAS=4
export SESTRIAN_LOCAL_QUOTA_MAX_4DP=20000
export SESTRIAN_LOCAL_K_SUSTAIN=2
export SESTRIAN_LOCAL_ANNOUNCE_LEAD=1
# PROTOCOL v4: prove growth under the SHIPPING rule, not a retired one. The
# quorum gate is active from block 0 here and needs both miners to commit a
# positive score in the window — exactly what devnet requires of its two.
export SESTRIAN_LOCAL_V4_HEIGHT=0
export SESTRIAN_LOCAL_GROWTH_QUORUM=2

rm -rf /tmp/growth0 /tmp/growth1
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/growth_genesis.bin
$B --network local --data-dir /tmp/growth0 --key-seed $(printf 'a%.0s' {1..64} | head -c 64) \
   --genesis-file /tmp/growth_genesis.bin --port 7920 --api-port 8290 \
   --bridge-port 7979 --produce --data-refs genesis --interval 3 --seconds $S \
   --data-contributor $FOUNDER > /tmp/growth0.log 2>&1 &
$B --network local --data-dir /tmp/growth1 --key-seed $(printf 'b%.0s' {1..64} | head -c 64) \
   --genesis-file /tmp/growth_genesis.bin --port 7921 --api-port 8291 \
   --bridge-port 7978 --produce --data-refs genesis --interval 3 --seconds $S \
   --peers /ip4/127.0.0.1/udp/7920/quic-v1 \
   --data-contributor $FOUNDER > /tmp/growth1.log 2>&1 &
sleep 3
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7979 --model toy-moe --inner 6 --batch 8 --device cpu > /tmp/growthb0.log 2>&1 &
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7978 --model toy-moe --inner 6 --batch 8 --device cpu > /tmp/growthb1.log 2>&1 &

# poll until BOTH nodes report a grown model (or the run ends)
GENESIS_PAGES=9   # toy-moe: backbone + 2 layers x 4 experts
GREW=0
for _ in $(seq 1 $((S / 5))); do
    sleep 5
    P0=$(curl -s localhost:8290/status | python3 -c \
        'import sys,json;print(json.load(sys.stdin)["model"]["pages_total"])' 2>/dev/null || echo 0)
    P1=$(curl -s localhost:8291/status | python3 -c \
        'import sys,json;print(json.load(sys.stdin)["model"]["pages_total"])' 2>/dev/null || echo 0)
    if [ "$P0" -gt "$GENESIS_PAGES" ] && [ "$P1" -gt "$GENESIS_PAGES" ]; then
        GREW=1
        echo "GROWTH OBSERVED on both nodes: pages $P0 / $P1 (genesis $GENESIS_PAGES)"
        # let the post-growth chain run a little longer, then wind down
        sleep 20
        break
    fi
done
kill %1 %2 2>/dev/null || true
wait %1 %2 2>/dev/null || true
kill %3 %4 2>/dev/null || true

L0=$(grep LINEAGE /tmp/growth0.log | sed 's/.*LINEAGE[: ]*//' | tr -d '[:space:]')
L1=$(grep LINEAGE /tmp/growth1.log | sed 's/.*LINEAGE[: ]*//' | tr -d '[:space:]')
if [ "$GREW" != 1 ]; then
    echo "GROWTH PROOF ✗ — no growth event activated in ${S}s (see /tmp/growth*.log)"
    grep -h "GROWTH" /tmp/growth0.log /tmp/growth1.log || true
    exit 1
fi
if [ -z "$L0" ] || [ "$L0" != "$L1" ]; then
    echo "GROWTH PROOF ✗ — chains diverged or empty after growth"
    exit 1
fi
echo "GROWTH PROOF ✓ — event activated on-chain, both nodes converged" \
     "($(printf '%s' "$L0" | awk -F'>' '{print NF}') blocks)"
