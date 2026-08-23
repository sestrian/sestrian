#!/bin/bash
# Rust devnet: 2 producing nodes + 2 PyTorch trainer bridges; asserts convergence.
set -e
cd "$(dirname "$0")/.."
( cd node && cargo build --release )
B=node/target/release/sestrian-node
S=${1:-90}
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
rm -rf /tmp/devnet0 /tmp/devnet1
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/devnet_genesis.bin
$B --network local --data-dir /tmp/devnet0 --key-seed $(printf 'a%.0s' {1..64} | head -c 64) \
   --genesis-file /tmp/devnet_genesis.bin --port 7930 --api-port 8390 \
   --bridge-port 7969 --produce --data-refs genesis --interval 6 --seconds $S \
   --data-contributor $FOUNDER > /tmp/devnet0.log 2>&1 &
$B --network local --data-dir /tmp/devnet1 --key-seed $(printf 'b%.0s' {1..64} | head -c 64) \
   --genesis-file /tmp/devnet_genesis.bin --port 7931 --api-port 8391 \
   --bridge-port 7968 --produce --data-refs genesis --interval 6 --seconds $S \
   --peers /ip4/127.0.0.1/udp/7930/quic-v1 \
   --data-contributor $FOUNDER > /tmp/devnet1.log 2>&1 &
sleep 3
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7969 --model toy-moe --inner 10 --batch 16 --device cpu > /tmp/devnetb0.log 2>&1 &
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7968 --model toy-moe --inner 10 --batch 16 --device cpu > /tmp/devnetb1.log 2>&1 &
wait %1 %2 || true
kill %3 %4 2>/dev/null || true
L0=$(grep LINEAGE /tmp/devnet0.log); L1=$(grep LINEAGE /tmp/devnet1.log)
echo "$L0"; echo "$L1"
# The lineage AFTER the label must be non-empty. Two nodes that produced NO
# blocks both print an empty lineage; those compare equal, so the old check
# reported CONVERGED on a chain that never advanced (exactly what happened when
# rev-5 provenance started filtering every delta for want of --data-refs).
B0=$(printf '%s' "$L0" | sed 's/.*LINEAGE[: ]*//' | tr -d '[:space:]')
B1=$(printf '%s' "$L1" | sed 's/.*LINEAGE[: ]*//' | tr -d '[:space:]')
if [ -z "$B0" ]; then
    echo "DEVNET PRODUCED NO BLOCKS ✗ (empty lineage — see /tmp/devnet0.log)"; exit 1
fi
if [ "$B0" = "$B1" ]; then
    echo "DEVNET CONVERGED ✓ ($(printf '%s' "$B0" | awk -F'>' '{print NF}') blocks)"
else
    echo "DEVNET DIVERGED ✗"; exit 1
fi
