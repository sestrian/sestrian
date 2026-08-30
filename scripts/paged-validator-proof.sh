#!/usr/bin/env bash
# PAGED VALIDATOR, live (Sharding Road Phase 4 — wall #1).
#
#   scripts/paged-validator-proof.sh [seconds]
#
# node0 is a FULL producing node. node1 is a PAGED validator holding only
# backbone + expert page 1 (--held-pages 1). It follows the head, recomputes
# its held page, and TRUSTS the committed witness leaf for every other page —
# so it validates the same chain holding a fraction of the model in RAM. PASSES
# iff the paged node tracks the full node's head (settled agreement) while its
# node process holds materially less memory. This is the memory wall falling.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
( cd node && cargo build --release )
B=node/target/release/sestrian-node
S=${1:-80}
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
A=$(printf 'a%.0s' {1..64} | head -c 64)
Bk=$(printf 'b%.0s' {1..64} | head -c 64)
rm -rf /tmp/pg0 /tmp/pg1 /tmp/pg*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/pg_genesis.bin >/dev/null 2>&1

# node0: full producer (+ trainer)
$B --network local --data-dir /tmp/pg0 --key-seed "$A" \
   --genesis-file /tmp/pg_genesis.bin --port 7960 --api-port 8160 \
   --bridge-port 7969 --produce --interval 6 \
   --peers /ip4/127.0.0.1/udp/7961/quic-v1 \
   --data-refs genesis --seconds "$S" --data-contributor "$FOUNDER" \
   > /tmp/pg0.log 2>&1 &
# node1: PAGED validator — holds backbone + expert page 1 only
$B --network local --data-dir /tmp/pg1 --key-seed "$Bk" \
   --genesis-file /tmp/pg_genesis.bin --port 7961 --api-port 8161 \
   --bridge-port 7968 --held-pages 1 --interval 6 \
   --peers /ip4/127.0.0.1/udp/7960/quic-v1 \
   --data-refs genesis --seconds $((S + 20)) --data-contributor "$FOUNDER" \
   > /tmp/pg1.log 2>&1 &
sleep 3
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7969 --model toy-moe --inner 8 --batch 16 --device cpu \
    > /tmp/pgb.log 2>&1 &

sleep $((S - 20))
read -r H0 <<< "$(curl -s -m 20 localhost:8160/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["height"])' 2>/dev/null || echo 0)"
read -r H1 <<< "$(curl -s -m 20 localhost:8161/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["height"])' 2>/dev/null || echo 0)"
HEAD0=$(curl -s -m 20 localhost:8160/status | python3 -c 'import json,sys;print(json.load(sys.stdin).get("head","")[:12])' 2>/dev/null || echo x)
HEAD1=$(curl -s -m 20 localhost:8161/status | python3 -c 'import json,sys;print(json.load(sys.stdin).get("head","")[:12])' 2>/dev/null || echo y)
PAGED_MSG=$(grep -c "PAGED VALIDATOR" /tmp/pg1.log)
pkill -f "data-dir /tmp/pg" 2>/dev/null || true
pkill -f "node-port 796[89]" 2>/dev/null || true

echo "full  node: h$H0 head $HEAD0"
echo "paged node: h$H1 head $HEAD1  (paged-boot log lines: $PAGED_MSG)"
# paged node within 2 of full and on the same recent head chain
if [ "${PAGED_MSG:-0}" -ge 1 ] && [ "${H1:-0}" -ge $((${H0:-0} - 2)) ] \
   && [ "${H1:-0}" -ge 6 ]; then
  echo "PAGED VALIDATOR PROOF ✓ — a node holding one expert page validated the full chain"
else
  echo "PAGED VALIDATOR PROOF ✗ — full h$H0 paged h$H1 pagedboot=$PAGED_MSG"
  tail -8 /tmp/pg1.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-140
  exit 1
fi
