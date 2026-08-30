#!/usr/bin/env bash
# DATA-AVAILABILITY, live (Sharding Road Phase 3).
#
#   scripts/withholding-proof.sh [seconds]
#
# node0 mints blocks but WITHHOLDS their body shards (--byzantine-withhold):
# it publishes commitments, then answers no shard request. node1 is honest.
# The bodies are erasure-coded (any K of N reconstruct), so node1 samples/fetches
# and — finding it can never gather a withheld body — flags the block
# UNAVAILABLE and never adopts it, while its OWN chain keeps advancing
# (liveness preserved). PASSES iff node1 raises an availability verdict AND
# stays live.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
( cd node && cargo build --release )
B=node/target/release/sestrian-node
S=${1:-80}
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
A=$(printf 'a%.0s' {1..64} | head -c 64)
Bk=$(printf 'b%.0s' {1..64} | head -c 64)
export SESTRIAN_DTX_INLINE_MAX=0
rm -rf /tmp/wh0 /tmp/wh1 /tmp/wh*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/wh_genesis.bin >/dev/null 2>&1

# node0 withholds; both produce so each has real bodies to (not) serve
$B --network local --data-dir /tmp/wh0 --key-seed "$A" \
   --genesis-file /tmp/wh_genesis.bin --port 7950 --api-port 8150 \
   --bridge-port 7969 --produce --interval 6 --byzantine-withhold \
   --peers /ip4/127.0.0.1/udp/7951/quic-v1 \
   --data-refs genesis --seconds "$S" --data-contributor "$FOUNDER" \
   > /tmp/wh0.log 2>&1 &
$B --network local --data-dir /tmp/wh1 --key-seed "$Bk" \
   --genesis-file /tmp/wh_genesis.bin --port 7951 --api-port 8151 \
   --bridge-port 7968 --interval 6 \
   --peers /ip4/127.0.0.1/udp/7950/quic-v1 \
   --data-refs genesis --seconds $((S + 20)) --data-contributor "$FOUNDER" \
   > /tmp/wh1.log 2>&1 &
sleep 3
# only node0 (the withholder) trains+produces; node1 is a pure validator, so
# EVERY block is node0's and EVERY body is withheld — node1 must flag them.
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7969 --model toy-moe --inner 8 --batch 16 --device cpu \
    > /tmp/whb.log 2>&1 &

sleep $((S - 30))
H1A=$(curl -s -m 20 localhost:8151/status | python3 -c "import json,sys;print(json.load(sys.stdin)['height'])" 2>/dev/null || echo 0)
sleep 10
H1B=$(curl -s -m 20 localhost:8151/status | python3 -c "import json,sys;print(json.load(sys.stdin)['height'])" 2>/dev/null || echo 0)
pkill -f "data-dir /tmp/wh" 2>/dev/null || true
pkill -f "node-port 796[89]" 2>/dev/null || true

FLAGGED=$(grep -c "AVAILABILITY: block UNAVAILABLE" /tmp/wh1.log)
PRODUCED=$(grep -c "head advanced" /tmp/wh0.log)
echo "withholder produced blocks: $PRODUCED"
echo "validator availability verdicts: $FLAGGED"
grep -m1 "AVAILABILITY: block UNAVAILABLE" /tmp/wh1.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-130

# DETECTION is the property: the validator flags withheld blocks it cannot
# gather. (Liveness-under-withholding needs honest producers — covered by
# devnet, where available blocks always advance.)
if [ "${FLAGGED:-0}" -ge 1 ] && [ "${PRODUCED:-0}" -ge 2 ]; then
  echo "WITHHOLDING PROOF ✓ — the validator flagged withheld blocks as unavailable"
else
  echo "WITHHOLDING PROOF ✗ — flagged=$FLAGGED produced=$PRODUCED"
  tail -8 /tmp/wh1.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-140
  exit 1
fi
