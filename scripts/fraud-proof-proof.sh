#!/usr/bin/env bash
# THE DISPUTE GAME, live (Sharding Road Phase 1).
#
#   scripts/fraud-proof-proof.sh [seconds]
#
# node0 mints with --byzantine-aggregation: every block it proposes commits a
# corrupted page aggregate (a wrong state_root the page_leaves witness matches,
# so the block is internally consistent — only re-aggregation catches it).
# node1 is honest. PASSES iff node1 both REJECTS the fraudulent blocks on their
# bad root AND emits/verifies a page fraud proof that convicts one — i.e. the
# proof machinery works end to end on a real gossip network, not just in golden
# vectors. The honest node keeps a clean chain of its own blocks throughout.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
( cd node && cargo build --release )
B=node/target/release/sestrian-node
S=${1:-70}
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
A=$(printf 'a%.0s' {1..64} | head -c 64)
Bk=$(printf 'b%.0s' {1..64} | head -c 64)
rm -rf /tmp/fraud0 /tmp/fraud1 /tmp/fraud*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/fraud_genesis.bin >/dev/null 2>&1

# node0: the attacker. Still needs a peer + a trainer so it has real deltas to
# corrupt. --produce with the byzantine flag.
$B --network local --data-dir /tmp/fraud0 --key-seed "$A" \
   --genesis-file /tmp/fraud_genesis.bin --port 7930 --api-port 8130 \
   --bridge-port 7969 --produce --interval 6 --byzantine-aggregation \
   --peers /ip4/127.0.0.1/udp/7931/quic-v1 \
   --data-refs genesis --seconds "$S" --data-contributor "$FOUNDER" \
   > /tmp/fraud0.log 2>&1 &
# node1: honest. It must reject node0's blocks and raise proofs.
$B --network local --data-dir /tmp/fraud1 --key-seed "$Bk" \
   --genesis-file /tmp/fraud_genesis.bin --port 7931 --api-port 8131 \
   --bridge-port 7968 --produce --interval 6 \
   --peers /ip4/127.0.0.1/udp/7930/quic-v1 \
   --data-refs genesis --seconds "$S" --data-contributor "$FOUNDER" \
   > /tmp/fraud1.log 2>&1 &
sleep 3
for port in 7969 7968; do
  uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port "$port" --model toy-moe --inner 8 --batch 16 --device cpu \
    > "/tmp/fraudb_$port.log" 2>&1 &
done

sleep "$S"
pkill -f "data-dir /tmp/fraud" 2>/dev/null || true
pkill -f "node-port 796[89]" 2>/dev/null || true
sleep 2

strip() { sed 's/\x1b\[[0-9;]*m//g'; }
ATTACKED=$(grep -c "BYZANTINE: corrupting" /tmp/fraud0.log)
REJECTED=$(grep -c "state_root does not reproduce" /tmp/fraud1.log)
# the honest node convicts either its own detection (emit) or a received proof
VERIFIED=$(grep -cE "FRAUD PROOF VERIFIED|FRAUD: page" /tmp/fraud1.log)
echo "attacker corrupted   : $ATTACKED blocks"
echo "honest rejected root : $REJECTED blocks"
echo "honest fraud verdicts: $VERIFIED"
grep -m1 -E "FRAUD: page|FRAUD PROOF VERIFIED" /tmp/fraud1.log | strip | cut -c1-120

if [ "${ATTACKED:-0}" -ge 1 ] && [ "${REJECTED:-0}" -ge 1 ] && [ "${VERIFIED:-0}" -ge 1 ]; then
  echo "FRAUD-PROOF PROOF ✓ — the dispute game convicted a byzantine block live"
else
  echo "FRAUD-PROOF PROOF ✗ — attacked=$ATTACKED rejected=$REJECTED verified=$VERIFIED"
  echo "--- node1 tail ---"; tail -8 /tmp/fraud1.log | strip | cut -c1-140
  exit 1
fi
