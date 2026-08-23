#!/usr/bin/env bash
# Chaos + soak harness: two producing nodes + trainers, with one node KILLED and
# restarted mid-run. Exercises the production-hardening paths under churn —
# fast-boot recovery, sync catch-up after falling behind, the data-dir lock
# releasing on exit, and SIGTERM final-snapshot — then asserts both nodes
# converge to identical lineage. Single-machine; the multi-HOST version runs the
# same two commands on two boxes with real network addresses.
#
#   scripts/soak.sh [total_seconds] [kill_at_seconds]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
( cd node && cargo build --release )
B=node/target/release/sestrian-node
S=${1:-90}
KILL_AT=${2:-35}
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
A=$(printf 'a%.0s' {1..64} | head -c 64)
Bk=$(printf 'b%.0s' {1..64} | head -c 64)
rm -rf /tmp/soak0 /tmp/soak1
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/soak_genesis.bin

start_n0() {
  # SYMMETRIC peers (the real topology: every node dials the bootstrap
  # anchors). A restarted node with no --peers can only wait for inbound —
  # and the surviving side's redial is defeated by its lingering QUIC
  # connection to the dead process, so an asymmetric harness partitions
  # itself for the idle-timeout window. (Node-side eager-redial-on-disconnect
  # is tracked in production-readiness.)
  $B --network local --data-dir /tmp/soak0 --key-seed "$A" --genesis-file /tmp/soak_genesis.bin \
     --port 7910 --api-port 8110 --bridge-port 7989 --produce --interval 6 \
     --peers /ip4/127.0.0.1/udp/7911/quic-v1 \
     --data-refs genesis --seconds "$1" --data-contributor "$FOUNDER" >> /tmp/soak0.log 2>&1 &
  echo $!
}
: > /tmp/soak0.log; : > /tmp/soak1.log
N0=$(start_n0 "$S")
$B --network local --data-dir /tmp/soak1 --key-seed "$Bk" --genesis-file /tmp/soak_genesis.bin \
   --port 7911 --api-port 8111 --bridge-port 7988 --produce --interval 6 \
   --data-refs genesis --seconds "$S" --peers /ip4/127.0.0.1/udp/7910/quic-v1 \
   --data-contributor "$FOUNDER" >> /tmp/soak1.log 2>&1 &
N1=$!
sleep 3
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7989 --model toy-moe --inner 10 --batch 16 --device cpu > /tmp/soakb0.log 2>&1 &
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7988 --model toy-moe --inner 10 --batch 16 --device cpu > /tmp/soakb1.log 2>&1 &

# CHAOS: after KILL_AT seconds, hard-kill node0, wait, and restart it. It must
# fast-boot from its snapshot, re-acquire the data-dir lock, and sync the blocks
# it missed from node1 — then both must still converge.
sleep "$KILL_AT"
echo ">>> CHAOS: SIGKILL node0 (pid $N0)"; kill -9 "$N0" 2>/dev/null || true
sleep 8
REMAIN=$(( S - KILL_AT - 8 ))
echo ">>> restarting node0 for ${REMAIN}s (must fast-boot + catch up)"
N0=$(start_n0 "$REMAIN")
grep -q "FAST-BOOT\|full validated replay" /tmp/soak0.log && echo ">>> node0 booted" || true

wait "$N1" 2>/dev/null || true
# The restarted node0 is NOT a child of this shell (start_n0 backgrounds it
# inside $()), so `wait` can't see it — poll for its final LINEAGE instead.
for _ in $(seq 1 45); do
  grep -q LINEAGE /tmp/soak0.log && break
  sleep 2
done
pkill -f miner_bridge 2>/dev/null || true

echo "=== node0 boot lines ==="; grep -E "FAST-BOOT|full validated replay|another sestrian" /tmp/soak0.log | tail -3
L0=$(grep LINEAGE /tmp/soak0.log | tail -1 | sed 's/.*LINEAGE[: ]*//' | tr -d '[:space:]')
L1=$(grep LINEAGE /tmp/soak1.log | tail -1 | sed 's/.*LINEAGE[: ]*//' | tr -d '[:space:]')
echo "node0: $L0"; echo "node1: $L1"
# Under v1's always-on proposing (no leader rotation), the two nodes exit at
# STAGGERED times, so their last block or two is an unsettled Nakamoto tip —
# exact full-lineage equality is the old rotation-era assertion. The real
# property: both chains agree on everything except that tip window, and the
# chain genuinely advanced (recovered through the mid-run SIGKILL).
python3 - "$L0" "$L1" <<'PY'
import sys
a, b = (x.split(">") for x in sys.argv[1:3])
if not a or not b or a == [""] or b == [""]:
    print("SOAK PRODUCED NO BLOCKS ✗ (empty lineage)"); sys.exit(1)
n = min(len(a), len(b))
settle = max(0, n - 2)                    # allow a 2-block unsettled tip
if n < 8:
    print(f"SOAK TOO SHORT ✗ (only {n} common-height blocks)"); sys.exit(1)
if a[:settle] != b[:settle]:
    print("SOAK DIVERGED ✗ (disagreement below the tip window)"); sys.exit(1)
print(f"SOAK CONVERGED ✓ (recovered from mid-run kill; {settle} settled blocks "
      f"agree, tips {len(a)}/{len(b)} blocks)")
PY
