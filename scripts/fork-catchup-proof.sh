#!/bin/bash
# Reproduces the EU-anchor wedge of 2026-08-25: a node that produced its own
# side branch while isolated must, on reconnect, walk back to the fork point
# and reorg onto the heavier fleet branch via request-response sync.
#
#   scripts/fork-catchup-proof.sh [isolation-blocks] [reconnect-seconds]
#
# Node A mines the "fleet" branch continuously. Node B first mines ALONE
# (no peers) to build a rival branch, is then restarted pointing at A, and
# must converge on A's chain. FAILS if B is still on its own branch at the
# end — which is exactly the live wedge.
set -e
cd "$(dirname "$0")/.."
( cd node && cargo build --release )
B=node/target/release/sestrian-node
ISO=${1:-4}
SECS=${2:-120}
IVL=5
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}

rm -rf /tmp/fcp_a /tmp/fcp_b /tmp/fcp_*.log
uv run --with torch --with numpy --with pynacl python -m client.make_genesis \
    --model toy-moe --seed 1337 --out /tmp/fcp_genesis.bin

# A: the fleet, mining from t=0 and for the whole run
$B --network local --data-dir /tmp/fcp_a --key-seed "$(printf '%064x' 1)" \
   --genesis-file /tmp/fcp_genesis.bin --port 7801 --api-port 8601 \
   --bridge-port 7601 --produce --data-refs genesis --interval $IVL \
   --seconds $((SECS + ISO * IVL + 20)) \
   --data-contributor $FOUNDER > /tmp/fcp_a.log 2>&1 &
A_PID=$!
sleep 3
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7601 --model toy-moe --inner 6 --batch 8 --device cpu \
    > /tmp/fcp_ta.log 2>&1 &
TA_PID=$!

# B phase 1: isolated, mining its own rival branch
$B --network local --data-dir /tmp/fcp_b --key-seed "$(printf '%064x' 2)" \
   --genesis-file /tmp/fcp_genesis.bin --port 7802 --api-port 8602 \
   --bridge-port 7602 --produce --data-refs genesis --interval $IVL \
   --seconds $((ISO * IVL + 10)) ${B_EXTRA:-} \
   --data-contributor $FOUNDER > /tmp/fcp_b1.log 2>&1 &
B1_PID=$!
sleep 3
uv run --with torch --with numpy --with pynacl python -m client.miner_bridge \
    --node-port 7602 --model toy-moe --inner 6 --batch 8 --device cpu \
    > /tmp/fcp_tb.log 2>&1 &
TB_PID=$!
wait $B1_PID || true
kill $TB_PID 2>/dev/null || true
echo "--- isolation over: A=$(curl -s localhost:8601/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["height"])' 2>/dev/null) B=$(grep -c 'head advanced' /tmp/fcp_b1.log) ---"

# B phase 2: reconnect to A; must abandon its branch and adopt A's
$B --network local --data-dir /tmp/fcp_b --key-seed "$(printf '%064x' 2)" \
   --genesis-file /tmp/fcp_genesis.bin --port 7802 --api-port 8602 \
   --bridge-port 7602 --interval $IVL --seconds $SECS ${B_EXTRA:-} \
   --peers /ip4/127.0.0.1/udp/7801/quic-v1 \
   --data-contributor $FOUNDER > /tmp/fcp_b2.log 2>&1 &
B_PID=$!

wait $B_PID || true
kill $A_PID $TA_PID 2>/dev/null || true
wait $A_PID 2>/dev/null || true

python3 - <<'PY'
import json
def head_chain(path):
    blocks, by_hash = [], {}
    with open(path) as f:
        for ln in f:
            b = json.loads(ln)
            by_hash[b.get("hash") or b["header"].get("hash", "")] = b
            blocks.append(b)
    return blocks
a = head_chain("/tmp/fcp_a/blocks.jsonl")
b = head_chain("/tmp/fcp_b/blocks.jsonl")
ah = {bb["header"]["height"]: bb for bb in a}
bh = {bb["header"]["height"]: bb for bb in b}
amax, bmax = max(ah), max(bh)
import sys
# B converged iff its top settled heights carry A's exact headers
probe = range(max(1, min(amax, bmax) - 2), min(amax, bmax) + 1)
same = all(json.dumps(ah[h]["header"], sort_keys=True)
           == json.dumps(bh[h]["header"], sort_keys=True) for h in probe)
print(f"A height {amax}, B height {bmax}, tail identical: {same}")
if same and bmax >= amax - 2:
    print("FORK-CATCHUP CONVERGED ✓")
else:
    print("FORK-CATCHUP WEDGED ✗ (the live EU failure, reproduced)")
    sys.exit(1)
PY
