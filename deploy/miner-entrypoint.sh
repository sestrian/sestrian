#!/usr/bin/env bash
# Run the node and the trainer together, and DIE TOGETHER if either dies.
#
# The failure this exists to prevent: the trainer exits, the node keeps running
# and keeps looking healthy, and the operator earns nothing while everything
# appears fine. Silent half-failure is the worst outcome for a volunteer miner,
# so the container exits instead — let the restart policy deal with it.
set -uo pipefail

DATA_DIR="${SESTRIAN_DATA_DIR:-/data}"
GENESIS="${SESTRIAN_GENESIS:-/genesis/genesis.bin}"
API_PORT="${SESTRIAN_API_PORT:-8090}"
BRIDGE_PORT="${SESTRIAN_BRIDGE_PORT:-7999}"
DATA_REFS="${SESTRIAN_DATA_REFS:-genesis}"
MODEL="${SESTRIAN_MODEL:-small}"
NETWORK="${SESTRIAN_NETWORK:-devnet}"

die() { echo "sestrian-miner: $*" >&2; exit 1; }

[ -s "$GENESIS" ] || die "no genesis at $GENESIS.
  Mount one:  -v \$PWD/genesis.bin:$GENESIS:ro
  Get it:     curl -fL -o genesis.bin.zst \\
                https://github.com/sestrian/sestrian/releases/download/devnet-genesis-1/genesis.bin.zst
              zstd -d genesis.bin.zst
  (the node verifies it against the id compiled in, so a wrong file fails fast)"

if [ -z "${SESTRIAN_KEY_SEED:-}" ] && [ ! -s "$DATA_DIR/wallet.json" ]; then
  die "no identity. Either pass -e SESTRIAN_KEY_SEED=\$(head -c32 /dev/urandom | xxd -p -c64)
  (and SAVE IT — it is your wallet), or mount a wallet at $DATA_DIR/wallet.json"
fi

WALLET_ARGS=()
[ -s "$DATA_DIR/wallet.json" ] && WALLET_ARGS=(--wallet "$DATA_DIR/wallet.json")

# What device will torch actually use? Say so loudly — a container silently
# training on CPU because the GPU was never passed through is a slow surprise.
DEVICE=$(python -c "
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)
if [ "$DEVICE" = "cpu" ]; then
  echo "sestrian-miner: WARNING — no CUDA visible, training on CPU."
  echo "  On Linux+NVIDIA pass --gpus all and use the cu121 image."
  echo "  On macOS, Docker cannot reach the Apple GPU at all: run"
  echo "  scripts/install.sh --mine natively instead."
else
  echo "sestrian-miner: CUDA available — training on GPU."
fi

# Preflight first: refuse to burn hours in a state that cannot earn.
sestrian-node --check --network "$NETWORK" --data-dir "$DATA_DIR" \
  --genesis-file "$GENESIS" --api-port "$API_PORT" \
  --produce --data-refs "$DATA_REFS" "${WALLET_ARGS[@]}" || die "preflight failed (see above)"

term() { kill -TERM "${NODE_PID:-}" "${TRAIN_PID:-}" 2>/dev/null; }
trap term TERM INT

sestrian-node --network "$NETWORK" --data-dir "$DATA_DIR" --genesis-file "$GENESIS" \
  --api-port "$API_PORT" --api-bind 0.0.0.0 --bridge-port "$BRIDGE_PORT" \
  --produce --data-refs "$DATA_REFS" "${WALLET_ARGS[@]}" &
NODE_PID=$!

# Wait for the node's bridge before starting the trainer; replaying a long chain
# can take a while, so be patient rather than crash-looping.
for _ in $(seq 1 240); do
  curl -sf -m 2 "http://127.0.0.1:${API_PORT}/status" >/dev/null && break
  kill -0 "$NODE_PID" 2>/dev/null || die "node exited during startup"
  sleep 5
done

python -m client.miner_bridge --node-port "$BRIDGE_PORT" --model "$MODEL" \
  ${SESTRIAN_CORPUS:+--data "$SESTRIAN_CORPUS"} &
TRAIN_PID=$!

# First one to exit takes the container down with it. Polled rather than
# `wait -n` so this works on any POSIX-ish bash (and so it is testable outside
# a container, where the shell may be older than 4.3).
while true; do
  if ! kill -0 "$NODE_PID" 2>/dev/null; then
    wait "$NODE_PID"; CODE=$?; WHO=node; break
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    wait "$TRAIN_PID"; CODE=$?; WHO=trainer; break
  fi
  sleep 2
done
echo "sestrian-miner: $WHO exited (status $CODE) — stopping both so this does" >&2
echo "  not sit here looking healthy while earning nothing." >&2
term
wait 2>/dev/null
exit "$CODE"
