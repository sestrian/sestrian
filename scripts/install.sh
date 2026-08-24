#!/usr/bin/env bash
# Sestrian one-command setup: build, wallet, genesis, preflight, (optional) service.
#
#   scripts/install.sh              # watch/sync node
#   scripts/install.sh --mine       # also train and earn (needs a GPU + PyTorch)
#   scripts/install.sh --service    # install systemd/launchd so it survives reboots
#
# Everything it does is what docs/joining.md documents by hand — this just does it
# in order, and refuses to leave you in a state where you'd silently earn nothing.
set -euo pipefail

# --- live devnet parameters (see docs/joining.md) ---------------------------
NETWORK="${SESTRIAN_NETWORK:-devnet}"
GENESIS_ID="${SESTRIAN_GENESIS_ID:-91bdcc281c0dbbd7b3bea3d38003e4c61565bcaa5fd8e7bfca296e6a4994ddb1}"  # devnet-genesis-3 (page-Merkle root; PREVIEW until ceremony)
MODEL="${SESTRIAN_MODEL:-small-moe}"
GENESIS_SEED="${SESTRIAN_GENESIS_SEED:-20260824}"
INTERVAL="${SESTRIAN_INTERVAL:-180}"

MINE=0; SERVICE=0
for arg in "$@"; do
  case "$arg" in
    --mine) MINE=1 ;;
    --service) SERVICE=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${SESTRIAN_HOME:-$HOME/.sestrian}"
mkdir -p "$HOME_DIR"
say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "toolchain"
command -v cargo >/dev/null || {
  echo "installing rust…"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
}
# RUN is a python interpreter invocation: "$RUN -m client.x" must always work.
UV=""
if command -v uv >/dev/null; then
  UV="uv run --python 3.11 --with torch --with numpy --with pynacl python"
elif python3 -c "import torch,numpy,nacl" 2>/dev/null; then UV="python3"
else
  echo "note: no uv and no torch — install uv (https://astral.sh/uv) or torch+numpy+pynacl."
  echo "      needed to generate the genesis and to mine."
  [ "$MINE" = 1 ] && { echo "cannot --mine without them"; exit 1; }
fi

say "build node"
( cd "$REPO/node" && cargo build --release )
BIN="$REPO/node/target/release/sestrian-node"

say "identity"
WALLET="$HOME_DIR/wallet.json"
if [ -f "$WALLET" ]; then
  echo "using existing wallet $WALLET"
else
  # non-interactive-safe: SESTRIAN_WALLET_PASSPHRASE encrypts it if you set
  # one, otherwise the file is plaintext (0600) — same as answering the prompt.
  ( cd "$REPO" && ${UV:-python3} -m client.wallet new --path "$WALLET" \
      --passphrase-env SESTRIAN_WALLET_PASSPHRASE )
  echo "BACK THIS FILE UP — it is your identity and your balance."
fi

say "genesis"
GEN="$HOME_DIR/genesis.bin"
if [ -f "$GEN" ]; then
  echo "using existing $GEN"
elif [ -n "${SESTRIAN_GENESIS_URL:-}" ]; then
  # Convenience path: a prebuilt artifact (scripts/release-genesis.sh). Trust is
  # NOT implied — the node verifies the weights against the state_root compiled
  # into the binary, so a tampered download fails at startup.
  echo "downloading prebuilt genesis: $SESTRIAN_GENESIS_URL"
  TMP="$HOME_DIR/.genesis.download"
  curl -fL --progress-bar "$SESTRIAN_GENESIS_URL" -o "$TMP"
  # verify the bytes AS DOWNLOADED (that is what the manifest publishes for the
  # artifact) — before spending time decompressing something tampered with
  if [ -n "${SESTRIAN_GENESIS_SHA256:-}" ]; then
    GOT=$(shasum -a 256 "$TMP" | awk '{print $1}')
    [ "$GOT" = "$SESTRIAN_GENESIS_SHA256" ] || {
      echo "FATAL: downloaded artifact sha256 mismatch"
      echo "  want $SESTRIAN_GENESIS_SHA256"; echo "  got  $GOT"
      rm -f "$TMP"; exit 1; }
    echo "artifact sha256 verified."
  fi
  case "$SESTRIAN_GENESIS_URL" in
    *.zst) command -v zstd >/dev/null || { echo "need zstd to decompress"; exit 1; }
           zstd -d -q -f "$TMP" -o "$GEN"; rm -f "$TMP" ;;
    *)     mv "$TMP" "$GEN" ;;
  esac
  echo "(the node also verifies it against the network's baked-in state_root)"
else
  [ -n "$UV" ] || { echo "need python+torch to generate the genesis, or set \
SESTRIAN_GENESIS_URL to a prebuilt artifact"; exit 1; }
  echo "reproducing it locally — deterministic, so this is trustless"
  ( cd "$REPO" && $UV -m client.make_genesis \
      --model "$MODEL" --seed "$GENESIS_SEED" --out "$GEN" ) | tee "$HOME_DIR/genesis.log"
  if ! grep -q "$GENESIS_ID" "$HOME_DIR/genesis.log"; then
    echo "FATAL: generated genesis does NOT match the published id $GENESIS_ID"
    echo "       you would be on a different chain — not continuing."
    exit 1
  fi
  echo "verified against the published genesis id."
fi

# --data-refs: provenance is required, so a miner must name a staked corpus.
# 'genesis' is the always-staked founding corpus — the correct starting point.
# Once you stake your own (client.wallet submit-data), name its hash instead and
# the data share flows to you.
REFS="${SESTRIAN_DATA_REFS:-genesis}"

say "preflight"
# Consensus params (bootstrap peer, genesis id, genesis-ledger contributor) are
# baked into the binary and selected by --network — nothing here can fork you.
ARGS=(--network "$NETWORK" --data-dir "$HOME_DIR/nodedata" --wallet "$WALLET"
      --genesis-file "$GEN" --api-port 8090)
[ "$MINE" = 1 ] && ARGS+=(--produce --interval "$INTERVAL" --data-refs "$REFS")
"$BIN" --check "${ARGS[@]}" || { echo "preflight failed — fix the above first."; exit 1; }

if [ "$SERVICE" = 0 ]; then
  say "ready — run it"
  echo "  $BIN ${ARGS[*]}"
  [ "$MINE" = 1 ] && echo "  # and in another shell, the trainer:
  cd $REPO && $UV -m client.miner_bridge --node-port 7999 --model $MODEL
  # (no --data needed: it fetches a public-domain corpus on first run.
  #  point --data at your own text once you stake it.)"
  echo
  echo "  watch:  curl -s localhost:8090/status   (stale_deltas must stay 0)"
  exit 0
fi

say "install service"
case "$(uname -s)" in
  Linux)
    sudo tee /etc/systemd/system/sestrian-node.service >/dev/null <<UNIT
[Unit]
Description=Sestrian node
After=network-online.target
Wants=network-online.target
[Service]
User=$USER
ExecStart=$BIN ${ARGS[*]}
Restart=always
RestartSec=5
LimitNOFILE=65536
WorkingDirectory=$REPO
[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload
    sudo systemctl enable --now sestrian-node
    echo "installed: systemctl status sestrian-node"
    ;;
  Darwin)
    PLIST="$HOME/Library/LaunchAgents/com.sestrian.node.plist"
    { echo '<?xml version="1.0" encoding="UTF-8"?>'
      echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
      echo '<plist version="1.0"><dict>'
      echo '  <key>Label</key><string>com.sestrian.node</string>'
      echo '  <key>ProgramArguments</key><array>'
      printf '    <string>%s</string>\n' "$BIN" "${ARGS[@]}"
      echo '  </array>'
      echo "  <key>WorkingDirectory</key><string>$REPO</string>"
      echo '  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>'
      echo "  <key>StandardOutPath</key><string>$HOME_DIR/node.log</string>"
      echo "  <key>StandardErrorPath</key><string>$HOME_DIR/node.log</string>"
      echo '</dict></plist>'
    } > "$PLIST"
    launchctl bootout "gui/$(id -u)/com.sestrian.node" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "installed: launchctl list | grep sestrian   (logs: $HOME_DIR/node.log)"
    ;;
  *) echo "unsupported OS for --service; run the command printed above manually" ;;
esac
