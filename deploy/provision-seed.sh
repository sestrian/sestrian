#!/bin/bash
# Provision a fresh Ubuntu 24.04 VPS into a Sestrian public seed node
# (bootstrap peer + circuit-relay v2 + HTTP API), from zero, idempotently.
#
#   scp deploy/provision-seed.sh root@<ip>:/root/ && ssh root@<ip> 'bash provision-seed.sh'
#
# The repo is private pre-launch: the script generates a machine deploy key and
# prints its PUBLIC half, then waits — add it at github.com/<repo>/settings/keys
# (read-only) and re-run; both steps are safe to repeat. After the repo goes
# public this step disappears (script falls back to https clone).
#
# What you get:
#   * sestrian-node (release build) under systemd, Restart=always
#   * --relay-server, public --external-address auto-detected
#   * genesis.bin materialized in-place from the published seed and VERIFIED
#   * ufw: 22/tcp, 9800/tcp+udp (gossip+relay), 8080/tcp (API)
set -euo pipefail

REPO_SSH="git@github.com:sestrian/sestrian.git"
REPO_HTTPS="https://github.com/sestrian/sestrian.git"

# --- genesis identity (env-overridable; same names scripts/install.sh uses) --
# A re-genesis is a values-only change: flip these three (env or defaults) and
# re-run. GENESIS_ID is the published genesis_state_root the generated artifact
# MUST hash to; $APP/genesis.id records which identity this box currently holds.
GENESIS_ID=${SESTRIAN_GENESIS_ID:-a597316003dbf12122b7cc6f39226ce7c8f7a871e58e7ddf364e56b08102527b}
GENESIS_MODEL=${SESTRIAN_MODEL:-small-moe}
GENESIS_SEED=${SESTRIAN_GENESIS_SEED:-20260822}

DATA_CONTRIBUTOR="3432d48fd6878b4f2e7a1e40cc15e112c512fae7"
# Optional: comma-separated multiaddrs of the OTHER anchors. The first seed runs
# inbound-only, but every subsequent anchor must dial an existing one or it sits
# at height 0 until a third party happens to connect it to the network.
PEERS=${SESTRIAN_PEERS:-}
NODE_PORT=9800
API_PORT=8080
APP=/opt/sestrian

echo "== swap =="
# Catch-up validation on the ~860MB model peaks at several GB of transient
# state; on an 8GB VPS that margin is thin enough that the OOM killer took an
# anchor down mid-sync (live finding). Swap is the backstop, not the plan.
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile
    swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -qy git build-essential pkg-config curl ufw python3-venv python3-pip zstd

echo "== rust toolchain =="
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "== source =="
if [ ! -d $APP/.git ]; then
    if git ls-remote -q "$REPO_HTTPS" >/dev/null 2>&1; then
        git clone --depth 1 "$REPO_HTTPS" $APP        # repo is public
    else
        # private repo: machine deploy key
        if [ ! -f /root/.ssh/sestrian_deploy ]; then
            ssh-keygen -t ed25519 -N "" -C "sestrian-seed-$(hostname)" \
                -f /root/.ssh/sestrian_deploy -q
        fi
        ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
        export GIT_SSH_COMMAND="ssh -i /root/.ssh/sestrian_deploy"
        if ! git clone --depth 1 "$REPO_SSH" $APP 2>/dev/null; then
            echo ""
            echo "############################################################"
            echo "# Add this READ-ONLY deploy key to the GitHub repo, then   #"
            echo "# re-run this script:                                      #"
            echo "############################################################"
            cat /root/.ssh/sestrian_deploy.pub
            exit 1
        fi
    fi
else
    ( cd $APP && GIT_SSH_COMMAND="ssh -i /root/.ssh/sestrian_deploy" git pull -q || true )
fi

echo "== build node =="
( cd $APP/node && cargo build --release )

echo "== genesis artifact =="
# RESET-AWARE: $APP/genesis.id stamps WHICH genesis this box holds. If the stamp
# matches $GENESIS_ID the step is a no-op; on any mismatch (including a missing
# stamp — a legacy pre-stamp deployment gets one reset on first run under this
# script) the old chain is backed up, wiped, and the configured genesis is
# regenerated + verified. $APP/seed.key is NEVER touched: peer identity survives
# a re-genesis.
STAMP=$APP/genesis.id
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$GENESIS_ID" ] && [ -f $APP/genesis.bin ]; then
    echo "genesis $GENESIS_ID already provisioned (stamp matches)"
else
    if [ -f $APP/genesis.bin ] || [ -d $APP/data ]; then
        echo "genesis identity change: $([ -f "$STAMP" ] && cat "$STAMP" || echo '<unstamped>') -> $GENESIS_ID"
        systemctl stop sestrian-seed 2>/dev/null || true
        if [ -d $APP/data ]; then
            mkdir -p $APP/backups
            BK=$APP/backups/pre-regenesis-$(date +%Y%m%d-%H%M%S).tar.zst
            DATA=$APP/data bash $APP/deploy/backup-restore.sh backup "$BK"
        fi
        rm -rf $APP/data $APP/genesis.bin
        rm -f "$STAMP"
    fi
    # ALWAYS rebuild the venv here: this branch only runs on a fresh install or
    # an identity change, and a venv predating an OS python upgrade is broken
    # in ways a health check on bin/python misses (bin/pip's shebang names the
    # old versioned interpreter). One extra torch install per re-genesis is
    # cheap; debugging a half-alive venv on a prod seed is not. (Live finding.)
    rm -rf $APP/.venv
    python3 -m venv $APP/.venv
    $APP/.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
    $APP/.venv/bin/pip install -q numpy pynacl
    ( cd $APP && .venv/bin/python -m client.make_genesis \
        --model $GENESIS_MODEL --seed $GENESIS_SEED --out genesis.bin ) | tee $APP/genesis.log
    # same guard scripts/install.sh applies: a genesis that doesn't hash to the
    # published id would put this seed on a different chain — never continue.
    if ! grep -q "$GENESIS_ID" $APP/genesis.log; then
        echo "FATAL: generated genesis does NOT match the published id $GENESIS_ID"
        echo "       (model=$GENESIS_MODEL seed=$GENESIS_SEED) — not continuing."
        rm -f $APP/genesis.bin
        exit 1
    fi
    printf '%s\n' "$GENESIS_ID" > "$STAMP"
    echo "genesis verified + stamped: $GENESIS_ID"
fi

echo "== identity =="
if [ ! -f $APP/seed.key ]; then
    head -c 32 /dev/urandom | xxd -p -c 64 > $APP/seed.key
    chmod 600 $APP/seed.key
fi

PUBLIC_IP=$(curl -4s https://ifconfig.me || curl -4s https://api.ipify.org)
echo "public ip: $PUBLIC_IP"

echo "== systemd =="
cat > /etc/systemd/system/sestrian-seed.service <<EOF
[Unit]
Description=Sestrian seed node (bootstrap + relay)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$APP/node/target/release/sestrian-node \\
  --data-dir $APP/data \\
  --key-file $APP/seed.key \\
  --genesis-file $APP/genesis.bin \\
  --port $NODE_PORT --api-port $API_PORT --bridge-port 7999 \\
  --relay-server \\
  --prune-depth 2 \\
  --external-address /ip4/$PUBLIC_IP/udp/$NODE_PORT/quic-v1 \\
  --data-contributor $DATA_CONTRIBUTOR${PEERS:+ --peers $PEERS}
Restart=always
RestartSec=5
LimitNOFILE=65536
WorkingDirectory=$APP

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now sestrian-seed

echo "== firewall =="
ufw allow 22/tcp >/dev/null
ufw allow $NODE_PORT/tcp >/dev/null
ufw allow $NODE_PORT/udp >/dev/null
ufw allow $API_PORT/tcp >/dev/null
ufw --force enable >/dev/null

sleep 4
echo "== verify =="
systemctl --no-pager -l status sestrian-seed | head -6
curl -s -m 5 http://127.0.0.1:$API_PORT/status && echo
echo ""
echo "SEED LIVE. Bootstrap multiaddr for node operators:"
echo "  /ip4/$PUBLIC_IP/udp/$NODE_PORT/quic-v1"
echo "  API: http://$PUBLIC_IP:$API_PORT/status"
