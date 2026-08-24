#!/usr/bin/env bash
# Build the publishable genesis artifact set for a network.
#
#   scripts/release-genesis.sh [outdir]        # defaults to ./dist
#
# Produces, for the network's published (model, seed):
#   genesis.bin            canonical raw i64-LE weights (what --genesis-file takes)
#   genesis.bin.zst        the same bytes, zstd -19 (~23% — quantized weights are
#                          small ints, so the high bytes are sign-extension)
#   genesis-manifest.json  model/seed/params/state_root/sizes/digests
#   SHA256SUMS             for `shasum -a 256 -c`
#
# The point of publishing this is CONVENIENCE, never trust: the genesis is
# deterministic, so anyone can regenerate it and must get identical bytes, and
# the node verifies whatever it is given against the state_root baked into the
# binary. A tampered artifact fails at startup. Nodes also serve the genesis as
# erasure-coded DA shards, so the download is not the only path.
set -euo pipefail

MODEL="${SESTRIAN_MODEL:-small-moe}"
SEED="${SESTRIAN_GENESIS_SEED:-20260824}"
# devnet-genesis-3 (protocol v1): state_root is the PAGE-MERKLE root over the
# consensus page table — the value node/net/src/main.rs bakes in. The default
# below is the pre-ceremony PREVIEW; docs/genesis-ceremony.md requires the
# ceremony to regenerate on BOTH founder machines and update it before publish.
EXPECT="${SESTRIAN_GENESIS_ID:-91bdcc281c0dbbd7b3bea3d38003e4c61565bcaa5fd8e7bfca296e6a4994ddb1}"
OUT="${1:-dist}"

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

if command -v uv >/dev/null; then
  RUN="uv run --python 3.11 --with torch --with numpy --with pynacl python"
else
  RUN="python3"
fi

say "generating genesis (model=$MODEL seed=$SEED)"
$RUN -m client.make_genesis --model "$MODEL" --seed "$SEED" \
     --out "$OUT/genesis.bin" | tee "$OUT/.genesis.log"

ROOT=$(grep -o 'genesis_state_root: [0-9a-f]*' "$OUT/.genesis.log" | awk '{print $2}')
MODEL_ROOT=$(grep -o 'genesis_model_root: [0-9a-f]*' "$OUT/.genesis.log" | awk '{print $2}')
if [ -z "$ROOT" ]; then echo "could not read state_root from make_genesis"; exit 1; fi
if [ -n "$EXPECT" ] && [ "$ROOT" != "$EXPECT" ]; then
  echo "FATAL: state_root $ROOT != expected $EXPECT — refusing to publish."
  echo "       (a different model/seed, or a consensus-affecting code change)"
  exit 1
fi
say "state_root verified: $ROOT"

sz() { if stat -f%z "$1" >/dev/null 2>&1; then stat -f%z "$1"; else stat -c%s "$1"; fi; }
digest() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' \
           || sha256sum "$1" | awk '{print $1}'; }

RAW_SIZE=$(sz "$OUT/genesis.bin")
RAW_SHA=$(digest "$OUT/genesis.bin")

ZST_SIZE=null; ZST_SHA=null
if command -v zstd >/dev/null; then
  say "compressing (zstd -19)"
  zstd -q -19 -f "$OUT/genesis.bin" -o "$OUT/genesis.bin.zst"
  ZST_SIZE=$(sz "$OUT/genesis.bin.zst")
  ZST_SHA=\"$(digest "$OUT/genesis.bin.zst")\"
  echo "  $RAW_SIZE -> $ZST_SIZE bytes"
else
  echo "zstd not installed — skipping the compressed artifact"
fi

say "manifest"
cat > "$OUT/genesis-manifest.json" <<JSON
{
  "network": "${SESTRIAN_NETWORK:-devnet}",
  "model": "$MODEL",
  "seed": $SEED,
  "params": $((RAW_SIZE / 8)),
  "state_root": "$ROOT",
  "model_root": "${MODEL_ROOT:-}",
  "raw": { "file": "genesis.bin", "bytes": $RAW_SIZE, "sha256": "$RAW_SHA" },
  "zstd": { "file": "genesis.bin.zst", "bytes": $ZST_SIZE, "sha256": $ZST_SHA },
  "reproduce": "python -m client.make_genesis --model $MODEL --seed $SEED --out genesis.bin"
}
JSON
( cd "$OUT" && shasum -a 256 genesis.bin genesis.bin.zst genesis-manifest.json \
    2>/dev/null > SHA256SUMS || sha256sum genesis.bin genesis.bin.zst \
    genesis-manifest.json > SHA256SUMS )
rm -f "$OUT/.genesis.log"

cat "$OUT/genesis-manifest.json"
say "done — publish $OUT/ as release assets"
echo "  testers then either:"
echo "    SESTRIAN_GENESIS_URL=<url-to-genesis.bin.zst> scripts/install.sh"
echo "  or regenerate it themselves (identical bytes, no trust needed)."
