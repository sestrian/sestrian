#!/bin/bash
# Fleet-level invariants, checked automatically.
#
#   scripts/fleet-health.sh [node-url ...]      # defaults to the devnet fleet
#
# Every serious problem on 2026-08-25 — a node forked onto its own branch, a
# miner locked out of proposing entirely, a scheduled version bump that never
# took effect, an anchor wedged behind a head tie — was caught by a human
# reading /status or a block file, never by the test suite. Unit tests check
# the rules; this checks the RUNNING FLEET against the properties those rules
# are supposed to produce. Exit non-zero if any invariant is violated, so it
# can gate a release or run from cron.
set -uo pipefail

NODES=("$@")
if [ ${#NODES[@]} -eq 0 ]; then
    NODES=(http://localhost:8090 http://169.58.211.248:8080 http://13.140.32.27:8080)
fi
WINDOW=${WINDOW:-16}          # blocks to inspect for diversity
MIN_PROPOSERS=${MIN_PROPOSERS:-2}
FAIL=0
note() { printf '  %-6s %s\n' "$1" "$2"; }
bad()  { note "FAIL" "$1"; FAIL=1; }
ok()   { note "ok" "$1"; }
warn() { note "WARN" "$1"; }

echo "== fleet health =="

# ---- 1. reachability + lockstep -------------------------------------------
# A node serving a resync is SLOW, not DOWN. One timed-out request used to be
# reported as an outage — twice today, while every node was in fact healthy and
# merely busy feeding a resyncing peer. Try twice, patiently, before condemning.
ask() {
    local out
    out=$(curl -s -m 25 "$1/status" 2>/dev/null)
    [ -z "$out" ] && sleep 3 && out=$(curl -s -m 25 "$1/status" 2>/dev/null)
    printf '%s' "$out"
}
heights=(); heads=(); reach=0
for n in "${NODES[@]}"; do
    s=$(ask "$n")
    if [ -z "$s" ]; then bad "$n unreachable (two attempts, 25s each)"; heights+=(-1); heads+=("?"); continue; fi
    reach=$((reach+1))
    h=$(echo "$s" | python3 -c "import json,sys;print(json.load(sys.stdin).get('height',-1))" 2>/dev/null)
    hd=$(echo "$s" | python3 -c "import json,sys;print((json.load(sys.stdin).get('head') or '')[:12])" 2>/dev/null)
    st=$(echo "$s" | python3 -c "import json,sys;print(json.load(sys.stdin).get('stale_deltas',0))" 2>/dev/null)
    heights+=("$h"); heads+=("$hd")
    [ "${st:-0}" -gt 0 ] 2>/dev/null && warn "$n stale_deltas=$st (a miner is working for nothing)"
done
[ "$reach" -lt 2 ] && bad "fewer than 2 nodes reachable — cannot judge consensus"

# CONSENSUS is judged on SETTLED HISTORY at a COMMON height, never on tips.
# Tips legitimately differ every block when more than one miner proposes, so
# tip comparison reported a fork on a healthy fleet repeatedly — including a
# "still disagree after a block" confirmation that was simply observing a NEW
# tie. Comparing one agreed height is the same discipline devnet.sh uses.
verdict=$(python3 - "${NODES[@]}" <<'PY' 2>/dev/null
import json, sys, urllib.request
maps, tips = {}, {}
for u in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen(u + "/chain", timeout=25))
        rows = d if isinstance(d, list) else d.get("blocks", [])
        m = {b["height"]: (b.get("hash") or "")[:12] for b in rows}
        if m:
            maps[u], tips[u] = m, max(m)
    except Exception:
        pass
if len(maps) < 2:
    print(f"UNKNOWN only {len(maps)} node(s) served a chain")
else:
    h = min(tips.values()) - 3
    at = {u: maps[u].get(h) for u in maps}
    if any(v is None for v in at.values()):
        print(f"UNKNOWN height {h} outside some node's served window (tips {list(tips.values())})")
    elif len({v for v in at.values()}) == 1:
        print(f"AGREE at h{h} across {len(at)} nodes (tips {list(tips.values())})")
    else:
        print(f"DISAGREE at h{h}: {at}")
PY
)
case "$verdict" in
    AGREE*)    ok "$verdict" ;;
    DISAGREE*) bad "FORK — $verdict" ;;
    *)         warn "${verdict:-could not compare chains}" ;;
esac

# ---- 2. liveness ----------------------------------------------------------
# Reference node = the reachable node at the GREATEST height. Taking the first
# reachable one meant a node resyncing from genesis became the reference, and
# its empty registry / short chain then failed every downstream check.
ref=""; refh=-1
for i in "${!NODES[@]}"; do
    [ "${heights[$i]}" -lt 0 ] 2>/dev/null && continue
    if [ "${heights[$i]}" -gt "$refh" ]; then refh=${heights[$i]}; ref="${NODES[$i]}"; fi
done
[ -z "$ref" ] && { echo "no reachable node"; exit 1; }
note "info" "reference node $ref (height $refh)"
h1=$(curl -s -m 10 "$ref/status" | python3 -c "import json,sys;print(json.load(sys.stdin).get('height',-1))" 2>/dev/null)
sleep "${LIVENESS_WAIT:-420}"
h2=$(curl -s -m 10 "$ref/status" | python3 -c "import json,sys;print(json.load(sys.stdin).get('height',-1))" 2>/dev/null)
# A window shorter than the block interval cannot observe advancement, so a
# flat reading there is INCONCLUSIVE, not a stall. Calling it a failure would
# make every quick run red and train the reader to ignore the check.
LW=${LIVENESS_WAIT:-420}
if [ "${h2:-0}" -gt "${h1:-0}" ]; then ok "chain advancing ($h1 -> $h2)"
elif [ "$LW" -lt 420 ]; then warn "no new block in ${LW}s — shorter than the OBSERVED block time, inconclusive"
else bad "chain STALLED at $h1 over ${LW}s"; fi

# ---- 3. proposer diversity (consensus-relevant since v4) -------------------
# The v4 quorum gate needs `growth_quorum` DISTINCT proposers to score inside a
# window. A single perpetual proposer does not just look unfair — it gates
# growth off permanently, which is exactly what a per-node round epoch caused.
div=$(curl -s -m 10 "$ref/chain" 2>/dev/null | python3 -c "
import json,sys
from collections import Counter
d=json.load(sys.stdin); rows = d if isinstance(d,list) else d.get('blocks',[])
c=Counter(b.get('proposer','?')[:8] for b in rows[-$WINDOW:])
print(len(c), dict(c))" 2>/dev/null)
dn=$(echo "$div" | awk '{print $1}')
if [ -z "$dn" ]; then warn "could not read /chain for proposer diversity"
elif [ "$dn" -ge "$MIN_PROPOSERS" ]; then ok "proposer diversity ${dn} over last $WINDOW ($(echo "$div" | cut -d' ' -f2-))"
else bad "SINGLE PROPOSER over last $WINDOW blocks $(echo "$div" | cut -d' ' -f2-) — v4 growth is gated off"; fi

# ---- 4. provenance is real, not the bare chainparam ------------------------
reg=$(curl -s -m 10 "$ref/data/registry" 2>/dev/null | python3 -c "
import json,sys
r=json.load(sys.stdin).get('registry',{})
real=[k for k,v in r.items() if v.get('da_root')]
print(len(r), len(real))" 2>/dev/null)
tot=$(echo "$reg" | awk '{print $1}'); real=$(echo "$reg" | awk '{print $2}')
if [ -z "$tot" ]; then warn "could not read the data registry"
elif [ "${real:-0}" -ge 1 ]; then ok "$real of $tot registry entries carry an availability commitment"
else bad "NO staked corpus has a da_root — provenance is nominal only"; fi

echo
if [ "$FAIL" = "0" ]; then echo "FLEET HEALTHY ✓"; else echo "FLEET UNHEALTHY ✗"; fi
exit $FAIL
