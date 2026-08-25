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
heights=(); heads=(); reach=0
for n in "${NODES[@]}"; do
    s=$(curl -s -m 10 "$n/status" 2>/dev/null)
    if [ -z "$s" ]; then bad "$n unreachable"; heights+=(-1); heads+=("?"); continue; fi
    reach=$((reach+1))
    h=$(echo "$s" | python3 -c "import json,sys;print(json.load(sys.stdin).get('height',-1))" 2>/dev/null)
    hd=$(echo "$s" | python3 -c "import json,sys;print((json.load(sys.stdin).get('head') or '')[:12])" 2>/dev/null)
    st=$(echo "$s" | python3 -c "import json,sys;print(json.load(sys.stdin).get('stale_deltas',0))" 2>/dev/null)
    heights+=("$h"); heads+=("$hd")
    [ "${st:-0}" -gt 0 ] 2>/dev/null && warn "$n stale_deltas=$st (a miner is working for nothing)"
done
[ "$reach" -lt 2 ] && bad "fewer than 2 nodes reachable — cannot judge consensus"

# heads agree, or heights close enough to be mid-propagation
uniq_heads=$(printf '%s\n' "${heads[@]}" | grep -v '^?$' | sort -u | wc -l | tr -d ' ')
max=-1; min=999999999
for h in "${heights[@]}"; do
    [ "$h" -lt 0 ] 2>/dev/null && continue
    [ "$h" -gt "$max" ] && max=$h
    [ "$h" -lt "$min" ] && min=$h
done
spread=$((max-min))
if [ "$uniq_heads" -eq 1 ]; then ok "all reachable nodes on one head ($max)"
elif [ "$spread" -le 2 ]; then warn "heads differ but heights within $spread — likely propagation, recheck"
else bad "FORK: heights $min..$max across ${#NODES[@]} nodes, $uniq_heads distinct heads"; fi

# ---- 2. liveness ----------------------------------------------------------
ref="${NODES[0]}"
for n in "${NODES[@]}"; do
    s=$(curl -s -m 10 "$n/status" 2>/dev/null) && [ -n "$s" ] && { ref="$n"; break; }
done
h1=$(curl -s -m 10 "$ref/status" | python3 -c "import json,sys;print(json.load(sys.stdin).get('height',-1))" 2>/dev/null)
sleep "${LIVENESS_WAIT:-200}"
h2=$(curl -s -m 10 "$ref/status" | python3 -c "import json,sys;print(json.load(sys.stdin).get('height',-1))" 2>/dev/null)
if [ "${h2:-0}" -gt "${h1:-0}" ]; then ok "chain advancing ($h1 -> $h2)"
else bad "chain STALLED at $h1 over ${LIVENESS_WAIT:-200}s"; fi

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
