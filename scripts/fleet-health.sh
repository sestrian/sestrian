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
# A FORK and a LAG look alike if you only compare heads. They are different
# conditions with different responses: a fork needs operator action, a node
# catching up needs patience. Distinguish them by height — nodes at (nearly)
# the same height with different heads are forked; a node far behind is simply
# behind. Getting this wrong makes the check cry wolf during every resync,
# which is how a health check stops being read.
same_h_diff_head=0
for i in "${!heights[@]}"; do
    for j in "${!heights[@]}"; do
        [ "$i" -ge "$j" ] && continue
        hi=${heights[$i]}; hj=${heights[$j]}
        [ "$hi" -lt 0 ] 2>/dev/null && continue
        [ "$hj" -lt 0 ] 2>/dev/null && continue
        d=$((hi-hj)); d=${d#-}
        if [ "$d" -le 2 ] && [ "${heads[$i]}" != "${heads[$j]}" ]; then same_h_diff_head=1; fi
    done
done
if [ "$uniq_heads" -eq 1 ]; then ok "all reachable nodes on one head ($max)"
elif [ "$same_h_diff_head" = "1" ]; then
    bad "FORK: nodes at the same height disagree on the head (heights $min..$max)"
elif [ "$spread" -gt 2 ]; then
    warn "a node is BEHIND (heights $min..$max) — catching up, not forked"
else warn "heads differ but heights within $spread — propagation in flight"; fi

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
sleep "${LIVENESS_WAIT:-200}"
h2=$(curl -s -m 10 "$ref/status" | python3 -c "import json,sys;print(json.load(sys.stdin).get('height',-1))" 2>/dev/null)
# A window shorter than the block interval cannot observe advancement, so a
# flat reading there is INCONCLUSIVE, not a stall. Calling it a failure would
# make every quick run red and train the reader to ignore the check.
LW=${LIVENESS_WAIT:-200}
if [ "${h2:-0}" -gt "${h1:-0}" ]; then ok "chain advancing ($h1 -> $h2)"
elif [ "$LW" -lt 200 ]; then warn "no new block in ${LW}s — shorter than the block interval, inconclusive"
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
