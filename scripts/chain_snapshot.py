#!/usr/bin/env python3
"""Capture a chain snapshot for the website's live panel.

sestrian.com is static and served over https; the node speaks plain http, and
a browser will not fetch http:// from an https page under any circumstances.
So the numbers are gathered HERE, on a CI runner that has no such restriction,
and baked into the deploy. The page renders this immediately and only then
tries to upgrade itself to live polling.

Never fails the caller. A deploy of a copy change must not be blocked because
the seed happens to be down — in that case we emit ok:false and the panel says
so in as many words, which is better than a page of dashes with no explanation.

    python3 scripts/chain_snapshot.py site/chain.json
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Both anchors, tried in order — the panel must not go dark because one
# region's seed is being restarted. SESTRIAN_SEED overrides with a single URL.
SEEDS = ([os.environ["SESTRIAN_SEED"].rstrip("/")]
         if os.environ.get("SESTRIAN_SEED")
         else ["http://169.58.211.248:8080", "http://13.140.32.27:8080"])
SEED = SEEDS[0]
TIMEOUT = float(os.environ.get("SESTRIAN_SEED_TIMEOUT", "20"))

# Only these reach the page. An allow-list rather than passing the node's
# response through verbatim: /status is an operator endpoint and may grow
# fields that have no business being republished on a public site.
STATUS_FIELDS = ("height", "head", "supply", "peers", "stale_deltas",
                 "quota_rejects", "producer", "model_attached")
MODEL_FIELDS = ("dim", "model_root", "expert_pages", "expert_pages_active",
                "pages_total", "growth_events", "pending_growth", "window_id")
# Per-miner rows for the network page's leaderboard. Addresses and block counts
# are public chain facts — anyone replaying the chain derives them — but keep it
# to an allow-list so an added operator-only field cannot leak by accident.
MINER_FIELDS = ("address", "blocks_proposed", "deltas", "share_pct",
                "balance", "last_height")

# Subdomains the site links to. Probed here rather than assumed, so a button
# never points at something that does not answer — the page renders it as
# "soon" until this says otherwise, and lights up on the next deploy.
ENDPOINTS = {"api": "https://api.sestrian.com/status",
             "anchor2": "https://anchor2.sestrian.com/status",
             "chat": "https://chat.sestrian.com/"}


def get(path):
    with urllib.request.urlopen(f"{SEED}{path}", timeout=TIMEOUT) as r:
        return json.loads(r.read())


def probe(url):
    """True if `url` answers at all. Any HTTP response counts — we are asking
    'does this host exist yet', not 'is every route healthy'."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=6) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500          # 401/404 still means something is there
    except Exception:                # noqa: BLE001 - DNS, TLS, timeout: not up
        return False


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "site/chain.json"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    global SEED
    raw = None
    for SEED in SEEDS:
        try:
            raw = get("/status")
            break
        except (urllib.error.URLError, OSError, ValueError,
                json.JSONDecodeError) as e:
            print(f"{SEED} unreachable ({e}) — trying next", file=sys.stderr)
    if raw is None:
        print("no anchor reachable — writing ok:false", file=sys.stderr)
        payload = {"ok": False, "captured_at": now, "seed": SEEDS[-1]}
    else:
        status = {k: raw[k] for k in STATUS_FIELDS if k in raw}
        model = raw.get("model") or {}
        if model:
            status["model"] = {k: model[k] for k in MODEL_FIELDS if k in model}

        # Miner count is a separate call and strictly a nice-to-have: if it
        # fails we still ship the chain numbers rather than nothing.
        # /miners answers {head_height, miners[], omissions, peers_connected} —
        # a bare list is accepted too, in case that shape ever comes back.
        try:
            m = get("/miners")
            lst = m.get("miners") if isinstance(m, dict) else m
            if isinstance(lst, list):
                status["miners"] = len(lst)
                status["miner_rows"] = sorted(
                    [{k: r[k] for k in MINER_FIELDS if k in r} for r in lst],
                    key=lambda r: r.get("blocks_proposed", 0), reverse=True)[:25]
        except Exception as e:                       # noqa: BLE001 - best effort
            print(f"/miners unavailable ({e}) — omitting", file=sys.stderr)

        payload = {"ok": True, "captured_at": now, "seed": SEED, "status": status}

    payload["endpoints"] = {name: probe(url) for name, url in ENDPOINTS.items()}

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
