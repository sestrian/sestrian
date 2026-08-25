#!/usr/bin/env python3
"""Generate site/sitemap.xml with real lastmod dates.

Built at deploy time rather than committed, because a hand-maintained sitemap
rots: the dates drift from the content and crawlers learn to distrust them.
`lastmod` here comes from the last git commit that actually touched each file,
so it means what it claims.

Pages are discovered from site/*.html rather than listed, so adding a page
cannot silently miss the sitemap.

    python3 scripts/make_sitemap.py
"""

import os
import subprocess
import sys
from datetime import datetime, timezone

BASE = "https://sestrian.com"
SITE = "site"
OUT = os.path.join(SITE, "sitemap.xml")

# Pages a crawler should treat as the canonical set, with a priority hint.
# 404 is excluded — indexing an error page is a classic own goal.
PRIORITY = {"index.html": "1.0", "technical.html": "0.8", "network.html": "0.8"}
SKIP = {"404.html"}


def last_commit_date(path: str) -> str:
    """ISO date of the last commit touching `path`, or today if untracked."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", path],
            capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return out.split("T")[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    pages = sorted(f for f in os.listdir(SITE)
                   if f.endswith(".html") and f not in SKIP)
    if not pages:
        sys.exit("no pages found — refusing to write an empty sitemap")

    rows = []
    for f in pages:
        loc = f"{BASE}/" if f == "index.html" else f"{BASE}/{f}"
        rows.append(
            "  <url>\n"
            f"    <loc>{loc}</loc>\n"
            f"    <lastmod>{last_commit_date(os.path.join(SITE, f))}</lastmod>\n"
            f"    <priority>{PRIORITY.get(f, '0.5')}</priority>\n"
            "  </url>")

    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           + "\n".join(rows) + "\n</urlset>\n")
    with open(OUT, "w") as fh:
        fh.write(xml)
    print(f"wrote {OUT} with {len(pages)} pages: {', '.join(pages)}")


if __name__ == "__main__":
    main()
