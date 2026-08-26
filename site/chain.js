/* Shared chain data source for sestrian.com.
 *
 * Two sources, deliberately ordered:
 *
 *   1. chain.json — baked into the Pages deploy by scripts/chain_snapshot.py.
 *      Same-origin, always present, renders instantly.
 *   2. https://api.sestrian.com — polled after, upgrades the page in place.
 *
 * It has to be this way round. The site is https and the node speaks plain
 * http; a browser refuses that fetch outright, the request never leaves, and
 * it does not even surface as a CORS error. So a page that only polled the
 * node would show nothing, confusingly. The snapshot is the floor; live is the
 * bonus, and arrives on its own the day an https endpoint exists.
 *
 * Pages supply their own render(data, meta). Kept in one file because two
 * copies of this logic would drift the moment either page changed.
 */
(function (global) {
  "use strict";

  // Every anchor, in preference order. Each terminates its own TLS and holds a
  // certificate for its own name, so the browser can fail over between them.
  //
  // Caddy on each anchor already falls back to the other anchor's NODE, but
  // that cannot help if the whole box is gone: api.sestrian.com resolves only
  // to anchor1. A second name here is what survives losing that machine. It is
  // also why each anchor has its own hostname rather than several A records on
  // one name — round-robin records make ACME validation a coin flip, since the
  // CA may challenge the box that is not currently requesting the certificate.
  var LIVE = ["https://api.sestrian.com", "https://anchor2.sestrian.com"];
  var POLL = 20000;
  var RETRY = 3000;           // after a failure, try the next anchor promptly
  var GIVE_UP_AFTER = 3;      // consecutive whole-sweep failures before stopping

  var nf = new Intl.NumberFormat("en-US");

  function compact(n) {
    if (n == null || !isFinite(n)) return "–";
    if (n >= 1e9) return (n / 1e9).toFixed(2).replace(/\.?0+$/, "") + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
    return String(n);
  }

  function ago(iso) {
    var s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 90) return Math.round(s) + "s ago";
    if (s < 5400) return Math.round(s / 60) + " min ago";
    if (s < 172800) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " days ago";
  }

  /* Coins are held in base units, 1e9 to the token — same as the wallet CLI. */
  function ses(base) {
    return nf.format(Math.round((base || 0) / 1e9));
  }

  function short(hash, head, tail) {
    if (!hash) return "—";
    if (hash.length <= (head + tail + 1)) return hash;
    return hash.slice(0, head) + "…" + hash.slice(-tail);
  }

  /* /miners answers {head_height, miners[], …}; older builds answered a bare
   * array. Reading it as an array outright silently produced an empty table. */
  function minerList(payload) {
    if (!payload) return null;
    var lst = Array.isArray(payload) ? payload : payload.miners;
    return Array.isArray(lst) ? lst : null;
  }

  function fetchJSON(url, ms) {
    var opts = { cache: "no-store" };
    var timer;
    if (ms && global.AbortController) {
      var ctl = new AbortController();
      opts.signal = ctl.signal;
      timer = setTimeout(function () { ctl.abort(); }, ms);
    }
    return fetch(url, opts).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).finally(function () { if (timer) clearTimeout(timer); });
  }

  /**
   * render(status, meta) — meta = {live, at, endpoints, ok}
   * Called once with the baked snapshot, then again per successful live poll.
   */
  function load(render) {
    var endpoints = {};

    // A page's render() runs inside these promise chains, so an exception in it
    // would otherwise be caught below and reported as "chain unavailable" —
    // sending you to debug the network when the bug is in the page. Keep the
    // two apart: a render fault is logged as itself and never rewrites state.
    function draw(d, meta) {
      try { render(d, meta); }
      catch (e) { console.error("sestrian: chain panel render failed", e); }
    }

    fetchJSON("chain.json")
      .then(function (s) {
        endpoints = s.endpoints || {};
        applyEndpoints(endpoints);
        if (s.ok && s.status) draw(s.status, { live: false, at: s.captured_at, ok: true });
        else draw(null, { live: false, ok: false });
      })
      .catch(function () { draw(null, { live: false, ok: false }); })
      .finally(goLive);

    function goLive() {
      var misses = 0;      // consecutive failures, counted across all anchors
      (function poll() {
        var failed = false;
        // Ask EVERY anchor and show the one at the GREATEST height. Rotating
        // only on FAILURE meant a lagging anchor — which answers perfectly
        // well, it is merely behind — got rendered indefinitely: a resyncing
        // node once had the site reporting a third of the real chain height.
        // The page must show the chain, not the slowest window onto it.
        Promise.all(LIVE.map(function (b) {
          return fetchJSON(b + "/status", 6000)
            .then(function (d) { return { base: b, d: d }; },
                  function () { return null; });
        }))
          .then(function (results) {
            var best = null;
            results.forEach(function (r) {
              if (!r || !r.d) return;
              var h = Number(r.d.height);
              if (!isFinite(h)) return;
              if (!best || h > Number(best.d.height)) best = r;
            });
            if (!best) throw new Error("no anchor answered");
            misses = 0;
            var base = best.base, d = best.d;
            // Miners is a nice-to-have: a failure there must not cost us the
            // chain numbers, and must not push us off a working anchor.
            return fetchJSON(base + "/miners", 6000)
              .then(minerList, function () { return null; })
              .then(function (lst) {
                if (lst) { d.miners = lst.length; d.miner_rows = lst; }
                draw(d, { live: true, ok: true, endpoints: endpoints, endpoint: base });
              });
          })
          .catch(function () {
            failed = true;
            misses++;
          })
          .finally(function () {
            // Give up only once every anchor has failed GIVE_UP_AFTER times in
            // a row — otherwise one flaky host would silence a healthy one. A
            // failure retries sooner than a success, so failing over to the
            // second anchor is quick rather than a full poll interval away.
            if (misses < GIVE_UP_AFTER * LIVE.length) {
              setTimeout(poll, failed ? RETRY : POLL);
            }
          });
      })();
    }
  }

  /* Nav buttons for subdomains that may not exist yet. They ship inert and
   * marked "soon"; the build-time probe in chain_snapshot.py is what promotes
   * them, so we never advertise a link that would land on a browser error. */
  function applyEndpoints(eps) {
    document.querySelectorAll("[data-endpoint]").forEach(function (el) {
      var up = !!eps[el.getAttribute("data-endpoint")];
      el.classList.toggle("soon", !up);
      if (up) {
        el.removeAttribute("aria-disabled");
        el.removeAttribute("tabindex");
      } else {
        el.setAttribute("aria-disabled", "true");
        el.setAttribute("tabindex", "-1");
        el.addEventListener("click", function (e) { e.preventDefault(); });
      }
    });
  }

  global.SestrianChain = {
    load: load, LIVE: LIVE, POLL: POLL,
    fmt: { n: nf, compact: compact, ago: ago, ses: ses, short: short }
  };
})(window);
