"""The chain watcher — a local web UI onto the live blockchain, wherever it is.

Joins the gossip network as an OBSERVER: a full node that receives transactions
and blocks, follows fork choice, and tracks the head — but never trains or
proposes. Point it at any peer (local, LAN, cloud) and open the browser:

  * watch blocks land in real time — height, root, lineage, mempool, peers;
  * watch the model's val loss fall as the chain trains it;
  * and the crazy part: TALK to the model at the current head. Every reply is
    generated from the weights the chain agrees on *right now* and stamped with
    the block it came from — a model that visibly grows while you chat with it.

The serving model is a separate CPU copy synced to the head on demand, so chat
never contends with a trainer's GPU. With --train the node also mines (a full
trainer with a window into itself).

  python -m client.watch --id 9 --port 9859 --peers 100.x.y.z:9851 --n 2 --http 8080
  → open http://localhost:8080
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import webbrowser

from rig.chain import dequantize
from . import gossip as g
from .data import ByteData
from .gpt import build
from .gossip import GossipNode, _dbg
from .trainer import set_flat_params

import torch

SERVE_DEVICE = "cpu"        # set by --serve-device; big models want mps/cuda for chat


# --------------------------------------------------------------------------
# Observer node: full gossip participant that never trains or proposes
# --------------------------------------------------------------------------
class WatchNode(GossipNode):
    def __init__(self, node_id, host, port, peers, n_total, train=False):
        super().__init__(node_id, host, port, peers, n_total)
        self.train = train
        # separate model for serving chat — its own device, never the trainer's
        self.serve_model, self.serve_device = build(g.MODEL_CFG, device=SERVE_DEVICE,
                                                    seed=g.GENESIS_SEED)
        self.serve_model.eval()
        self.serve_height = -1
        self.val_history = []                      # [(height, val_loss)]
        self._val_data = None                      # lazy ByteData for eval

    # ---- chain facts for the UI ------------------------------------------
    def status(self):
        tree = self.core.tree
        head = tree.blocks[tree.head]
        lineage = [{"h": b.header.height, "root": b.hash[:10],
                    "txs": len(b.txs), "proposer": str(b.header.proposer)[:10]}
                   for b in tree.chain_from_genesis()[-16:]]
        self._track_val(head.header.height)
        return {
            "height": head.header.height,
            "head": tree.head[:16],
            "state_root": head.header.state_root[:16],
            "peers": len(self.writers),
            "mempool": len(self.core.mempool),
            "seen_tx": len(self.core.seen_tx),
            "seen_block": len(self.core.seen_block),
            "params": self.serve_model.num_params(),
            "mode": "trainer" if self.train else "observer",
            "lineage": lineage,
            "val": self.val_history[-60:],
        }

    def _sync_serve(self):
        h = self.core.tree.blocks[self.core.tree.head].header.height
        if h != self.serve_height:
            set_flat_params(self.serve_model, dequantize(self.core.tree.head_state()))
            self.serve_height = h
        return h

    def _track_val(self, height):
        """Cheap CPU val-loss point per new head, for the chart."""
        if self.val_history and self.val_history[-1][0] == height:
            return
        if self._val_data is None:
            kwargs = {"path": g.DATA_PATH} if g.DATA_PATH else {}
            self._val_data = ByteData(block_size=g.MODEL_CFG.block_size,
                                      device=self.serve_device, **kwargs)
        self._sync_serve()
        v = self._val_data.estimate_loss(self.serve_model, batch_size=8, iters=3)["val"]
        self.val_history.append((height, round(float(v), 4)))

    def submit_transfer(self, q: dict):
        """Accept a signed transfer, gossip it to the network, and let the next
        proposer include it — the block's ledger_root then COMMITS it (settled)."""
        from rig.token import TransferTx, address as token_address
        try:
            tx = TransferTx(from_pub=str(q["from_pub"]), to_addr=str(q["to_addr"]),
                            amount=int(q["amount"]), nonce=int(q["nonce"]),
                            sig=bytes.fromhex(q["sig"]))
        except (KeyError, ValueError) as e:
            return {"ok": False, "error": f"malformed: {e}"}
        if not tx.verify():
            return {"ok": False, "error": "bad signature"}
        if self.core.head_ledger().balance(token_address(tx.from_pub)) < tx.amount:
            return {"ok": False, "error": "insufficient balance"}
        outbox = []
        self.core.recv_transfer(tx, outbox)
        for m in outbox:
            self._bcast(m)                          # gossip to trainer peers
        return {"ok": True, "txid": tx.txid(),
                "status": "in mempool — settles in the next block that includes it"}

    def submit_data(self, kind: str, q: dict):
        """Accept a signed data-lane tx (submit/challenge/vote), gossip it, and
        let the next proposer include it — the ledger_root then commits it."""
        from rig.token import DataChallengeTx, DataSubmitTx, DataVoteTx
        try:
            if kind == "submit":
                tx = DataSubmitTx(owner_pub=str(q["owner_pub"]),
                                  data_hash=str(q["data_hash"]),
                                  size_bytes=int(q["size_bytes"]),
                                  media_type=str(q.get("media_type", "text")),
                                  stake=int(q["stake"]), nonce=int(q["nonce"]),
                                  sig=bytes.fromhex(q["sig"]))
            elif kind == "challenge":
                tx = DataChallengeTx(challenger_pub=str(q["challenger_pub"]),
                                     data_id=str(q["data_id"]),
                                     stake=int(q["stake"]),
                                     reason=str(q.get("reason", "validity")),
                                     nonce=int(q["nonce"]),
                                     sig=bytes.fromhex(q["sig"]))
            else:
                tx = DataVoteTx(voter_pub=str(q["voter_pub"]),
                                challenge_id=str(q["challenge_id"]),
                                support=bool(q["support"]), nonce=int(q["nonce"]),
                                sig=bytes.fromhex(q["sig"]))
        except (KeyError, ValueError) as e:
            return {"ok": False, "error": f"malformed: {e}"}
        if not tx.verify():
            return {"ok": False, "error": "bad signature"}
        outbox = []
        self.core.recv_data_tx(tx, outbox)
        for m in outbox:
            self._bcast(m)
        return {"ok": True, "txid": tx.txid(),
                "status": "in mempool — settles in the next block that includes it"}

    def chat(self, prompt: str, n_new=220, temperature=0.85):
        """Generate from the CURRENT HEAD weights; stamp the reply with the block
        it came from — the model you talked to is the one the chain agrees on."""
        h = self._sync_serve()
        raw = prompt.encode("utf-8")[-g.MODEL_CFG.block_size + 1:] or b" "
        idx = torch.tensor([list(raw)], dtype=torch.long, device=self.serve_device)
        with torch.no_grad():
            out = self.serve_model.generate(idx, n_new, temperature=temperature)
        text = bytes(out[0].tolist()[len(raw):]).decode("utf-8", errors="replace")
        return {"reply": text, "height": h, "root": self.core.tree.head[:12]}

    # ---- observer main loop (no training, no proposing) ------------------
    async def _loop(self, seconds, settle=0):
        if self.train:                               # full trainer + window
            await super()._loop(seconds, settle=12.0)
            return
        end = time.time() + seconds
        while time.time() < end:
            await asyncio.sleep(1.0)                 # everything happens in _handle
        self._stop.set()


# --------------------------------------------------------------------------
# A deliberately tiny asyncio HTTP server (stdlib only)
# --------------------------------------------------------------------------
async def _http(node: WatchNode, host, port):
    async def handle(reader, writer):
        try:
            req = (await reader.readline()).decode()
            headers = {}
            while True:
                line = (await reader.readline()).decode().strip()
                if not line:
                    break
                k, _, v = line.partition(":")
                headers[k.lower().strip()] = v.strip()
            body = b""
            n = int(headers.get("content-length", 0))
            if n:
                body = await reader.readexactly(n)
            method, full = req.split()[0], req.split()[1]
            path, _, query = full.partition("?")
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)

            if method == "GET" and path == "/":
                payload, ctype = PAGE.encode(), "text/html; charset=utf-8"
            elif method == "GET" and path == "/status":
                payload, ctype = json.dumps(node.status()).encode(), "application/json"
            elif method == "GET" and path == "/balance":
                led = node.core.head_ledger()
                addr = params.get("addr", "")
                h = node.core.tree.blocks[node.core.tree.head].header.height
                payload = json.dumps({"addr": addr, "grains": led.balance(addr),
                                      "nonce": led.nonces.get(addr, 0),
                                      "supply": led.supply(), "height": h}).encode()
                ctype = "application/json"
            elif method == "POST" and path == "/transfer":
                payload, ctype = json.dumps(node.submit_transfer(
                    json.loads(body or b"{}"))).encode(), "application/json"
            elif method == "POST" and path in ("/data/submit", "/data/challenge",
                                               "/data/vote"):
                payload, ctype = json.dumps(node.submit_data(
                    path.rsplit("/", 1)[1],
                    json.loads(body or b"{}"))).encode(), "application/json"
            elif method == "GET" and path == "/data/registry":
                led = node.core.head_ledger()
                payload = json.dumps({"registry": led.registry,
                                      "challenges": led.challenges}).encode()
                ctype = "application/json"
            elif method == "POST" and path == "/chat":
                q = json.loads(body or b"{}")
                payload = json.dumps(node.chat(str(q.get("prompt", ""))[:2000],
                                               temperature=float(q.get("temp", 0.85)))).encode()
                ctype = "application/json"
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length:0\r\n\r\n")
                await writer.drain(); writer.close(); return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: " + ctype.encode() +
                         b"\r\nContent-Length: " + str(len(payload)).encode() +
                         b"\r\nConnection: close\r\n\r\n" + payload)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError,
                IndexError, json.JSONDecodeError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(handle, host, port)
    print(f"watch UI → http://localhost:{port}", flush=True)
    return server


# --------------------------------------------------------------------------
# The page — one dark, crude, pleasing screen
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>sestrian · chain watch</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧠</text></svg>">
<style>
:root{--bg:#0a0d12;--s:#111721;--s2:#0d1219;--ink:#dbe4ee;--mut:#6d7f92;--line:#1d2836;
--a:#3fe6cd;--a2:#8f80ff;--mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:var(--mono);font-size:14px;line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:18px}
header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;
border-bottom:1px solid var(--line);padding-bottom:12px}
h1{font-size:16px;margin:0;letter-spacing:.06em;display:flex;align-items:center;gap:9px}
h1 b{color:var(--a)}
#dot{width:9px;height:9px;border-radius:50%;background:var(--a);
box-shadow:0 0 8px var(--a);animation:pulse 1.6s ease-in-out infinite}
#dot.dead{background:#c0392b;box-shadow:none;animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){#dot{animation:none}}
#mode{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0}
.stat{background:var(--s);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
.stat .v{font-size:22px;margin-top:2px;color:var(--a);font-variant-numeric:tabular-nums}
.panel{background:var(--s);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}
.panel h2{font-size:12px;margin:0 0 10px;color:var(--mut);text-transform:uppercase;letter-spacing:.12em}
#blocks{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px}
.blk{flex:0 0 auto;background:var(--s2);border:1px solid var(--line);border-radius:8px;
padding:8px 10px;min-width:86px;text-align:center}
.blk.new{border-color:var(--a);box-shadow:0 0 12px rgba(63,230,205,.25)}
.blk .h{color:var(--a);font-size:15px}.blk .r{color:var(--mut);font-size:10.5px}
.blk .t{color:var(--mut);font-size:10.5px}
#loss{width:100%;height:90px;display:block}
#chatlog{max-height:340px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;margin-bottom:12px}
.msg{border-radius:10px;padding:10px 12px;max-width:88%;white-space:pre-wrap;word-break:break-word}
.msg.you{align-self:flex-end;background:#1a2436;border:1px solid #263650}
.msg.model{align-self:flex-start;background:var(--s2);border:1px solid var(--line)}
.msg .stamp{display:block;margin-top:6px;color:var(--a2);font-size:10.5px}
#bar{display:flex;gap:8px}
#prompt{flex:1;background:var(--s2);border:1px solid var(--line);border-radius:9px;
color:var(--ink);font-family:var(--mono);font-size:14px;padding:10px 12px;outline:none}
#prompt:focus{border-color:var(--a)}
button{background:var(--a);color:#04120f;border:0;border-radius:9px;padding:10px 18px;
font-family:var(--mono);font-weight:700;cursor:pointer;font-size:13px}
button:disabled{opacity:.5}
.note{color:var(--mut);font-size:11.5px;margin-top:8px}
</style></head><body><div class="wrap">
<header><h1><span id="dot"></span><b>sestrian</b> chain watch</h1><div id="mode">connecting…</div></header>
<div class="grid">
 <div class="stat"><div class="k">height</div><div class="v" id="height">–</div></div>
 <div class="stat"><div class="k">head</div><div class="v" id="head" style="font-size:14px">–</div></div>
 <div class="stat"><div class="k">peers</div><div class="v" id="peers">–</div></div>
 <div class="stat"><div class="k">mempool</div><div class="v" id="mempool">–</div></div>
 <div class="stat"><div class="k">deltas seen</div><div class="v" id="seen_tx">–</div></div>
 <div class="stat"><div class="k">params</div><div class="v" id="params">–</div></div>
</div>
<div class="panel"><h2>chain — newest blocks land on the right</h2><div id="blocks"></div></div>
<div class="panel"><h2>model val loss — falling means the chain is learning</h2>
<canvas id="loss"></canvas></div>
<div class="panel"><h2>talk to the model at the head — it grows while you chat</h2>
<div id="chatlog"></div>
<div id="bar"><input id="prompt" placeholder="say something to the chain…"
 autocomplete="off"><button id="send">send</button></div>
<div class="note">every reply is generated from the exact weights the chain agrees on at that
moment, stamped with its block. same prompt, later block → different (better) model.</div></div>
</div><script>
var lastH=-1;
function poll(){fetch('/status').then(function(r){return r.json()}).then(function(s){
 document.getElementById('dot').className='';
 document.getElementById('mode').textContent=s.mode+' node · live';
 ['height','peers','mempool','seen_tx'].forEach(function(k){
   document.getElementById(k).textContent=s[k]});
 document.getElementById('head').textContent=s.head.slice(0,10);
 document.getElementById('params').textContent=(s.params/1e6).toFixed(1)+'M';
 var bl=document.getElementById('blocks');bl.innerHTML='';
 s.lineage.forEach(function(b,i){var d=document.createElement('div');
   d.className='blk'+(b.h===s.height&&s.height!==lastH?' new':'');
   d.innerHTML='<div class="h">#'+b.h+'</div><div class="r">'+b.root+'</div><div class="t">'+b.txs+' Δ</div>';
   bl.appendChild(d)});
 bl.scrollLeft=bl.scrollWidth;
 lastH=s.height;drawLoss(s.val);
}).catch(function(){document.getElementById('dot').className='dead';
 document.getElementById('mode').textContent='disconnected…'})}
function drawLoss(v){var c=document.getElementById('loss'),x=c.getContext('2d');
 var W=c.width=c.clientWidth*2,H=c.height=180;x.clearRect(0,0,W,H);
 if(!v||v.length<2)return;var vs=v.map(function(p){return p[1]});
 var mn=Math.min.apply(0,vs),mx=Math.max.apply(0,vs);if(mx-mn<1e-6)mx=mn+1e-6;
 x.strokeStyle='#3fe6cd';x.lineWidth=3;x.beginPath();
 v.forEach(function(p,i){var px=i/(v.length-1)*(W-20)+10,
   py=14+((p[1]-mn)/(mx-mn))*(H-28);
   i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
 x.fillStyle='#6d7f92';x.font='20px ui-monospace';
 x.fillText(vs[vs.length-1].toFixed(3),W-90,24);}
function send(){var inp=document.getElementById('prompt'),btn=document.getElementById('send');
 var p=inp.value.trim();if(!p)return;inp.value='';btn.disabled=true;
 var log=document.getElementById('chatlog');
 var m=document.createElement('div');m.className='msg you';m.textContent=p;log.appendChild(m);
 log.scrollTop=log.scrollHeight;
 fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({prompt:p})}).then(function(r){return r.json()}).then(function(a){
  var d=document.createElement('div');d.className='msg model';
  d.textContent=a.reply;
  var s=document.createElement('span');s.className='stamp';
  s.textContent='— head @ block #'+a.height+' · '+a.root;d.appendChild(s);
  log.appendChild(d);log.scrollTop=log.scrollHeight;btn.disabled=false;
 }).catch(function(){btn.disabled=false})}
document.getElementById('send').onclick=send;
document.getElementById('prompt').addEventListener('keydown',function(e){
 if(e.key==='Enter')send()});
poll();setInterval(poll,1500);
</script></body></html>"""


async def run(a):
    peers = [(h, int(p)) for h, p in (x.split(":") for x in a.peers.split(",") if x)]
    node = WatchNode(a.id, "0.0.0.0", a.port, peers, a.n, train=a.train)
    http_server = await _http(node, "0.0.0.0", a.http)
    if a.open:
        webbrowser.open(f"http://localhost:{a.http}")
    try:
        await node.run(a.seconds)
    finally:
        http_server.close()


def _demo(a):
    """The one-command app: spin up a small local chain (2 trainer nodes as
    subprocesses), attach the observer + web UI to it, and open the browser.
    Ctrl-C tears the whole thing down."""
    t0 = time.time() + 4
    extra = []
    for flag in ("model", "data", "genesis", "inner", "batch"):
        v = getattr(a, flag, None)
        if v:
            extra += [f"--{flag}", str(v)]
    kids = []
    for i, port in enumerate((a.port + 1, a.port + 2)):
        other = a.port + 2 if i == 0 else a.port + 1
        kids.append(subprocess.Popen(
            [sys.executable, "-m", "client.gossip", "--id", str(i),
             "--port", str(port), "--peers",
             f"127.0.0.1:{other},127.0.0.1:{a.port}",
             "--n", "2", "--seconds", str(int(a.seconds)),
             "--interval", "2.0", "--device", "cpu", "--t0", str(t0)] + extra,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    print(f"demo chain: 2 trainer nodes launched — watch it live in the browser", flush=True)
    a.id, a.peers = 9, f"127.0.0.1:{a.port+1},127.0.0.1:{a.port+2}"
    a.n, a.open = 2, True
    try:
        asyncio.run(run(a))
    finally:
        for k in kids:
            k.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="one command: local chain + web UI, opens your browser")
    ap.add_argument("--id", type=int, default=9)
    ap.add_argument("--port", type=int, default=9850)       # gossip port
    ap.add_argument("--peers", default="")                  # host:port,... of chain peers
    ap.add_argument("--n", type=int, default=2)             # trainer count (for leader rotation)
    ap.add_argument("--http", type=int, default=8080)       # the browser UI
    ap.add_argument("--seconds", type=float, default=1e9)   # run ~forever by default
    ap.add_argument("--train", action="store_true")         # also mine, not just watch
    ap.add_argument("--open", action="store_true")          # auto-open the browser
    ap.add_argument("--model", default=None, choices=list(g.MODEL_PRESETS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--genesis", default=None)
    ap.add_argument("--inner", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--serve-device", default="cpu")        # mps/cuda for big-model chat
    ap.add_argument("--data-contributor", default=None)     # same genesis param as trainers
    a = ap.parse_args()
    g.apply_flags(a)
    global SERVE_DEVICE
    SERVE_DEVICE = a.serve_device
    if a.demo:
        _demo(a)
    else:
        asyncio.run(run(a))


if __name__ == "__main__":
    main()
