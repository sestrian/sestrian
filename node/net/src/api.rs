//! HTTP API — the same JSON routes the Python watcher exposes, so the wallet
//! CLI and any UI work against a Rust node unchanged:
//!
//!   GET  /status           chain summary
//!   GET  /balance?addr=    grains, nonce, supply, height
//!   GET  /data/registry    data registry + open challenges
//!   POST /transfer         signed transfer -> mempool + gossip
//!   POST /data/submit | /data/challenge | /data/vote
//!
//! Handlers talk to the node loop over an mpsc command channel with oneshot
//! replies — the node's state is single-owner, no locks.

use axum::{
    extract::{Query, State},
    http::HeaderMap,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

#[derive(Debug)]
pub enum ApiCmd {
    Status(oneshot::Sender<Value>),
    Balance(String, oneshot::Sender<Value>),
    Registry(oneshot::Sender<Value>),
    Chain(oneshot::Sender<Value>),
    Miners(oneshot::Sender<Value>),
    Chat(String, oneshot::Sender<Value>),
    Upload(Vec<u8>, u64, String, oneshot::Sender<Value>),
    SubmitAccountTx(Value, oneshot::Sender<Value>),
    Metrics(oneshot::Sender<String>),
}

#[derive(Clone)]
struct Api {
    tx: mpsc::Sender<ApiCmd>,
    /// Bearer token gating operator-only endpoints (/upload spends the node's
    /// wallet; /chat monopolizes the trainer). None => those endpoints are
    /// disabled entirely (safe default). Read + signature-authenticated tx
    /// endpoints are always open — that's how a decentralized mempool works.
    admin_token: Option<String>,
}

/// True iff the request carries `Authorization: Bearer <admin_token>`. An
/// unset admin token disables the guarded endpoints rather than opening them.
fn authorized(api: &Api, headers: &HeaderMap) -> bool {
    let Some(tok) = &api.admin_token else { return false };
    headers.get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|h| h.strip_prefix("Bearer ").unwrap_or(h) == tok)
        .unwrap_or(false)
}

fn forbidden() -> Json<Value> {
    Json(json!({"ok": false,
        "error": "unauthorized: this endpoint requires the operator's \
                  Authorization: Bearer token (set SESTRIAN_API_TOKEN)"}))
}

async fn ask(tx: &mpsc::Sender<ApiCmd>, make: impl FnOnce(oneshot::Sender<Value>) -> ApiCmd)
    -> Json<Value>
{
    let (otx, orx) = oneshot::channel();
    if tx.send(make(otx)).await.is_err() {
        return Json(json!({"ok": false, "error": "node shutting down"}));
    }
    Json(orx.await.unwrap_or_else(|_| json!({"ok": false, "error": "node dropped request"})))
}

async fn status(State(api): State<Api>) -> Json<Value> {
    ask(&api.tx, ApiCmd::Status).await
}

async fn balance(State(api): State<Api>, Query(q): Query<HashMap<String, String>>)
    -> Json<Value>
{
    let addr = q.get("addr").cloned().unwrap_or_default();
    ask(&api.tx, |o| ApiCmd::Balance(addr, o)).await
}

async fn registry(State(api): State<Api>) -> Json<Value> {
    ask(&api.tx, ApiCmd::Registry).await
}

async fn chain(State(api): State<Api>) -> Json<Value> {
    ask(&api.tx, ApiCmd::Chain).await
}

async fn miners(State(api): State<Api>) -> Json<Value> {
    ask(&api.tx, ApiCmd::Miners).await
}

async fn chat(State(api): State<Api>, headers: HeaderMap, Json(b): Json<Value>) -> Json<Value> {
    if !authorized(&api, &headers) {
        return forbidden();
    }
    let prompt = b["prompt"].as_str().unwrap_or("").chars().take(1000).collect();
    ask(&api.tx, |o| ApiCmd::Chat(prompt, o)).await
}

async fn upload(
    State(api): State<Api>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    body: axum::body::Bytes,
) -> Json<Value> {
    if !authorized(&api, &headers) {
        return forbidden();
    }
    let stake = q.get("stake").and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
    let grains = (stake * 1e9) as u64;
    let media = q.get("media").cloned().unwrap_or_else(|| "text".into());
    ask(&api.tx, |o| ApiCmd::Upload(body.to_vec(), grains, media, o)).await
}

async fn dashboard() -> axum::response::Html<&'static str> {
    axum::response::Html(PAGE)
}

/// Prometheus text-format metrics (open by design — read-only operational data).
async fn metrics(State(api): State<Api>) -> String {
    let (otx, orx) = oneshot::channel();
    if api.tx.send(ApiCmd::Metrics(otx)).await.is_err() {
        return String::new();
    }
    orx.await.unwrap_or_default()
}

fn tag(kind: &str, mut body: Value) -> Value {
    body["kind"] = json!(kind);
    body
}

async fn transfer(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("transfer", b), o)).await
}

async fn data_submit(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("data_submit", b), o)).await
}

async fn data_challenge(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("data_challenge", b), o)).await
}

async fn data_vote(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("data_vote", b), o)).await
}

/// Submit a signed fee-bearing inference receipt (payer -> serving node).
async fn inference(State(api): State<Api>, Json(b): Json<Value>) -> Json<Value> {
    ask(&api.tx, |o| ApiCmd::SubmitAccountTx(tag("inference", b), o)).await
}

/// Routes any origin may read from a browser.
///
/// Read-only chain facts, all of them derivable by replaying the chain — there
/// is nothing here a visitor could not compute from a node of their own. The
/// mutating and operator routes are deliberately absent: they must stay
/// same-origin so a page on another site cannot drive somebody's node.
const PUBLIC_READS: [&str; 5] = ["/status", "/metrics", "/chain", "/miners", "/data/registry"];

/// Allow cross-origin reads of the public routes.
///
/// Without this a dashboard on any other host — sestrian.com included — gets
/// the response and then has it withheld by the browser. Serving the node over
/// https does NOT fix that; CORS is a separate refusal from mixed content, and
/// solving one without the other still leaves the page blank.
///
/// `*` rather than a named origin on purpose: these are public facts, and any
/// operator's own dashboard should work without us curating a list. Requests
/// carry no cookies and credentials are never allowed, so `*` grants a reader
/// nothing it could not get with curl.
async fn cors(req: axum::extract::Request,
              next: axum::middleware::Next) -> axum::response::Response {
    let public = PUBLIC_READS.contains(&req.uri().path());
    let mut res = next.run(req).await;
    if public {
        let h = res.headers_mut();
        h.insert("access-control-allow-origin", "*".parse().unwrap());
        // let a browser cache the permission rather than re-ask every poll
        h.insert("access-control-max-age", "86400".parse().unwrap());
    }
    res
}

pub async fn run(bind: String, port: u16, admin_token: Option<String>,
                 tx: mpsc::Sender<ApiCmd>) {
    let guarded = admin_token.is_some();
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/status", get(status))
        .route("/metrics", get(metrics))
        .route("/balance", get(balance))
        .route("/chain", get(chain))
        .route("/miners", get(miners))
        .route("/chat", post(chat))
        .route("/upload", post(upload))
        .route("/data/registry", get(registry))
        .route("/transfer", post(transfer))
        .route("/data/submit", post(data_submit))
        .route("/data/challenge", post(data_challenge))
        .route("/data/vote", post(data_vote))
        .route("/inference", post(inference))
        .layer(axum::middleware::from_fn(cors))
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(Api { tx, admin_token });
    // retry the bind: fast restarts leave the old socket lingering briefly, and
    // a silently-dead API made live nodes look wedged during the rehearsal
    let listener = loop {
        match tokio::net::TcpListener::bind((bind.as_str(), port)).await {
            Ok(l) => break l,
            Err(e) => {
                tracing::warn!("api {bind}:{port} busy ({e}); retrying in 3s");
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }
    };
    info!("http api on {bind}:{port} (admin endpoints {})",
          if guarded { "token-gated" } else { "DISABLED — set SESTRIAN_API_TOKEN" });
    let _ = axum::serve(listener, app).await;
}

/// The always-on chain dashboard, served by the node itself at `/`.
const PAGE: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>sestrian &middot; chain</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#129504;</text></svg>">
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
#sub{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.stat{background:var(--s);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
.stat .v{font-size:22px;margin-top:2px;color:var(--a);font-variant-numeric:tabular-nums;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stat .v.small{font-size:14px}
.panel{background:var(--s);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}
.panel h2{font-size:12px;margin:0 0 10px;color:var(--mut);text-transform:uppercase;letter-spacing:.12em}
#blocks{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px}
.blk{flex:0 0 auto;background:var(--s2);border:1px solid var(--line);border-radius:8px;
padding:8px 10px;min-width:96px;text-align:center}
.blk.new{border-color:var(--a);box-shadow:0 0 12px rgba(63,230,205,.25)}
.blk .h{color:var(--a);font-size:15px}.blk .r,.blk .t{color:var(--mut);font-size:10.5px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);color:var(--mut)}
th{text-transform:uppercase;font-size:10.5px;letter-spacing:.08em}
td.hi{color:var(--ink)}td.me{color:var(--a)}
.note{color:var(--mut);font-size:11.5px;margin-top:8px}
#chatlog{max-height:300px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;margin-bottom:12px}
.msg{border-radius:10px;padding:10px 12px;max-width:88%;white-space:pre-wrap;word-break:break-word}
.msg.you{align-self:flex-end;background:#1a2436;border:1px solid #263650}
.msg.model{align-self:flex-start;background:var(--s2);border:1px solid var(--line)}
.msg .stamp{display:block;margin-top:6px;color:var(--a2);font-size:10.5px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input[type=text],input[type=number]{background:var(--s2);border:1px solid var(--line);
border-radius:9px;color:var(--ink);font-family:var(--mono);font-size:14px;padding:10px 12px;outline:none}
input[type=text]{flex:1;min-width:200px}
input[type=text]:focus{border-color:var(--a)}
input[type=number]{width:110px}
input[type=file]{color:var(--mut);font-size:12px}
button{background:var(--a);color:#04120f;border:0;border-radius:9px;padding:10px 18px;
font-family:var(--mono);font-weight:700;cursor:pointer;font-size:13px}
button:disabled{opacity:.5}
select{background:var(--s2);border:1px solid var(--line);border-radius:9px;color:var(--ink);
font-family:var(--mono);padding:10px}
</style></head><body><div class="wrap">
<header><h1><span id="dot"></span><b>sestrian</b> chain</h1>
<div id="sub">connecting&hellip;</div></header>
<div class="grid">
 <div class="stat"><div class="k">height</div><div class="v" id="height">&ndash;</div></div>
 <div class="stat"><div class="k">head</div><div class="v small" id="head">&ndash;</div></div>
 <div class="stat"><div class="k">total supply</div><div class="v" id="supply">&ndash;</div></div>
 <div class="stat"><div class="k">peers</div><div class="v" id="peers">&ndash;</div></div>
 <div class="stat"><div class="k">delta mempool</div><div class="v" id="dpool">&ndash;</div></div>
 <div class="stat"><div class="k">model</div><div class="v small" id="model">&ndash;</div></div>
</div>
<div class="panel"><h2>chain &mdash; newest blocks land on the right</h2><div id="blocks"></div></div>
<div class="panel"><h2>our nodes &mdash; who does the work</h2>
<table id="miners"><thead><tr><th>miner</th><th>blocks proposed</th><th>deltas</th>
<th>share</th><th>earned</th><th>last seen</th></tr></thead><tbody></tbody></table>
<div class="note">computed from chain history: every block\u2019s proposer and every
delta\u2019s signer, with earnings from the live ledger. \u201cme\u201d = this node.</div></div>
<div class="panel"><h2>talk to the model at the head</h2>
<div id="chatlog"></div>
<div class="bar"><input type="text" id="prompt" placeholder="say something to the chain&hellip;"
 autocomplete="off"><button id="send">send</button></div>
<div class="note" id="chatnote">replies are generated from the exact weights the chain agrees
on right now, stamped with their block. on a CPU-only seed a reply can take a minute.</div></div>
<div class="panel"><h2>upload data to the chain</h2>
<div class="bar">
 <input type="file" id="file">
 <select id="media"><option>text</option><option>csv</option><option>code</option>
 <option>image</option><option>other</option></select>
 <input type="number" id="stake" value="1" min="0.1" step="0.1" title="stake (SESTRIAN)">
 <button id="up">upload + stake</button>
</div>
<div class="note" id="upnote">the node stores the bytes (content-addressed) and submits a
STAKED registry entry signed by its own wallet &mdash; it earns the data share by weight and
is challengeable like any entry. dev mode: needs this node\u2019s wallet to hold the stake.</div></div>
</div><script>
var lastH=-1;
function g(u){return fetch(u).then(function(r){return r.json()})}
function poll(){
 Promise.all([g('/status'),g('/chain'),g('/miners')])
 .then(function(rs){
  var s=rs[0],c=rs[1],m=rs[2];
  document.getElementById('dot').className='';
  document.getElementById('sub').textContent=(s.producer?'producer':'seed / relay')+
    ' node \u00b7 live \u00b7 '+String(s.miner).slice(0,10)+'\u2026';
  document.getElementById('height').textContent=s.height;
  document.getElementById('head').textContent=String(s.head).slice(0,14);
  document.getElementById('supply').textContent=(s.supply/1e9).toLocaleString();
  document.getElementById('peers').textContent=s.peers;
  document.getElementById('dpool').textContent=s.delta_pool;
  document.getElementById('model').textContent=s.model_attached?'attached':'none';
  var bl=document.getElementById('blocks');bl.innerHTML='';
  (c.blocks||[]).forEach(function(x){
    var d=document.createElement('div');
    d.className='blk'+(x.height===s.height&&s.height!==lastH?' new':'');
    d.innerHTML='<div class="h">#'+x.height+'</div><div class="r">'+
      String(x.hash).slice(0,10)+'</div><div class="t">'+x.n_txs+
      ' \u0394 \u00b7 '+String(x.proposer).slice(0,8)+'</div>';
    bl.appendChild(d)});
  bl.scrollLeft=bl.scrollWidth;lastH=s.height;
  var tb=document.querySelector('#miners tbody');tb.innerHTML='';
  (m.miners||[]).forEach(function(x){
    var tr=document.createElement('tr');
    var name=String(x.miner).slice(0,12)+'\u2026'+(x.is_me?' (me)':'');
    tr.innerHTML='<td class="'+(x.is_me?'me':'hi')+'">'+name+'</td><td>'+
      x.blocks_proposed+'</td><td>'+x.deltas+'</td><td>'+x.share_pct+
      '%</td><td class="hi">'+(x.balance/1e9).toLocaleString()+'</td><td>h'+
      x.last_height+'</td>';
    tb.appendChild(tr)});
 }).catch(function(){
  document.getElementById('dot').className='dead';
  document.getElementById('sub').textContent='disconnected\u2026 retrying';
 })}
function send(){
 var inp=document.getElementById('prompt'),btn=document.getElementById('send');
 var p=inp.value.trim();if(!p)return;inp.value='';btn.disabled=true;
 var log=document.getElementById('chatlog');
 var me=document.createElement('div');me.className='msg you';me.textContent=p;
 log.appendChild(me);log.scrollTop=log.scrollHeight;
 fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({prompt:p})}).then(function(r){return r.json()})
 .then(function(a){
  var d=document.createElement('div');d.className='msg model';
  d.textContent=a.ok?a.reply:('\u26a0 '+a.error);
  if(a.ok){var st=document.createElement('span');st.className='stamp';
   st.textContent='\u2014 head @ block #'+a.height;d.appendChild(st)}
  log.appendChild(d);log.scrollTop=log.scrollHeight;btn.disabled=false;
 }).catch(function(){btn.disabled=false})}
document.getElementById('send').onclick=send;
document.getElementById('prompt').addEventListener('keydown',function(e){
 if(e.key==='Enter')send()});
document.getElementById('up').onclick=function(){
 var f=document.getElementById('file').files[0];
 var note=document.getElementById('upnote');
 if(!f){note.textContent='pick a file first';return}
 if(f.size>60*1024*1024){note.textContent='dev limit: 60MB per upload';return}
 var stake=document.getElementById('stake').value;
 var media=document.getElementById('media').value;
 note.textContent='uploading '+f.name+' ('+f.size.toLocaleString()+' bytes)\u2026';
 f.arrayBuffer().then(function(buf){
  return fetch('/upload?stake='+stake+'&media='+media,
    {method:'POST',body:buf})
 }).then(function(r){return r.json()}).then(function(a){
  note.textContent=a.ok?('\u2713 custodied + staked: '+a.data_hash.slice(0,16)+
    '\u2026 (tx '+a.txid.slice(0,12)+'\u2026, settles next block)')
    :('\u26a0 '+a.error+(a.hint?' \u2014 '+a.hint:''));
 }).catch(function(e){note.textContent='upload failed: '+e})};
poll();setInterval(poll,5000);
</script></body></html>"#;

#[cfg(test)]
mod api_auth_tests {
    use super::*;

    fn api(token: Option<&str>) -> Api {
        let (tx, _rx) = mpsc::channel(1);
        Api { tx, admin_token: token.map(|s| s.to_string()) }
    }
    fn hdr(v: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(v) = v {
            h.insert("authorization", v.parse().unwrap());
        }
        h
    }

    #[test]
    fn admin_endpoints_require_matching_token() {
        // no token configured => guarded endpoints are closed, even with a header
        assert!(!authorized(&api(None), &hdr(Some("Bearer anything"))));
        let a = api(Some("s3cret"));
        assert!(authorized(&a, &hdr(Some("Bearer s3cret"))), "Bearer form accepted");
        assert!(authorized(&a, &hdr(Some("s3cret"))), "bare token accepted");
        assert!(!authorized(&a, &hdr(Some("Bearer wrong"))), "wrong token denied");
        assert!(!authorized(&a, &hdr(None)), "missing header denied");
    }
}

#[cfg(test)]
mod cors_tests {
    use super::*;

    /// The allow-list must never quietly grow to include a route that changes
    /// state or needs the operator token. Asserting membership both ways so
    /// adding a route to PUBLIC_READS is a deliberate act with a failing test
    /// behind it, not a one-word edit nobody reviews.
    #[test]
    fn only_read_only_routes_are_public() {
        for r in ["/status", "/metrics", "/chain", "/miners", "/data/registry"] {
            assert!(PUBLIC_READS.contains(&r), "{r} should be publicly readable");
        }
        for r in ["/chat", "/upload", "/transfer", "/inference",
                  "/data/submit", "/data/challenge", "/data/vote", "/balance", "/"] {
            assert!(!PUBLIC_READS.contains(&r),
                    "{r} must NOT be cross-origin readable — it mutates state, \
                     needs the operator token, or exposes an account");
        }
    }
}
