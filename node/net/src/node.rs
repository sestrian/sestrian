//! The node loop — single owner of all chain state, driving:
//! gossip (delta txs + payloads, account txs, commitment-only blocks),
//! range sync (request-response), the trainer bridge, block production,
//! persistence, and the HTTP API. NAT traversal behaviours (AutoNAT, DCUtR,
//! relay client, optional relay server for seeds) ride the same swarm.

use crate::api::ApiCmd;
use crate::bridge::{FromBridge, ToBridge};
use crate::proto::*;
use crate::store::{Store, SNAPSHOT_EVERY};
use libp2p::{
    autonat, dcutr, gossipsub, identify, ping, relay,
    futures::StreamExt,
    request_response::{self, ProtocolSupport},
    swarm::{behaviour::toggle::Toggle, NetworkBehaviour, SwarmEvent},
    Multiaddr, PeerId, StreamProtocol, Swarm,
};
use sestrian_core::{self as core, blocktree::BlockTree, token::AccountTx};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

pub const INCLUDE_K: usize = 8;

// --- mempool / cache bounds (DoS hardening) --------------------------------
// Every pool and dedup set is size-capped with eviction, and deltas/txs are
// admitted only within a narrow, includable window around the head — otherwise
// an unauthenticated peer can mint unlimited well-formed txs and grow memory +
// disk without bound. A delta is includable only when base_height == head
// height (validate_block requires it), so anything materially older can never
// be used and anything far in the future is spam.
const DELTA_STALE_SLACK: u64 = 2; // tolerate this many blocks of head lag
const DELTA_FUTURE_WINDOW: u64 = 4; // ...and this much look-ahead
const MAX_DELTA_POOL: usize = 512;
const MAX_ACCOUNT_POOL: usize = 4096;
const MAX_PENDING: usize = 256;
const MAX_SEEN: usize = 100_000;

// --- sync catch-up ----------------------------------------------------------
// A response is bounded by BYTES, not a fixed block count: a small-model chain
// packs many tiny blocks per response (fast catch-up) while a big-model chain
// (~18MB/block) sends only a few (protects small peers / slow uplinks). We keep
// at most one request in flight per peer, but re-request as soon as the previous
// response lands — so a node that fell behind actually recovers.
const SYNC_MAX_BLOCKS: usize = 64;
// 16MB (was 48): with the catch-up cursor + immediate re-request, small batches
// cost only an extra round-trip each, while a giant response is exactly what
// dies on a lossy WAN path (see the quic max_stream_data note in main.rs).
// Always >= 1 block regardless, so payload-heavy chains still make progress.
const SYNC_BYTE_BUDGET: usize = 16 * 1024 * 1024;
/// A delta body above this travels as an ANNOUNCEMENT (Dtx with an empty
/// payload) + shard fetch, and sync serves the block without it. Inline
/// paths die above this size: at a 4.7x retarget quota a body is ~92MB —
/// over the 64MB gossip cap, over the sync response cap with block overhead,
/// and 1.33x over everything once base64'd. The live network forked at its
/// first quota rise because every one of those paths failed at once.
const DTX_INLINE_MAX: usize = 8 * 1024 * 1024;
/// Env-overridable accessor (SESTRIAN_DTX_INLINE_MAX) — a transport knob, not
/// consensus. Tests force it to 0 so toy-size deltas exercise the
/// announce+shard-fetch path that production only hits at high quotas.
fn dtx_inline_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("SESTRIAN_DTX_INLINE_MAX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(DTX_INLINE_MAX))
}
/// b64 bytes of shards per ShardResponse (one oversized shard may exceed it —
/// at least one shard is always served so reconstruction can progress).
const SHARD_SERVE_BUDGET: usize = 12 * 1024 * 1024;
/// How long an announced-but-unfetched delta stays wanted before giving up.
const WANT_DELTA_TTL: f64 = 600.0;
const SYNC_INFLIGHT_TIMEOUT: f64 = 30.0;

/// Coordinates per aggregation chunk — the memory/latency knob (K deltas ×
/// this many i64 at once, ~8MB for the default, vs K × the whole 86M state).
const AGG_CHUNK: usize = 1 << 20;

/// Chunked, bounded-memory aggregation over SPARSE delta payloads for the
/// coordinate range [lo, hi). Bit-identical to `core::trimmed_mean` over the
/// same range of the full dense deltas (each coordinate's sort/trim/mean is
/// unchanged), but never materializes more than AGG_CHUNK coordinates × K
/// deltas at once — so a small peer can aggregate a 100M-param block without
/// holding K × ~0.8GB of dense deltas. In protocol v1 this is applied PER
/// PAGE over that page's claimants (build_candidate), mirroring
/// `core::paged_transition` exactly.
pub fn chunked_aggregate_range(payloads: &[&Payload], lo: usize, hi: usize,
                               chunk_size: usize) -> Vec<i64> {
    let mut out = vec![0i64; hi.saturating_sub(lo)];
    let step = chunk_size.max(1);
    let mut c = lo;
    while c < hi {
        let e = (c + step).min(hi);
        let chunk: Vec<Vec<i64>> = payloads.iter().map(|p| p.dense_range(c, e)).collect();
        out[c - lo..e - lo].copy_from_slice(&core::trimmed_mean(&chunk, 0.2));
        c = e;
    }
    out
}

/// A delta is worth holding only if its base_height sits in the includable
/// window around the current head. Pure + total so it can be unit-tested.
pub fn delta_in_window(base_height: u64, head_height: u64) -> bool {
    base_height + DELTA_STALE_SLACK >= head_height
        && base_height <= head_height + DELTA_FUTURE_WINDOW
}

#[cfg(test)]
mod mempool_bounds_tests {
    use super::*;

    #[test]
    fn delta_window_admits_near_head_rejects_far() {
        let head = 100;
        assert!(delta_in_window(head, head), "at-head delta is includable");
        assert!(delta_in_window(head - DELTA_STALE_SLACK, head), "within slack kept");
        assert!(!delta_in_window(head - DELTA_STALE_SLACK - 1, head), "too stale dropped");
        assert!(delta_in_window(head + DELTA_FUTURE_WINDOW, head), "near-future kept");
        assert!(!delta_in_window(head + DELTA_FUTURE_WINDOW + 1, head), "far-future dropped");
    }

    #[test]
    fn delta_window_safe_at_genesis() {
        // head 0 must not underflow / panic
        assert!(delta_in_window(0, 0));
        assert!(delta_in_window(3, 0));
        assert!(!delta_in_window(0 + DELTA_FUTURE_WINDOW + 1, 0));
    }

    #[test]
    fn chunked_aggregate_equals_dense() {
        // chunked aggregation over sparse payloads must be bit-identical to the
        // dense trimmed_mean, across MULTIPLE chunks (chunk_size < n).
        let n = 5000usize;
        let mk = |seed: i64| -> Vec<i64> {
            (0..n as i64)
                .map(|i| if (i + seed) % 5 == 0 { (i * seed) % 97 - 48 } else { 0 })
                .collect()
        };
        let dense: Vec<Vec<i64>> = (1..=5).map(mk).collect();
        let sparse = |d: &[i64]| -> Payload {
            let (mut idx, mut val) = (Vec::new(), Vec::new());
            for (i, &x) in d.iter().enumerate() {
                if x != 0 {
                    idx.extend_from_slice(&(i as u32).to_le_bytes());
                    val.extend_from_slice(&(x as i32).to_le_bytes());
                }
            }
            Payload { n: d.len(), idx: b64(&idx), val: b64(&val) }
        };
        let payloads: Vec<Payload> = dense.iter().map(|d| sparse(d)).collect();
        let refs: Vec<&Payload> = payloads.iter().collect();
        let dense_mean = core::trimmed_mean(&dense, 0.2);
        // a chunk size that divides the range unevenly, forcing several chunks
        assert_eq!(chunked_aggregate_range(&refs, 0, n, 1000), dense_mean, "chunk=1000");
        assert_eq!(chunked_aggregate_range(&refs, 0, n, 777), dense_mean, "uneven chunk");
        assert_eq!(chunked_aggregate_range(&refs, 0, n, n), dense_mean, "single chunk");
        // v1: a page-range aggregation equals the same slice of the dense mean
        let (lo, hi) = (1234usize, 4321usize);
        assert_eq!(chunked_aggregate_range(&refs, lo, hi, 500), dense_mean[lo..hi],
                   "page range");
    }
}

/// Length-prefixed JSON request-response codec with configurable size caps —
/// the stock JSON codec caps responses ~10MB, but an 86M-model compressed
/// payload is ~18MB. Generic so the block-sync and DA-shard protocols share it.
#[derive(Clone)]
pub struct JsonCodec<Req, Resp> {
    req_max: u64,
    resp_max: u64,
    _p: std::marker::PhantomData<fn() -> (Req, Resp)>,
}

impl<Req, Resp> JsonCodec<Req, Resp> {
    pub fn new(req_max: u64, resp_max: u64) -> Self {
        Self { req_max, resp_max, _p: std::marker::PhantomData }
    }
}

// A sync/shard REQUEST is small; a RESPONSE is bounded to cap the allocation a
// peer can force (block sync ~48MB serve budget + overhead; shards similar).
const SYNC_REQ_MAX: u64 = 256 * 1024;
const SYNC_RESP_MAX: u64 = 96 * 1024 * 1024;
const SHARD_REQ_MAX: u64 = 256 * 1024;
const SHARD_RESP_MAX: u64 = 96 * 1024 * 1024;

#[async_trait::async_trait]
impl<Req, Resp> request_response::Codec for JsonCodec<Req, Resp>
where
    Req: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    Resp: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    type Protocol = StreamProtocol;
    type Request = Req;
    type Response = Resp;

    async fn read_request<T>(&mut self, _: &StreamProtocol, io: &mut T) -> std::io::Result<Req>
    where T: futures::AsyncRead + Unpin + Send {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.take(self.req_max).read_to_end(&mut buf).await?;
        serde_json::from_slice(&buf).map_err(std::io::Error::other)
    }

    async fn read_response<T>(&mut self, _: &StreamProtocol, io: &mut T) -> std::io::Result<Resp>
    where T: futures::AsyncRead + Unpin + Send {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.take(self.resp_max).read_to_end(&mut buf).await?;
        serde_json::from_slice(&buf).map_err(std::io::Error::other)
    }

    async fn write_request<T>(&mut self, _: &StreamProtocol, io: &mut T, req: Req)
        -> std::io::Result<()>
    where T: futures::AsyncWrite + Unpin + Send {
        use futures::AsyncWriteExt;
        io.write_all(&serde_json::to_vec(&req)?).await?;
        io.close().await
    }

    async fn write_response<T>(&mut self, _: &StreamProtocol, io: &mut T, resp: Resp)
        -> std::io::Result<()>
    where T: futures::AsyncWrite + Unpin + Send {
        use futures::AsyncWriteExt;
        io.write_all(&serde_json::to_vec(&resp)?).await?;
        io.close().await
    }
}

#[derive(NetworkBehaviour)]
pub struct Behaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub identify: identify::Behaviour,
    pub sync: request_response::Behaviour<JsonCodec<SyncRequest, SyncResponse>>,
    pub shards: request_response::Behaviour<JsonCodec<ShardRequest, ShardResponse>>,
    pub autonat: autonat::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub relay_server: Toggle<relay::Behaviour>,
    pub ping: ping::Behaviour,
}

pub fn behaviour(
    key: &libp2p::identity::Keypair,
    relay_client: relay::client::Behaviour,
    relay_server: bool,
) -> Behaviour {
    let peer_id = key.public().to_peer_id();
    let gs_cfg = gossipsub::ConfigBuilder::default()
        .max_transmit_size(64 * 1024 * 1024)     // 86M compressed payloads fit
        .validation_mode(gossipsub::ValidationMode::Permissive)
        .build()
        .unwrap();
    // Peer scoring: track delivery/validity per peer and graylist the worst.
    // This is the real defense against a peer flooding max-size messages now
    // that size alone doesn't bound abuse. Conservative params (positive decay
    // so scores recover) with library-default thresholds so honest peers aren't
    // punished; if params are ever invalid we log and run without scoring rather
    // than refusing to start.
    let mut gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(key.clone()), gs_cfg).unwrap();
    let score_params = gossipsub::PeerScoreParams {
        decay_interval: Duration::from_secs(12),
        decay_to_zero: 0.01,
        ..Default::default()
    };
    if let Err(e) = gossipsub.with_peer_score(
        score_params, gossipsub::PeerScoreThresholds::default()) {
        warn!("gossipsub peer scoring disabled: {e}");
    }
    Behaviour {
        gossipsub,
        identify: identify::Behaviour::new(identify::Config::new(
            "/sestrian/1.0.0".into(), key.public())),
        sync: request_response::Behaviour::with_codec(
            JsonCodec::new(SYNC_REQ_MAX, SYNC_RESP_MAX),
            [(StreamProtocol::new("/sestrian/sync/1"), ProtocolSupport::Full)],
            request_response::Config::default()
                .with_request_timeout(Duration::from_secs(300)),
        ),
        shards: request_response::Behaviour::with_codec(
            JsonCodec::new(SHARD_REQ_MAX, SHARD_RESP_MAX),
            [(StreamProtocol::new("/sestrian/shards/1"), ProtocolSupport::Full)],
            request_response::Config::default()
                .with_request_timeout(Duration::from_secs(120)),
        ),
        autonat: autonat::Behaviour::new(peer_id, autonat::Config::default()),
        dcutr: dcutr::Behaviour::new(peer_id),
        relay_client,
        relay_server: Toggle::from(relay_server.then(|| {
            relay::Behaviour::new(peer_id, relay::Config::default())
        })),
        ping: ping::Behaviour::default(),
    }
}

fn now() -> f64 {
    // guarded: a clock set before 1970 would make duration_since Err and the old
    // `.unwrap()` panic the whole node. Treat a pre-epoch clock as t=0.
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// If a training round is dispatched to the bridge but no Delta/resync comes
/// back within this long, the trainer is assumed hung; clear the in-flight flag
/// so the node resumes contributing instead of going silent forever.
const TRAIN_TIMEOUT_SECS: f64 = 180.0;

/// Chain -> Net commands. The chain actor NEVER touches the swarm: it asks
/// the net loop, which enforces the per-peer in-flight gates and owns all
/// transport state. Unbounded: commands are tiny and the net loop drains fast.
pub enum ToNet {
    Publish(Gossip),
    /// request a sync pull; `from` is echoed back with the response batch so
    /// the chain's catch-up state machine sees the window it asked for
    SendSync(PeerId, u64),
    SendShards(PeerId, Vec<String>),
    /// the connected head changed — the net loop announces this each round
    /// and anchors its mesh-blind pulls on the height
    HeadInfo { hash: String, height: u64 },
}

/// Net -> Chain events. Everything stateful happens on the chain actor; the
/// net loop forwards, annotated with the transport facts only it knows.
pub enum ToChain {
    Dtx(crate::proto::WireDeltaTx, Payload, Option<PeerId>),
    Atx(serde_json::Value),
    Blk(StoredBlock, Option<PeerId>),
    /// a peer announced a head we may not have
    Head { peer: PeerId, hash: String, height: u64 },
    PeerConnected(PeerId),
    SyncServe(SyncRequest, tokio::sync::oneshot::Sender<SyncResponse>),
    ShardServe(ShardRequest, tokio::sync::oneshot::Sender<ShardResponse>),
    SyncBatch {
        peer: PeerId,
        /// true iff this response matched the outstanding request (stale
        /// batches still carry data but must not drive the state machine)
        current: bool,
        from: u64,
        blocks: Vec<StoredBlock>,
        payloads: HashMap<String, Payload>,
        their_head: u64,
    },
    ShardBatch { peer: PeerId, current: bool, bodies: Vec<BodyShards> },
    /// the net loop's round metronome, with the transport facts of the moment
    RoundTick { round: i64, elapsed_in_round: f64, num_peers: usize,
                connected: Vec<PeerId>, mesh_blind: bool },
}

pub struct NodeConfig {
    pub produce: bool,
    pub interval: f64,
    pub seconds: f64,               // 0 = run forever
    pub peers: String,              // configured peers — re-dialed when lost
    pub data_refs: Vec<String>,     // rev 5: staked corpora this miner names on its deltas
    /// DA retention: delete a body's shards once its block is this deep
    /// (0 = archive node, keep everything). Shard-zone retention is LINEAR
    /// growth — at retarget-window quotas it filled a founder disk in a day —
    /// so home nodes prune to a window and the anchors archive. A pruned node
    /// serves catch-up only inside its window; joining from genesis leans on
    /// the archive anchors until checkpoint sync exists (tracked).
    pub da_retain_blocks: u64,
}

pub struct Node {
    pub tree: BlockTree,
    pub store: Store,
    pub key: core::Key,
    pub blocks_full: HashMap<String, StoredBlock>,
    /// height up to which DA retention has already deleted old bodies
    pub da_pruned_to: u64,
    pub payloads: HashMap<String, Payload>,       // txid -> compressed payload
    pub delta_pool: HashMap<String, core::BackpropTx>,
    /// rev 7: held-out-loss scores from OUR trainer for pool deltas (txid ->
    /// micro-nats). Filled asynchronously via the bridge Eval verb; a missing
    /// score is 0 and the uniform fallback covers an entirely unscored block.
    /// Evicted in lockstep with delta_pool.
    pub delta_scores: HashMap<String, u64>,
    /// rev 8: influence sketches from OUR trainer for pool deltas (txid ->
    /// [i32; SKETCH_DIM]), arriving with the eval scores; missing = zeros
    /// (unsketched). Evicted in lockstep with delta_pool.
    pub delta_sketches: HashMap<String, Vec<i32>>,
    /// #114 (observable half): per-proposer count of pool deltas we had
    /// gossiped BEFORE a proposer's block arrived that the block then omitted.
    /// Censorship-suspicion observability, NOT a consensus rule.
    pub omitted_deltas: std::collections::BTreeMap<String, u64>,
    pub account_pool: HashMap<String, AccountTx>,
    pub pending: HashMap<String, (StoredBlock, PeerId)>, // blocks awaiting payloads
    /// when each pending block was parked — eviction is by AGE, not by height:
    /// during a fork heal the rival chain's blocks legitimately sit far below
    /// our head (a height rule evicted the whole fork on every head advance,
    /// found live)
    pub pending_at: HashMap<String, f64>,
    pub seen: HashSet<String>,
    /// insertion order for `seen`, so it can be bounded as a recency ring
    pub seen_order: VecDeque<String>,
    pub cfg: NodeConfig,
    pub topic: gossipsub::IdentTopic,
    pub bridge_tx: mpsc::Sender<ToBridge>,
    pub bridge_synced: bool,
    pub train_inflight: bool,
    /// wall-clock deadline for the in-flight training round (watchdog)
    pub train_deadline: f64,
    pub t0: f64,
    pub last_proposed_round: i64,
    /// once-per-round training dispatch (production is a separate, per-tick
    /// eligibility ladder — see the run loop)
    pub last_trained_round: i64,
    /// wall-clock of the last Head gossip we RECEIVED. Gossipsub mesh re-graft
    /// after churn (same-PeerId reconnect) is flaky; if no foreign head has
    /// been heard for a couple of rounds while peers are connected, the node
    /// PULLS via direct request-response sync — heartbeat healing must not
    /// depend on the mesh it is trying to heal.
    /// consecutive announce rounds with peers connected but NO foreign head
    /// heard. At 3, the connections themselves are treated as suspect: a
    /// SIGKILLed peer's QUIC connection lingers looking healthy (same PeerId
    /// as its restarted successor), and both gossip and request-response can
    /// be routed onto the corpse — so the node RECYCLES them (disconnect +
    /// redial) to force fresh transport. Found live in the CI soak.
    /// per-peer (timestamp, from_height) of the last sync we requested —
    /// heartbeat-triggered catch-up must not stack concurrent multi-hundred-MB
    /// transfers, and the response handler needs the window we asked for to
    /// detect a stalled overlap (see sync_cursor)
    /// The OUTSTANDING request id per peer. A gate cleared by a STALE response
    /// (one that had already timed out) spawns a second in-flight request; the
    /// stale responses multiply and a peer ends up served 4-6 concurrent syncs,
    /// all timing out — which re-forked the miners at max quota (found live).
    /// Only a response/failure matching the outstanding id touches the gate.
    /// Per-peer floor for the next sync request's from_height. Requests anchor
    /// at head−2 (reorg margin), but the server packs oldest-first under a byte
    /// budget — with payload-heavy blocks a batch can be EXACTLY the overlap
    /// window, so every response re-delivers known blocks and the head never
    /// moves (found live: the first WAN fresh-join on small-moe wedged at
    /// height 3 forever). When a response teaches nothing new while the peer is
    /// ahead, the cursor jumps past the served range; any response containing
    /// a new block clears it (a real reorg always teaches new blocks, so the
    /// margin is preserved exactly when it matters).
    pub sync_cursor: HashMap<PeerId, u64>,
    pub peers_connected: usize,
    pub chat_pending: Vec<tokio::sync::oneshot::Sender<Value>>,
    pub chat_inflight: bool,
    /// consecutive deltas dropped as stale — a slow trainer mining for nothing.
    /// Surfaced in /status + /metrics so the failure is visible, not silent.
    pub stale_deltas: u64,
    /// v1: deltas rejected for page-claim/quota violations (frozen-page claim,
    /// stray coordinates, below required_nnz) — spam or a misconfigured miner;
    /// visible in /status + /metrics, never silent.
    pub quota_rejects: u64,
    /// per-peer rotation cursor for serving GENESIS shards one at a time, so a
    /// bootstrapping peer that keeps asking us collects K distinct shards.
    /// the chain actor's outbound line to the net loop
    pub net: tokio::sync::mpsc::UnboundedSender<ToNet>,
    /// Exponential walk-back step per peer while hunting a fork point below
    /// our head (batches whose blocks can't connect to anything we know).
    pub sync_walkback: HashMap<PeerId, u64>,
    /// Rotating serve cursor per body — repeated asks get distinct shards.
    pub serve_shard_cursor: HashMap<String, usize>,
    /// Announced deltas whose bodies we are fetching by shards: txid ->
    /// (signed tx held until the body reconstructs, give-up deadline). Bounded.
    pub want_deltas: HashMap<String, (crate::proto::WireDeltaTx, f64)>,
}

impl Node {
    fn head_height(&self) -> u64 {
        self.tree.blocks[&self.tree.head].height
    }

    fn publish(&mut self, msg: &Gossip) {
        let _ = self.net.send(ToNet::Publish(msg.clone()));
    }

    /// Record a txid as seen, bounding the set as an insertion-ordered ring so a
    /// peer streaming unique txids can't grow it without limit.
    fn mark_seen(&mut self, txid: String) {
        if self.seen.insert(txid.clone()) {
            self.seen_order.push_back(txid);
            while self.seen_order.len() > MAX_SEEN {
                if let Some(old) = self.seen_order.pop_front() {
                    self.seen.remove(&old);
                }
            }
        }
    }

    /// Drop a never-included delta from the mempool AND reclaim its disk payload
    /// (it was written on accept; if it's never mined it is pure garbage).
    fn drop_pool_delta(&mut self, txid: &str) {
        self.delta_pool.remove(txid);
        self.delta_scores.remove(txid);
        self.delta_sketches.remove(txid);
        self.payloads.remove(txid);
        self.store.remove_payload(txid);
    }

    /// rev 7: the eval seed for the current proposal round — deterministic from
    /// (head hash, next height) so one round scores against one held-out batch.
    /// NOT consensus (scores are committed claims); it only needs to be stable
    /// within a round and not miner-chosen.
    fn eval_seed(&self) -> u64 {
        let hh = core::delta_hash(
            format!("{}|{}", self.tree.head, self.head_height() + 1).as_bytes());
        u64::from_str_radix(&hh[..16], 16).unwrap_or(0)
    }

    /// Ask our trainer to score every still-unscored pool delta at the head
    /// height. Fire-and-forget: production NEVER blocks on eval — deltas
    /// without a score in time weigh 0 (uniform fallback if all are unscored).
    fn request_evals(&mut self) {
        if !self.bridge_synced {
            return;
        }
        let hh = self.head_height();
        let want: Vec<(String, SparseI64, Vec<u32>)> = self.delta_pool.iter()
            .filter(|(id, t)| t.base_height == hh && !self.delta_scores.contains_key(*id))
            .filter_map(|(id, t)| {
                let p = self.payloads.get(id)?;
                let coords = p.coords()?;
                Some((id.clone(),
                      Payload::from_coords_i64(p.n, &coords),
                      t.canonical_pages()))
            })
            .collect();
        if !want.is_empty() {
            let _ = self.bridge_tx.try_send(ToBridge::Eval {
                height: hh, seed: self.eval_seed(), deltas: want,
            });
        }
    }

    /// Evict stale/over-cap deltas. Stale = outside the includable window (can
    /// never be mined); over-cap = keep the freshest MAX_DELTA_POOL by height.
    fn evict_delta_pool(&mut self) {
        let head = self.head_height();
        let stale: Vec<String> = self.delta_pool.iter()
            .filter(|(_, t)| !delta_in_window(t.base_height, head))
            .map(|(id, _)| id.clone()).collect();
        for id in stale {
            self.drop_pool_delta(&id);
        }
        if self.delta_pool.len() > MAX_DELTA_POOL {
            let mut by_h: Vec<(String, u64)> = self.delta_pool.iter()
                .map(|(id, t)| (id.clone(), t.base_height)).collect();
            by_h.sort_by_key(|(_, h)| *h);                 // stalest first
            let excess = self.delta_pool.len() - MAX_DELTA_POOL;
            for (id, _) in by_h.into_iter().take(excess) {
                self.drop_pool_delta(&id);
            }
        }
    }

    /// Evict account txs whose nonce is now below the sender's ledger nonce
    /// (can never apply), then cap by dropping the most speculative (highest
    /// nonce) first.
    fn evict_account_pool(&mut self) {
        use sestrian_core::token::address;
        let stale: Vec<String> = {
            let led = self.tree.head_ledger();
            self.account_pool.iter()
                .filter(|(_, t)| t.nonce()
                        < led.nonces.get(&address(t.sender_pub())).copied().unwrap_or(0))
                .map(|(id, _)| id.clone()).collect()
        };
        for id in stale {
            self.account_pool.remove(&id);
        }
        if self.account_pool.len() > MAX_ACCOUNT_POOL {
            let mut by_n: Vec<(String, u64)> = self.account_pool.iter()
                .map(|(id, t)| (id.clone(), t.nonce())).collect();
            by_n.sort_by_key(|(_, n)| std::cmp::Reverse(*n));  // most future first
            let excess = self.account_pool.len() - MAX_ACCOUNT_POOL;
            for (id, _) in by_n.into_iter().take(excess) {
                self.account_pool.remove(&id);
            }
        }
    }

    /// Queue an orphan/missing-payload block, bounded: when full, evict the
    /// lowest-height pending block (least likely to ever become live).
    fn queue_pending(&mut self, bh: String, sb: StoredBlock, peer: PeerId) {
        if self.pending.len() >= MAX_PENDING && !self.pending.contains_key(&bh) {
            // evict the HIGHEST parked block: low blocks apply first, so they
            // are the ones a heal needs; a dropped tip is re-served in seconds
            if let Some(drop) = self.pending.iter()
                .max_by_key(|(_, (s, _))| s.header.height).map(|(h, _)| h.clone()) {
                self.pending.remove(&drop);
                self.pending_at.remove(&drop);
            }
        }
        self.pending_at.entry(bh.clone()).or_insert_with(now);
        self.pending.insert(bh, (sb, peer));
    }

    // ---- delta txs (from our bridge or from gossip) ----------------------
    fn accept_delta(&mut self, tx: core::BackpropTx, payload: Payload) -> bool {
        let txid = tx.txid();
        if self.seen.contains(&txid) || !tx.verify() {
            return false;
        }
        let Some(coords) = payload.coords() else { return false };
        if core::delta_hash_sparse(payload.n, &coords) != tx.delta_hash {
            warn!("delta payload hash mismatch from {}", &tx.miner[..8]);
            return false;
        }
        // height gate: only admit deltas that can plausibly be mined onto head
        if !delta_in_window(tx.base_height, self.head_height()) {
            return false;
        }
        // v1 gate against the HEAD ModelState: canonical nonempty claim set,
        // every claimed page active, body zero outside claims, and the work
        // quota met — a delta that fails any of these can never be included,
        // so the node must never hold or gossip it (doomed-delta hygiene).
        {
            let model = &self.tree.model[&self.tree.head];
            let pages = tx.canonical_pages();
            if pages.is_empty() || tx.pages != pages {
                return false;
            }
            if pages.iter().any(|p| !model.is_active(*p as usize)) {
                self.quota_rejects += 1; // visible: frozen/missing-page claim
                return false;
            }
            if payload.n as u64 != model.dim() {
                return false;
            }
            let mut nnz: u64 = 0;
            let mut outside = false;
            let spans: Vec<(u64, u64)> = pages.iter()
                .map(|p| model.page_span(*p as usize)).collect();
            'coords: for &(i, x) in &coords {
                if x == 0 {
                    continue;
                }
                let i = i as u64;
                for (s, e) in &spans {
                    if i >= *s && i < *e {
                        nnz += 1;
                        continue 'coords;
                    }
                }
                outside = true;
                break;
            }
            if outside || nnz < model.required_nnz(&pages)
                || nnz > self.tree.params.delta_max_nnz {
                self.quota_rejects += 1; // below-quota / stray / over-envelope
                return false;
            }
        }
        // the tx's (body, hash) pair is now verified — connect need not
        // repeat the O(dim) streaming hash for this txid
        self.tree.hash_verified.insert(txid.clone());
        self.mark_seen(txid.clone());
        self.store.put_payload(&txid, &payload);
        // Disperse oversized bodies into erasure shards NOW, not at prune time:
        // a body over the inline gossip cap travels only by shard fetch, so the
        // shards must exist the moment the announcement goes out.
        if payload.wire_bytes() > dtx_inline_max() {
            self.store.disperse_payload(&txid, &payload);
        }
        self.payloads.insert(txid.clone(), payload);
        self.delta_pool.insert(txid, tx);
        self.evict_delta_pool();
        // rev 7: score the newcomer (and any other unscored delta) so its
        // committed score is ready by propose time. Async; never blocks.
        self.request_evals();
        true
    }

    fn accept_account_tx(&mut self, tx: AccountTx) -> Option<String> {
        use sestrian_core::token::address;
        let txid = tx.txid();
        if self.seen.contains(&txid) || !tx.verify() {
            return None;
        }
        // nonce gate: a tx below the sender's current nonce can never apply
        let cur = self.tree.head_ledger().nonces
            .get(&address(tx.sender_pub())).copied().unwrap_or(0);
        if tx.nonce() < cur {
            return None;
        }
        self.mark_seen(txid.clone());
        self.account_pool.insert(txid.clone(), tx);
        self.evict_account_pool();
        Some(txid)
    }

    // ---- block production ------------------------------------------------
    fn build_candidate(&self, attempt: u64)
        -> Option<(StoredBlock, sestrian_core::blocktree::Block)> {
        let head = self.tree.head.clone();
        let hh = self.head_height();
        let parent_model = &self.tree.model[&head];
        // rev 5: a delta that names no staked/active corpus would invalidate the
        // whole block (provenance required) — never build on one.
        let active_hashes: std::collections::BTreeSet<String> =
            self.tree.ledger[&head].registry.values()
                .filter(|e| e["status"] == "active")
                .filter_map(|e| e["data_hash"].as_str().map(|s| s.to_string()))
                .collect();
        // v1: never build on a delta the validator would reject — claims must
        // be canonical + active, and the work quota must hold against the
        // PARENT ModelState (the quota may have risen at a window boundary
        // since the delta entered the pool).
        let quota_ok = |t: &core::BackpropTx| -> bool {
            let pages = t.canonical_pages();
            if pages.is_empty() || t.pages != pages
                || pages.iter().any(|p| !parent_model.is_active(*p as usize)) {
                return false;
            }
            let Some(p) = self.payloads.get(&t.txid()) else { return false };
            let Some(val) = crate::proto::unb64(&p.val) else { return false };
            let nnz = val.chunks_exact(4)
                .filter(|c| i32::from_le_bytes((*c).try_into().unwrap()) != 0)
                .count() as u64;
            nnz >= parent_model.required_nnz(&pages)
                && nnz <= self.tree.params.delta_max_nnz
        };
        let mut cands: Vec<&core::BackpropTx> = self.delta_pool.values()
            .filter(|t| t.base_height == hh)
            .filter(|t| t.canonical_refs().iter().any(|r| active_hashes.contains(r)))
            .filter(|t| quota_ok(t))
            .collect();
        if cands.is_empty() {
            return None;
        }
        cands.sort_by_key(|t| t.txid());
        let mut chosen = Vec::new();
        let mut miners = HashSet::new();
        for t in cands {                            // one delta per miner
            if miners.insert(t.miner.clone()) {
                chosen.push((*t).clone());
                if chosen.len() >= INCLUDE_K {
                    break;
                }
            }
        }
        // weight-state transition (v1): PER-PAGE trimmed mean over each page's
        // actual claimants, chunked within pages for bounded memory — bit-
        // identical to core::paged_transition over the full dense bodies, so
        // the committed state_root reproduces on any validator.
        let parent_w = self.tree.head_state();
        let mut mean = vec![0i64; parent_w.len()];
        for (pid, page) in parent_model.pages.iter().enumerate() {
            let claimants: Vec<&Payload> = chosen.iter()
                .filter(|t| t.canonical_pages().contains(&(pid as u32)))
                .map(|t| &self.payloads[&t.txid()])
                .collect();
            if claimants.is_empty() {
                continue;
            }
            let (s, e) = (page.start as usize, page.end as usize);
            mean[s..e].copy_from_slice(
                &chunked_aggregate_range(&claimants, s, e, AGG_CHUNK));
        }
        // wrapping_add mirrors numpy int64 (matches validate_block exactly)
        let mut w: Vec<i64> = parent_w.iter().zip(&mean)
            .map(|(a, b)| a.wrapping_add(*b)).collect();
        // account lanes: dry-run in the validator's exact order (blocktree::apply)
        let mut scratch = self.tree.ledger[&head].clone();
        scratch.resolve_expired_challenges(hh + 1);
        scratch.resolve_expired_bonds(hh + 1);
        let miner_pubs: Vec<String> = chosen.iter().map(|t| t.miner.clone()).collect();
        // rev 5/6: the validator derives data_credits from the deltas' named
        // corpora and drains the fee pools — the producer must mirror it EXACTLY
        // or its own ledger_root never reproduces and every block it builds is
        // rejected (including by itself).
        let data_addrs: Vec<String> = self.tree.data_contributor.clone()
            .map(|d| vec![d]).unwrap_or_default();
        // rev 7: commit our trainer's held-out scores for the chosen deltas
        // (missing -> 0; the uniform fallback covers an all-unscored block),
        // then weight the dry-run rewards EXACTLY as the validator will.
        let blk_scores: std::collections::BTreeMap<String, u64> = chosen.iter()
            .map(|t| {
                let id = t.txid();
                let s = self.delta_scores.get(&id).copied().unwrap_or(0)
                    .min(core::blocktree::SCORE_CAP);
                (id, s)
            })
            .collect();
        let eff = core::blocktree::effective_scores(&chosen, &blk_scores);
        let active_set: std::collections::BTreeSet<String> = scratch.registry.values()
            .filter(|e| e["status"] == "active" && e["weight"].as_u64().unwrap_or(0) > 0)
            .filter_map(|e| e["data_hash"].as_str().map(String::from))
            .collect();
        // rev 8: commit our trainer's influence sketches (missing -> zeros) and
        // run the validator's registry accrual on the scratch ledger — the
        // ledger_root must reproduce EXACTLY (producer/validator asymmetry is
        // the self-rejecting-blocks bug class).
        let blk_sketches: std::collections::BTreeMap<String, Vec<i32>> = chosen.iter()
            .map(|t| {
                let id = t.txid();
                let mut sk = self.delta_sketches.get(&id).cloned()
                    .unwrap_or_else(|| vec![0; core::blocktree::SKETCH_DIM]);
                sk.resize(core::blocktree::SKETCH_DIM, 0);
                (id, sk)
            })
            .collect();
        let hash_to_key: std::collections::BTreeMap<String, String> = scratch.registry.iter()
            .filter(|(_, e)| e["status"] == "active")
            .filter_map(|(k, e)| Some((e["data_hash"].as_str()?.to_string(), k.clone())))
            .collect();
        let mut miner_weights: std::collections::BTreeMap<String, u64> = Default::default();
        let mut data_credits: std::collections::BTreeMap<String, u64> = Default::default();
        for t in &chosen {
            let txid = t.txid();
            let s = eff[&txid];
            *miner_weights.entry(t.miner.clone()).or_insert(0) += s;
            let named: Vec<String> = t.canonical_refs().into_iter()
                .filter(|r| active_set.contains(r)).collect();
            for r in &named {
                *data_credits.entry(r.clone()).or_insert(0) += s * 10_000 / named.len() as u64;
            }
            let sk = &blk_sketches[&txid];
            if sk.iter().any(|x| *x != 0) {
                let named_keys: Vec<&String> = t.canonical_refs().iter()
                    .filter_map(|r| hash_to_key.get(r)).collect::<Vec<_>>();
                let n = named_keys.len() as i128;
                if n > 0 {
                    for key in named_keys {
                        let e = scratch.registry.get_mut(key).unwrap();
                        let acc: Vec<i64> = match e.get("sketch").and_then(|v| v.as_array()) {
                            Some(a) if !a.is_empty() =>
                                a.iter().map(|x| x.as_i64().unwrap_or(0)).collect(),
                            _ => vec![0i64; core::blocktree::SKETCH_DIM],
                        };
                        let new: Vec<i64> = acc.iter().zip(sk.iter())
                            .map(|(a, x)| (*a as i128
                                 + (*x as i128 * core::blocktree::SKETCH_SCALE).div_euclid(n))
                                 .clamp(i64::MIN as i128, i64::MAX as i128) as i64)
                            .collect();
                        e["sketch"] = serde_json::json!(new);
                    }
                }
            }
        }
        scratch.apply_reward(hh + 1, &miner_pubs, &self.key.pub_hex(), &data_addrs,
                             &data_credits, &miner_weights);
        for t in &chosen {
            // bonds are 0 during bootstrap; a delta whose miner cannot afford its
            // bond would make the block invalid, so it must not be built on.
            if !scratch.lock_bond(&t.txid(), &core::token::address(&t.miner), t.bond, hh + 1) {
                warn!("candidate delta {} bond unaffordable; block would be invalid",
                      &t.txid()[..8]);
                return None;
            }
        }
        let jurors = self.tree.recent_proposers(&head);
        let mut transfers = Vec::new();
        let mut data_txs = Vec::new();
        let pool: Vec<AccountTx> = self.account_pool.values().cloned().collect();
        for t in sestrian_core::token::canonical_account_txs(&pool) {
            let ok = match &t {
                AccountTx::Transfer(x) => scratch.apply_transfer(x),
                _ => scratch.apply_data_tx(&t, hh + 1, &jurors),
            };
            if ok {
                match &t {
                    AccountTx::Transfer(_) =>
                        transfers.push(account_tx_to_json(&t)),
                    _ => data_txs.push(account_tx_to_json(&t)),
                }
            }
        }
        let core_transfers: Vec<_> = transfers.iter()
            .filter_map(|v| match account_tx_from_json(v) {
                Some(AccountTx::Transfer(x)) => Some(x),
                _ => None,
            }).collect();
        let core_data: Vec<AccountTx> = data_txs.iter()
            .filter_map(account_tx_from_json).collect();
        // v1 MODEL FOLD: the ModelState transition, then any growth activation
        // due THIS block appends its deterministically-initialized expert page
        // AFTER aggregation and BEFORE the root — the exact validate_block
        // order (one ordering slip here forks the chain at the first growth).
        let zero_scored = chosen.iter()
            .filter(|t| blk_scores[&t.txid()] == 0).count() as u64;
        let score_sum: u64 = chosen.iter()
            .map(|t| blk_scores[&t.txid()]).sum();
        let (post_model, activations) = core::model_state::fold(
            parent_model, &self.tree.params, hh + 1,
            chosen.len() as u64, zero_scored, &head, score_sum);
        for (page_id, layer, _expert, trigger) in &activations {
            info!(height = hh + 1, page_id, layer,
                  "GROWTH EVENT activates in our candidate block");
            w.extend(core::model_state::page_init(
                trigger, *page_id, &self.tree.params.spec));
        }
        // proposer lottery (v1): the proof binds to (height, ATTEMPT); work is
        // the attempt-discounted non-forgeable weight derived from it.
        let vrf_proof = core::lottery::vrf_prove(&self.key, &head, hh + 1, attempt);
        let header = core::Header {
            height: hh + 1,
            prev_hash: head.clone(),
            state_root: core::model_state::page_state_root(&w, &post_model),
            txset_root: core::txset_root(
                &chosen.iter().map(|t| t.txid()).collect::<Vec<_>>()),
            n_txs: chosen.len() as u64,
            work: core::lottery::attempt_work(&vrf_proof, attempt),
            proposer: self.key.pub_hex(),
            transfer_root: sestrian_core::token::transfer_root(&core_transfers),
            ledger_root: scratch.root(),
            data_root: sestrian_core::token::data_root(&core_data),
            vrf_proof: hex::encode(&vrf_proof),
            score_root: core::blocktree::scores_root(&blk_scores),
            sketch_root: core::blocktree::sketch_root(&blk_sketches),
            model_root: post_model.model_root(),
            vrf_attempt: attempt,
            version: core::expected_version_at(hh + 1, &self.tree.params),
        };
        let stored = StoredBlock {
            header: WireHeader::from_core(&header),
            txs: chosen.iter().map(WireDeltaTx::from_core).collect(),
            transfers,
            data_txs,
            scores: blk_scores.clone(),
            sketches: blk_sketches.clone(),
        };
        let mut bodies = HashMap::new();
        for t in chosen.iter() {
            bodies.insert(t.da_pointer.clone(), self.payloads[&t.txid()].dense().unwrap());
        }
        let block = sestrian_core::blocktree::Block {
            header, txs: chosen, bodies,
            sparse: Default::default(),
            transfers: core_transfers, data_txs: core_data,
            scores: blk_scores, sketches: blk_sketches,
        };
        Some((stored, block))
    }

    // ---- installation ----------------------------------------------------
    /// Try to install a stored block (bodies from the payload store). Returns
    /// true if installed; queues it as pending when payloads are missing.
    fn install(&mut self, sb: StoredBlock, from: Option<PeerId>) -> bool {
        let bh = sb.hash();
        if self.blocks_full.contains_key(&bh) {
            return false;
        }
        // Bodies live on DISK between fetch and apply; RAM holds only what is
        // actively validating. During the quota-fork heal ~80 fetched 92MB
        // bodies accumulated in this map while their blocks waited for
        // ancestors, and the OOM killer took an anchor down. Preload from the
        // store here; the post-apply eviction below returns them to disk-only.
        for t in &sb.txs {
            if let Some(tc) = t.to_core() {
                let id = tc.txid();
                if !self.payloads.contains_key(&id) {
                    if let Some(p) = self.store.get_payload(&id) {
                        self.payloads.insert(id, p);
                    }
                }
            }
        }
        // SPARSE bodies straight from payloads — the incremental engine's
        // native input; no dense materialization on the install path
        let Some(block) = sb.to_core_sparse(&self.payloads) else {
            // body missing: try to reconstruct it from erasure shards we already
            // hold, and if we can't, ask ALL peers for its shards (any one may
            // hold a few) in parallel with the full-block sync fallback.
            let missing: Vec<String> = sb.txs.iter()
                .filter_map(|t| t.to_core().map(|tc| tc.txid()))
                .filter(|id| {
                    if self.payloads.contains_key(id) || self.store.has_payload(id) {
                        return false;
                    }
                    if let Some(p) = self.store.reconstruct_payload(id) {
                        self.payloads.insert(id.clone(), p); // recovered locally
                        return false;
                    }
                    true
                })
                .collect();
            if !missing.is_empty() {
                if let Some(peer) = from {
                    let _ = self.net.send(ToNet::SendShards(peer, missing.clone()));
                }
                // the round refetch asks every connected peer; this is the
                // eager first ask of the peer that sent us the block
            }
            if let Some(peer) = from {
                let fh = sb.header.height.saturating_sub(1);
                let _ = self.net.send(ToNet::SendSync(peer, fh));
                self.queue_pending(bh, sb, peer);
            }
            return false;
        };
        let old_head = self.tree.head.clone();
        match self.tree.add_block(block) {
            Ok(_) => {
                // DURABILITY: a dropped block write silently truncates the chain
                // on the next boot, so a persist failure is fatal — halt loudly
                // and let the operator fix the disk and restart (replay recovers
                // the last durably-persisted head).
                if let Err(e) = self.store.append_block(&sb) {
                    error!("FATAL: cannot persist block h{}: {e}; halting to avoid \
                            silent chain truncation", sb.header.height);
                    std::process::exit(1);
                }
                // #114 (observable half): a VALIDATED foreign block that omits
                // deltas we had already gossiped for its height is censorship-
                // suspicious. Count per proposer — observability, NOT consensus.
                if from.is_some() && sb.header.proposer != self.key.pub_hex() {
                    let included: HashSet<String> = sb.txs.iter()
                        .filter_map(|t| t.to_core().map(|tc| tc.txid())).collect();
                    let omitted = self.delta_pool.iter()
                        .filter(|(id, t)| t.base_height + 1 == sb.header.height
                                && !included.contains(*id))
                        .count() as u64;
                    if omitted > 0 {
                        *self.omitted_deltas
                            .entry(sb.header.proposer.clone()).or_insert(0) += omitted;
                    }
                }
                for t in &sb.txs {
                    if let Some(tc) = t.to_core() {
                        let id = tc.txid();
                        self.delta_pool.remove(&id);
                        self.delta_scores.remove(&id);
                        self.delta_sketches.remove(&id);
                        // drop the in-memory payload only once it is confirmed on
                        // disk (the block references it; sync/replay read it back).
                        if let Some(p) = self.payloads.get(&id) {
                            if self.store.put_payload(&id, p) {
                                self.payloads.remove(&id);
                            }
                        }
                    }
                }
                for v in sb.transfers.iter().chain(sb.data_txs.iter()) {
                    if let Some(t) = account_tx_from_json(v) {
                        self.account_pool.remove(&t.txid());
                        self.mark_seen(t.txid());
                    }
                }
                self.blocks_full.insert(bh.clone(), sb);
                if self.tree.head != old_head {
                    self.on_head_advance(&old_head);
                }
                true
            }
            Err(e) => {
                if e.0.contains("orphan") {
                    if let Some(peer) = from {
                        let _ = self.net.send(ToNet::SendSync(
                            peer, self.head_height().saturating_sub(8)));
                        self.queue_pending(bh, sb, peer);
                    }
                } else {
                    warn!("invalid block h{}: {}", sb.header.height, e.0);
                }
                false
            }
        }
    }

    fn retry_pending(&mut self) {
        let ready: Vec<String> = self.pending.iter()
            .filter(|(_, (sb, _))| {
                sb.txs.iter().all(|t| t.to_core()
                    .map(|tc| { let id = tc.txid();
                        self.payloads.contains_key(&id)
                            || self.store.has_payload(&id) })
                    .unwrap_or(false))
                && self.tree.blocks.contains_key(&sb.header.prev_hash)
            })
            .map(|(h, _)| h.clone()).collect();
        for h in ready {
            if let Some((sb, peer)) = self.pending.remove(&h) {
                self.pending_at.remove(&h);
                self.install(sb, Some(peer));
            }
        }
    }

    fn on_head_advance(&mut self, old_head: &str) {
        let h = self.head_height();
        info!(height = h, head = &self.tree.head[..10],
              supply = self.tree.head_ledger().supply(), "head advanced");
        // the net loop announces our head each round — keep its cache honest
        let _ = self.net.send(ToNet::HeadInfo {
            hash: self.tree.head.clone(), height: h });
        // keep the bridge synced with a sparse state diff — or, across a v1
        // GROWTH boundary, an explicit Grow message. (The old zip silently
        // TRUNCATED on a length change, which would have desynced the trainer
        // exactly at the first growth event.)
        if self.bridge_synced {
            // the incremental tree hands us the exact sparse delta the head
            // block applied (its REDO log) plus any appended growth tail —
            // no dense diff of two 860MB vectors ever again.
            let sequential = self.tree.blocks.get(&self.tree.head)
                .map(|hh| hh.prev_hash == *old_head).unwrap_or(false);
            let applied = self.tree.applied_delta(&self.tree.head).cloned();
            let tail = self.tree.appended_tail(&self.tree.head)
                .map(|t| t.to_vec()).unwrap_or_default();
            match (sequential, applied) {
                (true, Some(coords)) if tail.is_empty() => {
                    let dim = self.tree.head_state().len() as u64;
                    let sparse = crate::proto::Payload::from_coords_i64(
                        dim as usize, &coords);
                    let _ = self.bridge_tx.try_send(ToBridge::Advance {
                        height: h, dim, sparse });
                }
                (true, Some(coords)) => {
                    let new_model = &self.tree.model[&self.tree.head];
                    let new_len = self.tree.head_state().len();
                    let old_len = new_len - tail.len();
                    let grew_one = tail.len() > 0
                        && new_model.pages.last()
                            .map(|p| (p.end - p.start) as usize == tail.len())
                            .unwrap_or(false);
                    if grew_one {
                        let sparse = crate::proto::Payload::from_coords_i64(
                            old_len, &coords);
                        let _ = self.bridge_tx.try_send(ToBridge::Advance {
                            height: h, dim: old_len as u64, sparse });
                        let page = new_model.pages.last().unwrap();
                        info!(height = h, page_id = new_model.pages.len() - 1,
                              layer = page.layer,
                              "GROWTH: syncing the new expert page to the trainer");
                        let _ = self.bridge_tx.try_send(ToBridge::Grow {
                            height: h,
                            new_dim: new_len as u64,
                            page_id: (new_model.pages.len() - 1) as u64,
                            layer: page.layer,
                            expert: page.expert,
                            init: tail,
                        });
                    } else {
                        self.send_bridge_state();
                    }
                }
                _ => {
                    // reorg or unknown diff — bridge resyncs from scratch
                    self.send_bridge_state();
                }
            }
        }
        if h % SNAPSHOT_EVERY == 0 {
            self.store.write_snapshot(&self.tree.head, h,
                                      self.tree.head_state(),
                                      self.tree.head_ledger(),
                                      &self.tree.model[&self.tree.head]);
        }
        // the head moved: prune mempools + pending against it
        self.evict_delta_pool();
        self.evict_account_pool();
        // rev 7: (re)score surviving pool deltas against the new head's round
        self.request_evals();
        // age-based only: rival-fork blocks legitimately sit far below head
        let drop_pending: Vec<String> = self.pending.keys()
            .filter(|k| self.pending_at.get(*k)
                .map(|t| now() - t > 1800.0).unwrap_or(true))
            .cloned().collect();
        for k in drop_pending {
            self.pending.remove(&k);
            self.pending_at.remove(&k);
        }
        self.prune_old_bodies(h);
    }

    /// The DA shard indices THIS node retains for old bodies — a deterministic,
    /// per-node window of K+2 shards (any node self-reconstructs, and a few
    /// nodes' windows cover all N so a peer with none can gather K from others).
    fn my_shard_zone(&self) -> Vec<u32> {
        let hh = core::delta_hash(self.key.pub_hex().as_bytes());
        let off = usize::from_str_radix(&hh[..8], 16).unwrap_or(0) % Store::DA_N;
        (0..(Store::DA_K + 2)).map(|j| ((off + j) % Store::DA_N) as u32).collect()
    }

    /// Once a block leaves the body-retention window, drop its MONOLITHIC bodies
    /// and keep only this node's shard zone — roughly halving old-body storage
    /// while every body stays reconstructable (locally from K+2 shards, or from
    /// peers via the shard exchange). Prunes exactly the block at the frontier.
    fn prune_old_bodies(&mut self, head_h: u64) {
        // PRUNED-NODE RETENTION (opt-in via --da-retain-blocks): once a block
        // is deeper than the window, delete its bodies' shards entirely.
        // Shard-zone retention is linear disk growth and filled a founder
        // machine within a day of the first quota rise. Archive nodes
        // (retain = 0, the anchors) keep serving deep history to joiners.
        if self.cfg.da_retain_blocks > 0 {
            let floor = head_h.saturating_sub(self.cfg.da_retain_blocks);
            if floor > self.da_pruned_to {
                let doomed: Vec<String> = self.blocks_full.values()
                    .filter(|sb| sb.header.height > self.da_pruned_to
                            && sb.header.height <= floor)
                    .flat_map(|sb| sb.txs.iter())
                    .filter_map(|t| t.to_core().map(|tc| tc.txid()))
                    .collect();
                if !doomed.is_empty() {
                    info!(bodies = doomed.len(), floor,
                          "DA retention: deleting shard sets beyond the window");
                }
                for txid in doomed {
                    self.store.delete_body_and_shards(&txid);
                    self.payloads.remove(&txid);
                }
                self.da_pruned_to = floor;
            }
        }
        const BODY_WINDOW: u64 = 16;
        let frontier = match head_h.checked_sub(BODY_WINDOW + 1) {
            Some(f) if f > 0 => f,
            _ => return,
        };
        let mut cur = self.tree.head.clone();
        let keep = self.my_shard_zone();
        while let Some(hdr) = self.tree.blocks.get(&cur) {
            if hdr.height < frontier {
                break;
            }
            if hdr.height == frontier {
                if let Some(sb) = self.blocks_full.get(&cur).cloned() {
                    for t in &sb.txs {
                        if let Some(tc) = t.to_core() {
                            self.store.prune_body_to_shards(&tc.txid(), &keep);
                        }
                    }
                }
                break;
            }
            cur = hdr.prev_hash.clone();
        }
    }

    fn send_bridge_state(&mut self) {
        let h = self.head_height();
        let state = self.tree.head_state().clone();
        let model = &self.tree.model[&self.tree.head];
        // v1: per-layer expert counts so the trainer builds the exact (possibly
        // grown, ragged) architecture before loading the chain-order state.
        let n_layers = self.tree.params.spec.n_layers;
        let mut epl = vec![0u64; n_layers as usize];
        for p in &model.pages {
            if p.kind == "expert" && p.layer >= 0 {
                epl[p.layer as usize] += 1;
            }
        }
        if self.bridge_tx.try_send(ToBridge::State {
            height: h, state, experts_per_layer: epl,
        }).is_ok() {
            self.bridge_synced = true;
        }
    }

    /// v1 producer sortition: the lowest attempt (0..=max_allowed, widening
    /// with time inside the round) at which OUR key is eligible for the next
    /// height — None if none yet. Deterministic per (head, height, attempt).
    fn eligible_attempt(&self, max_allowed: u64) -> Option<u64> {
        let led = self.tree.head_ledger();
        let stake = led.balance(&core::token::address(&self.key.pub_hex()));
        let total = led.supply();
        let h = self.head_height() + 1;
        for a in 0..=max_allowed.min(core::lottery::ATTEMPT_MAX) {
            let proof = core::lottery::vrf_prove(&self.key, &self.tree.head, h, a);
            if core::lottery::eligible(&self.key.pub_hex(), &proof,
                                       &self.tree.head, h, a, stake, total) {
                return Some(a);
            }
        }
        None
    }

    // ---- api -------------------------------------------------------------
    fn api_status(&self) -> Value {
        let led = self.tree.head_ledger();
        json!({
            "height": self.head_height(),
            "head": &self.tree.head[..16],
            "supply": led.supply(),
            "delta_pool": self.delta_pool.len(),
            "account_pool": self.account_pool.len(),
            "pending_blocks": self.pending.len(),
            "producer": self.cfg.produce,
            "miner": self.key.pub_hex(),
            "peers": self.peers_connected,
            "model_attached": self.bridge_synced,
            // >0 means this node is training but its deltas keep missing the
            // block window — mining for nothing. Visible, not silent.
            "stale_deltas": self.stale_deltas,
            "quota_rejects": self.quota_rejects,
            // v1 capacity telemetry: the model's shape + controller state, so
            // the retarget is observable ("the network grew its brain at block
            // N" must be visible before it's ever exciting).
            "model": {
                "model_root": self.tree.model[&self.tree.head].model_root(),
                "dim": self.tree.model[&self.tree.head].dim(),
                "pages_total": self.tree.model[&self.tree.head].pages.len(),
                "expert_pages": self.tree.model[&self.tree.head].n_expert_pages(),
                "expert_pages_active":
                    self.tree.model[&self.tree.head].n_active_expert_pages(),
                "quota_4dp": self.tree.model[&self.tree.head].quota_4dp,
                "window_id": self.tree.model[&self.tree.head].window_id,
                "pending_growth":
                    self.tree.model[&self.tree.head].pending_growth.len(),
                "growth_events": self.tree.model[&self.tree.head].events_total,
            },
        })
    }

    /// Prometheus text-format snapshot of node health for scraping/alerting.
    fn api_metrics(&self) -> String {
        let led = self.tree.head_ledger();
        let g = |help: &str, name: &str, v: u64| {
            format!("# HELP sestrian_{name} {help}\n# TYPE sestrian_{name} gauge\n\
                     sestrian_{name} {v}\n")
        };
        [
            g("chain head height", "height", self.head_height()),
            g("connected peers", "peers", self.peers_connected as u64),
            g("delta mempool size", "delta_pool", self.delta_pool.len() as u64),
            g("account mempool size", "account_pool", self.account_pool.len() as u64),
            g("orphan blocks awaiting parents/payloads", "pending_blocks",
              self.pending.len() as u64),
            g("dedup set size", "seen", self.seen.len() as u64),
            g("total token supply (grains)", "supply_grains", led.supply()),
            g("1 if this node produces blocks", "producer", self.cfg.produce as u64),
            g("1 if a training bridge is attached", "model_attached",
              self.bridge_synced as u64),
            g("consecutive deltas dropped as stale (trainer slower than the \
               block interval — alert on this: the miner earns nothing)",
              "stale_deltas", self.stale_deltas),
            g("1 if a training round is in flight", "train_inflight",
              self.train_inflight as u64),
            g("pool deltas scored by our trainer", "scored_deltas",
              self.delta_scores.len() as u64),
            g("gossiped deltas omitted from foreign blocks (censorship \
               suspicion; #114 observable half)", "omitted_deltas_total",
              self.omitted_deltas.values().sum::<u64>()),
            g("deltas rejected for page-claim/quota violations (v1)",
              "quota_rejects", self.quota_rejects),
            g("model dimension (total parameters)", "model_dim",
              self.tree.model[&self.tree.head].dim()),
            g("total pages in the model's page table", "model_pages",
              self.tree.model[&self.tree.head].pages.len() as u64),
            g("ACTIVE expert pages (frozen ones serve but reject deltas)",
              "model_expert_pages_active",
              self.tree.model[&self.tree.head].n_active_expert_pages() as u64),
            g("capacity work quota in 1e-4 units (10000 = 1.0)",
              "capacity_quota_4dp",
              self.tree.model[&self.tree.head].quota_4dp.max(0) as u64),
            g("retarget window id", "capacity_window",
              self.tree.model[&self.tree.head].window_id),
            g("growth events scheduled on this chain (ratchet count)",
              "capacity_growth_events",
              self.tree.model[&self.tree.head].events_total),
            g("growth events announced, awaiting activation",
              "capacity_pending_growth",
              self.tree.model[&self.tree.head].pending_growth.len() as u64),
        ].concat()
    }

    fn api_balance(&self, addr: &str) -> Value {
        let led = self.tree.head_ledger();
        json!({"addr": addr, "grains": led.balance(addr),
               "nonce": led.nonces.get(addr).copied().unwrap_or(0),
               "supply": led.supply(), "height": self.head_height()})
    }

    fn api_registry(&self) -> Value {
        let led = self.tree.head_ledger();
        json!({"registry": led.registry, "challenges": led.challenges})
    }

    fn api_miners(&self) -> Value {
        // work accounting straight from chain history: for every miner ever
        // seen, blocks proposed, deltas contributed, tokens earned, last height
        use sestrian_core::token::address;
        let mut stats: HashMap<String, (u64, u64, u64)> = HashMap::new(); // pub -> (proposed, deltas, last_h)
        for sb in self.blocks_full.values() {
            let h = sb.header.height;
            if sb.header.proposer != "genesis" {
                let e = stats.entry(sb.header.proposer.clone()).or_default();
                e.0 += 1;
                e.2 = e.2.max(h);
            }
            for t in &sb.txs {
                let e = stats.entry(t.miner.clone()).or_default();
                e.1 += 1;
                e.2 = e.2.max(h);
            }
        }
        let led = self.tree.head_ledger();
        let total_blocks = self.head_height().max(1);
        let mut miners: Vec<Value> = stats.into_iter().map(|(pub_hex, (p, d, lh))| {
            let addr = address(&pub_hex);
            json!({"miner": pub_hex, "address": addr,
                   "blocks_proposed": p, "deltas": d, "last_height": lh,
                   "balance": led.balance(&addr),
                   "share_pct": (p as f64 * 100.0 / total_blocks as f64).round(),
                   "is_me": pub_hex == self.key.pub_hex()})
        }).collect();
        miners.sort_by_key(|m| std::cmp::Reverse(m["blocks_proposed"].as_u64().unwrap_or(0)));
        json!({"miners": miners, "peers_connected": self.peers_connected,
               "head_height": self.head_height(),
               // #114 observable half: per-proposer omission counts (deltas we
               // gossiped that their blocks left out) — censorship suspicion,
               // not consensus.
               "omissions": self.omitted_deltas})
    }

    fn api_upload(&mut self, bytes: Vec<u8>, stake: u64, media: String) -> (Value, Option<Gossip>) {
        use sestrian_core::token::{address, AccountTx, DataSubmitTx};
        if bytes.is_empty() {
            return (json!({"ok": false, "error": "empty file"}), None);
        }
        let hash = core::delta_hash(&bytes);
        // check the balance BEFORE writing attacker-supplied bytes to disk, so a
        // caller can't fill the disk with unfunded uploads.
        let led = self.tree.head_ledger();
        let my_addr = address(&self.key.pub_hex());
        if led.balance(&my_addr) < stake {
            return (json!({"ok": false,
                "error": format!("node wallet balance {} < stake {}",
                                 led.balance(&my_addr), stake),
                "data_hash": hash,
                "hint": "fund the node wallet, or submit on-chain from a funded \
                         wallet with: wallet submit-data"}), None);
        }
        if let Err(e) = self.store.save_upload(&hash, &bytes) {
            return (json!({"ok": false, "error": format!("store: {e}")}), None);
        }
        let mut tx = DataSubmitTx {
            owner_pub: self.key.pub_hex(),
            data_hash: hash.clone(),
            size_bytes: bytes.len() as u64,
            media_type: media,
            stake,
            nonce: *led.nonces.get(&my_addr).unwrap_or(&0),
            sig: vec![],
        };
        tx.sig = self.key.sign(&AccountTx::DataSubmit(tx.clone()).signing_bytes());
        let atx = AccountTx::DataSubmit(tx);
        match self.accept_account_tx(atx.clone()) {
            Some(txid) => (json!({"ok": true, "txid": txid, "data_hash": hash,
                                  "bytes": bytes.len(),
                                  "status": "custodied + staked submission in mempool"}),
                           Some(Gossip::Atx { tx: account_tx_to_json(&atx) })),
            None => (json!({"ok": false, "error": "tx rejected (duplicate?)",
                            "data_hash": hash}), None),
        }
    }

    fn api_chain(&self) -> Value {
        // the last 16 headers along the head lineage, oldest first
        let mut out = Vec::new();
        let mut cur = self.tree.head.clone();
        for _ in 0..16 {
            if cur == self.tree.genesis_hash {
                break;
            }
            let h = &self.tree.blocks[&cur];
            out.push(json!({"height": h.height, "hash": cur,
                            "proposer": h.proposer, "n_txs": h.n_txs,
                            "work": h.work}));
            cur = h.prev_hash.clone();
        }
        out.reverse();
        json!({"blocks": out})
    }
}

// ---------------------------------------------------------------------------
// The main loop
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
/// THE ACTOR SPLIT. Two tasks, two owners, typed channels between them:
///
///  * the NET loop owns the swarm and nothing else — per-peer in-flight gates,
///    dial/recycle, gossip mechanics, a cached head for announcements. Its
///    worst-case pause is microseconds: it validates nothing, hashes nothing,
///    serves nothing itself.
///  * the CHAIN actor owns the tree, the store, the mempool, the trainer
///    bridge and the API — every stateful decision. A stall here (snapshot
///    fsync, a dense proposer aggregation) no longer starves the transport,
///    which is what turned every big-payload incident into a fork.
///
/// The chain actor asks for network sends via ToNet; the net loop annotates
/// inbound traffic with the transport facts only it knows (request identity,
/// mesh silence) via ToChain.
pub async fn run(
    mut node: Node,
    mut swarm: Swarm<Behaviour>,
    api_rx: mpsc::Receiver<ApiCmd>,
    bridge_rx: mpsc::Receiver<FromBridge>,
) {
    use futures::stream::FuturesUnordered;

    let end = if node.cfg.seconds > 0.0 { now() + node.cfg.seconds } else { f64::MAX };
    let jitter: f64 = rand::random::<f64>() * 0.5;
    let interval = node.cfg.interval;
    let t0 = node.t0;
    let cfg_peers = node.cfg.peers.clone();
    let topic = node.topic.clone();

    let (net_tx, mut net_rx) = tokio::sync::mpsc::unbounded_channel::<ToNet>();
    let (chain_tx, chain_rx) = tokio::sync::mpsc::unbounded_channel::<ToChain>();
    node.net = net_tx;
    let initial_head = (node.tree.head.clone(), node.head_height());

    // ---- the chain actor ----
    let chain = tokio::spawn(run_chain(node, chain_rx, api_rx, bridge_rx));

    // ---- net-side state ----
    let mut cached_head = initial_head;
    let mut last_announced_round: i64 = -1;
    let mut last_foreign_head = now();
    let mut silent_rounds: u64 = 0;
    let mut last_sync_req: HashMap<PeerId, (f64, u64)> = HashMap::new();
    let mut sync_req_id: HashMap<PeerId, request_response::OutboundRequestId> =
        HashMap::new();
    let mut last_shard_req: HashMap<PeerId, f64> = HashMap::new();
    let mut shard_req_id: HashMap<PeerId, request_response::OutboundRequestId> =
        HashMap::new();

    enum Reply {
        Sync(request_response::ResponseChannel<SyncResponse>, Option<SyncResponse>),
        Shards(request_response::ResponseChannel<ShardResponse>, Option<ShardResponse>),
    }
    let mut replies: FuturesUnordered<
        std::pin::Pin<Box<dyn std::future::Future<Output = Reply> + Send>>,
    > = FuturesUnordered::new();

    let mut tick = tokio::time::interval(Duration::from_millis(400));
    #[cfg(unix)]
    let mut stop_signal = tokio::signal::unix::signal(
        tokio::signal::unix::SignalKind::terminate())
        .expect("install SIGTERM handler");
    #[cfg(windows)]
    let mut stop_signal = tokio::signal::windows::ctrl_shutdown()
        .expect("install CTRL_SHUTDOWN handler");

    let send_sync = |swarm: &mut Swarm<Behaviour>,
                     last_sync_req: &mut HashMap<PeerId, (f64, u64)>,
                     sync_req_id: &mut HashMap<PeerId, request_response::OutboundRequestId>,
                     peer: PeerId, from: u64| {
        let inflight = last_sync_req.get(&peer)
            .map(|(t, _)| now() - t < SYNC_INFLIGHT_TIMEOUT)
            .unwrap_or(false);
        if !inflight {
            last_sync_req.insert(peer, (now(), from));
            let rid = swarm.behaviour_mut().sync.send_request(&peer, SyncRequest {
                from_height: from, count: SYNC_MAX_BLOCKS as u64,
                want_genesis: false });
            sync_req_id.insert(peer, rid);
        }
    };

    loop {
        if now() >= end {
            break;
        }
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("SIGINT — shutting down");
                break;
            }
            _ = stop_signal.recv() => {
                info!("terminate signal — shutting down");
                break;
            }
            Some(reply) = replies.next() => match reply {
                Reply::Sync(ch, Some(resp)) => {
                    let _ = swarm.behaviour_mut().sync.send_response(ch, resp);
                }
                Reply::Shards(ch, Some(resp)) => {
                    let _ = swarm.behaviour_mut().shards.send_response(ch, resp);
                }
                _ => {} // chain actor gone — shutting down
            },
            Some(cmd) = net_rx.recv() => match cmd {
                ToNet::Publish(msg) => {
                    let bytes = serde_json::to_vec(&msg).unwrap();
                    if let Err(e) = swarm.behaviour_mut().gossipsub
                        .publish(topic.clone(), bytes) {
                        debug!("publish: {e}");
                    }
                }
                ToNet::SendSync(peer, from) => {
                    send_sync(&mut swarm, &mut last_sync_req, &mut sync_req_id,
                              peer, from);
                }
                ToNet::SendShards(peer, txids) => {
                    let busy = last_shard_req.get(&peer)
                        .map(|t| now() - t < 120.0).unwrap_or(false);
                    if !busy && !txids.is_empty() {
                        last_shard_req.insert(peer, now());
                        let rid = swarm.behaviour_mut().shards
                            .send_request(&peer, ShardRequest { txids });
                        shard_req_id.insert(peer, rid);
                    }
                }
                ToNet::HeadInfo { hash, height } => {
                    cached_head = (hash, height);
                }
            },
            _ = tick.tick() => {
                let round = ((now() - t0 - jitter) / interval).floor() as i64;
                let elapsed_in_round = (now() - t0 - jitter) - round as f64 * interval;
                if round >= 0 && round != last_announced_round {
                    last_announced_round = round;
                    // the self-healing heartbeat: announce our head every round
                    let bytes = serde_json::to_vec(&Gossip::Head {
                        hash: cached_head.0.clone(),
                        height: cached_head.1,
                    }).unwrap();
                    let _ = swarm.behaviour_mut().gossipsub
                        .publish(topic.clone(), bytes);
                    // …and re-dial configured peers when connections are lost
                    let expected = cfg_peers.split(',')
                        .filter(|s| !s.is_empty()).count();
                    if swarm.network_info().num_peers() < expected {
                        dial_peers(&mut swarm, &cfg_peers);
                    }
                    // MESH-BLINDNESS + STALE-TRANSPORT RECYCLING (net-owned:
                    // it is a statement about connections, not the chain)
                    let connected: Vec<PeerId> =
                        swarm.connected_peers().copied().collect();
                    let mut mesh_blind = false;
                    if !connected.is_empty()
                        && now() - last_foreign_head > 2.0 * interval {
                        silent_rounds += 1;
                        if silent_rounds >= 3 && !cfg_peers.is_empty() {
                            warn!(rounds = silent_rounds,
                                  "no foreign heads for 3+ rounds — recycling \
                                   peer connections (stale-transport heal)");
                            for p in &connected {
                                let _ = swarm.disconnect_peer_id(*p);
                            }
                            last_sync_req.clear();
                            sync_req_id.clear();
                            silent_rounds = 0;
                            dial_peers(&mut swarm, &cfg_peers);
                        } else {
                            mesh_blind = true;
                        }
                    } else if now() - last_foreign_head <= 2.0 * interval {
                        silent_rounds = 0;
                    }
                    let _ = chain_tx.send(ToChain::RoundTick {
                        round, elapsed_in_round,
                        num_peers: swarm.network_info().num_peers(),
                        connected: connected.clone(),
                        mesh_blind,
                    });
                } else if round >= 0 {
                    // sub-round ticks still drive the chain (training watchdog,
                    // proposal politeness ladder)
                    let _ = chain_tx.send(ToChain::RoundTick {
                        round, elapsed_in_round,
                        num_peers: swarm.network_info().num_peers(),
                        connected: swarm.connected_peers().copied().collect(),
                        mesh_blind: false,
                    });
                }
            }
            event = swarm.select_next_some() => match event {
                SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(
                        gossipsub::Event::Message { propagation_source, message, .. })) => {
                    if let Ok(msg) = serde_json::from_slice::<Gossip>(&message.data) {
                        match msg {
                            Gossip::Dtx { tx, payload } => {
                                let _ = chain_tx.send(ToChain::Dtx(
                                    tx, payload, Some(propagation_source)));
                            }
                            Gossip::Atx { tx } => {
                                let _ = chain_tx.send(ToChain::Atx(tx));
                            }
                            Gossip::Blk { block } => {
                                let _ = chain_tx.send(ToChain::Blk(
                                    block, Some(propagation_source)));
                            }
                            Gossip::Head { hash, height } => {
                                last_foreign_head = now();
                                silent_rounds = 0;
                                let _ = chain_tx.send(ToChain::Head {
                                    peer: propagation_source, hash, height });
                            }
                        }
                    }
                }
                SwarmEvent::Behaviour(BehaviourEvent::Sync(
                        request_response::Event::OutboundFailure {
                            peer, error, request_id, .. })) => {
                    if sync_req_id.get(&peer) == Some(&request_id) {
                        warn!(%peer, %error, "sync request failed");
                        sync_req_id.remove(&peer);
                        last_sync_req.remove(&peer);
                    }
                }
                SwarmEvent::Behaviour(BehaviourEvent::Sync(
                        request_response::Event::InboundFailure { peer, error, .. })) => {
                    warn!(%peer, %error, "sync response delivery failed");
                }
                SwarmEvent::Behaviour(BehaviourEvent::Sync(
                        request_response::Event::Message { peer, message, .. })) => {
                    match message {
                        request_response::Message::Request { request, channel, .. } => {
                            let (otx, orx) = tokio::sync::oneshot::channel();
                            let _ = chain_tx.send(ToChain::SyncServe(request, otx));
                            replies.push(Box::pin(async move {
                                Reply::Sync(channel, orx.await.ok())
                            }));
                        }
                        request_response::Message::Response {
                                response, request_id, .. } => {
                            let current = sync_req_id.get(&peer)
                                == Some(&request_id);
                            let from = if current {
                                sync_req_id.remove(&peer);
                                last_sync_req.remove(&peer).map(|(_, f)| f)
                                    .unwrap_or(0)
                            } else { 0 };
                            let _ = chain_tx.send(ToChain::SyncBatch {
                                peer, current, from,
                                blocks: response.blocks,
                                payloads: response.payloads,
                                their_head: response.head_height,
                            });
                        }
                    }
                }
                SwarmEvent::Behaviour(BehaviourEvent::Shards(
                        request_response::Event::OutboundFailure {
                            peer, error, request_id, .. })) => {
                    if shard_req_id.get(&peer) == Some(&request_id) {
                        warn!(%peer, %error, "shard request failed");
                        shard_req_id.remove(&peer);
                        last_shard_req.remove(&peer);
                    }
                }
                SwarmEvent::Behaviour(BehaviourEvent::Shards(
                        request_response::Event::InboundFailure { peer, error, .. })) => {
                    warn!(%peer, %error, "shard response delivery failed");
                }
                SwarmEvent::Behaviour(BehaviourEvent::Shards(
                        request_response::Event::Message { peer, message, .. })) => {
                    match message {
                        request_response::Message::Request { request, channel, .. } => {
                            let (otx, orx) = tokio::sync::oneshot::channel();
                            let _ = chain_tx.send(ToChain::ShardServe(request, otx));
                            replies.push(Box::pin(async move {
                                Reply::Shards(channel, orx.await.ok())
                            }));
                        }
                        request_response::Message::Response {
                                response, request_id, .. } => {
                            let current = shard_req_id.get(&peer)
                                == Some(&request_id);
                            if current {
                                shard_req_id.remove(&peer);
                                last_shard_req.remove(&peer);
                            }
                            let _ = chain_tx.send(ToChain::ShardBatch {
                                peer, current, bodies: response.bodies });
                        }
                    }
                }
                SwarmEvent::NewListenAddr { address, .. } => {
                    info!(%address, "listening");
                }
                SwarmEvent::Behaviour(BehaviourEvent::Identify(
                        identify::Event::Received { peer_id, info, .. })) => {
                    debug!(%peer_id, agent = %info.agent_version, "peer identified");
                }
                SwarmEvent::ConnectionClosed { peer_id, cause, num_established, .. } => {
                    info!(%peer_id, remaining = num_established,
                          cause = ?cause, "peer connection closed");
                }
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    info!(%peer_id, "peer connected");
                    let _ = chain_tx.send(ToChain::PeerConnected(peer_id));
                }
                _ => {}
            }
        }
    }
    // dropping chain_tx ends the chain actor's loop; it writes the final
    // snapshot + report before returning
    drop(chain_tx);
    let _ = chain.await;
}

/// The chain actor: owns every stateful decision. See `run` for the split.
async fn run_chain(
    mut node: Node,
    mut chain_rx: tokio::sync::mpsc::UnboundedReceiver<ToChain>,
    mut api_rx: mpsc::Receiver<ApiCmd>,
    mut bridge_rx: mpsc::Receiver<FromBridge>,
) {
    loop {
        tokio::select! {
            cmd = chain_rx.recv() => {
                let Some(cmd) = cmd else { break }; // net loop gone: shut down
                node.handle_chain_cmd(cmd);
            }
            Some(ev) = bridge_rx.recv() => {
                node.handle_bridge(ev);
            }
            Some(cmd) = api_rx.recv() => {
                node.handle_api(cmd);
            }
        }
    }
    // final report + snapshot (moved here from the old single loop)
    let h = node.head_height();
    node.store.write_snapshot(&node.tree.head, h,
                              node.tree.head_state(),
                              node.tree.head_ledger(),
                              &node.tree.model[&node.tree.head]);
    let mut lineage = Vec::new();
    let mut cur = node.tree.head.clone();
    while cur != node.tree.genesis_hash {
        lineage.push(cur[..6].to_string());
        let Some(hh) = node.tree.blocks.get(&cur) else { break };
        cur = hh.prev_hash.clone();
    }
    lineage.reverse();
    println!("LINEAGE {}", lineage.join(">"));
}

impl Node {
    fn handle_chain_cmd(&mut self, cmd: ToChain) {
        match cmd {
            ToChain::Dtx(tx, payload, source) => {
                if let Some(t) = tx.to_core() {
                    let empty = payload.idx.is_empty() && payload.val.is_empty();
                    if !empty {
                        self.accept_delta(t, payload);
                    } else {
                        // announcement of a body too big to gossip inline:
                        // reconstruct from shards we hold, or start fetching
                        let txid = t.txid();
                        if let Some(p) = self.payloads.get(&txid).cloned()
                            .or_else(|| self.store.reconstruct_payload(&txid)) {
                            self.payloads.insert(txid.clone(), p.clone());
                            self.accept_delta(t, p);
                        } else if self.want_deltas.len() < 64
                            && !self.want_deltas.contains_key(&txid) {
                            info!(txid = &txid[..12],
                                  "big delta announced — fetching body by shards");
                            self.want_deltas.insert(txid.clone(),
                                (tx, now() + WANT_DELTA_TTL));
                            if let Some(p) = source {
                                let _ = self.net.send(ToNet::SendShards(
                                    p, vec![txid]));
                            }
                        }
                    }
                    self.retry_pending();
                }
            }
            ToChain::Atx(v) => {
                if let Some(t) = account_tx_from_json(&v) {
                    self.accept_account_tx(t);
                }
            }
            ToChain::Blk(block, source) => {
                self.install(block, source);
                self.retry_pending();
            }
            ToChain::Head { peer, hash, height } => {
                if !self.tree.blocks.contains_key(&hash) {
                    let from = self.sync_cursor.get(&peer).copied()
                        .unwrap_or_else(|| self.head_height()
                            .min(height).saturating_sub(2));
                    info!(peer = %peer, their_h = height, from,
                          "unknown head — requesting sync");
                    let _ = self.net.send(ToNet::SendSync(peer, from));
                }
            }
            ToChain::PeerConnected(peer) => {
                // opportunistic catch-up from every new peer — anchor BELOW
                // our head: an equal-height fork needs the peer's blocks at
                // heights we already have, not just above them
                let from = self.sync_cursor.get(&peer).copied()
                    .unwrap_or_else(|| self.head_height().saturating_sub(8));
                let _ = self.net.send(ToNet::SendSync(peer, from));
            }
            ToChain::SyncServe(request, reply) => {
                let _ = reply.send(self.serve_sync(&request));
            }
            ToChain::ShardServe(request, reply) => {
                let _ = reply.send(self.serve_shards(&request));
            }
            ToChain::SyncBatch { peer, current, from, blocks, payloads, their_head } => {
                self.handle_sync_batch(peer, current, from, blocks, payloads,
                                       their_head);
            }
            ToChain::ShardBatch { peer, current, bodies } => {
                self.handle_shard_batch(peer, current, bodies);
            }
            ToChain::RoundTick { round, elapsed_in_round, num_peers,
                                 connected, mesh_blind } => {
                self.peers_connected = num_peers;
                self.round_tick(round, elapsed_in_round, connected, mesh_blind);
            }
        }
    }

    fn handle_sync_batch(&mut self, peer: PeerId, current: bool, from: u64,
                         blocks: Vec<StoredBlock>,
                         payloads: HashMap<String, Payload>, their_head: u64) {
        info!(blocks = blocks.len(), their_head, current,
              "sync response received");
        for (txid, p) in payloads {
            if !self.payloads.contains_key(&txid) {
                self.store.put_payload(&txid, &p);
                self.payloads.insert(txid, p);
            }
        }
        let served = blocks.len() as u64;
        let batch_hashes: HashSet<String> = blocks.iter().map(|sb| sb.hash()).collect();
        let known_before = self.tree.blocks.len() + self.pending.len();
        let batch_blocks: Vec<StoredBlock> = blocks;
        for sb in batch_blocks.clone() {
            self.install(sb, Some(peer));
        }
        self.retry_pending();
        let learned = self.tree.blocks.len() + self.pending.len() > known_before;
        // The state machine runs for EVERY batch, current or not. With serve
        // budgets near the block size, every intermediate batch of a long
        // catch-up is current=false — skipping them starved the orphan
        // walkback of exactly the batches carrying the fork evidence, and a
        // node behind an equal-height fork stalled forever (found live, EU
        // anchor). Stale responses are already dropped by request-id matching
        // in the net loop, which is what this early-out originally guarded.
        let last_from = Some(from);
        let behind = their_head > self.head_height();
        // "orphaned": some served block's parent is nowhere — not applied,
        // not pending, not in this batch. Its ancestors are missing, so the
        // fork point is below the request window.
        let orphaned = batch_blocks.iter().any(|sb| {
            let par = &sb.header.prev_hash;
            sb.header.height > 0
                && par != &self.tree.genesis_hash
                && !self.tree.blocks.contains_key(par)
                && !self.pending.contains_key(par)
                && !batch_hashes.contains(par)
        });
        if orphaned && served > 0 {
            let base = last_from.unwrap_or_else(|| self.head_height());
            let step = self.sync_walkback.entry(peer).or_insert(4);
            let cur = base.saturating_sub(*step);
            *step = (*step * 2).min(4096);
            self.sync_cursor.insert(peer, cur);
            warn!(peer = %peer, from = cur,
                  "sync batch unconnectable — walking back to find the fork point");
        } else if learned {
            // Still behind: keep marching from where this batch ended. Clearing
            // the cursor here re-anchors the next request at head-2, which
            // re-downloads the same overlap and livelocks a lagging node (each
            // climb re-learns 2 blocks while the fleet mints one). The reorg
            // margin is only needed once we are caught up, so clear it then.
            if behind {
                if let Some(f) = last_from {
                    self.sync_cursor.insert(peer, (f + served).min(their_head));
                }
                self.sync_walkback.remove(&peer);
            } else {
                self.sync_cursor.remove(&peer);
                self.sync_walkback.remove(&peer);
            }
        } else if behind && served > 0 {
            if let Some(f) = last_from {
                let cur = (f + served).min(their_head);
                self.sync_cursor.insert(peer, cur);
                warn!(peer = %peer, cursor = cur, their_head,
                      "sync batch taught nothing — advancing request cursor \
                       past the overlap");
            }
        }
        let cursor_now = self.sync_cursor.get(&peer).copied();
        let moved = learned || (cursor_now.is_some() && cursor_now != last_from);
        if (behind || orphaned) && served > 0 && moved {
            let f = cursor_now.unwrap_or_else(||
                self.head_height().min(their_head).saturating_sub(2));
            let _ = self.net.send(ToNet::SendSync(peer, f));
        }
    }

    fn handle_shard_batch(&mut self, peer: PeerId, current: bool,
                          bodies: Vec<BodyShards>) {
        let mut got = false;
        info!(bodies = bodies.len(), "shard response received");
        for b in bodies {
            for (i, data) in b.shards {
                if let Some(bytes) = unb64(&data) {
                    self.store.put_shard(&b.txid, i, &bytes,
                        b.k as usize, b.n as usize, b.orig_len);
                    got = true;
                }
            }
            if b.txid != crate::store::Store::GENESIS_DA_KEY
                && !self.payloads.contains_key(&b.txid)
                && !self.store.has_payload(&b.txid) {
                if let Some(p) = self.store.reconstruct_payload(&b.txid) {
                    self.store.put_payload(&b.txid, &p);
                }
            }
            if let Some((wtx, _)) = self.want_deltas.remove(&b.txid) {
                match (wtx.to_core(),
                       self.payloads.get(&b.txid).cloned()
                           .or_else(|| self.store.get_payload(&b.txid))) {
                    (Some(t), Some(p)) => {
                        info!(txid = &b.txid[..12],
                              "announced delta body reconstructed from shards");
                        self.accept_delta(t, p);
                    }
                    (_, None) => {
                        self.want_deltas.insert(b.txid.clone(),
                            (wtx, now() + WANT_DELTA_TTL));
                    }
                    _ => {}
                }
            }
        }
        if got {
            self.retry_pending();
            if current {
                let still = self.missing_bodies(32);
                if !still.is_empty() {
                    let _ = self.net.send(ToNet::SendShards(peer, still));
                }
            }
        }
    }

    /// Bodies still missing for pending blocks (lowest block first) and for
    /// announced deltas — what the refetch machinery asks peers for.
    fn missing_bodies(&mut self, cap: usize) -> Vec<String> {
        self.want_deltas.retain(|_, (_, dl)| *dl > now());
        let mut by_h: Vec<(u64, String)> = self.pending.values()
            .flat_map(|(sb, _)| sb.txs.iter()
                .filter_map(|t| t.to_core().map(|tc|
                    (sb.header.height, tc.txid()))))
            .filter(|(_, id)| !self.payloads.contains_key(id)
                    && !self.store.has_payload(id))
            .collect();
        by_h.sort();
        let mut seen_ids = HashSet::new();
        let mut want: Vec<String> = by_h.into_iter()
            .filter(|(_, id)| seen_ids.insert(id.clone()))
            .map(|(_, id)| id).collect();
        want.extend(self.want_deltas.keys().cloned());
        want.truncate(cap);
        want
    }
}

impl Node {
    fn round_tick(&mut self, round: i64, elapsed_in_round: f64,
                  connected: Vec<PeerId>, mesh_blind: bool) {
        if round >= 0 && round != self.last_trained_round {
            self.last_trained_round = round;
            // BODY REFETCH: once per round, ask connected peers for the shards
            // of bodies pending blocks are stuck on + announced deltas wanted
            let want = self.missing_bodies(32);
            if !want.is_empty() {
                info!(bodies = want.len(), peers = connected.len(),
                      "refetching missing delta bodies by shards");
                for p in &connected {
                    let _ = self.net.send(ToNet::SendShards(*p, want.clone()));
                }
            }
            // republish unconfirmed deltas for the current height: a publish
            // can silently fail before the gossip mesh forms
            let hh = self.head_height();
            let resend: Vec<(WireDeltaTx, Payload)> = self.delta_pool.values()
                .filter(|t| t.base_height == hh)
                .filter_map(|t| self.payloads.get(&t.txid())
                    .map(|p| (WireDeltaTx::from_core(t), p.clone())))
                .collect();
            for (tx, payload) in resend {
                let payload = if payload.wire_bytes() > dtx_inline_max() {
                    Payload { n: payload.n, idx: String::new(), val: String::new() }
                } else { payload };
                self.publish(&Gossip::Dtx { tx, payload });
            }
            // train EVERY round (the delta gossips to whoever proposes)
            if self.cfg.produce && self.bridge_synced && !self.train_inflight {
                self.train_inflight = true;
                self.train_deadline = now() + TRAIN_TIMEOUT_SECS;
                let model = &self.tree.model[&self.tree.head];
                let active_pages: Vec<u32> = model.pages.iter().enumerate()
                    .filter(|(_, p)| p.status == "A")
                    .map(|(i, _)| i as u32).collect();
                let min_nnz = model.required_nnz(&active_pages);
                let _ = self.bridge_tx.try_send(ToBridge::Train {
                    height: self.head_height(),
                    seed: round as u64,
                    budget_s: self.cfg.interval * 0.6,
                    min_nnz,
                    max_nnz: self.tree.params.delta_max_nnz,
                    quota_4dp: model.quota_4dp as u64,
                    active_pages,
                });
            }
        }
        // MESH-BLINDNESS PULL: the net loop says peers are connected but no
        // foreign head has been heard — pull directly from every peer, with
        // our catch-up cursor as the request start where one is set
        if mesh_blind {
            for p in &connected {
                let from = self.sync_cursor.get(p).copied()
                    .unwrap_or_else(|| self.head_height().saturating_sub(8));
                info!(peer = %p, from, "no foreign heads heard — direct sync pull");
                let _ = self.net.send(ToNet::SendSync(*p, from));
            }
        }
        // v1 PROPOSING: the eligibility ladder widens inside the round; the
        // per-miner phase offset staggers proposals so ties are rare
        if self.cfg.produce && round >= 0 && round != self.last_proposed_round {
            let phase = {
                let h = core::delta_hash(self.key.pub_hex().as_bytes());
                (u64::from_str_radix(&h[..8], 16).unwrap_or(0) % 1000)
                    as f64 / 1000.0 * (self.cfg.interval / 4.0)
            };
            if elapsed_in_round >= phase {
                let max_allowed =
                    (elapsed_in_round / (self.cfg.interval / 8.0)).floor()
                        .max(0.0) as u64;
                if let Some(attempt) = self.eligible_attempt(max_allowed) {
                    self.last_proposed_round = round;
                    if let Some((stored, block)) = self.build_candidate(attempt) {
                        let bh = stored.hash();
                        match self.tree.add_block(block) {
                            Ok(_) => {
                                if let Err(e) = self.store.append_block(&stored) {
                                    error!("FATAL: cannot persist our block h{}: {e}; \
                                            halting", stored.header.height);
                                    std::process::exit(1);
                                }
                                for t in &stored.txs {
                                    if let Some(tc) = t.to_core() {
                                        let id = tc.txid();
                                        self.delta_pool.remove(&id);
                                        self.delta_scores.remove(&id);
                                        self.delta_sketches.remove(&id);
                                    }
                                }
                                for v in stored.transfers.iter()
                                        .chain(stored.data_txs.iter()) {
                                    if let Some(t) = account_tx_from_json(v) {
                                        self.account_pool.remove(&t.txid());
                                    }
                                }
                                self.blocks_full.insert(bh, stored.clone());
                                let old = stored.header.prev_hash.clone();
                                self.on_head_advance(&old);
                                self.publish(&Gossip::Blk { block: stored });
                            }
                            Err(e) => warn!("own block rejected: {}", e.0),
                        }
                    }
                }
            }
        }
        // WATCHDOG: a hung trainer must not silence the node forever
        if self.train_inflight && now() > self.train_deadline {
            warn!("training round timed out after {TRAIN_TIMEOUT_SECS}s — \
                   clearing in-flight flag and resuming");
            self.train_inflight = false;
        }
    }

    fn handle_bridge(&mut self, ev: FromBridge) {
        match ev {
            FromBridge::Connected | FromBridge::NeedState => {
                self.train_inflight = false;
                self.chat_inflight = false;
                for tx in self.chat_pending.drain(..) {
                    let _ = tx.send(json!({"ok": false,
                        "error": "model reconnected — try again"}));
                }
                self.send_bridge_state();
            }
            FromBridge::Generated { text, height } => {
                self.chat_inflight = false;
                if let Some(tx) = self.chat_pending.pop() {
                    let _ = tx.send(json!({"ok": true, "reply": text,
                                           "height": height}));
                }
            }
            FromBridge::Scores { height, scores, sketches } => {
                if height == self.head_height() {
                    for (txid, s) in scores {
                        if self.delta_pool.contains_key(&txid) {
                            self.delta_scores.insert(
                                txid, s.min(core::blocktree::SCORE_CAP));
                        }
                    }
                    for (txid, mut sk) in sketches {
                        if self.delta_pool.contains_key(&txid) {
                            sk.resize(core::blocktree::SKETCH_DIM, 0);
                            self.delta_sketches.insert(txid, sk);
                        }
                    }
                }
            }
            FromBridge::Delta { height, loss, pages, payload } => {
                self.train_inflight = false;
                if height != self.head_height() {
                    self.stale_deltas += 1;
                    warn!(trained_at = height, head = self.head_height(),
                          consecutive = self.stale_deltas,
                          "DELTA DROPPED (stale): your training round finished \
                           after the head moved on, so it cannot be included \
                           and earns nothing. Your GPU is slower than the \
                           block interval — lower --inner/--batch on the \
                           trainer (or raise the node's --interval).");
                } else {
                    self.stale_deltas = 0;
                    // hash from the SPARSE form — no dense materialization
                    let dh = payload.coords()
                        .map(|c| core::delta_hash_sparse(payload.n, &c))
                        .unwrap_or_default();
                    let mut claim: Vec<u32> = pages.clone();
                    claim.sort_unstable();
                    claim.dedup();
                    let mut tx = core::BackpropTx {
                        miner: self.key.pub_hex(),
                        base_height: height,
                        delta_hash: dh.clone(),
                        da_pointer: format!("da://{dh}"),
                        bond: 0,
                        pages: claim,
                        data_refs: self.cfg.data_refs.clone(),
                        sig: vec![],
                    };
                    tx.sig = self.key.sign(&tx.signing_bytes());
                    info!(height, loss, kb = payload.wire_bytes() / 1024,
                          "trained delta");
                    let wire = WireDeltaTx::from_core(&tx);
                    if self.accept_delta(tx, payload.clone()) {
                        let payload = if payload.wire_bytes() > dtx_inline_max() {
                            Payload { n: payload.n,
                                      idx: String::new(), val: String::new() }
                        } else { payload };
                        self.publish(&Gossip::Dtx { tx: wire, payload });
                    }
                }
            }
        }
    }

    fn handle_api(&mut self, cmd: ApiCmd) {
        match cmd {
            ApiCmd::Status(o) => { let _ = o.send(self.api_status()); }
            ApiCmd::Metrics(o) => { let _ = o.send(self.api_metrics()); }
            ApiCmd::Balance(addr, o) => { let _ = o.send(self.api_balance(&addr)); }
            ApiCmd::Registry(o) => { let _ = o.send(self.api_registry()); }
            ApiCmd::Chain(o) => { let _ = o.send(self.api_chain()); }
            ApiCmd::Miners(o) => { let _ = o.send(self.api_miners()); }
            ApiCmd::Chat(prompt, o) => {
                if !self.bridge_synced {
                    let _ = o.send(json!({"ok": false,
                        "error": "no model attached to this node yet"}));
                } else if self.chat_inflight {
                    let _ = o.send(json!({"ok": false,
                        "error": "model is generating for someone else — try again"}));
                } else {
                    self.chat_inflight = true;
                    self.chat_pending.push(o);
                    let _ = self.bridge_tx.try_send(ToBridge::Generate {
                        prompt, n: 120,
                    });
                }
            }
            ApiCmd::Upload(bytes, stake, media, o) => {
                let (reply, gossip) = self.api_upload(bytes, stake, media);
                if let Some(msg) = gossip {
                    self.publish(&msg);
                }
                let _ = o.send(reply);
            }
            ApiCmd::SubmitAccountTx(v, o) => {
                let reply = match account_tx_from_json(&v) {
                    None => json!({"ok": false, "error": "malformed tx"}),
                    Some(tx) => match self.accept_account_tx(tx.clone()) {
                        None => json!({"ok": false,
                                       "error": "bad signature or duplicate"}),
                        Some(txid) => {
                            self.publish(&Gossip::Atx {
                                tx: account_tx_to_json(&tx) });
                            json!({"ok": true, "txid": txid,
                                   "status": "in mempool — settles in the next block"})
                        }
                    },
                };
                let _ = o.send(reply);
            }
        }
    }

    fn serve_sync(&mut self, request: &SyncRequest) -> SyncResponse {
        // serve OUR head chain from `from_height` upward, oldest-first,
        // bounded by SYNC_BYTE_BUDGET (always at least one block)
        let mut ascending: Vec<String> = Vec::new();
        let mut cur = self.tree.head.clone();
        while cur != self.tree.genesis_hash {
            let hdr = &self.tree.blocks[&cur];
            if hdr.height < request.from_height {
                break;
            }
            ascending.push(cur.clone());
            cur = hdr.prev_hash.clone();
        }
        ascending.reverse();
        let want = (request.count as usize).min(SYNC_MAX_BLOCKS);
        let mut chain = Vec::new();
        let mut payloads = HashMap::new();
        let mut bytes = 0usize;
        for h in ascending {
            if chain.len() >= want {
                break;
            }
            let Some(sb) = self.blocks_full.get(&h).cloned() else { continue };
            for t in &sb.txs {
                if let Some(tc) = t.to_core() {
                    let txid = tc.txid();
                    if let Some(p) = self.payloads.get(&txid).cloned()
                        .or_else(|| self.store.get_payload(&txid)) {
                        // oversized bodies travel by shards, not inline
                        if p.wire_bytes() > dtx_inline_max() {
                            continue;
                        }
                        bytes += p.wire_bytes();
                        payloads.insert(txid, p);
                    }
                }
            }
            chain.push(sb);
            if bytes >= SYNC_BYTE_BUDGET {
                break;
            }
        }
        info!(from = request.from_height, served = chain.len(),
              kb = bytes / 1024, "serving sync request");
        let genesis = if request.want_genesis {
            match self.tree.genesis_state() {
                Some(w) if w.len() * 8 <= SYNC_BYTE_BUDGET => Some(w.clone()),
                Some(w) => {
                    warn!(params = w.len(),
                          "peer asked for the genesis but it is too large to \
                           serve over sync; they must generate it locally");
                    None
                }
                None => None,
            }
        } else {
            None
        };
        SyncResponse {
            blocks: chain, payloads,
            head_height: self.head_height(),
            genesis,
        }
    }

    fn serve_shards(&mut self, request: &ShardRequest) -> ShardResponse {
        // byte-budgeted, rotating per-body cursor: any size body moves in
        // bounded responses; >= 1 shard always served (see the fork incident)
        let mut bodies = Vec::new();
        let mut budget_used = 0usize;
        for txid in request.txids.iter().take(32) {
            if budget_used >= SHARD_SERVE_BUDGET {
                break;
            }
            let Some((k, n, orig_len)) = self.store.shard_meta(txid)
                else { continue };
            let have = self.store.list_shard_indices(txid);
            if have.is_empty() {
                continue;
            }
            let cur = self.serve_shard_cursor
                .entry(txid.clone()).or_insert(0);
            let mut shards: Vec<(u32, String)> = Vec::new();
            for step in 0..have.len() {
                if budget_used >= SHARD_SERVE_BUDGET && !shards.is_empty() {
                    break;
                }
                let pick = have[(*cur + step) % have.len()];
                if let Some(d) = self.store.read_shard(txid, pick) {
                    budget_used += d.len() * 4 / 3;
                    shards.push((pick, b64(&d)));
                }
            }
            *cur = (*cur + shards.len().max(1)) % have.len().max(1);
            if !shards.is_empty() {
                bodies.push(BodyShards {
                    txid: txid.clone(), k: k as u32, n: n as u32,
                    orig_len, shards });
            }
        }
        info!(asked = request.txids.len(), served = bodies.len(),
              kb = budget_used / 1024, "serving shard request");
        ShardResponse { bodies }
    }
}

pub fn dial_peers(swarm: &mut Swarm<Behaviour>, peers: &str) {
    for p in peers.split(',').filter(|s| !s.is_empty()) {
        match p.parse::<Multiaddr>() {
            Ok(addr) => {
                if let Err(e) = swarm.dial(addr) {
                    warn!("dial {p}: {e}");
                }
            }
            Err(e) => warn!("bad multiaddr {p}: {e}"),
        }
    }
}
