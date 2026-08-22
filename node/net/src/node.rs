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
const SYNC_BYTE_BUDGET: usize = 48 * 1024 * 1024;
const SYNC_INFLIGHT_TIMEOUT: f64 = 30.0;

/// Coordinates per aggregation chunk — the memory/latency knob (K deltas ×
/// this many i64 at once, ~8MB for the default, vs K × the whole 86M state).
const AGG_CHUNK: usize = 1 << 20;

/// Chunked, bounded-memory aggregation over SPARSE delta payloads. Bit-identical
/// to `core::trimmed_mean` over the full dense deltas (each coordinate's sort/
/// trim/mean is unchanged), but never materializes more than AGG_CHUNK
/// coordinates × K deltas at once — so a small peer can aggregate an 86M-param
/// block without holding K × ~0.7GB of dense deltas.
pub fn chunked_aggregate(payloads: &[&Payload], n: usize, chunk_size: usize) -> Vec<i64> {
    let mut out = vec![0i64; n];
    let step = chunk_size.max(1);
    let mut c = 0;
    while c < n {
        let hi = (c + step).min(n);
        let chunk: Vec<Vec<i64>> = payloads.iter().map(|p| p.dense_range(c, hi)).collect();
        out[c..hi].copy_from_slice(&core::trimmed_mean(&chunk, 0.2));
        c = hi;
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
        assert_eq!(chunked_aggregate(&refs, n, 1000), dense_mean, "chunk=1000");
        assert_eq!(chunked_aggregate(&refs, n, 777), dense_mean, "uneven chunk");
        assert_eq!(chunked_aggregate(&refs, n, n), dense_mean, "single chunk");
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

pub struct NodeConfig {
    pub produce: bool,
    pub interval: f64,
    pub rotate: Option<(u64, u64)>, // (n, id): deterministic devnet rotation
    pub seconds: f64,               // 0 = run forever
    pub peers: String,              // configured peers — re-dialed when lost
    pub data_refs: Vec<String>,     // rev 5: staked corpora this miner names on its deltas
}

pub struct Node {
    pub tree: BlockTree,
    pub store: Store,
    pub key: core::Key,
    pub blocks_full: HashMap<String, StoredBlock>,
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
    pub last_announced_round: i64,
    /// per-peer timestamp of the last sync we requested — heartbeat-triggered
    /// catch-up must not stack concurrent multi-hundred-MB transfers
    pub last_sync_req: HashMap<PeerId, f64>,
    pub peers_connected: usize,
    pub chat_pending: Vec<tokio::sync::oneshot::Sender<Value>>,
    pub chat_inflight: bool,
    /// consecutive deltas dropped as stale — a slow trainer mining for nothing.
    /// Surfaced in /status + /metrics so the failure is visible, not silent.
    pub stale_deltas: u64,
    /// per-peer rotation cursor for serving GENESIS shards one at a time, so a
    /// bootstrapping peer that keeps asking us collects K distinct shards.
    pub genesis_shard_cursor: HashMap<PeerId, usize>,
}

impl Node {
    fn head_height(&self) -> u64 {
        self.tree.blocks[&self.tree.head].height
    }

    fn publish(&mut self, swarm: &mut Swarm<Behaviour>, msg: &Gossip) {
        let bytes = serde_json::to_vec(msg).unwrap();
        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(self.topic.clone(), bytes) {
            debug!("publish: {e}");                // e.g. no peers yet — fine
        }
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
        let want: Vec<(String, SparseI64)> = self.delta_pool.iter()
            .filter(|(id, t)| t.base_height == hh && !self.delta_scores.contains_key(*id))
            .filter_map(|(id, _)| {
                let dense = self.payloads.get(id)?.dense()?;
                Some((id.clone(), Payload::from_dense_i64(&dense)))
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
            if let Some(drop) = self.pending.iter()
                .min_by_key(|(_, (s, _))| s.header.height).map(|(h, _)| h.clone()) {
                self.pending.remove(&drop);
            }
        }
        self.pending.insert(bh, (sb, peer));
    }

    // ---- delta txs (from our bridge or from gossip) ----------------------
    fn accept_delta(&mut self, tx: core::BackpropTx, payload: Payload) -> bool {
        let txid = tx.txid();
        if self.seen.contains(&txid) || !tx.verify() {
            return false;
        }
        let Some(dense) = payload.dense() else { return false };
        if core::delta_hash(&core::int64_bytes(&dense)) != tx.delta_hash {
            warn!("delta payload hash mismatch from {}", &tx.miner[..8]);
            return false;
        }
        // height gate: only admit deltas that can plausibly be mined onto head
        if !delta_in_window(tx.base_height, self.head_height()) {
            return false;
        }
        self.mark_seen(txid.clone());
        self.store.put_payload(&txid, &payload);
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
    fn build_candidate(&self) -> Option<(StoredBlock, sestrian_core::blocktree::Block)> {
        let head = self.tree.head.clone();
        let hh = self.head_height();
        // rev 5: a delta that names no staked/active corpus would invalidate the
        // whole block (provenance required) — never build on one.
        let active_hashes: std::collections::BTreeSet<String> =
            self.tree.ledger[&head].registry.values()
                .filter(|e| e["status"] == "active")
                .filter_map(|e| e["data_hash"].as_str().map(|s| s.to_string()))
                .collect();
        let mut cands: Vec<&core::BackpropTx> = self.delta_pool.values()
            .filter(|t| t.base_height == hh)
            .filter(|t| t.canonical_refs().iter().any(|r| active_hashes.contains(r)))
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
        // weight-state transition
        let parent_w = &self.tree.state[&head];
        // chunked aggregation over the sparse payloads — bounded memory, but
        // bit-identical to core::trimmed_mean over the full dense deltas, so the
        // committed state_root reproduces on any validator.
        let payload_refs: Vec<&Payload> =
            chosen.iter().map(|t| &self.payloads[&t.txid()]).collect();
        let mean = chunked_aggregate(&payload_refs, parent_w.len(), AGG_CHUNK);
        // wrapping_add mirrors numpy int64 (matches validate_block exactly)
        let w: Vec<i64> = parent_w.iter().zip(&mean).map(|(a, b)| a.wrapping_add(*b)).collect();
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
        // proposer lottery: sign the VRF proof over this height's seed; work is
        // the non-forgeable weight derived from it.
        let vrf_proof = core::lottery::vrf_prove(&self.key, &head, hh + 1);
        let header = core::Header {
            height: hh + 1,
            prev_hash: head.clone(),
            state_root: core::state_root(&w),
            txset_root: core::txset_root(
                &chosen.iter().map(|t| t.txid()).collect::<Vec<_>>()),
            n_txs: chosen.len() as u64,
            work: core::lottery::vrf_work(&vrf_proof),
            proposer: self.key.pub_hex(),
            transfer_root: sestrian_core::token::transfer_root(&core_transfers),
            ledger_root: scratch.root(),
            data_root: sestrian_core::token::data_root(&core_data),
            vrf_proof: hex::encode(&vrf_proof),
            score_root: core::blocktree::scores_root(&blk_scores),
            sketch_root: core::blocktree::sketch_root(&blk_sketches),
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
            transfers: core_transfers, data_txs: core_data,
            scores: blk_scores, sketches: blk_sketches,
        };
        Some((stored, block))
    }

    // ---- installation ----------------------------------------------------
    /// Try to install a stored block (bodies from the payload store). Returns
    /// true if installed; queues it as pending when payloads are missing.
    fn install(&mut self, sb: StoredBlock, from: Option<PeerId>,
               swarm: &mut Swarm<Behaviour>) -> bool {
        let bh = sb.hash();
        if self.blocks_full.contains_key(&bh) {
            return false;
        }
        let Some(block) = sb.to_core(&self.payloads) else {
            // body missing: try to reconstruct it from erasure shards we already
            // hold, and if we can't, ask ALL peers for its shards (any one may
            // hold a few) in parallel with the full-block sync fallback.
            let missing: Vec<String> = sb.txs.iter()
                .filter_map(|t| t.to_core().map(|tc| tc.txid()))
                .filter(|id| {
                    if self.payloads.contains_key(id) {
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
                let peers: Vec<PeerId> = swarm.connected_peers().copied().collect();
                for p in &peers {
                    swarm.behaviour_mut().shards
                        .send_request(p, ShardRequest { txids: missing.clone() });
                }
            }
            if let Some(peer) = from {
                let req = SyncRequest {
                    from_height: sb.header.height.saturating_sub(1),
                    count: 32,
                    want_genesis: false,
                };
                swarm.behaviour_mut().sync.send_request(&peer, req);
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
                        let req = SyncRequest {
                            from_height: self.head_height().saturating_sub(8),
                            count: 64,
                            want_genesis: false,
                        };
                        swarm.behaviour_mut().sync.send_request(&peer, req);
                        self.queue_pending(bh, sb, peer);
                    }
                } else {
                    warn!("invalid block h{}: {}", sb.header.height, e.0);
                }
                false
            }
        }
    }

    fn retry_pending(&mut self, swarm: &mut Swarm<Behaviour>) {
        let ready: Vec<String> = self.pending.iter()
            .filter(|(_, (sb, _))| {
                sb.txs.iter().all(|t| t.to_core()
                    .map(|tc| self.payloads.contains_key(&tc.txid()))
                    .unwrap_or(false))
                && self.tree.blocks.contains_key(&sb.header.prev_hash)
            })
            .map(|(h, _)| h.clone()).collect();
        for h in ready {
            if let Some((sb, peer)) = self.pending.remove(&h) {
                self.install(sb, Some(peer), swarm);
            }
        }
    }

    fn on_head_advance(&mut self, old_head: &str) {
        let h = self.head_height();
        info!(height = h, head = &self.tree.head[..10],
              supply = self.tree.head_ledger().supply(), "head advanced");
        // keep the bridge synced with a sparse state diff
        if self.bridge_synced {
            if let (Some(new_w), Some(old_w)) =
                (self.tree.state.get(&self.tree.head), self.tree.state.get(old_head))
            {
                let diff: Vec<i64> = new_w.iter().zip(old_w)
                    .map(|(a, b)| a - b).collect();
                let sparse = Payload::from_dense_i64(&diff);
                let _ = self.bridge_tx.try_send(ToBridge::Advance { height: h, sparse });
            } else {
                // reorg past pruned state — bridge must resync from scratch
                self.send_bridge_state();
            }
        }
        if h % SNAPSHOT_EVERY == 0 {
            self.store.write_snapshot(&self.tree.head, h,
                                      &self.tree.state[&self.tree.head],
                                      self.tree.head_ledger());
        }
        // the head moved: prune mempools + pending against it
        self.evict_delta_pool();
        self.evict_account_pool();
        // rev 7: (re)score surviving pool deltas against the new head's round
        self.request_evals();
        let drop_pending: Vec<String> = self.pending.iter()
            .filter(|(_, (s, _))| s.header.height + DELTA_STALE_SLACK < h)
            .map(|(k, _)| k.clone()).collect();
        for k in drop_pending {
            self.pending.remove(&k);
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
        let state = self.tree.state[&self.tree.head].clone();
        if self.bridge_tx.try_send(ToBridge::State { height: h, state }).is_ok() {
            self.bridge_synced = true;
        }
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
pub async fn run(
    mut node: Node,
    mut swarm: Swarm<Behaviour>,
    mut api_rx: mpsc::Receiver<ApiCmd>,
    mut bridge_rx: mpsc::Receiver<FromBridge>,
) {
    let end = if node.cfg.seconds > 0.0 { now() + node.cfg.seconds } else { f64::MAX };
    let mut tick = tokio::time::interval(Duration::from_millis(400));
    let jitter: f64 = rand::random::<f64>() * 0.5;
    // SIGTERM is what systemd/k8s send on stop — handle it (not just SIGINT) so
    // the post-loop final snapshot runs and the next boot fast-boots at head.
    // Windows has no SIGTERM; CTRL_SHUTDOWN is its analogue (sent when a service
    // is stopped or the machine shuts down), so graceful shutdown works there too.
    #[cfg(unix)]
    let mut stop_signal = tokio::signal::unix::signal(
        tokio::signal::unix::SignalKind::terminate())
        .expect("install SIGTERM handler");
    #[cfg(windows)]
    let mut stop_signal = tokio::signal::windows::ctrl_shutdown()
        .expect("install CTRL_SHUTDOWN handler");

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
            _ = tick.tick() => {
                let round = ((now() - node.t0 - jitter) / node.cfg.interval).floor() as i64;
                if round >= 0 && round != node.last_announced_round {
                    node.last_announced_round = round;
                    // the self-healing heartbeat: announce our head every round
                    let head_msg = Gossip::Head {
                        hash: node.tree.head.clone(),
                        height: node.head_height(),
                    };
                    node.publish(&mut swarm, &head_msg);
                    // …and re-dial configured peers when connections are lost
                    // (restarts on either side otherwise orphan the mesh forever)
                    let expected = node.cfg.peers.split(',')
                        .filter(|s| !s.is_empty()).count();
                    if swarm.network_info().num_peers() < expected {
                        dial_peers(&mut swarm, &node.cfg.peers.clone());
                    }
                }
                if node.cfg.produce && round >= 0 && round != node.last_proposed_round {
                    node.last_proposed_round = round;
                    // republish unconfirmed deltas for the current height: a
                    // publish can silently fail before the gossip mesh forms
                    // (InsufficientPeers), so retry each round until included
                    let hh = node.head_height();
                    let resend: Vec<(WireDeltaTx, Payload)> = node.delta_pool.values()
                        .filter(|t| t.base_height == hh)
                        .filter_map(|t| node.payloads.get(&t.txid())
                            .map(|p| (WireDeltaTx::from_core(t), p.clone())))
                        .collect();
                    for (tx, payload) in resend {
                        node.publish(&mut swarm, &Gossip::Dtx { tx, payload });
                    }
                    // train EVERY round (the delta gossips to whoever proposes);
                    // proposing itself may rotate (devnet) or be open (mainnet)
                    if node.bridge_synced && !node.train_inflight {
                        node.train_inflight = true;
                        node.train_deadline = now() + TRAIN_TIMEOUT_SECS;
                        let _ = node.bridge_tx.try_send(ToBridge::Train {
                            height: node.head_height(),
                            seed: round as u64,
                            // spend at most ~60% of the round on inner steps:
                            // the rest covers compression, scoring/sketching and
                            // gossip, so the delta lands while it's still
                            // includable rather than arriving stale.
                            budget_s: node.cfg.interval * 0.6,
                        });
                    }
                    let my_turn = match node.cfg.rotate {
                        Some((n, id)) => (round as u64) % n == id,
                        None => true,                   // open proposing; fork choice settles
                    };
                    if my_turn {
                        if let Some((stored, block)) = node.build_candidate() {
                            let bh = stored.hash();
                            match node.tree.add_block(block) {
                                Ok(_) => {
                                    // durability: never gossip a block we didn't persist
                                    if let Err(e) = node.store.append_block(&stored) {
                                        error!("FATAL: cannot persist our block h{}: {e}; \
                                                halting", stored.header.height);
                                        std::process::exit(1);
                                    }
                                    for t in &stored.txs {
                                        if let Some(tc) = t.to_core() {
                                            let id = tc.txid();
                                            node.delta_pool.remove(&id);
                                            node.delta_scores.remove(&id);
                                            node.delta_sketches.remove(&id);
                                        }
                                    }
                                    for v in stored.transfers.iter()
                                            .chain(stored.data_txs.iter()) {
                                        if let Some(t) = account_tx_from_json(v) {
                                            node.account_pool.remove(&t.txid());
                                        }
                                    }
                                    node.blocks_full.insert(bh, stored.clone());
                                    let old = stored.header.prev_hash.clone();
                                    node.on_head_advance(&old);
                                    node.publish(&mut swarm, &Gossip::Blk { block: stored });
                                }
                                Err(e) => warn!("own block rejected: {}", e.0),
                            }
                        }
                    }
                }
                // WATCHDOG: a hung trainer must not silence the node forever
                if node.train_inflight && now() > node.train_deadline {
                    warn!("training round timed out after {TRAIN_TIMEOUT_SECS}s — \
                           clearing in-flight flag and resuming");
                    node.train_inflight = false;
                }
            }
            Some(ev) = bridge_rx.recv() => match ev {
                FromBridge::Connected | FromBridge::NeedState => {
                    node.train_inflight = false;
                    node.chat_inflight = false;
                    for tx in node.chat_pending.drain(..) {
                        let _ = tx.send(json!({"ok": false,
                            "error": "model reconnected — try again"}));
                    }
                    node.send_bridge_state();
                }
                FromBridge::Generated { text, height } => {
                    node.chat_inflight = false;
                    if let Some(tx) = node.chat_pending.pop() {
                        let _ = tx.send(json!({"ok": true, "reply": text,
                                               "height": height}));
                    }
                }
                FromBridge::Scores { height, scores, sketches } => {
                    // rev 7/8: cache our trainer's held-out scores + influence
                    // sketches for pool deltas; build_candidate commits them.
                    // Clamped/shaped; only for deltas we still hold (stale
                    // responses no-op harmlessly).
                    if height == node.head_height() {
                        for (txid, s) in scores {
                            if node.delta_pool.contains_key(&txid) {
                                node.delta_scores.insert(
                                    txid, s.min(core::blocktree::SCORE_CAP));
                            }
                        }
                        for (txid, mut sk) in sketches {
                            if node.delta_pool.contains_key(&txid) {
                                sk.resize(core::blocktree::SKETCH_DIM, 0);
                                node.delta_sketches.insert(txid, sk);
                            }
                        }
                    }
                }
                FromBridge::Delta { height, loss, payload } => {
                    node.train_inflight = false;
                    if height != node.head_height() {
                        // A delta is includable only at base_height == head, so a
                        // trainer slower than the block interval produces deltas
                        // that are ALWAYS stale — it mines forever and earns
                        // nothing. This used to be debug!, i.e. an invisible
                        // failure: the operator saw "trained, loss …" and no
                        // rewards, with no explanation. Say it loudly, and say
                        // what to do about it.
                        node.stale_deltas += 1;
                        warn!(trained_at = height, head = node.head_height(),
                              consecutive = node.stale_deltas,
                              "DELTA DROPPED (stale): your training round finished \
                               after the head moved on, so it cannot be included \
                               and earns nothing. Your GPU is slower than the \
                               block interval — lower --inner/--batch on the \
                               trainer (or raise the node's --interval).");
                    } else {
                        node.stale_deltas = 0;
                        let dense = payload.dense().unwrap_or_default();
                        let dh = core::delta_hash(&core::int64_bytes(&dense));
                        let mut tx = core::BackpropTx {
                            miner: node.key.pub_hex(),
                            base_height: height,
                            shard_id: 0,
                            delta_hash: dh.clone(),
                            da_pointer: format!("da://{dh}"),
                            bond: 0, // bootstrap: no bond; a bonded-miner policy is config
                            // rev 5: name the staked corpora this miner trains on so
                            // the delta is provenanced and the data share pays their owners
                            data_refs: node.cfg.data_refs.clone(),
                            sig: vec![],
                        };
                        tx.sig = node.key.sign(&tx.signing_bytes());
                        info!(height, loss, kb = payload.wire_bytes() / 1024,
                              "trained delta");
                        let wire = WireDeltaTx::from_core(&tx);
                        if node.accept_delta(tx, payload.clone()) {
                            node.publish(&mut swarm,
                                         &Gossip::Dtx { tx: wire, payload });
                        }
                    }
                }
            },
            Some(cmd) = api_rx.recv() => match cmd {
                ApiCmd::Status(o) => { let _ = o.send(node.api_status()); }
                ApiCmd::Metrics(o) => { let _ = o.send(node.api_metrics()); }
                ApiCmd::Balance(addr, o) => { let _ = o.send(node.api_balance(&addr)); }
                ApiCmd::Registry(o) => { let _ = o.send(node.api_registry()); }
                ApiCmd::Chain(o) => { let _ = o.send(node.api_chain()); }
                ApiCmd::Miners(o) => { let _ = o.send(node.api_miners()); }
                ApiCmd::Chat(prompt, o) => {
                    if !node.bridge_synced {
                        let _ = o.send(json!({"ok": false,
                            "error": "no model attached to this node yet"}));
                    } else if node.chat_inflight {
                        let _ = o.send(json!({"ok": false,
                            "error": "model is generating for someone else — try again"}));
                    } else {
                        node.chat_inflight = true;
                        node.chat_pending.push(o);
                        let _ = node.bridge_tx.try_send(ToBridge::Generate {
                            prompt, n: 120,
                        });
                    }
                }
                ApiCmd::Upload(bytes, stake, media, o) => {
                    let (reply, gossip) = node.api_upload(bytes, stake, media);
                    if let Some(msg) = gossip {
                        node.publish(&mut swarm, &msg);
                    }
                    let _ = o.send(reply);
                }
                ApiCmd::SubmitAccountTx(v, o) => {
                    let reply = match account_tx_from_json(&v) {
                        None => json!({"ok": false, "error": "malformed tx"}),
                        Some(tx) => match node.accept_account_tx(tx.clone()) {
                            None => json!({"ok": false,
                                           "error": "bad signature or duplicate"}),
                            Some(txid) => {
                                node.publish(&mut swarm,
                                             &Gossip::Atx { tx: account_tx_to_json(&tx) });
                                json!({"ok": true, "txid": txid,
                                       "status": "in mempool — settles in the next block"})
                            }
                        },
                    };
                    let _ = o.send(reply);
                }
            },
            ev = swarm.select_next_some() => match ev {
                SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(
                        gossipsub::Event::Message { message, propagation_source, .. })) => {
                    if let Ok(g) = serde_json::from_slice::<Gossip>(&message.data) {
                        match g {
                            Gossip::Dtx { tx, payload } => {
                                if let Some(t) = tx.to_core() {
                                    node.accept_delta(t, payload);
                                    node.retry_pending(&mut swarm);
                                }
                            }
                            Gossip::Atx { tx } => {
                                if let Some(t) = account_tx_from_json(&tx) {
                                    node.accept_account_tx(t);
                                }
                            }
                            Gossip::Blk { block } => {
                                node.install(block, Some(propagation_source), &mut swarm);
                                node.retry_pending(&mut swarm);
                            }
                            Gossip::Head { hash, height } => {
                                // unknown head -> pull the sender's recent chain,
                                // BUT at most one in-flight catch-up per peer per
                                // 90s — payload batches are tens of MB and stacked
                                // transfers saturate home uplinks without landing
                                // one request in flight per peer; re-request as
                                // soon as the last response landed (last_sync_req
                                // is cleared on receipt) or after a lost-response
                                // timeout — so a lagging node keeps catching up.
                                let inflight = node.last_sync_req
                                    .get(&propagation_source)
                                    .map(|t| now() - t < SYNC_INFLIGHT_TIMEOUT)
                                    .unwrap_or(false);
                                if !node.tree.blocks.contains_key(&hash) && !inflight {
                                    node.last_sync_req
                                        .insert(propagation_source, now());
                                    let from = node.head_height()
                                        .min(height).saturating_sub(2);
                                    info!(peer = %propagation_source, their_h = height,
                                          from, "unknown head — requesting sync");
                                    let req = SyncRequest {
                                        from_height: from, count: SYNC_MAX_BLOCKS as u64,
                                        want_genesis: false };
                                    swarm.behaviour_mut().sync
                                        .send_request(&propagation_source, req);
                                }
                            }
                        }
                    }
                }
                SwarmEvent::Behaviour(BehaviourEvent::Sync(
                        request_response::Event::Message { peer, message, .. })) => {
                    match message {
                        request_response::Message::Request { request, channel, .. } => {
                            // serve OUR head chain from `from_height` upward,
                            // oldest-first, bounded by SYNC_BYTE_BUDGET (always at
                            // least one block, so progress is guaranteed).
                            let mut ascending: Vec<String> = Vec::new();
                            let mut cur = node.tree.head.clone();
                            while cur != node.tree.genesis_hash {
                                let hdr = &node.tree.blocks[&cur];
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
                                let Some(sb) = node.blocks_full.get(&h).cloned() else { continue };
                                for t in &sb.txs {
                                    if let Some(tc) = t.to_core() {
                                        let txid = tc.txid();
                                        if let Some(p) = node.payloads.get(&txid).cloned()
                                            .or_else(|| node.store.get_payload(&txid)) {
                                            bytes += p.wire_bytes();
                                            payloads.insert(txid, p);
                                        }
                                    }
                                }
                                chain.push(sb);
                                if bytes >= SYNC_BYTE_BUDGET {
                                    break; // packed a budget's worth (>=1 block sent)
                                }
                            }
                            info!(from = request.from_height, served = chain.len(),
                                  kb = bytes / 1024, "serving sync request");
                            // a fresh node bootstraps the genesis from us — the
                            // shared trust anchor, self-verified by the requester
                            // against the published genesis id.
                            // Serving the genesis is only possible when it fits
                            // the response cap: JSON-encoded i64s are ~6-10
                            // bytes each, so a ~650MB production genesis blows
                            // past SYNC_RESP_MAX and the requester would just
                            // time out. Refuse loudly instead of blackholing —
                            // the requester's error tells them to generate it
                            // locally (it's deterministic).
                            let genesis = if request.want_genesis {
                                let g = node.tree.state.get(&node.tree.genesis_hash);
                                match g {
                                    Some(w) if w.len() * 8 <= SYNC_BYTE_BUDGET =>
                                        Some(w.clone()),
                                    Some(w) => {
                                        warn!(params = w.len(),
                                              "peer asked for the genesis but it is \
                                               too large to serve over sync; they \
                                               must generate it locally from the \
                                               published seed");
                                        None
                                    }
                                    None => None,
                                }
                            } else {
                                None
                            };
                            let resp = SyncResponse {
                                blocks: chain, payloads,
                                head_height: node.head_height(),
                                genesis,
                            };
                            let _ = swarm.behaviour_mut().sync
                                .send_response(channel, resp);
                        }
                        request_response::Message::Response { response, .. } => {
                            info!(blocks = response.blocks.len(),
                                  their_head = response.head_height,
                                  "sync response received");
                            for (txid, p) in response.payloads {
                                if !node.payloads.contains_key(&txid) {
                                    node.store.put_payload(&txid, &p);
                                    node.payloads.insert(txid, p);
                                }
                            }
                            for sb in response.blocks {
                                node.install(sb, None, &mut swarm);
                            }
                            node.retry_pending(&mut swarm);
                            // clear the in-flight marker so the next Head from this
                            // peer immediately pulls the next batch (continuous
                            // catch-up instead of one batch per throttle window)
                            node.last_sync_req.remove(&peer);
                        }
                    }
                }
                SwarmEvent::Behaviour(BehaviourEvent::Shards(
                        request_response::Event::Message { message, peer, .. })) => {
                    match message {
                        // serve whatever shards we hold for the requested bodies
                        request_response::Message::Request { request, channel, .. } => {
                            let mut bodies = Vec::new();
                            for txid in request.txids.iter().take(128) {
                                // The GENESIS shard set is ~2GB (48 x ~43MB), so
                                // it can never be served the way a delta body is.
                                // Return exactly ONE shard per response, and walk
                                // a per-peer cursor so a requester asking
                                // repeatedly collects K DISTINCT shards from us
                                // (the request carries no "shards I already have"
                                // list, so the server drives the rotation).
                                if txid == crate::store::Store::GENESIS_DA_KEY {
                                    let Some((k, n, orig_len)) = node.store.shard_meta(txid)
                                        else { continue };
                                    let have = node.store.list_shard_indices(txid);
                                    if have.is_empty() {
                                        continue;
                                    }
                                    let cur = node.genesis_shard_cursor.entry(peer).or_insert(0);
                                    let pick = have[*cur % have.len()];
                                    *cur = cur.wrapping_add(1);
                                    if let Some(d) = node.store.read_shard(txid, pick) {
                                        bodies.push(BodyShards {
                                            txid: txid.clone(), k: k as u32, n: n as u32,
                                            orig_len, shards: vec![(pick, b64(&d))] });
                                    }
                                    continue;
                                }
                                if let Some((k, n, orig_len)) = node.store.shard_meta(txid) {
                                    let shards: Vec<(u32, String)> = node.store.list_shards(txid)
                                        .into_iter().map(|(i, d)| (i, b64(&d))).collect();
                                    if !shards.is_empty() {
                                        bodies.push(BodyShards {
                                            txid: txid.clone(), k: k as u32, n: n as u32,
                                            orig_len, shards });
                                    }
                                }
                            }
                            let _ = swarm.behaviour_mut().shards
                                .send_response(channel, ShardResponse { bodies });
                        }
                        // store fetched shards; reconstruct any body now >= K
                        request_response::Message::Response { response, .. } => {
                            let mut got = false;
                            for b in response.bodies {
                                for (i, data) in b.shards {
                                    if let Some(bytes) = unb64(&data) {
                                        node.store.put_shard(&b.txid, i, &bytes,
                                            b.k as usize, b.n as usize, b.orig_len);
                                        got = true;
                                    }
                                }
                                // the genesis key is raw weights, not a Payload —
                                // reconstructing it is the bootstrap path's job
                                if b.txid != crate::store::Store::GENESIS_DA_KEY
                                    && !node.payloads.contains_key(&b.txid) {
                                    if let Some(p) = node.store.reconstruct_payload(&b.txid) {
                                        node.payloads.insert(b.txid.clone(), p);
                                    }
                                }
                            }
                            if got {
                                node.retry_pending(&mut swarm);
                            }
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
                SwarmEvent::ConnectionClosed { .. } => {
                    node.peers_connected = node.peers_connected.saturating_sub(1);
                }
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    node.peers_connected += 1;
                    info!(%peer_id, "peer connected");
                    // opportunistic catch-up from every new peer — anchor BELOW
                    // our head: an equal-height fork needs the peer's blocks at
                    // heights we already have, not just above them
                    let req = SyncRequest {
                        from_height: node.head_height().saturating_sub(8), count: 64,
                        want_genesis: false,
                    };
                    swarm.behaviour_mut().sync.send_request(&peer_id, req);
                }
                _ => {}
            },
        }
    }

    // final report + snapshot
    let h = node.head_height();
    node.store.write_snapshot(&node.tree.head, h,
                              &node.tree.state[&node.tree.head], node.tree.head_ledger());
    let mut lineage = Vec::new();
    let mut cur = node.tree.head.clone();
    while cur != node.tree.genesis_hash {
        lineage.push(cur[..6].to_string());
        cur = node.tree.blocks[&cur].prev_hash.clone();
    }
    lineage.reverse();
    println!("LINEAGE {}", lineage.join(">"));
    println!("done — height {} head {} supply {} ledger {}",
             h, &node.tree.head[..16], node.tree.head_ledger().supply(),
             &node.tree.head_ledger().root()[..12]);
}

/// Dial the configured peers (multiaddrs, comma-separated).
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
