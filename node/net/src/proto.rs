//! Wire types and JSON codecs — gossip messages, sync request/response, and
//! the commitment-only block representation (bodies never ride in blocks; they
//! are reconstructed from the compressed payload store, exactly as the Python
//! client does).

use base64::Engine;
use sestrian_core::{
    self as core,
    blocktree::Block,
    token::{AccountTx, DataChallengeTx, DataSubmitTx, DataVoteTx, InferenceReceiptTx, TransferTx},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn b64(v: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(v)
}

pub fn unb64(s: &str) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD.decode(s).ok()
}

// ---------------------------------------------------------------------------
// Compressed delta payloads (top-k sparse, the transmission form; densifies to
// the exact int64 vector the chain commits to)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Payload {
    pub n: usize,        // dense length
    pub idx: String,     // base64 of u32-LE indices
    pub val: String,     // base64 of i32-LE values
}

impl Payload {
    pub fn dense(&self) -> Option<Vec<i64>> {
        let idx = unb64(&self.idx)?;
        let val = unb64(&self.val)?;
        Some(core::decompress(self.n, &idx, &val))
    }

    /// Decompress only coordinates [lo, hi) — the bounded-memory building block
    /// for chunked aggregation, so a small peer never materializes the full
    /// dense delta. Identical to `dense()[lo..hi]` without allocating the whole.
    pub fn dense_range(&self, lo: usize, hi: usize) -> Vec<i64> {
        let mut out = vec![0i64; hi.saturating_sub(lo)];
        let (Some(idx), Some(val)) = (unb64(&self.idx), unb64(&self.val)) else { return out };
        for i in 0..idx.len() / 4 {
            let j = u32::from_le_bytes(idx[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
            if j >= lo && j < hi {
                out[j - lo] = i32::from_le_bytes(val[i * 4..i * 4 + 4].try_into().unwrap()) as i64;
            }
        }
        out
    }

    pub fn wire_bytes(&self) -> usize {
        self.idx.len() * 3 / 4 + self.val.len() * 3 / 4
    }

    /// Sparse-encode an int64 vector (used for aggregate deltas to the bridge —
    /// values may exceed i32 so this variant carries i64 values).
    /// Canonical sparse coords of this payload: sorted unique indices,
    /// LAST WRITE WINS on duplicates (byte-identical to decompress), zeros
    /// dropped. The incremental engine's native body form — no dense vector.
    pub fn coords(&self) -> Option<Vec<(u32, i64)>> {
        let idx = unb64(&self.idx)?;
        let val = unb64(&self.val)?;
        if idx.len() / 4 != val.len() / 4 {
            return None;
        }
        let k = idx.len() / 4;
        let mut m: std::collections::BTreeMap<u32, i64> = Default::default();
        for i in 0..k {
            let ix = u32::from_le_bytes(idx[i * 4..i * 4 + 4].try_into().ok()?);
            if (ix as usize) >= self.n {
                return None; // out-of-range index would panic dense paths
            }
            let v = i32::from_le_bytes(val[i * 4..i * 4 + 4].try_into().ok()?) as i64;
            m.insert(ix, v);
        }
        Some(m.into_iter().filter(|(_, v)| *v != 0).collect())
    }

    pub fn from_coords_i64(n: usize, coords: &[(u32, i64)]) -> SparseI64 {
        let mut idx = Vec::with_capacity(coords.len() * 4);
        let mut val = Vec::with_capacity(coords.len() * 8);
        for &(i, x) in coords {
            idx.extend_from_slice(&i.to_le_bytes());
            val.extend_from_slice(&x.to_le_bytes());
        }
        SparseI64 { n, idx: b64(&idx), val: b64(&val) }
    }

}

/// Sparse vector with full i64 values (aggregates; bridge advances).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SparseI64 {
    pub n: usize,
    pub idx: String,
    pub val: String,
}

// ---------------------------------------------------------------------------
// Commitment-only stored blocks (what gossips, persists, and syncs)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WireHeader {
    pub height: u64,
    pub prev_hash: String,
    pub state_root: String,
    pub txset_root: String,
    pub n_txs: u64,
    pub work: u64,
    pub proposer: String,
    pub transfer_root: String,
    pub ledger_root: String,
    pub data_root: String,
    pub vrf_proof: String,
    pub score_root: String,    // rev 7: commitment to the proposer's delta scores
    pub sketch_root: String,   // rev 8: commitment to the deltas' influence sketches
    // PROTOCOL v1 — deliberately NO serde defaults on any field from here on
    // (or above): a pre-fork peer/disk artifact must fail to parse loudly, not
    // half-deserialize into a hash that can never validate.
    pub model_root: String,    // ModelState commitment AFTER this block
    pub vrf_attempt: u64,      // sortition attempt the proof binds to
    pub version: u64,          // protocol version (VERSION_SCHEDULE-checked)
}

impl WireHeader {
    pub fn to_core(&self) -> core::Header {
        core::Header {
            height: self.height,
            prev_hash: self.prev_hash.clone(),
            state_root: self.state_root.clone(),
            txset_root: self.txset_root.clone(),
            n_txs: self.n_txs,
            work: self.work,
            proposer: self.proposer.clone(),
            transfer_root: self.transfer_root.clone(),
            ledger_root: self.ledger_root.clone(),
            data_root: self.data_root.clone(),
            vrf_proof: self.vrf_proof.clone(),
            score_root: self.score_root.clone(),
            sketch_root: self.sketch_root.clone(),
            model_root: self.model_root.clone(),
            vrf_attempt: self.vrf_attempt,
            version: self.version,
        }
    }

    pub fn from_core(h: &core::Header) -> Self {
        WireHeader {
            height: h.height,
            prev_hash: h.prev_hash.clone(),
            state_root: h.state_root.clone(),
            txset_root: h.txset_root.clone(),
            n_txs: h.n_txs,
            work: h.work,
            proposer: h.proposer.clone(),
            transfer_root: h.transfer_root.clone(),
            ledger_root: h.ledger_root.clone(),
            data_root: h.data_root.clone(),
            vrf_proof: h.vrf_proof.clone(),
            score_root: h.score_root.clone(),
            sketch_root: h.sketch_root.clone(),
            model_root: h.model_root.clone(),
            vrf_attempt: h.vrf_attempt,
            version: h.version,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WireDeltaTx {
    pub miner: String,
    pub base_height: u64,
    pub delta_hash: String,
    pub da_pointer: String,
    pub bond: u64,
    /// v1: the claimed page ids (sorted, deduped) — no serde default; a tx
    /// without a claim set is a pre-fork artifact and must fail to parse.
    pub pages: Vec<u32>,
    #[serde(default)]
    pub data_refs: Vec<String>,   // rev 5: provenance — corpora this gradient trained on
    pub sig: String, // hex
}

impl WireDeltaTx {
    pub fn to_core(&self) -> Option<core::BackpropTx> {
        Some(core::BackpropTx {
            miner: self.miner.clone(),
            base_height: self.base_height,
            delta_hash: self.delta_hash.clone(),
            da_pointer: self.da_pointer.clone(),
            bond: self.bond,
            pages: self.pages.clone(),
            data_refs: self.data_refs.clone(),
            sig: hex::decode(&self.sig).ok()?,
        })
    }

    pub fn from_core(t: &core::BackpropTx) -> Self {
        WireDeltaTx {
            miner: t.miner.clone(),
            base_height: t.base_height,
            delta_hash: t.delta_hash.clone(),
            da_pointer: t.da_pointer.clone(),
            bond: t.bond,
            pages: t.pages.clone(),
            data_refs: t.data_refs.clone(),
            sig: hex::encode(&t.sig),
        }
    }
}

/// Account txs on the wire: tagged JSON.
pub fn account_tx_to_json(t: &AccountTx) -> Value {
    match t {
        AccountTx::Transfer(x) => json!({"kind": "transfer", "from_pub": x.from_pub,
            "to_addr": x.to_addr, "amount": x.amount, "nonce": x.nonce,
            "sig": hex::encode(&x.sig)}),
        AccountTx::DataSubmit(x) => json!({"kind": "data_submit", "owner_pub": x.owner_pub,
            "data_hash": x.data_hash, "size_bytes": x.size_bytes,
            "media_type": x.media_type, "stake": x.stake, "nonce": x.nonce,
            "da_root": x.da_root,
            "sig": hex::encode(&x.sig)}),
        AccountTx::DataChallenge(x) => json!({"kind": "data_challenge",
            "challenger_pub": x.challenger_pub, "data_id": x.data_id,
            "stake": x.stake, "reason": x.reason, "nonce": x.nonce,
            "sig": hex::encode(&x.sig)}),
        AccountTx::DataVote(x) => json!({"kind": "data_vote", "voter_pub": x.voter_pub,
            "challenge_id": x.challenge_id, "support": x.support, "nonce": x.nonce,
            "sig": hex::encode(&x.sig)}),
        AccountTx::InferenceReceipt(x) => json!({"kind": "inference",
            "payer_pub": x.payer_pub, "server_addr": x.server_addr, "fee": x.fee,
            "output_hash": x.output_hash, "head_root": x.head_root, "nonce": x.nonce,
            "answer_sketch": x.answer_sketch,
            "sig": hex::encode(&x.sig)}),
    }
}

pub fn account_tx_from_json(v: &Value) -> Option<AccountTx> {
    let sig = hex::decode(v["sig"].as_str()?).ok()?;
    Some(match v["kind"].as_str()? {
        "transfer" => AccountTx::Transfer(TransferTx {
            from_pub: v["from_pub"].as_str()?.into(),
            to_addr: v["to_addr"].as_str()?.into(),
            amount: v["amount"].as_u64()?,
            nonce: v["nonce"].as_u64()?, sig,
        }),
        "data_submit" => AccountTx::DataSubmit(DataSubmitTx {
            owner_pub: v["owner_pub"].as_str()?.into(),
            data_hash: v["data_hash"].as_str()?.into(),
            size_bytes: v["size_bytes"].as_u64()?,
            media_type: v["media_type"].as_str()?.into(),
            stake: v["stake"].as_u64()?,
            nonce: v["nonce"].as_u64()?,
            // Missing on a pre-§7.2a peer: decode to "" and let the ledger's
            // well-formedness rule reject it. Defaulting to anything acceptable
            // here would let an old-format submission become a registry entry
            // with no availability commitment — the exact hole this closes.
            da_root: v["da_root"].as_str().unwrap_or("").into(),
            sig,
        }),
        "data_challenge" => AccountTx::DataChallenge(DataChallengeTx {
            challenger_pub: v["challenger_pub"].as_str()?.into(),
            data_id: v["data_id"].as_str()?.into(),
            stake: v["stake"].as_u64()?,
            reason: v["reason"].as_str()?.into(),
            nonce: v["nonce"].as_u64()?, sig,
        }),
        "data_vote" => AccountTx::DataVote(DataVoteTx {
            voter_pub: v["voter_pub"].as_str()?.into(),
            challenge_id: v["challenge_id"].as_str()?.into(),
            support: v["support"].as_bool()?,
            nonce: v["nonce"].as_u64()?, sig,
        }),
        "inference" => AccountTx::InferenceReceipt(InferenceReceiptTx {
            payer_pub: v["payer_pub"].as_str()?.into(),
            server_addr: v["server_addr"].as_str()?.into(),
            fee: v["fee"].as_u64()?,
            output_hash: v["output_hash"].as_str()?.into(),
            head_root: v["head_root"].as_str()?.into(),
            answer_sketch: v.get("answer_sketch").and_then(|a| a.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default(),
            nonce: v["nonce"].as_u64()?, sig,
        }),
        _ => return None,
    })
}

/// A block as gossiped/stored: commitments only, no bodies.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StoredBlock {
    pub header: WireHeader,
    pub txs: Vec<WireDeltaTx>,
    pub transfers: Vec<Value>, // account-tx JSON (kind=transfer)
    pub data_txs: Vec<Value>,  // account-tx JSON (data lane)
    #[serde(default)]
    pub scores: std::collections::BTreeMap<String, u64>, // rev 7: txid -> score
    #[serde(default)]
    pub sketches: std::collections::BTreeMap<String, Vec<i32>>, // rev 8: txid -> sketch
}

impl StoredBlock {
    pub fn hash(&self) -> String {
        self.header.to_core().block_hash()
    }

    /// Materialize a validatable core Block by densifying bodies from payloads.
    /// Returns None if any payload is missing or malformed.
    /// Like `to_core`, but bodies stay SPARSE — the incremental engine's
    /// input. The connect path never materializes a dense vector per delta.
    pub fn to_core_sparse(&self, payloads: &HashMap<String, Payload>) -> Option<Block> {
        let mut b = self.to_core_inner(payloads, false)?;
        for wt in &self.txs {
            let t = wt.to_core()?;
            let p = payloads.get(&t.txid())?;
            b.sparse.insert(t.da_pointer.clone(), (p.n as u64, p.coords()?));
        }
        Some(b)
    }

    pub fn to_core(&self, payloads: &HashMap<String, Payload>) -> Option<Block> {
        self.to_core_inner(payloads, true)
    }

    fn to_core_inner(&self, payloads: &HashMap<String, Payload>, dense: bool) -> Option<Block> {
        let mut txs = Vec::new();
        let mut bodies = HashMap::new();
        for wt in &self.txs {
            let t = wt.to_core()?;
            if dense {
                let p = payloads.get(&t.txid())?;
                bodies.insert(t.da_pointer.clone(), p.dense()?);
            }
            txs.push(t);
        }
        let mut transfers = Vec::new();
        for v in &self.transfers {
            match account_tx_from_json(v)? {
                AccountTx::Transfer(t) => transfers.push(t),
                _ => return None,
            }
        }
        let mut data_txs = Vec::new();
        for v in &self.data_txs {
            data_txs.push(account_tx_from_json(v)?);
        }
        Some(Block { header: self.header.to_core(), txs, bodies,
                     sparse: HashMap::new(), transfers, data_txs,
                     scores: self.scores.clone(), sketches: self.sketches.clone() })
    }
}

// ---------------------------------------------------------------------------
// Gossip envelope + sync protocol
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(tag = "t")]
pub enum Gossip {
    /// A miner's delta commitment + its compressed payload.
    Dtx { tx: WireDeltaTx, payload: Payload },
    /// An account tx (transfer or data lane).
    Atx { tx: Value },
    /// A block — commitments only.
    Blk { block: StoredBlock },
    /// Tiny per-round head announcement — the self-healing heartbeat: any node
    /// seeing an unknown head syncs from its sender, so divergence resolves
    /// within a round regardless of what earlier messages were lost.
    Head { hash: String, height: u64 },
}

/// Range chain sync over libp2p request-response (JSON codec).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SyncRequest {
    pub from_height: u64,
    pub count: u64,
    /// a fresh node with no genesis asks for it here — served + self-verified
    /// against the published genesis id, so the genesis is public + fetchable.
    #[serde(default)]
    pub want_genesis: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SyncResponse {
    pub blocks: Vec<StoredBlock>,
    /// BUSY is not "nothing to send". A sync response carries ~31MB, and a
    /// peer stuck in a retry loop asked every ~15s: serving it unbounded grew
    /// a HEALTHY node's RSS by 2.4GB in 15 minutes (measured), which is how a
    /// single broken peer walks its neighbours into the OOM killer. Over the
    /// cap we answer BUSY so the client retries rather than concluding the
    /// window is empty and marching its cursor past blocks it still needs.
    #[serde(default)]
    pub busy: bool,
    pub payloads: HashMap<String, Payload>, // txid -> payload for those blocks
    pub head_height: u64,
    /// the genesis weight vector, included when the requester set want_genesis
    #[serde(default)]
    pub genesis: Option<Vec<i64>>,
}

/// Data-availability shard exchange (§3.3). A node missing a body asks peers for
/// its erasure shards; each peer returns whatever shards it holds. The requester
/// gathers K across peers and reconstructs — so a body stays recoverable even
/// when no single node retains it whole.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ShardRequest {
    pub txids: Vec<String>,
}

/// PEER EXCHANGE. Nodes previously knew only the baked-in anchors plus any
/// explicit `--peers`, so the topology was a STAR through two hosts: joiners
/// could not find each other, every node's traffic funnelled through the
/// anchors, and two miners on the same LAN sat with no direct link — which is
/// how they forked when anchor connectivity churned. A node asks a peer who
/// else it is connected to and dials a bounded number of them, so the mesh
/// closes itself without a DHT.
#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct PeerRequest {
    /// The asker's OWN dialable multiaddr (ending /p2p/<id>), self-declared.
    /// libp2p identify advertises only CONFIRMED external addresses, which on
    /// a private or NAT'd network is nothing at all — measured: identify
    /// returned zero addresses, so peer exchange had nothing to hand out.
    /// Self-declaring is exactly what identify does when it works, and the
    /// responder verifies the embedded peer id matches the actual sender.
    #[serde(default)]
    pub me: String,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct PeerResponse {
    /// dialable multiaddrs, each ending in /p2p/<peer-id>. Bounded by the
    /// responder; a peer that floods this is simply ignored past the cap.
    pub peers: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ShardResponse {
    pub bodies: Vec<BodyShards>,
    /// BUSY is not ABSENT. The first backpressure attempt replied with an
    /// EMPTY body list when over its cap, which a client cannot distinguish
    /// from "I do not have these" — so it stopped asking, delta bodies
    /// stopped flowing, and a routine head tie could not heal. An explicit
    /// flag lets the client retry instead of giving up.
    #[serde(default)]
    pub busy: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BodyShards {
    pub txid: String,
    pub k: u32,
    pub n: u32,
    pub orig_len: u64,
    pub shards: Vec<(u32, String)>, // (index, base64 shard bytes)
}
