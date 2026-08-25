//! The native token ledger — chain state, mirroring `rig/token.py` bit-exactly.
//!
//! State maps use BTreeMap (sorted keys) and the registry/challenge entries are
//! serde_json Values, so the canonical ledger root — Python's
//! `json.dumps(state, sort_keys=True, separators=(",",":"))` — falls out of
//! `serde_json::to_string` structurally (serde_json's default Map is a BTreeMap).

use crate::verify_sig;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const GRAIN: u64 = 1_000_000_000;
pub const BASE_REWARD: u64 = 50 * GRAIN;
// rev 6: 1M-block epochs (~2yrs at 60s blocks) + TAIL EMISSION instead of a
// hard sunset — the reward floors at the final epoch's value forever, a
// perpetual training wage (inflation asymptotes to ~0%/yr).
pub const HALVING_BLOCKS: u64 = 1_000_000;
pub const TAIL_EPOCH: u64 = 9;
pub const TAIL_REWARD: u64 = BASE_REWARD >> TAIL_EPOCH;
pub const SHARE_MINERS: u64 = 7_000;
pub const SHARE_PROPOSER: u64 = 1_000;
pub const SHARE_DATA: u64 = 2_000;
// rev 6: inference-fee split (basis points, sum 10_000). Server paid instantly
// (absorbs division dust — supply-exact); data + training slices accumulate in
// the ledger fee pools, drained each block to provenance-named data owners and
// delta miners. Usage revenue funds training + data, not just serving.
pub const FEE_SHARE_SERVER: u64 = 6_000;
pub const FEE_SHARE_DATA: u64 = 2_000;
pub const FEE_SHARE_TRAIN: u64 = 2_000;
pub const CHALLENGE_WINDOW: u64 = 20;
pub const PROPOSER_LOOKBACK: usize = 32;
pub const GENESIS_DATA_WEIGHT: u64 = 1_000_000;
/// Minimum affirmative juror votes to uphold a challenge — one juror must never
/// be able to seize an owner's stake; below quorum the challenge is rejected.
pub const CHALLENGE_QUORUM: usize = 3;
/// Blocks a delta's admission bond stays locked (slashable) before it returns.
pub const BOND_WINDOW: u64 = 20;

/// Wallet address: sha256 of the raw pubkey bytes, first 20 bytes, hex.
pub fn address(pub_hex: &str) -> String {
    let bytes = hex::decode(pub_hex).unwrap_or_default();
    hex::encode(&Sha256::digest(&bytes)[..20])
}

/// Deterministic block reward: halves every HALVING_BLOCKS, then floors at
/// TAIL_REWARD forever (tail emission — never zero for h >= 1).
pub fn emission(height: u64) -> u64 {
    if height < 1 {
        return 0;
    }
    (BASE_REWARD >> ((height - 1) / HALVING_BLOCKS).min(62)).max(TAIL_REWARD)
}

// ---------------------------------------------------------------------------
// Account transactions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TransferTx {
    pub from_pub: String,
    pub to_addr: String,
    pub amount: u64,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DataSubmitTx {
    pub owner_pub: String,
    pub data_hash: String,
    pub size_bytes: u64,
    pub media_type: String,
    pub stake: u64,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DataChallengeTx {
    pub challenger_pub: String,
    pub data_id: String,
    pub stake: u64,
    pub reason: String,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DataVoteTx {
    pub voter_pub: String,
    pub challenge_id: String,
    pub support: bool,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct InferenceReceiptTx {
    pub payer_pub: String,
    pub server_addr: String,
    pub fee: u64,
    pub output_hash: String,
    pub head_root: String,
    pub nonce: u64,
    // rev 8: the ANSWER SKETCH — the emitted answer's loss-gradient projected
    // through the shared seeded matrix, quantized. The fee's data slice splits
    // across corpora by positive alignment with their accumulated ledger
    // sketches; empty = unsketched → the slice pools. Committed in the payer's
    // signature; recomputable from the output + head_root model (challengeable).
    pub answer_sketch: Vec<i64>,
    pub sig: Vec<u8>,
}

/// The merged account-tx lane: one nonce sequence per wallet totally orders
/// everything it does.
#[derive(Clone, Debug)]
pub enum AccountTx {
    Transfer(TransferTx),
    DataSubmit(DataSubmitTx),
    DataChallenge(DataChallengeTx),
    DataVote(DataVoteTx),
    InferenceReceipt(InferenceReceiptTx),
}

impl InferenceReceiptTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        // rev 8: one framed field appends the compact JSON of the answer sketch —
        // serde_json's "[1,-2]" is byte-identical to the rig's
        // json.dumps(sk, separators=(",",":")), and "[]" when empty.
        let sk = serde_json::to_string(&self.answer_sketch).unwrap();
        crate::frame(&[b"inference", self.payer_pub.as_bytes(), self.server_addr.as_bytes(),
                       self.fee.to_string().as_bytes(), self.output_hash.as_bytes(),
                       self.head_root.as_bytes(), self.nonce.to_string().as_bytes(),
                       sk.as_bytes()])
    }
}

impl TransferTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        crate::frame(&[b"transfer", self.from_pub.as_bytes(), self.to_addr.as_bytes(),
                       self.amount.to_string().as_bytes(), self.nonce.to_string().as_bytes()])
    }
}

impl DataSubmitTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        crate::frame(&[b"data_submit", self.owner_pub.as_bytes(), self.data_hash.as_bytes(),
                       self.size_bytes.to_string().as_bytes(), self.media_type.as_bytes(),
                       self.stake.to_string().as_bytes(), self.nonce.to_string().as_bytes()])
    }
}

impl DataChallengeTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        crate::frame(&[b"data_challenge", self.challenger_pub.as_bytes(), self.data_id.as_bytes(),
                       self.stake.to_string().as_bytes(), self.reason.as_bytes(),
                       self.nonce.to_string().as_bytes()])
    }
}

impl DataVoteTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        let support = if self.support { 1u8 } else { 0u8 };
        crate::frame(&[b"data_vote", self.voter_pub.as_bytes(), self.challenge_id.as_bytes(),
                       support.to_string().as_bytes(), self.nonce.to_string().as_bytes()])
    }
}

impl AccountTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        match self {
            AccountTx::Transfer(t) => t.signing_bytes(),
            AccountTx::DataSubmit(t) => t.signing_bytes(),
            AccountTx::DataChallenge(t) => t.signing_bytes(),
            AccountTx::DataVote(t) => t.signing_bytes(),
            AccountTx::InferenceReceipt(t) => t.signing_bytes(),
        }
    }

    pub fn txid(&self) -> String {
        hex::encode(Sha256::digest(&self.signing_bytes()))
    }

    pub fn sender_pub(&self) -> &str {
        match self {
            AccountTx::Transfer(t) => &t.from_pub,
            AccountTx::DataSubmit(t) => &t.owner_pub,
            AccountTx::DataChallenge(t) => &t.challenger_pub,
            AccountTx::DataVote(t) => &t.voter_pub,
            AccountTx::InferenceReceipt(t) => &t.payer_pub,
        }
    }

    pub fn nonce(&self) -> u64 {
        match self {
            AccountTx::Transfer(t) => t.nonce,
            AccountTx::DataSubmit(t) => t.nonce,
            AccountTx::DataChallenge(t) => t.nonce,
            AccountTx::DataVote(t) => t.nonce,
            AccountTx::InferenceReceipt(t) => t.nonce,
        }
    }

    pub fn sig(&self) -> &[u8] {
        match self {
            AccountTx::Transfer(t) => &t.sig,
            AccountTx::DataSubmit(t) => &t.sig,
            AccountTx::DataChallenge(t) => &t.sig,
            AccountTx::DataVote(t) => &t.sig,
            AccountTx::InferenceReceipt(t) => &t.sig,
        }
    }

    pub fn verify(&self) -> bool {
        verify_sig(self.sender_pub(), &self.signing_bytes(), self.sig())
    }
}

/// Consensus ordering of ALL account txs in a block: (sender address, nonce, txid).
pub fn canonical_account_txs(txs: &[AccountTx]) -> Vec<AccountTx> {
    let mut out = txs.to_vec();
    out.sort_by_key(|t| (address(t.sender_pub()), t.nonce(), t.txid()));
    out
}

fn set_root(txids: &mut Vec<String>) -> String {
    txids.sort();
    hex::encode(Sha256::digest(txids.join("|").as_bytes()))
}

/// Order-independent commitment to a transfer set.
pub fn transfer_root(transfers: &[TransferTx]) -> String {
    let mut ids: Vec<String> = transfers.iter()
        .map(|t| AccountTx::Transfer(t.clone()).txid()).collect();
    set_root(&mut ids)
}

/// Order-independent commitment to a data-lane tx set.
pub fn data_root(data_txs: &[AccountTx]) -> String {
    let mut ids: Vec<String> = data_txs.iter().map(|t| t.txid()).collect();
    set_root(&mut ids)
}

// ---------------------------------------------------------------------------
// The ledger
// ---------------------------------------------------------------------------

#[derive(Clone, Default, Debug)]
pub struct TokenLedger {
    pub balances: BTreeMap<String, u64>,
    pub nonces: BTreeMap<String, u64>,
    pub registry: BTreeMap<String, Value>,   // data_id -> entry object
    pub challenges: BTreeMap<String, Value>, // challenge_id -> challenge object
    pub bonds: BTreeMap<String, Value>,      // delta_txid -> {miner, amount, expiry}
    // rev 6: inference-fee slices awaiting distribution — the data slice drains
    // to the next block's provenance-named data owners, the training slice to
    // its delta miners. Consensus state (in root + supply).
    pub fee_data_pool: u64,
    pub fee_train_pool: u64,
}

impl TokenLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seed_genesis_data(&mut self, owner_addr: &str) {
        self.registry.insert("genesis".into(), json!({
            "owner": owner_addr, "data_hash": "genesis", "size": 0,
            "media_type": "text", "stake": 0,
            "weight": GENESIS_DATA_WEIGHT, "status": "active"}));
    }

    pub fn balance(&self, addr: &str) -> u64 {
        *self.balances.get(addr).unwrap_or(&0)
    }

    fn credit(&mut self, addr: &str, amount: u64) {
        if amount > 0 {
            // saturating guard: schedule supply is capped far below u64::MAX
            // (10 epochs × HALVING_BLOCKS × BASE_REWARD ≈ 1e17, plus a ~5e13/yr
            // tail ≪ 1.8e19), so a single balance can never actually reach the
            // ceiling — this is a can't-happen defense that stays deterministic
            // instead of panicking/wrapping if the emission constants change.
            let bal = self.balances.entry(addr.to_string()).or_insert(0);
            *bal = bal.saturating_add(amount);
        }
    }

    /// Mint + split the block reward; data share across the weighted registry.
    ///
    /// The three shares (miners / proposer / data) and the per-recipient splits
    /// are floor divisions; the remainders (dust) are DELIBERATELY not minted, so
    /// realized supply is always ≤ the emission schedule, never above it. This is
    /// deterministic (identical on every node) and is the reason `supply()` runs a
    /// hair under the nominal curve — a known, documented property, not a leak.
    pub fn apply_reward(&mut self, height: u64, miner_pubs: &[String],
                        proposer_pub: &str, legacy_data_addrs: &[String],
                        data_credits: &BTreeMap<String, u64>,
                        miner_weights: &BTreeMap<String, u64>) {
        let total = emission(height);
        if total == 0 && self.fee_train_pool == 0 && self.fee_data_pool == 0 {
            return;
        }
        let mut miners_pool = total * SHARE_MINERS / 10_000;
        let proposer_cut = total * SHARE_PROPOSER / 10_000;
        let mut data_pool = total * SHARE_DATA / 10_000;
        // rev 6: drain the fee pools into this block's payouts when they have
        // recipients (a block without miners / named data carries them forward).
        // Division dust is burned, same doctrine as emission dust.
        if !miner_pubs.is_empty() && self.fee_train_pool > 0 {
            miners_pool = miners_pool.saturating_add(self.fee_train_pool);
            self.fee_train_pool = 0;
        }
        if !miner_pubs.is_empty() {
            // rev 7: split ∝ committed delta score when weights are given (and
            // nonzero); equal split otherwise. BTreeMap over the block's miner
            // pubs dedups + iterates sorted — identical to the rig's
            // `sorted(set(miner_pubs))`. Dust burned either way.
            let weights: BTreeMap<&String, u64> = miner_pubs.iter()
                .map(|p| (p, *miner_weights.get(p).unwrap_or(&0)))
                .collect();
            let wsum: u128 = weights.values().map(|w| *w as u128).sum();
            if wsum > 0 {
                for (p, w) in weights {
                    let share = (miners_pool as u128 * w as u128 / wsum) as u64;
                    let a = address(p);
                    self.credit(&a, share);
                }
            } else {
                let each = miners_pool / miner_pubs.len() as u64;
                let mut sorted: Vec<&String> = miner_pubs.iter().collect();
                sorted.sort();
                for p in sorted {
                    let a = address(p);
                    self.credit(&a, each);
                }
            }
        }
        if !proposer_pub.is_empty() && proposer_pub != "genesis" {
            let a = address(proposer_pub);
            self.credit(&a, proposer_cut);
        }
        // rev 5 PROVENANCE: the data share pays the owners of the corpora THIS
        // block's deltas named, ∝ their credit weight — resolved to active
        // registry entries (the on-chain availability proxy). An unbacked hash
        // pays nobody. Mirrors rig.token.apply_reward.
        let hash_to_owner: BTreeMap<String, String> = self.registry.values()
            .filter(|e| e["status"] == "active")
            .filter_map(|e| Some((e["data_hash"].as_str()?.to_string(),
                                  e["owner"].as_str()?.to_string())))
            .collect();
        let mut paid: Vec<(&String, u64)> = data_credits.iter()
            .filter(|(h, w)| **w > 0 && hash_to_owner.contains_key(*h))
            .map(|(h, w)| (h, *w))
            .collect();
        if !paid.is_empty() && self.fee_data_pool > 0 {
            data_pool = data_pool.saturating_add(self.fee_data_pool);
            self.fee_data_pool = 0;
        }
        if !paid.is_empty() {
            paid.sort();                                   // by data_hash, canonical
            let wsum: u128 = paid.iter().map(|(_, w)| *w as u128).sum();
            for (h, w) in paid {
                // u128 intermediate: pool×weight can exceed u64 (Python bigints
                // don't overflow; the floor-divided result always fits u64)
                let share = (data_pool as u128 * w as u128 / wsum) as u64;
                self.credit(&hash_to_owner[h], share);
            }
        } else if !legacy_data_addrs.is_empty() {
            let each = data_pool / legacy_data_addrs.len() as u64;
            let mut sorted: Vec<&String> = legacy_data_addrs.iter().collect();
            sorted.sort();
            for a in sorted {
                self.credit(a, each);
            }
        }
    }

    /// Settle every expired challenge (sorted id order) — FIRST step per block.
    /// Return every stake bond whose lock window has closed (sorted txid order).
    pub fn resolve_expired_bonds(&mut self, height: u64) {
        let ids: Vec<String> = self.bonds.keys().cloned().collect();
        for tid in ids {
            let b = self.bonds[&tid].clone();
            if b["expiry"].as_u64().unwrap() <= height {
                let miner = b["miner"].as_str().unwrap().to_string();
                self.credit(&miner, b["amount"].as_u64().unwrap());
                self.bonds.remove(&tid);
            }
        }
    }

    /// Lock a delta's admission bond from the miner's balance (the Bitcoin analog
    /// of paying to participate, but recoverable). False => block invalid. A zero
    /// bond is a no-op so the fair-launch bootstrap still works.
    pub fn lock_bond(&mut self, delta_txid: &str, miner_addr: &str, amount: u64, height: u64) -> bool {
        if amount == 0 {
            return true;
        }
        if self.balance(miner_addr) < amount {
            return false;
        }
        *self.balances.get_mut(miner_addr).unwrap() -= amount;
        self.bonds.insert(delta_txid.to_string(), json!({
            "miner": miner_addr, "amount": amount, "expiry": height + BOND_WINDOW}));
        true
    }

    pub fn resolve_expired_challenges(&mut self, height: u64) {
        let ids: Vec<String> = self.challenges.keys().cloned().collect();
        for cid in ids {
            let ch = self.challenges[&cid].clone();
            if ch["expiry"].as_u64().unwrap() > height {
                continue;
            }
            let data_id = ch["data_id"].as_str().unwrap().to_string();
            // QUORUM: upheld only with a strict majority AND at least
            // CHALLENGE_QUORUM affirmative juror votes; below quorum → rejected.
            let vf = ch["votes_for"].as_array().unwrap().len();
            let va = ch["votes_against"].as_array().unwrap().len();
            let upheld = vf >= CHALLENGE_QUORUM && vf > va;
            if let Some(entry) = self.registry.get_mut(&data_id) {
                if upheld {
                    let stake = entry["stake"].as_u64().unwrap();
                    entry["status"] = json!("revoked");
                    entry["stake"] = json!(0);
                    let challenger = ch["challenger"].as_str().unwrap().to_string();
                    self.credit(&challenger, stake.saturating_add(ch["stake"].as_u64().unwrap()));
                } else {
                    let owner = entry["owner"].as_str().unwrap().to_string();
                    self.credit(&owner, ch["stake"].as_u64().unwrap());
                }
            }
            self.challenges.remove(&cid);
        }
    }

    pub fn apply_transfer(&mut self, tx: &TransferTx) -> bool {
        let atx = AccountTx::Transfer(tx.clone());
        if !atx.verify() || tx.amount == 0 {
            return false;
        }
        let src = address(&tx.from_pub);
        if tx.nonce != *self.nonces.get(&src).unwrap_or(&0)
            || self.balance(&src) < tx.amount {
            return false;
        }
        *self.balances.get_mut(&src).unwrap() -= tx.amount;
        self.credit(&tx.to_addr, tx.amount);
        self.nonces.insert(src, tx.nonce + 1);
        true
    }

    pub fn apply_data_tx(&mut self, tx: &AccountTx, height: u64,
                         recent_proposers: &std::collections::HashSet<String>) -> bool {
        if !tx.verify() {
            return false;
        }
        let src = address(tx.sender_pub());
        if tx.nonce() != *self.nonces.get(&src).unwrap_or(&0) {
            return false;
        }
        match tx {
            AccountTx::DataSubmit(t) => {
                if t.stake == 0 || self.balance(&src) < t.stake
                    || self.registry.contains_key(&tx.txid()) {
                    return false;
                }
                *self.balances.get_mut(&src).unwrap() -= t.stake;
                self.registry.insert(tx.txid(), json!({
                    "owner": src, "data_hash": t.data_hash, "size": t.size_bytes,
                    "media_type": t.media_type, "stake": t.stake,
                    "weight": t.stake, "status": "active"}));
            }
            AccountTx::DataChallenge(t) => {
                let ok = self.registry.get(&t.data_id)
                    .map(|e| e["status"] == "active").unwrap_or(false);
                let already = self.challenges.values()
                    .any(|c| c["data_id"] == t.data_id.as_str());
                if !ok || already || t.stake == 0 || self.balance(&src) < t.stake {
                    return false;
                }
                *self.balances.get_mut(&src).unwrap() -= t.stake;
                self.challenges.insert(tx.txid(), json!({
                    "data_id": t.data_id, "challenger": src, "stake": t.stake,
                    "reason": t.reason, "expiry": height + CHALLENGE_WINDOW,
                    "votes_for": [], "votes_against": []}));
            }
            AccountTx::DataVote(t) => {
                if !recent_proposers.contains(&t.voter_pub) {
                    return false;
                }
                let Some(ch) = self.challenges.get(&t.challenge_id) else {
                    return false;
                };
                let voted = ch["votes_for"].as_array().unwrap().iter().any(|v| v == src.as_str())
                    || ch["votes_against"].as_array().unwrap().iter().any(|v| v == src.as_str());
                // DISINTERESTED JURORS ONLY: neither the challenger nor the data
                // owner may vote on their own challenge — both are interested
                // parties. Jurors are disinterested recent proposers.
                let is_challenger = ch["challenger"].as_str() == Some(src.as_str());
                let data_id = ch["data_id"].as_str().unwrap().to_string();
                if voted || is_challenger {
                    return false;
                }
                if let Some(entry) = self.registry.get(&data_id) {
                    if entry["owner"].as_str() == Some(src.as_str()) {
                        return false;
                    }
                }
                let ch = self.challenges.get_mut(&t.challenge_id).unwrap();
                let k = if t.support { "votes_for" } else { "votes_against" };
                let arr = ch[k].as_array_mut().unwrap();
                arr.push(json!(src));
                arr.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
            }
            AccountTx::InferenceReceipt(t) => {
                // a signed usage fee: the payer pays for an attested inference.
                // rev 6: the fee splits 60/20/20 — the serving node is paid
                // instantly (absorbing division dust, keeping the split
                // supply-exact); the data + training slices accumulate in the
                // fee pools, drained by the next block's reward to its
                // provenance-named data owners + delta miners. Mirrors
                // rig.token.apply_data_tx.
                if t.fee == 0 || self.balance(&src) < t.fee {
                    return false;
                }
                *self.balances.get_mut(&src).unwrap() -= t.fee;
                let data_cut = t.fee * FEE_SHARE_DATA / 10_000;
                let train_cut = t.fee * FEE_SHARE_TRAIN / 10_000;
                self.credit(&t.server_addr, t.fee - data_cut - train_cut);
                self.fee_train_pool = self.fee_train_pool.saturating_add(train_cut);
                // rev 8 USAGE ATTRIBUTION: if the receipt carries an answer
                // sketch, the data slice pays the corpora whose accumulated
                // ledger sketches POSITIVELY align with it (∝ dot product —
                // data that pushed against the answer earns nothing), directly.
                // Unsketched receipts / no positive alignment → the slice pools
                // as before. i128 dots; mirrors rig.token.apply_data_tx.
                let mut paid_direct = false;
                if t.answer_sketch.iter().any(|x| *x != 0) {
                    let mut aligns: BTreeMap<String, i128> = BTreeMap::new();
                    for e in self.registry.values() {
                        if e["status"] != "active" {
                            continue;
                        }
                        let Some(sk) = e.get("sketch").and_then(|v| v.as_array()) else { continue };
                        if sk.is_empty() {
                            continue;
                        }
                        let d: i128 = sk.iter().zip(t.answer_sketch.iter())
                            .map(|(a, b)| a.as_i64().unwrap_or(0) as i128 * *b as i128)
                            .sum();
                        if d > 0 {
                            if let Some(owner) = e["owner"].as_str() {
                                *aligns.entry(owner.to_string()).or_insert(0) += d;
                            }
                        }
                    }
                    let total: i128 = aligns.values().sum();
                    if total > 0 {
                        for (owner, a) in &aligns {        // BTreeMap = sorted owners
                            let share = (data_cut as i128 * a / total) as u64;
                            self.credit(owner, share);      // dust burned
                        }
                        paid_direct = true;
                    }
                }
                if !paid_direct {
                    self.fee_data_pool = self.fee_data_pool.saturating_add(data_cut);
                }
            }
            AccountTx::Transfer(_) => return false,
        }
        self.nonces.insert(src, tx.nonce() + 1);
        true
    }

    /// Canonical root — byte-identical to the Python reference: compact JSON,
    /// all keys sorted (BTreeMaps + serde_json's sorted Map make it structural).
    pub fn root(&self) -> String {
        let state = json!({
            "balances": self.balances,
            "bonds": self.bonds,
            "challenges": self.challenges,
            "fee_data_pool": self.fee_data_pool,
            "fee_train_pool": self.fee_train_pool,
            "nonces": self.nonces,
            "registry": self.registry,
        });
        hex::encode(Sha256::digest(serde_json::to_string(&state).unwrap().as_bytes()))
    }

    pub fn supply(&self) -> u64 {
        // pool balances are minted/paid tokens in flight, so they count
        self.balances.values().sum::<u64>() + self.fee_data_pool + self.fee_train_pool
    }

    /// Serialize the full ledger for a snapshot (fast-boot). Structural, not the
    /// hashed root form — this round-trips the state itself.
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::json!({
            "balances": self.balances, "nonces": self.nonces,
            "registry": self.registry, "challenges": self.challenges,
            "bonds": self.bonds,
            "fee_data_pool": self.fee_data_pool,
            "fee_train_pool": self.fee_train_pool,
        })
    }

    /// Reconstruct a ledger from a snapshot value (inverse of to_value).
    ///
    /// Returns None if the value is malformed in ANY way — wrong top-level
    /// shape, a non-integer balance/nonce, or a registry/challenge entry missing
    /// a field (or of the wrong type) that a later ledger path unwraps. A
    /// snapshot is untrusted input on the fast-boot path; seeding a partially-
    /// built ledger from it would either corrupt balances or panic a
    /// snapshot-booted node inside validate_block while block-synced peers march
    /// on. On None the caller falls back to full validated replay from genesis.
    pub fn from_value(v: &serde_json::Value) -> Option<Self> {
        let mut led = TokenLedger::new();
        for (k, x) in v["balances"].as_object()? {
            led.balances.insert(k.clone(), x.as_u64()?);
        }
        for (k, x) in v["nonces"].as_object()? {
            led.nonces.insert(k.clone(), x.as_u64()?);
        }
        for (k, e) in v["registry"].as_object()? {
            if !valid_registry_entry(e) {
                return None;
            }
            led.registry.insert(k.clone(), e.clone());
        }
        for (k, c) in v["challenges"].as_object()? {
            if !valid_challenge(c) {
                return None;
            }
            led.challenges.insert(k.clone(), c.clone());
        }
        // bonds are optional for backward compatibility with pre-rev-4 snapshots
        if let Some(bonds) = v.get("bonds").and_then(|b| b.as_object()) {
            for (k, b) in bonds {
                if !(b["miner"].is_string() && b["amount"].is_u64() && b["expiry"].is_u64()) {
                    return None;
                }
                led.bonds.insert(k.clone(), b.clone());
            }
        }
        // fee pools (rev 6) are optional for pre-rev-6 snapshots (default 0),
        // but if present they must be plain u64 — anything else is malformed.
        for key in ["fee_data_pool", "fee_train_pool"] {
            if let Some(x) = v.get(key) {
                let n = x.as_u64()?;
                if key == "fee_data_pool" {
                    led.fee_data_pool = n;
                } else {
                    led.fee_train_pool = n;
                }
            }
        }
        Some(led)
    }
}

/// Every field the reward/resolve/apply/root paths read off a registry entry,
/// with the type they unwrap — validated once at snapshot load so no later
/// `.unwrap()` can panic on a crafted snapshot.
fn valid_registry_entry(e: &serde_json::Value) -> bool {
    e["owner"].is_string()
        && e["data_hash"].is_string()
        && e["size"].is_u64()
        && e["media_type"].is_string()
        && e["stake"].is_u64()
        && e["weight"].is_u64()
        && e["status"].is_string()
}

/// Likewise for an open challenge, including vote lists that must be arrays of
/// address strings (resolve/apply iterate and compare them).
fn valid_challenge(c: &serde_json::Value) -> bool {
    let str_array = |x: &serde_json::Value| {
        x.as_array().is_some_and(|a| a.iter().all(|v| v.is_string()))
    };
    c["data_id"].is_string()
        && c["challenger"].is_string()
        && c["stake"].is_u64()
        && c["reason"].is_string()
        && c["expiry"].is_u64()
        && str_array(&c["votes_for"])
        && str_array(&c["votes_against"])
}
