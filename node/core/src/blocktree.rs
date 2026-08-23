//! Blocks, first-principles validation, and Nakamoto fork choice — mirroring
//! `rig/blockchain.py`. A node holding no special trust validates every block
//! completely: signatures, DA bodies against their hashes, the weight-state
//! transition (trimmed mean), the tx-set root, and — rev 2 — the full token
//! transition (rewards + canonical transfers) against the committed ledger_root.

use crate::model_state::{
    fold as model_fold, page_init, page_state_root, GenesisParams, ModelState,
};
use crate::token::{
    address, canonical_account_txs, data_root, transfer_root, AccountTx, TokenLedger,
    TransferTx, PROPOSER_LOOKBACK,
};
use crate::{
    delta_hash, expected_version, int64_bytes, paged_transition, txset_root, BackpropTx, Header,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub struct Block {
    pub header: Header,
    pub txs: Vec<BackpropTx>,
    pub bodies: HashMap<String, Vec<i64>>, // da_pointer -> dense delta
    pub transfers: Vec<TransferTx>,
    pub data_txs: Vec<AccountTx>,          // rev 3: Data{Submit,Challenge,Vote}
    pub scores: BTreeMap<String, u64>,     // rev 7: txid -> micro-nat held-out score
    pub sketches: BTreeMap<String, Vec<i32>>, // rev 8: txid -> [i32; SKETCH_DIM]
}

/// rev 7: a delta's score = its held-out loss improvement in micro-nats, >= 0,
/// clamped by consensus so a lying proposer can't mint unbounded weight.
pub const SCORE_CAP: u64 = 1_000_000_000;

/// rev 8: influence-sketch dimensionality (one sha256 of sign bits per index in
/// the rig's implicit projection; consensus only carries the committed values).
pub const SKETCH_DIM: usize = 256;
/// Published projection seed + fixed-point scale (rig/sketch.py) — consensus
/// never recomputes sketches, but the constants are part of the protocol.
pub const SKETCH_SEED: u64 = 1234;
pub const SKETCH_SCALE: i128 = 10_000;

/// Canonical commitment to {txid: [ints; SKETCH_DIM]}: sorted compact JSON —
/// byte-identical to rig `json.dumps(sketches, sort_keys=True, separators=(",",":"))`.
pub fn sketch_root(sketches: &BTreeMap<String, Vec<i32>>) -> String {
    crate::sha256_hex_pub(serde_json::to_string(sketches).unwrap().as_bytes())
}

/// Saturate a big-int accumulation into i64 — mirrors rig `_sat64` exactly
/// (clamped to [i64::MIN, i64::MAX]) so ledger sketch accumulators can never
/// overflow/diverge between implementations.
fn sat64(x: i128) -> i64 {
    x.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Canonical commitment to {txid: score}: sorted compact JSON, hashed. BTreeMap
/// + serde_json compact output is byte-identical to the rig's
/// `json.dumps(scores, sort_keys=True, separators=(",",":"))`.
pub fn scores_root(scores: &BTreeMap<String, u64>) -> String {
    crate::sha256_hex_pub(serde_json::to_string(scores).unwrap().as_bytes())
}

/// Consensus scores used for reward weighting: the committed score per txid,
/// with a UNIFORM fallback (all 1) when every score is zero — an unscored block
/// (bootstrap, eval timeout) still splits rewards equally rather than burning
/// them. Deterministic from block content only. Mirrors rig effective_scores.
pub fn effective_scores(txs: &[BackpropTx], scores: &BTreeMap<String, u64>) -> BTreeMap<String, u64> {
    let mut eff: BTreeMap<String, u64> = txs.iter()
        .map(|t| { let id = t.txid(); let s = *scores.get(&id).unwrap_or(&0); (id, s) })
        .collect();
    if !eff.is_empty() && eff.values().all(|s| *s == 0) {
        for v in eff.values_mut() {
            *v = 1;
        }
    }
    eff
}

impl Block {
    pub fn hash(&self) -> String {
        self.header.block_hash()
    }
}

#[derive(Debug)]
pub struct ValidationError(pub String);

fn err(msg: &str) -> ValidationError {
    ValidationError(msg.to_string())
}

/// Full validation against the parent's state (protocol v1 — the live
/// protocol, mirroring the `parent_model is not None` branch of
/// `rig/blockchain.py::validate_block`); returns
/// (post-weights, post-ledger, post-model). Enforces the header version,
/// proposer ELIGIBILITY (stake-weighted VRF sortition with the attempt-widening
/// liveness fallback), page claims (existence, active status, body zero outside
/// claims), the WORK QUOTA, the per-page state transition, growth activation,
/// and the ModelState fold + model_root commitment.
pub fn validate_block(
    block: &Block,
    parent_w: &[i64],
    parent_height: u64,
    parent_ledger: &TokenLedger,
    data_contributor: Option<&str>,
    recent_proposers: &HashSet<String>,
    parent_model: &ModelState,
    params: &GenesisParams,
) -> Result<(Vec<i64>, TokenLedger, ModelState), ValidationError> {
    let h = &block.header;
    let dim = parent_w.len();
    // 0. STRUCTURAL invariants binding the header to its parent and body.
    //    v1: the version must be the scheduled version for this height — the
    //    whole upgrade mechanism (unknown-to-us versions fail loudly upstream).
    if h.version != expected_version(h.height) {
        return Err(ValidationError(format!(
            "header version {} != scheduled {}",
            h.version,
            expected_version(h.height)
        )));
    }
    //    height must advance by exactly one — otherwise a miner could pin a low
    //    height on every block and mint the height-keyed reward forever (halving
    //    /sunset are only meaningful if height is monotone), and a height-0
    //    non-genesis block would underflow `h.height - 1` below.
    if h.height != parent_height + 1 {
        return Err(err("height must be parent height + 1"));
    }
    //    n_txs must match the real tx count (it is committed in the block hash).
    if h.n_txs as usize != block.txs.len() {
        return Err(err("n_txs does not match tx count"));
    }
    // PROPOSER LOTTERY: the VRF proof must be a valid signature by the proposer
    // over this (height, attempt) seed, and header.work must be the
    // attempt-discounted weight derived from it — so work is NON-FORGEABLE.
    // v1 ADDS the eligibility gate itself: the proof must clear the
    // stake-weighted threshold for its attempt (cold-start and ATTEMPT_MAX
    // rules inside lottery::eligible). Genesis is constructed and exempt.
    if h.proposer != "genesis" {
        let proof = hex::decode(&h.vrf_proof).unwrap_or_default();
        if h.vrf_attempt > crate::lottery::ATTEMPT_MAX {
            return Err(err("vrf_attempt out of range"));
        }
        let stake = parent_ledger.balance(&address(&h.proposer));
        let total = parent_ledger.supply();
        if !crate::lottery::eligible(
            &h.proposer, &proof, &h.prev_hash, h.height, h.vrf_attempt, stake, total,
        ) {
            return Err(err("proposer not eligible at this attempt"));
        }
        if h.work != crate::lottery::attempt_work(&proof, h.vrf_attempt) {
            return Err(err("header.work is not the VRF-derived weight"));
        }
    }
    // 1. every delta tx well-formed and signed; its DA body must have the model
    //    dimension (so aggregation can't be made to panic/diverge by a short or
    //    long body) and match its hash. v1 ADDS the page-claim rules: the claim
    //    set is canonical and nonempty, every claimed page exists and is ACTIVE
    //    (frozen pages reject deltas), the body is EXACTLY ZERO outside the
    //    claimed spans (a non-claimant's zero is absence, not a vote for zero),
    //    and the claimed region carries at least the quota's worth of nonzero
    //    work (required_nnz).
    for tx in &block.txs {
        if !tx.verify() {
            return Err(err("bad signature on tx"));
        }
        if tx.base_height != h.height - 1 {
            return Err(err("tx base_height does not match parent"));
        }
        let body = block
            .bodies
            .get(&tx.da_pointer)
            .ok_or_else(|| err("missing DA body"))?;
        if body.len() != dim {
            return Err(err("delta body length != model dimension"));
        }
        if delta_hash(&int64_bytes(body)) != tx.delta_hash {
            return Err(err("delta body hash mismatch"));
        }
        let pages = tx.canonical_pages();
        if pages.is_empty() || tx.pages != pages {
            return Err(err("tx pages must be canonical and nonempty"));
        }
        for &p in &pages {
            if !parent_model.is_active(p as usize) {
                return Err(err("tx claims missing/frozen page"));
            }
        }
        let mut mask = vec![false; dim];
        for &p in &pages {
            let (s, e) = parent_model.page_span(p as usize);
            for m in &mut mask[s as usize..e as usize] {
                *m = true;
            }
        }
        if body.iter().zip(&mask).any(|(&x, &m)| !m && x != 0) {
            return Err(err("delta body nonzero outside claimed pages"));
        }
        let nnz = body.iter().filter(|&&x| x != 0).count() as u64;
        if nnz < parent_model.required_nnz(&pages) {
            return Err(err("delta below work quota"));
        }
    }
    // 2. tx-set root
    let ids: Vec<String> = block.txs.iter().map(|t| t.txid()).collect();
    if txset_root(&ids) != h.txset_root {
        return Err(err("txset_root mismatch"));
    }
    // 2b. DELTA SCORES (rev 7): exactly one committed score per included tx,
    //     in [0, SCORE_CAP], and the commitment reproduces. Scores are block
    //     data — validators never recompute the float eval (cross-GPU
    //     nondeterminism stays outside consensus); a fraudulent score is a
    //     bonded, challengeable claim.
    let txid_set: BTreeSet<&String> = ids.iter().collect();
    let score_keys: BTreeSet<&String> = block.scores.keys().collect();
    if score_keys != txid_set {
        return Err(err("scores must cover exactly the included txs"));
    }
    if block.scores.values().any(|v| *v > SCORE_CAP) {
        return Err(err("score out of range"));
    }
    if scores_root(&block.scores) != h.score_root {
        return Err(err("score_root mismatch"));
    }
    // 2c. INFLUENCE SKETCHES (rev 8): one committed sketch per included tx,
    //     SKETCH_DIM ints each within i32 (an all-zero sketch = "unsketched",
    //     contributing nothing to attribution), and the commitment reproduces.
    //     Entry range is enforced by the i32 type; only shape can be wrong.
    let sketch_keys: BTreeSet<&String> = block.sketches.keys().collect();
    if sketch_keys != txid_set {
        return Err(err("sketches must cover exactly the included txs"));
    }
    if block.sketches.values().any(|v| v.len() != SKETCH_DIM) {
        return Err(err("sketch malformed"));
    }
    if sketch_root(&block.sketches) != h.sketch_root {
        return Err(err("sketch_root mismatch"));
    }
    // 3. the state transition reproduces the committed roots. v1: per-page
    //    trimmed mean over each page's actual claimants, computed against the
    //    PARENT page table; then the ModelState fold — any growth event due
    //    this block appends its deterministically-initialized expert page(s)
    //    AFTER aggregation and BEFORE the root; state_root commits the
    //    page-Merkle root over the (possibly extended) page set. THE ORDER OF
    //    THESE THREE STEPS IS CONSENSUS — must match the rig exactly.
    let bodies: Vec<Vec<i64>> = block
        .txs
        .iter()
        .map(|t| block.bodies[&t.da_pointer].clone())
        .collect();
    let claims: Vec<Vec<u32>> = block.txs.iter().map(|t| t.canonical_pages()).collect();
    let spans: Vec<(u64, u64)> = parent_model.pages.iter().map(|p| (p.start, p.end)).collect();
    let mut w = paged_transition(parent_w, &bodies, &claims, &spans);
    let zero_scored = block
        .txs
        .iter()
        .filter(|t| *block.scores.get(&t.txid()).unwrap_or(&0) == 0)
        .count() as u64;
    let (post_model, activations) = model_fold(
        parent_model,
        params,
        h.height,
        block.txs.len() as u64,
        zero_scored,
        &h.prev_hash,
    );
    for (page_id, _layer, _expert, trigger) in &activations {
        w.extend(page_init(trigger, *page_id, &params.spec));
    }
    if page_state_root(&w, &post_model) != h.state_root {
        return Err(err("state_root does not reproduce from txs"));
    }
    if post_model.model_root() != h.model_root {
        return Err(err("model_root does not reproduce (fold divergence)"));
    }
    // 4. the transfer + data lanes: set roots + full token transition, in the
    //    exact reference order (resolve expired -> rewards -> merged canonical
    //    account txs)
    if transfer_root(&block.transfers) != h.transfer_root {
        return Err(err("transfer_root mismatch"));
    }
    if data_root(&block.data_txs) != h.data_root {
        return Err(err("data_root mismatch"));
    }
    let mut led = parent_ledger.clone();
    led.resolve_expired_challenges(h.height);
    led.resolve_expired_bonds(h.height); // return matured delta bonds first
    let miner_pubs: Vec<String> = block.txs.iter().map(|t| t.miner.clone()).collect();
    let data_addrs: Vec<String> = data_contributor.map(|d| vec![d.to_string()]).unwrap_or_default();
    // rev 5 PROVENANCE: every delta must name data that is staked + active in the
    // registry; the data share is credited to those named corpora (∝ registry
    // weight; loss-score replaces this when delta scoring lands). Mirrors
    // rig.blockchain.apply_ledger.
    let active_hashes: std::collections::BTreeSet<String> = led.registry.values()
        .filter(|e| e["status"] == "active")
        .filter_map(|e| e["data_hash"].as_str().map(|s| s.to_string()))
        .collect();
    for tx in &block.txs {
        if !tx.canonical_refs().iter().any(|r| active_hashes.contains(r)) {
            return Err(err("delta names no staked/available data (provenance required)"));
        }
    }
    // DELTA SCORING (rev 7): rewards are weighted by each delta's committed
    // held-out-loss score. Miners: pool split ∝ their deltas' scores. Data: each
    // delta's score splits equally across its named active corpora (scaled by
    // 10_000 so integer division doesn't vanish small scores). All-zero scores
    // fall back to uniform — deterministic from block content alone. Mirrors
    // rig.blockchain.apply_ledger.
    let eff = effective_scores(&block.txs, &block.scores);
    let active_set: BTreeSet<String> = led.registry.values()
        .filter(|e| e["status"] == "active" && e["weight"].as_u64().unwrap_or(0) > 0)
        .filter_map(|e| e["data_hash"].as_str().map(String::from))
        .collect();
    let mut miner_weights: BTreeMap<String, u64> = Default::default();
    let mut data_credits: BTreeMap<String, u64> = Default::default();
    // rev 8: data_hash -> registry KEY of its active entry, for sketch accrual
    // (active-only, NOT weight-gated — mirrors rig hash_to_entry).
    let hash_to_key: BTreeMap<String, String> = led.registry.iter()
        .filter(|(_, e)| e["status"] == "active")
        .filter_map(|(k, e)| Some((e["data_hash"].as_str()?.to_string(), k.clone())))
        .collect();
    for tx in &block.txs {
        let txid = tx.txid();
        let s = eff[&txid];
        *miner_weights.entry(tx.miner.clone()).or_insert(0) += s;
        let named: Vec<String> = tx.canonical_refs().into_iter()
            .filter(|r| active_set.contains(r)).collect();
        for r in &named {
            *data_credits.entry(r.clone()).or_insert(0) += s * 10_000 / named.len() as u64;
        }
        // rev 8: accrue this delta's committed influence sketch onto the corpora
        // it named — a corpus's ledger sketch = Σ (its deltas' sketches), the
        // projection of its total contribution to the weights. Saturating i64;
        // floor division (rig `//`, positive divisor) = div_euclid. Mirrors
        // rig.blockchain.apply_ledger.
        let sk = &block.sketches[&txid];
        if sk.iter().any(|x| *x != 0) {
            let named_keys: Vec<&String> = tx.canonical_refs().iter()
                .filter_map(|r| hash_to_key.get(r))
                .collect::<Vec<_>>();
            let n = named_keys.len() as i128;
            if n > 0 {
                for key in named_keys {
                    let e = led.registry.get_mut(key).unwrap();
                    let acc: Vec<i64> = match e.get("sketch").and_then(|v| v.as_array()) {
                        Some(a) if !a.is_empty() =>
                            a.iter().map(|x| x.as_i64().unwrap_or(0)).collect(),
                        _ => vec![0i64; SKETCH_DIM],
                    };
                    let new: Vec<i64> = acc.iter().zip(sk.iter())
                        .map(|(a, x)| sat64(*a as i128
                             + (*x as i128 * SKETCH_SCALE).div_euclid(n)))
                        .collect();
                    e["sketch"] = serde_json::json!(new);
                }
            }
        }
    }
    led.apply_reward(h.height, &miner_pubs, &h.proposer, &data_addrs, &data_credits,
                     &miner_weights);
    // lock each included delta's admission bond (after the reward, so this
    // block's reward can fund its bond); an unaffordable bond invalidates the block
    for tx in &block.txs {
        if !led.lock_bond(&tx.txid(), &crate::token::address(&tx.miner), tx.bond, h.height) {
            return Err(err("miner cannot afford delta bond"));
        }
    }
    let mut merged: Vec<AccountTx> = block.data_txs.clone();
    merged.extend(block.transfers.iter().cloned().map(AccountTx::Transfer));
    for tx in canonical_account_txs(&merged) {
        let ok = match &tx {
            AccountTx::Transfer(t) => led.apply_transfer(t),
            _ => led.apply_data_tx(&tx, h.height, recent_proposers),
        };
        if !ok {
            return Err(err("invalid account tx (sig/nonce/balance/gating)"));
        }
    }
    if led.root() != h.ledger_root {
        return Err(err("ledger_root does not reproduce from block"));
    }
    Ok((w, led, post_model))
}

/// All known blocks with heaviest-valid-chain (Nakamoto) fork choice.
pub struct BlockTree {
    pub blocks: HashMap<String, Header>, // header per hash (bodies not retained)
    pub state: HashMap<String, Vec<i64>>,
    pub ledger: HashMap<String, TokenLedger>,
    /// v1: per-block ModelState — small, NEVER pruned (like ledgers/headers).
    pub model: HashMap<String, ModelState>,
    pub cum_work: HashMap<String, u64>,
    pub head: String,
    pub genesis_hash: String,
    pub data_contributor: Option<String>,
    /// v1: the genesis parameters (ModelSpec + retarget constants) — identical
    /// on every node; the page table is a pure function of these + growth events.
    pub params: GenesisParams,
    /// keep full state vectors only this many blocks below the head (plus
    /// genesis) — an 86M state is ~0.7GB, so retaining one per block OOMs.
    /// Headers/ledgers/cum_work are kept forever (fork choice needs them).
    pub prune_depth: Option<u64>,
}

/// The genesis header for an initial weight vector — the network's shared trust
/// anchor. Its block_hash is the genesis id; a joining node fetches the genesis
/// weights from a peer and verifies `genesis_block_hash(w, params)` equals the
/// published id before adopting them, so the genesis is public + self-verifying.
/// v1: the state commitment is the page-Merkle root and the header carries the
/// ModelState commitment from block 0.
pub fn genesis_header(genesis_w: &[i64], params: &GenesisParams) -> Header {
    let model0 = ModelState::genesis(&params.spec);
    assert_eq!(
        model0.dim() as usize,
        genesis_w.len(),
        "genesis weight length must equal the ModelSpec page table"
    );
    Header {
        height: 0,
        prev_hash: "0".repeat(64),
        state_root: page_state_root(genesis_w, &model0),
        txset_root: crate::sha256_hex_pub(b""),
        n_txs: 0,
        work: 0,
        proposer: "genesis".into(),
        transfer_root: String::new(),
        ledger_root: String::new(),
        data_root: String::new(),
        vrf_proof: String::new(),
        score_root: String::new(),
        sketch_root: String::new(),
        model_root: model0.model_root(),
        vrf_attempt: 0,
        version: 1,
    }
}

/// The genesis block id (hash) for an initial weight vector.
pub fn genesis_block_hash(genesis_w: &[i64], params: &GenesisParams) -> String {
    genesis_header(genesis_w, params).block_hash()
}

impl BlockTree {
    pub fn new(
        genesis_w: Vec<i64>,
        data_contributor: Option<String>,
        params: GenesisParams,
    ) -> Self {
        let gh = genesis_header(&genesis_w, &params);
        let ghash = gh.block_hash();
        let model0 = ModelState::genesis(&params.spec);
        let mut t = BlockTree {
            blocks: HashMap::new(),
            state: HashMap::new(),
            ledger: HashMap::new(),
            model: HashMap::new(),
            cum_work: HashMap::new(),
            head: ghash.clone(),
            genesis_hash: ghash.clone(),
            data_contributor,
            params,
            prune_depth: None,
        };
        t.blocks.insert(ghash.clone(), gh);
        t.state.insert(ghash.clone(), genesis_w);
        t.model.insert(ghash.clone(), model0);
        // fair launch: empty balances; the founding corpus is registry entry zero
        let mut genesis_ledger = TokenLedger::new();
        if let Some(dc) = &t.data_contributor {
            genesis_ledger.seed_genesis_data(dc);
        }
        t.ledger.insert(ghash.clone(), genesis_ledger);
        t.cum_work.insert(ghash, 0);
        t
    }

    /// Proposer pubkeys of the last PROPOSER_LOOKBACK blocks ending at `tip` —
    /// the deterministic juror set for data-challenge votes.
    pub fn recent_proposers(&self, tip: &str) -> HashSet<String> {
        let mut out = HashSet::new();
        let mut cur = tip.to_string();
        for _ in 0..PROPOSER_LOOKBACK {
            if cur == self.genesis_hash {
                break;
            }
            let Some(h) = self.blocks.get(&cur) else { break };
            out.insert(h.proposer.clone());
            cur = h.prev_hash.clone();
        }
        out
    }

    /// Validate + attach; returns Ok(true) if the block became the new head.
    pub fn add_block(&mut self, block: Block) -> Result<bool, ValidationError> {
        let bh = block.hash();
        if self.blocks.contains_key(&bh) {
            return Ok(false);
        }
        let parent = block.header.prev_hash.clone();
        let parent_w = self
            .state
            .get(&parent)
            .ok_or_else(|| err("orphan: parent unknown"))?
            .clone();
        let parent_led = self.ledger[&parent].clone();
        let jurors = self.recent_proposers(&parent);
        let parent_height = self.blocks[&parent].height;
        let parent_model = self.model[&parent].clone();
        let (w, led, model) = validate_block(
            &block,
            &parent_w,
            parent_height,
            &parent_led,
            self.data_contributor.as_deref(),
            &jurors,
            &parent_model,
            &self.params,
        )?;
        // saturating: keeps fork-choice bookkeeping panic-free and
        // deterministic even at the u64 boundary (work itself is validated
        // against the VRF proof above).
        let work = self.cum_work[&parent].saturating_add(block.header.work.max(1));
        self.blocks.insert(bh.clone(), block.header);
        self.state.insert(bh.clone(), w);
        self.ledger.insert(bh.clone(), led);
        self.model.insert(bh.clone(), model);
        self.cum_work.insert(bh.clone(), work);
        // heaviest chain wins; ties broken by lexicographically smaller hash
        let head_work = self.cum_work[&self.head];
        let became = work > head_work || (work == head_work && bh < self.head);
        if became {
            self.head = bh;
        }
        self.prune_deep();
        Ok(became)
    }

    /// Drop heavy state vectors more than prune_depth below the head.
    fn prune_deep(&mut self) {
        let Some(depth) = self.prune_depth else { return };
        let head_h = self.blocks[&self.head].height;
        let floor = head_h.saturating_sub(depth);
        let doomed: Vec<String> = self.state.keys()
            .filter(|h| **h != self.genesis_hash
                    && self.blocks[*h].height < floor)
            .cloned().collect();
        for h in doomed {
            self.state.remove(&h);
        }
    }

    pub fn head_state(&self) -> &Vec<i64> {
        &self.state[&self.head]
    }

    pub fn head_ledger(&self) -> &TokenLedger {
        &self.ledger[&self.head]
    }

    pub fn head_model(&self) -> &ModelState {
        &self.model[&self.head]
    }
}
