//! Page fraud-proof verifier (Sharding Road, Phase 1) — the Rust side of the
//! dispute game, bit-exact with `rig/fraud.py::verify`.
//!
//! A proof convicts a block of committing a wrong post-state for ONE page,
//! checkable by a node that holds only the two block headers. The verdict
//! must match the reference exactly: acting on a false conviction, or missing
//! a real one, both fork the chain in a sharded world.

use crate::merkle;
use crate::int64_bytes;
use crate::{delta_hash, trimmed_mean_scalar, BackpropTx, Header};
use sha2::{Digest, Sha256};

fn sha_hex(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

/// A claimant's sparse body inside a proof.
pub struct Body {
    pub n: usize,
    pub idx: Vec<u64>,
    pub val: Vec<i64>,
}

/// A tx record carried by a proof (core has no serde derive on BackpropTx;
/// the net/test layer parses JSON and hands these in).
pub struct TxRec {
    pub tx: BackpropTx,
}

/// A page fraud proof, fully parsed. The net layer and the golden test build
/// this from JSON; the verifier is pure logic over owned data.
pub struct PageFraudProof {
    pub header: Header,
    pub parent_header: Header,
    pub parent_model_json: String,
    pub committed_leaves: Vec<String>,
    pub page_id: usize,
    pub txids: Vec<String>,
    pub txs: Vec<TxRec>,
    pub bodies: std::collections::HashMap<String, Body>,
    pub parent_page: Vec<i64>,
    pub parent_path: Vec<(bool, String)>,
}

/// (fraud_proven, reason). `true` == the accused block is provably fraudulent.
/// A `false` with an "invalid:" reason means the PROOF is malformed and must
/// never be acted on.
pub fn verify(p: &PageFraudProof) -> (bool, String) {
    let inv = |m: &str| (false, format!("invalid: {m}"));

    if p.header.prev_hash != p.parent_header.block_hash() {
        return inv("header does not extend parent");
    }
    if sha_hex(p.parent_model_json.as_bytes()) != p.parent_header.model_root {
        return inv("parent model json does not match model_root");
    }
    let mj: serde_json::Value = match serde_json::from_str(&p.parent_model_json) {
        Ok(m) => m,
        Err(_) => return inv("unparseable parent model json"),
    };
    let pages = match mj.get("pages").and_then(|x| x.as_array()) {
        Some(a) if !a.is_empty() => a,
        _ => return inv("parent model has no pages"),
    };
    let span = |i: usize| -> (usize, usize) {
        let r = pages[i].as_array().unwrap();
        (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize)
    };
    let pid = p.page_id;
    if pid >= pages.len() {
        return inv("page id out of range");
    }
    let (s, e) = span(pid);
    let dim = span(pages.len() - 1).1;

    // committed leaves bind through the accused state_root itself
    if p.committed_leaves.len() != pages.len() {
        return inv("growth blocks are not disputable in fraud v1");
    }
    let mut level: Vec<[u8; 32]> = Vec::with_capacity(p.committed_leaves.len());
    for h in &p.committed_leaves {
        match hex::decode(h).ok().and_then(|b| b.try_into().ok()) {
            Some(a) => level.push(a),
            None => return inv("malformed committed leaf"),
        }
    }
    if hex::encode(merkle::root_from_hashes(level.clone())) != p.header.state_root {
        return inv("committed leaves do not fold to state_root");
    }

    // tx-set completeness through txset_root
    let mut sorted_ids = p.txids.clone();
    sorted_ids.sort();
    if sha_hex(sorted_ids.join("|").as_bytes()) != p.header.txset_root {
        return inv("txids do not reproduce txset_root");
    }
    if p.txs.len() != p.txids.len() || p.txids.len() as u64 != p.header.n_txs {
        return inv("tx list incomplete");
    }

    // claimant bodies bind through delta_hash; recompute the page column set
    let mut claim_cols: Vec<Vec<i64>> = Vec::new();
    for rec in &p.txs {
        let tx = &rec.tx;
        if !p.txids.contains(&tx.txid()) {
            return inv("tx not in committed set");
        }
        if !tx.verify() {
            return inv("bad tx signature");
        }
        if tx.base_height != p.parent_header.height {
            return inv("tx base height mismatch");
        }
        if !tx.canonical_pages().contains(&(pid as u32)) {
            continue;
        }
        let body = match p.bodies.get(&tx.txid()) {
            Some(b) => b,
            None => return inv("missing claimant body"),
        };
        if body.n != dim || body.idx.len() != body.val.len() {
            return inv("malformed body");
        }
        // dense image (for the hash) + the page column
        let mut dense = vec![0i64; dim];
        let mut last: i64 = -1;
        for (i, v) in body.idx.iter().zip(&body.val) {
            let i = *i as usize;
            if i >= dim || i as i64 <= last {
                return inv("body coords not sorted-unique in range");
            }
            last = i as i64;
            dense[i] = *v;
        }
        if delta_hash(&int64_bytes(&dense)) != tx.delta_hash {
            return inv("body does not match delta_hash");
        }
        claim_cols.push(dense[s..e].to_vec());
    }

    // parent page binds through the parent state_root
    if p.parent_page.len() != e - s {
        return inv("parent page span mismatch");
    }
    let mut path: Vec<(bool, [u8; 32])> = Vec::with_capacity(p.parent_path.len());
    for (side, sib) in &p.parent_path {
        match hex::decode(sib).ok().and_then(|b| b.try_into().ok()) {
            Some(a) => path.push((*side, a)),
            None => return inv("malformed branch hash"),
        }
    }
    let parent_leaf = merkle::leaf_hash(&int64_bytes(&p.parent_page));
    let root: [u8; 32] = match hex::decode(&p.parent_header.state_root)
        .ok().and_then(|b| b.try_into().ok())
    {
        Some(a) => a,
        None => return inv("malformed parent state_root"),
    };
    if !merkle::verify_leaf(parent_leaf, &path, &root) {
        return inv("parent page not in parent state_root");
    }

    // THE RECOMPUTATION — the exact per-page consensus rule, one page wide
    let width = e - s;
    let mut new_page = p.parent_page.clone();
    if !claim_cols.is_empty() {
        for j in 0..width {
            let mut col: Vec<i64> = claim_cols.iter().map(|c| c[j]).collect();
            new_page[j] = new_page[j].wrapping_add(trimmed_mean_scalar(&mut col, 0.2));
        }
    }
    let recomputed = merkle::leaf_hash(&int64_bytes(&new_page));
    let committed = level[pid];
    if recomputed == committed {
        return (false, "invalid: page aggregates correctly — no fraud".into());
    }
    (true, format!("FRAUD: page {pid} committed {}… but honest {}…",
                   &hex::encode(committed)[..12], &hex::encode(recomputed)[..12]))
}
