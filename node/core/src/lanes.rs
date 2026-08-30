//! Training lanes (Sharding Road, Phase 2) — mirrors `rig/lanes.py`.
//!
//! Lanes partition the EXPERT pages so a v5 block only accepts a miner's delta
//! if its claimed pages lie in (backbone ∪ that miner's beacon-assigned lane).
//! Pure functions of (epoch, miner pubkey, active page table): identical on
//! every node, golden-vectored.

use crate::model_state::ModelState;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Lanes the current model supports: expert pages / lane_width, at least 1.
pub fn n_lanes(active_expert_pages: usize, lane_width: usize) -> usize {
    if active_expert_pages == 0 {
        return 1;
    }
    (active_expert_pages / lane_width.max(1)).max(1)
}

/// Deterministic lane for a miner in an epoch (hash, so a key can't be picked
/// to own a lane, and it rotates every epoch).
pub fn lane_of_miner(epoch: u64, miner_pub: &str, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let d = Sha256::digest(format!("sestrian-lane|v5|{epoch}|{miner_pub}").as_bytes());
    let x = u64::from_be_bytes(d[..8].try_into().unwrap());
    (x % n as u64) as usize
}

/// The full page set a miner may claim this epoch: backbone + its lane's
/// experts (round-robin stripe over the ACTIVE expert page ids).
pub fn claimable_pages(epoch: u64, miner_pub: &str, model: &ModelState,
                       lane_width: usize) -> BTreeSet<u32> {
    let expert_ids: Vec<usize> = (0..model.pages.len())
        .filter(|&i| model.pages[i].kind != "backbone" && model.is_active(i))
        .collect();
    let n = n_lanes(expert_ids.len(), lane_width);
    let lane = lane_of_miner(epoch, miner_pub, n);
    let mut out: BTreeSet<u32> = if n <= 1 {
        expert_ids.iter().map(|&p| p as u32).collect()
    } else {
        expert_ids.iter().enumerate()
            .filter(|(k, _)| k % n == lane)
            .map(|(_, &p)| p as u32).collect()
    };
    for i in 0..model.pages.len() {
        if model.pages[i].kind == "backbone" && model.is_active(i) {
            out.insert(i as u32);
        }
    }
    out
}
