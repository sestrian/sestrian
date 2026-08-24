//! A binary Merkle tree over weight pages (§3.1) — mirrors `rig/merkle.py`.
//!
//! The chain commits `state_root` = the Merkle root over the model's pages
//! (protocol v1). Domain-separated hashing (0x00 for leaves, 0x01 for internal
//! nodes) prevents a leaf/node confusion; an odd node on a level is promoted by
//! hashing it with ITSELF, exactly as the reference does.

use sha2::{Digest, Sha256};

pub fn leaf_hash(page_bytes: &[u8]) -> [u8; 32] {
    let mut m = Sha256::new();
    m.update([0x00]);
    m.update(page_bytes);
    m.finalize().into()
}

fn node_hash(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut m = Sha256::new();
    m.update([0x01]);
    m.update(a);
    m.update(b);
    m.finalize().into()
}

/// Merkle root over the leaves, in order. Panics on zero leaves (the page
/// table always has at least the backbone page).
pub fn root(leaves: &[&[u8]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "need at least one leaf");
    let level: Vec<[u8; 32]> = leaves.iter().map(|l| leaf_hash(l)).collect();
    root_from_hashes(level)
}

/// Merkle root over PRECOMPUTED leaf hashes — the incremental engine caches a
/// leaf hash per page and rehashes only pages a block touched, so the root is
/// O(touched pages + log P) instead of O(model).
pub fn root_from_hashes(mut level: Vec<[u8; 32]>) -> [u8; 32] {
    assert!(!level.is_empty(), "need at least one leaf");
    while level.len() > 1 {
        let mut nxt = Vec::with_capacity(level.len().div_ceil(2));
        for i in (0..level.len()).step_by(2) {
            let a = &level[i];
            let b = if i + 1 < level.len() { &level[i + 1] } else { &level[i] };
            nxt.push(node_hash(a, b));
        }
        level = nxt;
    }
    level[0]
}
