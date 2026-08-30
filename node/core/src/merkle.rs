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

/// The full tree as levels[0] = leaf hashes .. levels[last] = [root] —
/// mirrors `rig/merkle.py::build` (odd node promoted by hashing with itself).
pub fn levels(leaf_hashes: Vec<[u8; 32]>) -> Vec<Vec<[u8; 32]>> {
    assert!(!leaf_hashes.is_empty(), "need at least one leaf");
    let mut out = vec![leaf_hashes];
    while out.last().unwrap().len() > 1 {
        let level = out.last().unwrap();
        let mut nxt = Vec::with_capacity(level.len().div_ceil(2));
        for i in (0..level.len()).step_by(2) {
            let a = &level[i];
            let b = if i + 1 < level.len() { &level[i + 1] } else { &level[i] };
            nxt.push(node_hash(a, b));
        }
        out.push(nxt);
    }
    out
}

/// Inclusion proof for leaf `index`: (sibling_is_left, sibling_hash) per
/// level — mirrors `rig/merkle.py::proof` ("L" == sibling on the left).
pub fn proof(levels: &[Vec<[u8; 32]>], index: usize) -> Vec<(bool, [u8; 32])> {
    let mut path = Vec::new();
    let mut idx = index;
    for level in &levels[..levels.len() - 1] {
        if idx % 2 == 0 {
            let sib = if idx + 1 < level.len() { level[idx + 1] } else { level[idx] };
            path.push((false, sib)); // sibling on the RIGHT
        } else {
            path.push((true, level[idx - 1])); // sibling on the LEFT
        }
        idx /= 2;
    }
    path
}

/// Fold a leaf hash up its proof; true iff it lands on the committed root —
/// mirrors `rig/merkle.py::verify` (which takes page bytes; callers here pass
/// `leaf_hash(bytes)` so disputes can also verify a COMMITTED leaf directly).
pub fn verify_leaf(leaf: [u8; 32], path: &[(bool, [u8; 32])],
                   expected_root: &[u8; 32]) -> bool {
    let mut h = leaf;
    for (sib_left, sib) in path {
        h = if *sib_left { node_hash(sib, &h) } else { node_hash(&h, sib) };
    }
    &h == expected_root
}
