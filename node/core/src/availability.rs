//! Availability sampling (Sharding Road, Phase 3) — mirrors `rig/da.py`.
//!
//! Delta bodies are already erasure-coded into N shards (any K reconstruct) and
//! Merkle-committed. A node samples a few random shards of each new block's
//! bodies; a body is unrecoverable only if more than N−K shards are missing, so
//! a handful of samples catches a withholding proposer with high probability.
//! This module pins the DETECTION-PROBABILITY math — the security parameter —
//! so the Rust sampler's margin matches the reference exactly.

/// C(n, r) as f64 (n, r small — shard counts). Saturates rather than overflows.
fn comb(n: u64, r: u64) -> f64 {
    if r > n {
        return 0.0;
    }
    let r = r.min(n - r);
    let mut acc = 1.0f64;
    for i in 0..r {
        acc = acc * (n - i) as f64 / (i + 1) as f64;
    }
    acc
}

/// P(random sampling hits at least one missing shard) when a body is
/// unrecoverable (available ≤ n−k). Mirrors `rig/da.detection_probability`.
pub fn detection_probability(available: u64, n: u64, k: u64, samples: u64) -> f64 {
    let miss = n.saturating_sub(available);
    if miss == 0 || samples > available {
        return 1.0;
    }
    let _ = k;
    1.0 - comb(available, samples) / comb(n, samples)
}

/// Recoverability: any K of N shards suffice.
pub fn recoverable(available: u64, k: u64) -> bool {
    available >= k
}
