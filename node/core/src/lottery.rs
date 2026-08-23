//! Verifiable proposer sortition (§7.4, protocol v1) — mirrors `rig/lottery.py`.
//!
//! Stake-weighted, verifiable eligibility replaces round-robin — ENFORCED in
//! validation (v1), with a deterministic liveness fallback so a tiny fleet can
//! never stall the chain:
//!
//!   * The per-height seed binds to (parent hash, height, ATTEMPT). A
//!     proposer's VRF proof is a DETERMINISTIC Ed25519 signature over that
//!     seed — only the key holder can produce it, anyone can verify it, and it
//!     is unique per (key, height, attempt). Each attempt is an honest re-roll.
//!   * Eligibility: vrf_output < 2^256 · TARGET_PROPOSERS · stake/total_stake
//!     · 2^attempt. The threshold doubles per attempt, widening the eligible
//!     set deterministically instead of the height stalling. Two absolute
//!     rules: total_stake == 0 (cold start) admits any verifying proof at
//!     attempt 0; attempt == ATTEMPT_MAX admits any verifying proof — the
//!     liveness floor for a 2-miner devnet.
//!   * Fork-choice weight: header.work MUST equal `attempt_work(proof,
//!     attempt)` = max(1, vrf_work >> attempt) — committed and validated, so a
//!     low-attempt (prompt) proposer strictly tends to dominate reorgs.

use crate::{verify_sig, Key};
use sha2::{Digest, Sha256};

pub const TARGET_PROPOSERS: u64 = 2; // expected eligible proposers, attempt 0
pub const ATTEMPT_MAX: u64 = 16; // unconditional-eligibility liveness floor

/// The per-height randomness seed, bound to the parent, height and attempt.
pub fn seed(prev_hash: &str, height: u64, attempt: u64) -> [u8; 32] {
    let mut m = Sha256::new();
    m.update(format!("sestrian-lottery|{prev_hash}|{height}|{attempt}").as_bytes());
    m.finalize().into()
}

/// The proposer's VRF proof: a deterministic signature over the seed.
pub fn vrf_prove(key: &Key, prev_hash: &str, height: u64, attempt: u64) -> Vec<u8> {
    key.sign(&seed(prev_hash, height, attempt))
}

/// A uniform 256-bit value only the key holder could have produced, as bytes
/// (big-endian) so it can be compared against the threshold without bignum deps.
pub fn vrf_output(proof: &[u8]) -> [u8; 32] {
    Sha256::digest(proof).into()
}

/// Raw fork-choice weight: leading zero bits of the VRF output + 1 (>= 1).
pub fn vrf_work(proof: &[u8]) -> u64 {
    let out = vrf_output(proof);
    let mut lz = 0u64;
    for &b in out.iter() {
        if b == 0 {
            lz += 8;
        } else {
            lz += b.leading_zeros() as u64;
            break;
        }
    }
    lz + 1
}

/// The committed header.work: the raw VRF work discounted by the attempt
/// (>> attempt), floored at 1 — validation requires exact equality.
pub fn attempt_work(proof: &[u8], attempt: u64) -> u64 {
    (vrf_work(proof) >> attempt.min(63)).max(1)
}

/// True iff `proof` is a valid VRF proof by `pub_hex` for (height, attempt)
/// AND it clears the stake-weighted, attempt-widened threshold (or a rule
/// applies: cold start at attempt 0, or the ATTEMPT_MAX liveness floor).
pub fn eligible(
    pub_hex: &str,
    proof: &[u8],
    prev_hash: &str,
    height: u64,
    attempt: u64,
    stake: u64,
    total_stake: u64,
) -> bool {
    if attempt > ATTEMPT_MAX {
        return false;
    }
    if !verify_sig(pub_hex, &seed(prev_hash, height, attempt), proof) {
        return false;
    }
    if attempt == ATTEMPT_MAX {
        return true; // absolute liveness floor
    }
    if total_stake == 0 {
        return attempt == 0; // cold start: everyone, first try
    }
    below_threshold(&vrf_output(proof), stake, total_stake, attempt)
}

/// Compare a 256-bit big-endian output against
/// min(2^256, 2^256 · TARGET · stake · 2^attempt / total_stake).
/// Done with integer long division over u128 limbs so there are NO floats and
/// no bignum dependency — bit-identical to the Python big-int compare.
fn below_threshold(output: &[u8; 32], stake: u64, total_stake: u64, attempt: u64) -> bool {
    if stake == 0 {
        return false; // threshold 0 — nothing is below it
    }
    // numerator = TARGET * stake * 2^attempt fits u128 comfortably:
    // 2 * 2^64 * 2^16 = 2^81. attempt < ATTEMPT_MAX here (checked upstream).
    let numerator = (TARGET_PROPOSERS as u128) * (stake as u128) << attempt;
    // If numerator >= total_stake, the threshold caps at 2^256 => everything
    // in [0, 2^256) is below it.
    if numerator >= total_stake as u128 {
        return true;
    }
    let q = mul2pow256_div(numerator, total_stake as u128); // 32-byte big-endian
    output.as_slice() < q.as_slice()
}

/// floor(numerator * 2^256 / divisor) as a 32-byte big-endian value, for
/// numerator < divisor (so the result is < 2^256). Schoolbook long division,
/// one output byte at a time, remainder carried in u128.
fn mul2pow256_div(numerator: u128, divisor: u128) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut rem = numerator; // numerator < divisor, so rem starts < divisor
    for byte in out.iter_mut() {
        // bring down 8 bits: rem = rem*256 (+ 0, since the low bytes are zero).
        // divisor is a u64-ranged value, so rem << 8 cannot overflow u128.
        let cur = rem << 8;
        *byte = (cur / divisor) as u8;
        rem = cur % divisor;
    }
    out
}
