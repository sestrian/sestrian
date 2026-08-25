//! Corpus data-availability: chunked, content-addressed, samplable (§7.2a).
//!
//! A faithful port of `rig/corpus.py` — the `corpus_da` golden vectors pin this
//! to it byte-for-byte, across chunk boundaries in both directions.
//!
//! The registry used to record only `sha256(corpus)`. A hash proves the bytes
//! existed once; it says nothing about whether they still do, so "stake the hash
//! then delete the corpus" was possible and the provenance rule leaned on a
//! three-juror vote to catch it. This is what replaces that vote with
//! arithmetic.
//!
//! Delta bodies get erasure-coded as ONE blob, which does not survive contact
//! with a multi-gigabyte corpus: Reed-Solomon over the whole thing needs it all
//! in memory and yields shards too big to move. So a corpus is split into fixed
//! CHUNKS, each chunk gets the ordinary DA treatment (the same `crate::da` code
//! delta bodies use), and the corpus commits to the ordered chunk roots with one
//! more Merkle root:
//!
//! ```text
//!   corpus ──split──▶ chunk₀ chunk₁ … chunkₘ
//!                       │      │        │      da::disperse (unchanged)
//!                     root₀  root₁ …  rootₘ
//!                       └──────┴────────┘
//!                            merkle::root ──▶ da_root  (in DataSubmitTx)
//! ```
//!
//! Two properties a flat hash cannot give: a verifier can SAMPLE (pick a random
//! chunk, then random shards within it, check Merkle proofs — cheap, and it
//! catches withholding with high probability without moving gigabytes), and
//! partial loss is both recoverable and detectable rather than all-or-nothing.
//!
//! This does not replace `data_hash`: that stays the plain content address
//! people quote and `--data-refs` names. `da_root` is the availability
//! commitment. Both are covered by the signature.

use crate::{da, merkle};

/// 4 MiB. Picked against three ceilings: under the node's 64MB API body cap,
/// close to the ~13MB delta bodies the DA path is already proven on, and small
/// enough that a chunk plus its shards fits in memory on a small VPS.
///
/// CONSENSUS-CRITICAL: chunk count is derived from `size_bytes`, so changing
/// this changes every `da_root` ever computed.
pub const CHUNK_BYTES: usize = 4 << 20;

/// Per-chunk erasure coding, matching the delta-body parameters: any 4 of 12
/// shards rebuild a chunk, so a chunk survives losing two thirds of its holders.
pub const CHUNK_K: usize = 4;
pub const CHUNK_N: usize = 12;

/// How many chunks a corpus of this size splits into.
///
/// Derived rather than stored: carrying it in the tx would let a submitter lie
/// about it and desynchronise sampling from the committed root. `size_bytes` is
/// already signed, so this is too.
pub fn chunk_count(size_bytes: u64) -> u64 {
    if size_bytes == 0 {
        return 0;
    }
    size_bytes.div_ceil(CHUNK_BYTES as u64)
}

/// DA Merkle root of one chunk — the same commitment a delta body receives.
pub fn chunk_root(chunk: &[u8]) -> Vec<u8> {
    da::disperse(chunk, CHUNK_K, CHUNK_N).root().to_vec()
}

/// Corpus commitment: Merkle root over the ordered chunk roots.
///
/// `None` for an empty corpus rather than a well-known empty root. A zero-byte
/// entry has nothing to withhold, so its availability could never fail a
/// challenge — precisely the unfalsifiable entry this module exists to prevent.
/// The ledger refuses `size_bytes == 0` for the same reason.
pub fn manifest_root(chunk_roots: &[Vec<u8>]) -> Option<String> {
    if chunk_roots.is_empty() {
        return None;
    }
    let leaves: Vec<&[u8]> = chunk_roots.iter().map(|r| r.as_slice()).collect();
    Some(hex::encode(merkle::root(&leaves)))
}

/// Everything needed to serve and verify a staked corpus.
#[derive(Clone, Debug)]
pub struct CorpusManifest {
    pub data_hash: String,
    pub size_bytes: u64,
    pub da_root: String,
    pub chunk_roots: Vec<Vec<u8>>,
}

/// Build a manifest from a reader, one chunk at a time.
///
/// Streaming is the point: corpora are routinely multi-GB and must never be read
/// whole just to be committed to. Memory stays O(CHUNK_BYTES).
pub fn build<R: std::io::Read>(mut r: R) -> std::io::Result<CorpusManifest> {
    use sha2::{Digest, Sha256};
    let mut sha = Sha256::new();
    let mut roots: Vec<Vec<u8>> = Vec::new();
    let mut size: u64 = 0;
    let mut buf = vec![0u8; CHUNK_BYTES];

    loop {
        // read_exact would fail on the final short chunk; fill as much as we can
        // and stop only on a genuine end of stream.
        let mut filled = 0usize;
        while filled < CHUNK_BYTES {
            match r.read(&mut buf[filled..])? {
                0 => break,
                n => filled += n,
            }
        }
        if filled == 0 {
            break;
        }
        sha.update(&buf[..filled]);
        size += filled as u64;
        roots.push(chunk_root(&buf[..filled]));
        if filled < CHUNK_BYTES {
            break;
        }
    }

    let da_root = manifest_root(&roots).unwrap_or_default();
    Ok(CorpusManifest {
        data_hash: hex::encode(sha.finalize()),
        size_bytes: size,
        da_root,
        chunk_roots: roots,
    })
}

/// A served chunk is authentic iff it re-disperses to the committed chunk root
/// AND that root sits at `index` under the committed manifest root.
///
/// Both halves matter. Checking only the chunk root lets a holder serve a valid
/// chunk from a DIFFERENT corpus; checking only the manifest position lets them
/// serve the wrong bytes for the right slot.
pub fn verify_chunk(chunk: &[u8], index: usize, chunk_roots: &[Vec<u8>],
                    da_root: &str) -> bool {
    if index >= chunk_roots.len() {
        return false;
    }
    if chunk_root(chunk) != chunk_roots[index] {
        return false;
    }
    // Recomputing the manifest root from the full list IS the inclusion proof
    // when you hold the whole list, which a node serving or auditing a corpus
    // does. A Merkle path only buys something for a LIGHT verifier holding just
    // da_root — worth adding when light clients exist, pointless before.
    matches!(manifest_root(chunk_roots), Some(r) if r == da_root)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Boundaries are where a chunking port goes wrong: exactly one chunk, one
    /// byte over, one byte under. The golden vectors cover the same sizes so a
    /// divergence from the Python reference shows up as a vector mismatch.
    #[test]
    fn chunk_count_boundaries() {
        assert_eq!(chunk_count(0), 0);
        assert_eq!(chunk_count(1), 1);
        assert_eq!(chunk_count(CHUNK_BYTES as u64 - 1), 1);
        assert_eq!(chunk_count(CHUNK_BYTES as u64), 1);
        assert_eq!(chunk_count(CHUNK_BYTES as u64 + 1), 2);
        assert_eq!(chunk_count(CHUNK_BYTES as u64 * 3), 3);
    }

    #[test]
    fn empty_corpus_has_no_commitment() {
        assert_eq!(manifest_root(&[]), None);
    }

    #[test]
    fn build_is_deterministic_and_tamper_evident() {
        let body: Vec<u8> = (0..(CHUNK_BYTES + 1000)).map(|i| (i % 251) as u8).collect();
        let a = build(&body[..]).unwrap();
        let b = build(&body[..]).unwrap();
        assert_eq!(a.da_root, b.da_root);
        assert_eq!(a.size_bytes, body.len() as u64);
        assert_eq!(a.chunk_roots.len() as u64, chunk_count(a.size_bytes));

        let mut tampered = body.clone();
        tampered[CHUNK_BYTES + 5] ^= 1;          // a byte in the SECOND chunk
        assert_ne!(build(&tampered[..]).unwrap().da_root, a.da_root,
                   "one flipped byte must move the commitment");
    }

    #[test]
    fn verify_chunk_rejects_wrong_bytes_and_wrong_slot() {
        let body: Vec<u8> = (0..(CHUNK_BYTES * 2)).map(|i| (i % 251) as u8).collect();
        let m = build(&body[..]).unwrap();
        let c0 = &body[..CHUNK_BYTES];
        assert!(verify_chunk(c0, 0, &m.chunk_roots, &m.da_root));
        assert!(!verify_chunk(c0, 1, &m.chunk_roots, &m.da_root), "wrong slot");
        assert!(!verify_chunk(&vec![0u8; CHUNK_BYTES], 0, &m.chunk_roots, &m.da_root),
                "wrong bytes");
        assert!(!verify_chunk(c0, 9, &m.chunk_roots, &m.da_root), "index past end");
    }
}
