"""Corpus data-availability: chunked, content-addressed, samplable (§7.2a).

The registry used to record only `sha256(corpus)`. A hash proves you once held
the bytes; it says nothing about whether they still exist, so "stake the hash
then delete the corpus" was possible and the provenance rule leaned on a
three-juror vote to catch it. This module is what replaces the vote with
arithmetic.

Why not simply reuse the delta-body DA path: that path erasure-codes ONE blob,
and corpora are gigabytes. Reed-Solomon over a multi-GB body needs the whole
thing in memory and produces shards too large to move. So a corpus is split
into fixed-size CHUNKS, each chunk gets the ordinary DA treatment (Reed-Solomon
+ Merkle commitment, exactly the code delta bodies use), and the corpus commits
to the ordered list of chunk roots via one more Merkle root:

    corpus ──split──▶ chunk₀ chunk₁ … chunkₘ
                        │      │        │
                     disperse disperse disperse      (rig.da.disperse, unchanged)
                        │      │        │
                      root₀  root₁ …  rootₘ
                        └──────┴────────┘
                              merkle.root  ─────▶  da_root  (goes in DataSubmitTx)

That gives two properties the flat hash could not:

  * A verifier can sample. Pick a random chunk, then random shards inside it,
    and check Merkle proofs — cheap, and it detects withholding with high
    probability without downloading gigabytes.
  * Partial loss is recoverable and *provable*. Losing more than n−k shards of
    any single chunk is detectable; the corpus is not all-or-nothing.

Note this does NOT replace data_hash. data_hash stays as the plain content
address people quote and `--data-refs` names; da_root is the availability
commitment. Both are covered by the signature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import da, merkle

# 4 MiB. Chosen against three ceilings: comfortably under the node's 64MB API
# body cap, close to the ~13MB delta bodies the DA path is already proven on,
# and small enough that a chunk plus its shards fits in memory on a small VPS.
# Consensus-critical: chunk count is derived from size_bytes, so changing this
# changes every da_root.
CHUNK_BYTES = 4 << 20

# Per-chunk erasure coding, matching the delta-body parameters (store.rs
# DA_K/DA_N): any 4 of 12 shards rebuild a chunk, so a chunk survives losing
# two thirds of its holders.
CHUNK_K = 4
CHUNK_N = 12


def chunk_count(size_bytes: int) -> int:
    """Chunks a corpus of this size splits into.

    Derived, not stored: putting it in the tx would let a submitter lie about it
    and desynchronise sampling from the committed root. size_bytes is already
    signed, so this is too.
    """
    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")
    if size_bytes == 0:
        return 0
    return (size_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES


def chunk_root(chunk: bytes) -> bytes:
    """DA Merkle root of one chunk — the same commitment a delta body gets."""
    return da.disperse(chunk, CHUNK_K, CHUNK_N).root


def manifest_root(chunk_roots: list[bytes]) -> bytes:
    """Corpus commitment: Merkle root over the ordered chunk roots.

    An empty corpus is rejected rather than given a well-known empty root. A
    zero-byte entry has nothing to sample, so it would be a registry entry whose
    availability can never fail a challenge — exactly the hash-with-no-bytes
    hole this module exists to close. `DataSubmitTx` refuses size_bytes == 0 for
    the same reason.
    """
    if not chunk_roots:
        raise ValueError("empty corpus cannot be staked: nothing to prove available")
    return merkle.root(chunk_roots)


@dataclass
class CorpusManifest:
    """Everything needed to serve and verify a staked corpus."""
    data_hash: str                  # sha256 of the whole corpus (content address)
    size_bytes: int
    da_root: str                    # hex manifest root — the availability commitment
    chunk_roots: list[bytes]        # per-chunk DA roots, in order

    def chunk_count(self) -> int:
        return len(self.chunk_roots)


def build(stream, size_hint: int | None = None) -> CorpusManifest:
    """Ingest a corpus from a byte stream, one chunk at a time.

    Streaming is the point: corpora are routinely multi-GB and must never be
    read whole just to be committed to. Memory stays O(CHUNK_BYTES).
    """
    sha = hashlib.sha256()
    roots: list[bytes] = []
    size = 0
    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        sha.update(chunk)
        size += len(chunk)
        roots.append(chunk_root(chunk))

    if size_hint is not None and size_hint != size:
        raise ValueError(f"stream produced {size} bytes, expected {size_hint}")
    if roots and chunk_count(size) != len(roots):
        raise ValueError(
            f"chunk accounting disagrees: {len(roots)} produced, "
            f"{chunk_count(size)} implied by size {size}")

    return CorpusManifest(data_hash=sha.hexdigest(), size_bytes=size,
                          da_root=manifest_root(roots).hex(), chunk_roots=roots)


def verify_chunk(chunk: bytes, index: int, chunk_roots: list[bytes],
                 da_root: str) -> bool:
    """A served chunk is authentic iff it re-disperses to the committed chunk
    root AND that root sits at `index` under the committed manifest root.

    Both halves matter. Checking only the chunk root lets a holder serve a
    valid chunk from a *different* corpus; checking only the manifest position
    lets them serve the wrong bytes for the right slot.
    """
    if index < 0 or index >= len(chunk_roots):
        return False
    if chunk_root(chunk) != chunk_roots[index]:
        return False
    # Recomputing the manifest root from the full list IS the inclusion proof
    # when you hold the whole list, which a node serving or auditing a corpus
    # does. A Merkle path only buys something for a LIGHT verifier holding just
    # da_root — worth adding when light clients exist, pointless before.
    return manifest_root(chunk_roots).hex() == da_root


def sample_plan(da_root: str, n_chunks: int, num_samples: int,
                rng) -> list[int]:
    """Which chunk indices a verifier should ask for.

    Seeded by the caller so a challenge is reproducible: a challenger who
    claims a corpus is unavailable must be forced to name the chunks they
    asked for, and anyone can replay the same draw.
    """
    if n_chunks <= 0:
        return []
    return [rng.randrange(n_chunks) for _ in range(min(num_samples, n_chunks))]
