"""Keys, signatures, and signed transactions (WHITEPAPER §4, §5).

Wraps the reference Ed25519 in a small API and defines the two signed message
types the chain authenticates: a BackpropTx (a miner's delta commitment) and a
generic signed envelope. Signing replaces the rig's earlier "trust the
miner_id" with real authentication — a delta counts only if it carries a valid
signature from the key that staked it.
"""

import hashlib
import os
import struct
from dataclasses import dataclass, field

from . import ed25519


def frame(*parts: bytes) -> bytes:
    """Unambiguous signing preimage: each field is length-prefixed (4-byte
    big-endian) so no field's contents can ever be confused with the structure.
    This replaces '|'-joined signing strings, where a free-form field containing
    the delimiter could make two logically different txs share a preimage / txid
    (a canonicalization / collision hazard)."""
    return b"".join(struct.pack(">I", len(p)) + p for p in parts)

# Prefer libsodium (pynacl) — same Ed25519, ~1000x faster than the pure-Python
# reference, which matters once a gossip network verifies thousands of sigs.
# Falls back to the self-contained reference when pynacl isn't installed.
try:
    from nacl.signing import SigningKey, VerifyKey     # type: ignore
    _HAVE_NACL = True
except Exception:
    _HAVE_NACL = False


@dataclass
class Key:
    """An Ed25519 keypair. `pub` (hex) is the on-chain identity of a node."""
    sk: bytes
    pk: bytes

    @property
    def pub(self) -> str:
        return self.pk.hex()

    @staticmethod
    def generate(seed: bytes | None = None) -> "Key":
        sk = seed if seed is not None else os.urandom(32)
        if len(sk) != 32:
            sk = hashlib.sha256(sk).digest()
        if _HAVE_NACL:
            pk = bytes(SigningKey(sk).verify_key)
        else:
            pk = ed25519.publickey(sk)
        return Key(sk=sk, pk=pk)

    def sign(self, msg: bytes) -> bytes:
        if _HAVE_NACL:
            return SigningKey(self.sk).sign(msg).signature
        return ed25519.signature(msg, self.sk, self.pk)


def verify(pub_hex: str, msg: bytes, sig: bytes) -> bool:
    try:
        if _HAVE_NACL:
            VerifyKey(bytes.fromhex(pub_hex)).verify(msg, sig)
            return True
        return ed25519.checkvalid(sig, msg, bytes.fromhex(pub_hex))
    except Exception:
        return False


@dataclass
class BackpropTx:
    """A signed delta commitment (§4.1, protocol v1). The body (delta) lives on
    the DA layer; the tx carries its hash, base height, the PAGE CLAIM SET, and
    a signature over all of it.

    v1: `shard_id` is gone; `pages` is the sorted set of page ids this delta
    trained. The dense body must be exactly zero outside the claimed pages'
    spans — that is what makes per-page aggregation over actual contributors
    well-defined (a non-claimant's zero is absence, not a vote for zero), and
    what makes freezing a page an enforceable rule."""
    miner: str            # signer pubkey (hex)
    base_height: int
    delta_hash: str       # sha256 of the delta body bytes
    da_pointer: str       # where the body can be fetched (DA layer key)
    bond: int = 0         # rev 4: stake bond the miner locks to submit (grains)
    # v1: the claimed page ids (canonicalized: sorted, unique). Empty = invalid.
    pages: list = field(default_factory=list)
    # rev 5: PROVENANCE — the content addresses (data_hash) of the data this
    # gradient was trained on. A gradient names its data so the data share can
    # be paid to the data's owners (not a single configured address) and the
    # link is auditable: fetch the named data, recompute the gradient, compare.
    # Empty = no provenance claim → the delta may still be accepted on its
    # loss-score merit, but earns no data share. Canonicalized (sorted, unique).
    data_refs: list = field(default_factory=list)
    sig: bytes = b""

    def canonical_refs(self) -> list:
        """Sorted, de-duplicated data_hash list — the canonical provenance set."""
        return sorted(set(self.data_refs))

    def canonical_pages(self) -> list:
        """Sorted, de-duplicated page-id list — the canonical claim set."""
        return sorted({int(p) for p in self.pages})

    def signing_bytes(self) -> bytes:
        refs = self.canonical_refs()
        pages = self.canonical_pages()
        return frame(b"backprop", self.miner.encode(), str(self.base_height).encode(),
                     self.delta_hash.encode(),
                     self.da_pointer.encode(), str(self.bond).encode(),
                     # count-prefixed lists so zero entries is unambiguous vs.
                     # the fields above; each entry its own length-framed field.
                     str(len(pages)).encode(), *[str(p).encode() for p in pages],
                     str(len(refs)).encode(), *[r.encode() for r in refs])

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "BackpropTx":
        assert key.pub == self.miner, "signer must match tx.miner"
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.miner, self.signing_bytes(), self.sig)


def delta_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()
