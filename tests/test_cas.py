"""Content-addressed DA layer (client/cas.py): CID integrity, Bitswap swarm
reconstruction of the model, and the model-as-Merkle-DAG-of-pages using the real
MoE PageMap spans."""

import numpy as np

from client.cas import (
    Bitswap, ContentStore, cid, get_model, put_model, read_manifest,
    state_from_pages, pages_from_state,
)
from client.moe import MoEGPT, MoEGPTConfig


def _chunks(n: int, size: int):
    """Chunk [0, n) into ≤ size-parameter spans — CAS transport granularity is
    arbitrary (it addresses bytes, not consensus pages), so plain chunking
    replaces the old PageMap.subdivide here."""
    return [(a, min(a + size, n)) for a in range(0, n, size)]
from client.trainer import flat_params
from rig.chain import quantize, state_root


def test_cid_is_content_hash_and_store_roundtrips():
    s = ContentStore()
    data = b"a delta body"
    c = s.put(data)
    assert c == cid(data)
    assert s.get(c) == data and s.has(c)
    assert s.get("deadbeef") is None


def test_on_block_rejects_forgery_for_free():
    """Content addressing = trustless bodies: a block whose bytes don't match the
    requested CID is dropped, no signature needed."""
    bs = Bitswap(ContentStore())
    bs.want("f" * 64)
    msgs, ok = bs.on_block("f" * 64, b"not the real bytes")
    assert not ok and not bs.store.has("f" * 64)          # forgery rejected
    # the honest block is accepted and re-announced
    good = b"real"
    msgs, ok = bs.on_block(cid(good), good)
    assert ok and ("have", cid(good)) in msgs


def test_put_get_model_roundtrip_over_pagemap():
    model = MoEGPT(MoEGPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8,
                                n_experts=4, top_k=2))
    state = quantize(flat_params(model))
    spans = _chunks(state.size, 4096)                     # page-granularity chunks
    store = ContentStore()
    root, page_cids = put_model(store, state, spans)
    assert len(page_cids) == len(spans)
    got, missing = get_model(store, root)
    assert not missing
    assert np.array_equal(got, state)
    assert state_root(got) == state_root(state)           # same committed root


def _route(nodes, seeds):
    """Tiny synchronous swarm: broadcast every emitted message to all other nodes
    until no node emits anything new. `nodes` maps name -> Bitswap. `seeds` is a
    list of (origin, message) to inject."""
    queue = list(seeds)
    while queue:
        origin, msg = queue.pop(0)
        for name, bs in nodes.items():
            if name == origin:
                continue
            out = []
            if msg[0] == "want":
                out = bs.on_want(origin, msg[1])
            elif msg[0] == "have":
                out = bs.on_have(origin, msg[1])
            elif msg[0] == "block":
                out, _ = bs.on_block(msg[1], msg[2])
            for m in out:
                queue.append((name, m))


def test_swarm_reconstructs_model_from_content_addresses():
    """Node A holds the whole model; B and C hold nothing. B wants it and fetches
    every page from the swarm by CID, reconstructing the exact state — no central
    server, no trust in the provider (each page self-verifies against its CID)."""
    model = MoEGPT(MoEGPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=8,
                                n_experts=6, top_k=2))
    state = quantize(flat_params(model))
    spans = _chunks(state.size, 2048)

    a_store = ContentStore()
    root, page_cids = put_model(a_store, state, spans)    # A stores everything
    A, B, C = Bitswap(a_store), Bitswap(ContentStore()), Bitswap(ContentStore())
    nodes = {"A": A, "B": B, "C": C}

    # A announces the root + pages; B asks for the root first, then its pages.
    seeds = [("A", m) for c in [root] + page_cids for m in A.announce(c)]
    seeds += [("B", m) for m in B.want(root)]
    _route(nodes, seeds)

    # B now has the manifest → discovers the page CIDs → wants any it still lacks
    parsed = read_manifest(B.store, root)
    assert parsed is not None                              # B got the root object
    _, _, cids = parsed
    seeds = [("B", m) for c in cids for m in B.want(c)]
    _route(nodes, seeds)

    got, missing = get_model(B.store, root)
    assert not missing                                    # B fetched every page
    assert np.array_equal(got, state)                     # exact model, from the swarm
    assert state_root(got) == state_root(state)


def test_partial_holdings_still_serve_the_swarm():
    """Even if no single peer (besides the origin) holds everything, content
    addressing lets a node collect pages from whoever has each one."""
    n = 100
    state = np.arange(n, dtype=np.int64)
    spans = [(0, 50), (50, 100)]
    full = ContentStore()
    root, page_cids = put_model(full, state, spans)

    # Two partial holders: X has page 0 + manifest, Y has page 1.
    X, Y, Z = ContentStore(), ContentStore(), ContentStore()
    X.put(full.get(root)); X.put(full.get(page_cids[0]))
    Y.put(full.get(page_cids[1]))
    bx, by, bz = Bitswap(X), Bitswap(Y), Bitswap(Z)
    nodes = {"X": bx, "Y": by, "Z": bz}

    seeds = [("X", m) for c in [root, page_cids[0]] for m in bx.announce(c)]
    seeds += [("Y", m) for m in by.announce(page_cids[1])]
    seeds += [("Z", m) for m in bz.want(root)]
    _route(nodes, seeds)
    parsed = read_manifest(bz.store, root)
    assert parsed is not None
    _, _, cids = parsed
    _route(nodes, [("Z", m) for c in cids for m in bz.want(c)])

    got, missing = get_model(bz.store, root)
    assert not missing and np.array_equal(got, state)     # assembled from X and Y
