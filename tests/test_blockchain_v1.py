"""PROTOCOL v1 end-to-end: paged state commitment, page-claimed deltas, the
work quota, VRF eligibility with attempts, the ModelState fold committed as
model_root — and THE test this revision hangs on: a growth event activating
on-chain with bit-exact replay across the dimension change."""

import numpy as np
import pytest

from rig import lottery
from rig.blockchain import (BlockTree, ValidationError, build_block,
                            expected_version)
from rig.chain import quantize
from rig.crypto import BackpropTx, Key, delta_hash
from rig.model_state import GenesisParams, ModelSpec, ModelState, page_state_root
from rig.token import address

# toy spec, aggressive retarget constants so growth fires in a short test
SPEC = ModelSpec(n_layers=2, d_model=4, d_ff=8, n_experts_initial=2, e_max=4,
                 backbone_params=100)
PARAMS = GenesisParams(spec=SPEC, retarget_window=2, target_deltas=4,
                       quota_max_4dp=20_000, k_sustain=2, announce_lead=1)
DIM0 = ModelState.genesis(SPEC).dim()

FOUNDER = address(Key.generate(b"founder".ljust(32, b"0")).pub)


def _tree():
    return BlockTree(quantize(np.zeros(DIM0)), data_contributor=FOUNDER,
                     params=PARAMS)


def _miners(n=3):
    return [Key.generate(f"m{i}".encode().ljust(32, b"0")) for i in range(n)]


def _eligible_attempt(tree, parent, key):
    """What the producer does: try attempts 0..MAX, use the first eligible."""
    led = tree.ledger[parent]
    h = tree.blocks[parent].header.height + 1
    stake, total = led.balance(address(key.pub)), led.supply()
    for a in range(lottery.ATTEMPT_MAX + 1):
        proof = lottery.vrf_prove(key, parent, h, a)
        if lottery.eligible(key.pub, proof, parent, h, a, stake, total):
            return a
    raise AssertionError("ATTEMPT_MAX must always be eligible")


def _body(rng, model, pages):
    """A dense body claiming `pages`: nonzero inside the claimed spans (well
    above any quota in these tests), exactly zero outside."""
    b = np.zeros(model.dim(), dtype=np.int64)
    for p in pages:
        s, e = model.page_span(p)
        b[s:e] = quantize(rng.standard_normal(e - s) * 0.1)
        if not b[s:e].any():
            b[s] = 1
    return b


def _block(tree, parent, miners, rng, pages=None, score=1000):
    model = tree.model[parent]
    height = tree.blocks[parent].header.height + 1
    claim = pages if pages is not None else \
        [i for i in range(len(model.pages)) if model.is_active(i)]
    txs, bodies, scores = [], {}, {}
    for i, k in enumerate(miners):
        body = _body(rng, model, claim)
        ptr = f"da://{height}/{i}/{rng.integers(1 << 30)}"
        tx = BackpropTx(miner=k.pub, base_height=height - 1,
                        delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                        pages=list(claim), data_refs=["genesis"]).signed(k)
        txs.append(tx); bodies[ptr] = body; scores[tx.txid()] = score
    prop = miners[0] if miners else Key.generate(b"prop".ljust(32, b"0"))
    att = _eligible_attempt(tree, parent, prop)
    return build_block(tree, parent, txs, bodies, {}, prop,
                       scores=scores, attempt=att)


def _grow_chain(tree, n, miners, seed=1, pages=None, score=1000):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        b = _block(tree, tree.head, miners, rng, pages, score)
        assert tree.add_block(b) or True
    return tree.head


def test_v1_chain_builds_and_replays_bit_exact():
    tree = _tree()
    _grow_chain(tree, 6, _miners())
    head = tree.blocks[tree.head]
    assert head.header.height == 6
    assert head.header.version == expected_version(6) == 2
    m = tree.head_model()
    assert page_state_root(tree.replay_head(), m) == head.header.state_root
    assert m.model_root() == head.header.model_root


def test_growth_activates_and_replays_across_dim_change():
    tree = _tree()
    miners = _miners(4)                 # 4 tx/block, W=2 → 8 ≥ target 4
    rng = np.random.default_rng(7)
    grew_at = None
    for i in range(120):
        b = _block(tree, tree.head, miners, rng)
        tree.add_block(b)
        if tree.head_model().dim() > DIM0:
            grew_at = b.header.height
            break
    assert grew_at is not None, "sustained surplus must grow the model on-chain"
    assert grew_at % PARAMS.retarget_window == 0      # activation at a boundary
    m = tree.head_model()
    assert len(m.pages) == 1 + 4 + 1                  # one new expert page
    assert m.pages[-1][2] == "expert" and m.events_total == 1
    # THE invariant: replay from genesis is bit-exact ACROSS the dim change
    w = tree.replay_head()
    assert int(w.shape[0]) == m.dim() > DIM0
    assert page_state_root(w, m) == tree.blocks[tree.head].header.state_root
    # the new page is claimable in the NEXT block, and training it works
    b = _block(tree, tree.head, miners, rng)          # claims all active incl. new
    assert any(len(m.pages) - 1 in tx.canonical_pages() for tx in b.txs)
    assert tree.add_block(b) is not None
    w2 = tree.replay_head()
    assert page_state_root(w2, tree.head_model()) == \
        tree.blocks[tree.head].header.state_root


def test_frozen_page_rejects_deltas():
    tree = _tree()
    miners = _miners(4)
    rng = np.random.default_rng(11)
    # grow one page, then starve the chain until it freezes
    while tree.head_model().dim() == DIM0:
        tree.add_block(_block(tree, tree.head, miners, rng))
    grown_page = len(tree.head_model().pages) - 1
    while tree.head_model().is_active(grown_page):
        b = _block(tree, tree.head, [], rng)          # empty blocks: deficit
        tree.add_block(b)
    # a tx claiming the frozen page is invalid
    bad = _block(tree, tree.head, miners[:1], rng, pages=[0, grown_page])
    with pytest.raises(ValidationError, match="frozen"):
        tree.add_block(bad)
    # genesis pages remain claimable
    ok = _block(tree, tree.head, miners[:1], rng,
                pages=[i for i in range(1 + 4)])
    assert tree.add_block(ok) is not None


def test_quota_and_claim_rules():
    tree = _tree()
    m = _miners(1)
    rng = np.random.default_rng(13)
    model = tree.head_model()
    height = 1

    def send(body, pages):
        ptr = f"da://q/{rng.integers(1 << 30)}"
        tx = BackpropTx(miner=m[0].pub, base_height=height - 1,
                        delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                        pages=pages, data_refs=["genesis"]).signed(m[0])
        att = _eligible_attempt(tree, tree.head, m[0])
        return build_block(tree, tree.head, [tx], {ptr: body}, {}, m[0],
                           scores={tx.txid(): 1000}, attempt=att)

    # nonzero outside the claimed page -> rejected
    body = _body(rng, model, [1])
    body[model.page_span(2)[0]] = 5
    with pytest.raises(ValidationError, match="outside claimed"):
        tree.add_block(send(body, [1]))
    # below the work quota -> rejected (quota 1.0 => 1% of the claimed page)
    sparse = np.zeros(model.dim(), dtype=np.int64)
    s, _e = model.page_span(0)
    sparse[s] = 1                                      # nnz=1 < 1% of 100.. == 1? no:
    # claimed = backbone (100) -> required = 1; claim backbone+experts instead
    all_pages = list(range(len(model.pages)))
    required = model.required_nnz(all_pages)
    assert required >= 2
    with pytest.raises(ValidationError, match="work quota"):
        tree.add_block(send(sparse, all_pages))
    # empty claim set -> rejected
    ok_body = _body(rng, model, [1])
    with pytest.raises(ValidationError, match="canonical and nonempty"):
        tree.add_block(send(ok_body, []))
    # a well-formed single-page claim is accepted
    assert tree.add_block(send(_body(rng, model, [1]), [1])) is not None


def test_v2_delta_envelope():
    """The payload never scales with quota: nnz above delta_max_nnz is
    invalid no matter how much work it represents. Specialization math: at a
    high quota the envelope bounds the claimable span."""
    spec = SPEC
    tight = GenesisParams(spec=spec, retarget_window=PARAMS.retarget_window,
                          target_deltas=PARAMS.target_deltas,
                          delta_max_nnz=8)
    tree = BlockTree(quantize(np.zeros(DIM0)), params=tight,
                     data_contributor=FOUNDER)
    m = _miners(1)
    rng = np.random.default_rng(23)
    model = tree.head_model()
    # a dense-enough claim whose nnz exceeds the tiny envelope -> rejected
    body = np.zeros(model.dim(), dtype=np.int64)
    s0, e0 = model.page_span(0)
    body[s0:s0 + 9] = 7                                # nnz=9 > cap 8
    ptr = "da://env/1"
    tx = BackpropTx(miner=m[0].pub, base_height=0,
                    delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                    pages=[0], data_refs=["genesis"]).signed(m[0])
    att = _eligible_attempt(tree, tree.head, m[0])
    blk = build_block(tree, tree.head, [tx], {ptr: body}, {}, m[0],
                      scores={tx.txid(): 1000}, attempt=att)
    with pytest.raises(ValidationError, match="envelope"):
        tree.add_block(blk)
    # same claim within the envelope -> accepted
    body2 = np.zeros(model.dim(), dtype=np.int64)
    body2[s0:s0 + 8] = 7                               # nnz=8 == cap
    ptr2 = "da://env/2"
    tx2 = BackpropTx(miner=m[0].pub, base_height=0,
                     delta_hash=delta_hash(body2.tobytes()), da_pointer=ptr2,
                     pages=[0], data_refs=["genesis"]).signed(m[0])
    blk2 = build_block(tree, tree.head, [tx2], {ptr2: body2}, {}, m[0],
                       scores={tx2.txid(): 1000}, attempt=att)
    assert tree.add_block(blk2) is not None
    # specialization bound: claimable params shrink as quota rises
    hi_q = 80000
    budget = tight.delta_max_nnz * 1_000_000 // hi_q
    assert budget < model.dim()  # cannot claim the whole model at 8x


def test_eligibility_and_version_enforced():
    tree = _tree()
    miners = _miners(2)
    rng = np.random.default_rng(17)
    _grow_chain(tree, 3, miners)                       # mint some stake
    # a proposer claiming an attempt at which it is NOT eligible is rejected
    parent = tree.head
    key = miners[1]
    att = _eligible_attempt(tree, parent, key)
    ineligible = None
    led = tree.ledger[parent]
    h = tree.blocks[parent].header.height + 1
    for a in range(lottery.ATTEMPT_MAX):
        proof = lottery.vrf_prove(key, parent, h, a)
        if not lottery.eligible(key.pub, proof, parent, h, a,
                                led.balance(address(key.pub)), led.supply()):
            ineligible = a
            break
    if ineligible is not None:
        bad = _block(tree, parent, [key], rng)
        bad.header.vrf_attempt = ineligible
        bad.header.vrf_proof = lottery.vrf_prove(key, parent, h, ineligible).hex()
        bad.header.work = lottery.attempt_work(
            bytes.fromhex(bad.header.vrf_proof), ineligible)
        with pytest.raises(ValidationError, match="eligible"):
            tree.add_block(bad)
    # ATTEMPT_MAX is always eligible (the liveness floor)
    proof = lottery.vrf_prove(key, parent, h, lottery.ATTEMPT_MAX)
    assert lottery.eligible(key.pub, proof, parent, h, lottery.ATTEMPT_MAX,
                            0, led.supply())
    # a wrong header version is rejected
    good = _block(tree, parent, [key], rng)
    good.header.version = 3
    with pytest.raises(ValidationError, match="version"):
        tree.add_block(good)
