"""The real client, coordinator-free — gossip consensus over real GPU training.

No central node. Every peer holds a BlockTree, trains the real GPT on its own
GPU, and gossips signed pseudo-gradient deltas and blocks to its peers over
async sockets. Leadership rotates by wall-clock round (leader = round mod N), so
the right to propose a block simply takes turns — nobody coordinates. Fork
choice is Nakamoto heaviest-valid-chain (rig/blockchain.py); a peer joining or
reconnecting re-announces its chain so forks reconcile.

This is client/chain_node.py with the coordinator removed. The delta bodies are
real (megabytes), so the model here is deliberately small to keep gossip light;
the mechanism is identical at any size.

  # on each machine (peers is host:port,host:port of the OTHER nodes):
  python -m client.gossip --id 0 --port 9850 --peers 100.x:9851 --n 2 --seconds 30
"""

import argparse
import asyncio
import os
import pickle
import struct
import time
from dataclasses import dataclass, field

_DBG = os.environ.get("GOSSIP_DEBUG")


def _dbg(nid, *a):
    if _DBG:
        print(f"    [dbg n{nid}]", *a, flush=True)

import numpy as np

from rig.blockchain import Block, BlockTree, ValidationError, build_block
from rig.chain import dequantize, quantize, state_root
from rig.crypto import BackpropTx, Key, delta_hash
from rig.token import canonical_account_txs
from .compress import Compressor, decompress
from .data import ByteData
from .gpt import GPTConfig, build
from .trainer import DiLoCoMiner, flat_params, set_flat_params

KEEP_FRAC = 0.02             # top-k delta compression (50x on the wire)

GOSSIP_CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=64)   # toy: light gossip
SMALL_CFG = GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=256)  # dense 86M (devnet-1)
# PROTOCOL v1 (devnet-genesis-2): MoE presets — the network model. e_max router
# columns are preallocated in the backbone so on-chain growth events add expert
# pages without touching backbone shapes. These numbers are CONSENSUS-frozen at
# genesis (they define the page table); see docs/genesis-ceremony.md.
from .moe import MoEGPTConfig, build_moe  # noqa: E402  (after GPTConfig import)
TOY_MOE_CFG = MoEGPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=64,
                           n_experts=4, e_max=8, top_k=2)
SMALL_MOE_CFG = MoEGPTConfig(n_layer=6, n_head=8, n_embd=512, block_size=256,
                             n_experts=8, e_max=16, top_k=2)   # ≈107.5M total
MODEL_PRESETS = {"toy": GOSSIP_CFG, "small": SMALL_CFG,
                 "toy-moe": TOY_MOE_CFG, "small-moe": SMALL_MOE_CFG}


def build_preset(name: str, device: str = None, seed: int = None,
                 experts_per_layer: list[int] | None = None):
    """Build a preset by name — dense presets via gpt.build, MoE presets via
    moe.build_moe (optionally with a ragged per-layer expert count, e.g. after
    on-chain growth). Returns (model, device)."""
    cfg = MODEL_PRESETS[name]
    if isinstance(cfg, MoEGPTConfig):
        return build_moe(cfg, device=device, seed=seed,
                         experts_per_layer=experts_per_layer)
    from .gpt import build as _build
    return _build(cfg, device=device, seed=seed)

# module config — set by CLI flags before nodes are built (watch.py shares these)
MODEL_CFG = GOSSIP_CFG
INNER_STEPS = 10
BATCH = 24
INCLUDE_K = 8
GENESIS_SEED = 1337          # network constant: every node's genesis weights match
GENESIS_FILE = None          # …or a shared pretrained genesis (convert_ckpt --genesis)
DATA_PATH = None             # corpus file (default: TinyShakespeare auto-download)
DEVICE_OVERRIDE = None       # set by --device
PRUNE_DEPTH = 8              # keep heavy per-block state only this deep (0.7GB each at 86M)
DATA_CONTRIBUTOR = None      # genesis parameter: address earning the data share (§9).
                             # MUST be identical on every node (it's part of the
                             # deterministic reward computation).


async def _send(w, obj):
    data = pickle.dumps(obj)
    w.write(struct.pack(">I", len(data)) + data); await w.drain()


async def _recv(r):
    (n,) = struct.unpack(">I", await r.readexactly(4))
    return pickle.loads(await r.readexactly(n))


class RealCore:
    """Consensus + real-model training for one gossip node."""

    def __init__(self, node_id, seed=0):
        self.node_id = node_id
        self.model, self.device = build(MODEL_CFG, device=DEVICE_OVERRIDE, seed=GENESIS_SEED)
        if GENESIS_FILE:                          # warm start: shared pretrained genesis
            from .convert_ckpt import load_genesis
            w, _ = load_genesis(GENESIS_FILE)
            set_flat_params(self.model, w)
        kwargs = {"path": DATA_PATH} if DATA_PATH else {}
        self.data = ByteData(block_size=MODEL_CFG.block_size, device=self.device, **kwargs)
        self.miner = DiLoCoMiner(self.model, self.data, self.device)
        self.key = Key.generate(f"node{node_id}".encode().ljust(32, b"0"))
        # the token ledger is chain state, owned by the BlockTree: validated per
        # block (ledger_root), empty at genesis (fair launch), deterministic.
        self.tree = BlockTree(quantize(flat_params(self.model)), prune_depth=PRUNE_DEPTH,
                              data_contributor=DATA_CONTRIBUTOR)
        self.mempool, self.seen_tx, self.seen_block = {}, set(), set()
        self.orphans, self.pending = {}, {}       # pending: blocks awaiting bodies
        # only COMPRESSED payloads are retained (an 86M dense body is ~0.7GB; its
        # payload ~14MB). Dense bodies are derived on demand and dropped.
        self.payload_store = {}                   # txid -> compressed payload, RETAINED
        self.comp = Compressor(keep_frac=KEEP_FRAC)
        self.transfer_pool = {}                   # txid -> TransferTx awaiting inclusion
        self.seen_xfer = set()
        self.data_pool = {}                       # txid -> Data{Submit,Challenge,Vote}Tx
        self.seen_dtx = set()

    def head_ledger(self):
        return self.tree.head_ledger()

    def recv_transfer(self, tx, outbox):
        """A signed transfer enters the mempool and gossips on; it SETTLES when a
        proposer includes it in a block (the ledger_root then commits it)."""
        if tx.txid() in self.seen_xfer or not tx.verify():
            return
        self.seen_xfer.add(tx.txid())
        self.transfer_pool[tx.txid()] = tx
        outbox.append(("xfer", tx))

    def recv_data_tx(self, tx, outbox):
        """A signed data-lane tx (submit/challenge/vote) enters the mempool and
        gossips on; a proposer includes it and the ledger_root commits it."""
        if tx.txid() in self.seen_dtx or not tx.verify():
            return
        self.seen_dtx.add(tx.txid())
        self.data_pool[tx.txid()] = tx
        outbox.append(("dtx", tx))

    def _body(self, txid):
        """Densify a retained payload on demand (transient — never stored)."""
        return decompress(self.payload_store[txid])

    def head_snapshot(self):
        """Read the current head on the CALLER's thread (cheap, touches the tree)."""
        hh = self.tree.blocks[self.tree.head].header.height
        return hh, dequantize(self.tree.head_state())

    def train_from(self, hh, weights):
        """The heavy GPU work — safe to run in an executor because it touches only
        self.model, never the tree. The main thread keeps installing gossiped
        blocks while this runs, so a fast peer can't starve a slow one. On CPU/CUDA
        PyTorch releases the GIL here; the network loop stays responsive."""
        set_flat_params(self.model, weights)
        delta, loss = self.miner.inner_train(INNER_STEPS, BATCH, seed=hh * 100 + self.node_id)
        return hh, delta, loss

    def train_delta(self):
        """Blocking convenience form (MPS path): snapshot + train on one thread."""
        hh, weights = self.head_snapshot()
        return self.train_from(hh, weights)

    def submit_delta(self, hh, delta_int, outbox):
        """Compress the delta (top-k + error feedback), sign a commitment tx, and
        gossip only the small payload — the body never rides in a block."""
        if hh != self.tree.blocks[self.tree.head].header.height:
            return
        payload = self.comp.compress(dequantize(delta_int))    # small on the wire
        dense = decompress(payload)                            # what everyone commits to
        dh = delta_hash(dense.tobytes())
        ptr = f"da://{dh}"                                     # CONTENT address — unique per body
        # rev 5 provenance: name the corpora this delta trained on. The sim
        # trainer draws from the whole staked pool, so it names every ACTIVE
        # registry corpus at its base height (at minimum the founding corpus).
        led = self.tree.ledger[self.tree.head]
        refs = sorted({e["data_hash"] for e in led.registry.values()
                       if e["status"] == "active"}) or ["genesis"]
        tx = BackpropTx(miner=self.key.pub, base_height=hh,
                        delta_hash=dh, da_pointer=ptr,
                        data_refs=refs).signed(self.key)
        if tx.txid() not in self.seen_tx:
            self.seen_tx.add(tx.txid())
            self.mempool[tx.txid()] = tx
            self.payload_store[tx.txid()] = payload
            outbox.append(("tx", tx, payload))

    def recv_tx(self, tx, payload, outbox):
        if tx.txid() in self.seen_tx:
            return
        if not tx.verify():
            _dbg(self.node_id, f"tx from {tx.miner[:8]} REJECT (bad sig)")
            return
        dense = decompress(payload)                           # transient — hash check only
        if delta_hash(dense.tobytes()) != tx.delta_hash:
            _dbg(self.node_id, f"tx from {tx.miner[:8]} REJECT (hash mismatch)")
            return
        del dense
        self.seen_tx.add(tx.txid())
        self.mempool[tx.txid()] = tx
        self.payload_store[tx.txid()] = payload
        outbox.append(("tx", tx, payload))
        self._retry_pending(outbox)                           # a block may now be complete

    def propose(self, outbox):
        head = self.tree.head
        hh = self.tree.blocks[head].header.height
        cands = [tx for tx in self.mempool.values() if tx.base_height == hh]
        if not cands:
            return
        cands.sort(key=lambda t: t.txid())
        chosen, seen_miners = [], set()
        for tx in cands:                           # at most one delta per miner per block
            if tx.miner in seen_miners:
                continue
            seen_miners.add(tx.miner)
            chosen.append(tx)
            if len(chosen) >= INCLUDE_K:
                break
        accepted = chosen
        bodies = {tx.da_pointer: self._body(tx.txid()) for tx in chosen}   # transient
        # account lanes: include pool txs (data + transfers) that apply cleanly
        # after this block's rewards — dry-run in the SAME canonical order the
        # validator uses, so we never build an invalid block
        from rig.token import TransferTx
        scratch = self.tree.ledger[head].copy()
        scratch.resolve_expired_challenges(hh + 1)
        # mirror apply_ledger's rev-5 data credits so the dry-run ledger
        # matches the validator's exactly (incl. rev-6 fee-pool drains)
        hash_weight = {e["data_hash"]: e["weight"]
                       for e in scratch.registry.values()
                       if e["status"] == "active" and e["weight"] > 0}
        credits: dict[str, int] = {}
        for tx in accepted:
            for r in tx.canonical_refs():
                if r in hash_weight:
                    credits[r] = credits.get(r, 0) + hash_weight[r]
        scratch.apply_reward(hh + 1, [tx.miner for tx in accepted], self.key.pub,
                             [DATA_CONTRIBUTOR] if DATA_CONTRIBUTOR else [],
                             data_credits=credits)
        jurors = self.tree.recent_proposers(head)
        xfers, dtxs = [], []
        for t in canonical_account_txs(list(self.data_pool.values()),
                                       list(self.transfer_pool.values())):
            if isinstance(t, TransferTx):
                if t.verify() and scratch.apply_transfer(t):
                    xfers.append(t)
            elif scratch.apply_data_tx(t, hh + 1, jurors):
                dtxs.append(t)
        block = build_block(self.tree, head, accepted, bodies,
                            {tx.txid(): 1.0 for tx in chosen}, self.key,
                            transfers=xfers, data_txs=dtxs)
        try:
            became_head = self.tree.add_block(block)           # our own block; guard anyway
        except ValidationError as e:
            _dbg(self.node_id, f"own block rejected: {e}")
            return
        for t in xfers:                                        # included -> out of pool
            self.transfer_pool.pop(t.txid(), None)
        for t in dtxs:
            self.data_pool.pop(t.txid(), None)
        if became_head:
            self.seen_block.add(block.hash)
            self._prune(block)
            outbox.append(("block", block.header, block.txs,
                           block.transfers, block.data_txs))

    def recv_block(self, header, txs, transfers, data_txs, outbox):
        bh = header.block_hash()
        if bh in self.seen_block:
            return
        bodies, missing = {}, False
        for tx in txs:                                 # densify from retained payloads
            if tx.txid() in self.payload_store:
                bodies[tx.da_pointer] = self._body(tx.txid())
            else:
                missing = True
        if missing:
            self.pending[bh] = (header, txs, transfers, data_txs)
            outbox.append(("getblock", bh))            # …and request the full block (getdata)
            return
        _dbg(self.node_id, f'block h{header.height} bodies-ready, installing')
        self._install(Block(header, txs, bodies, list(transfers), list(data_txs)), outbox)

    def serve_block(self, bh, outbox):
        """Answer a getblock request with a compressed full block if we have it."""
        b = self.tree.blocks.get(bh)
        if b is not None and all(tx.txid() in self.payload_store for tx in b.txs):
            payloads = {tx.txid(): self.payload_store[tx.txid()] for tx in b.txs}
            outbox.append(("fullblock", b.header, b.txs, payloads,
                           b.transfers, b.data_txs))

    def recv_fullblock(self, header, txs, payloads, transfers, data_txs, outbox):
        """Initial block download / getdata reply: a block plus the COMPRESSED
        payloads for its txs, so a node that missed the txs can reconstruct the
        bodies and catch up — small on the wire (payloads, not dense bodies)."""
        bh = header.block_hash()
        if bh in self.seen_block:
            return
        bodies = {}
        for tx in txs:
            dense = decompress(payloads[tx.txid()])
            if delta_hash(dense.tobytes()) != tx.delta_hash:
                return
            self.payload_store[tx.txid()] = payloads[tx.txid()]
            bodies[tx.da_pointer] = dense              # transient — used for install only
        _dbg(self.node_id, f'fullblock h{header.height} received, installing')
        self._install(Block(header, txs, bodies, list(transfers), list(data_txs)), outbox)

    def _install(self, block, outbox):
        try:
            self.tree.add_block(block)                 # validates weights AND ledger
        except ValidationError as e:
            if "orphan" in str(e):
                self.orphans.setdefault(block.header.prev_hash, []).append(
                    (block.header, block.txs, block.transfers, block.data_txs))
            else:
                _dbg(self.node_id, f"h{block.header.height} INVALID: {e}")
            return
        self.seen_block.add(block.hash)
        _dbg(self.node_id, f'INSTALLED h{block.header.height}, head=h{self.tree.blocks[self.tree.head].header.height}')
        self._prune_txs(block.txs)
        for t in block.transfers:                      # settled -> out of the pools
            self.transfer_pool.pop(t.txid(), None)
            self.seen_xfer.add(t.txid())
        for t in block.data_txs:
            self.data_pool.pop(t.txid(), None)
            self.seen_dtx.add(t.txid())
        outbox.append(("block", block.header, block.txs,
                       block.transfers, block.data_txs))
        for ch, ct, cx, cd in self.orphans.pop(block.hash, []):
            self.recv_block(ch, ct, cx, cd, outbox)

    def _retry_pending(self, outbox):
        for bh, (header, txs, transfers, data_txs) in list(self.pending.items()):
            if all(tx.txid() in self.payload_store for tx in txs):
                del self.pending[bh]
                self.recv_block(header, txs, transfers, data_txs, outbox)

    def _prune(self, block):
        self._prune_txs(block.txs)

    def _prune_txs(self, txs):
        for tx in txs:
            self.mempool.pop(tx.txid(), None)

    def rebroadcast(self, outbox):
        for b in self.tree.chain_from_genesis():
            if all(tx.txid() in self.payload_store for tx in b.txs):
                payloads = {tx.txid(): self.payload_store[tx.txid()] for tx in b.txs}
                outbox.append(("fullblock", b.header, b.txs, payloads,
                               b.transfers, b.data_txs))

    def val_loss(self):
        set_flat_params(self.model, dequantize(self.tree.head_state()))
        return self.data.estimate_loss(self.model, iters=6)["val"]


class GossipNode:
    def __init__(self, node_id, host, port, peers, n_total, interval=1.5, t0=None):
        self.core = RealCore(node_id)
        self.host, self.port, self.peers, self.n_total = host, port, peers, n_total
        self.interval, self.t0 = interval, t0 or time.time()
        self.writers = set()
        self.peer_ids = set()             # dedup: at most one connection per peer
        self._stop = asyncio.Event()

    async def _peer(self, reader, writer, dialer=False):
        await _send(writer, ("hello", self.core.node_id))
        try:
            hello = await _recv(reader)
            pid = hello[1] if hello and hello[0] == "hello" else None
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            writer.close(); return
        if pid is None:
            writer.close(); return
        # Simultaneous open: both peers dial AND accept, so two connections form.
        # Resolve deterministically — keep the one whose DIALER has the smaller id.
        # Both endpoints compute the same verdict, so exactly one full-duplex
        # channel survives (mismatched closes would leave no working link).
        keep = (self.core.node_id < pid) if dialer else (pid < self.core.node_id)
        _dbg(self.core.node_id, f"hello pid={pid} dialer={dialer} keep={keep} "
                                f"already={pid in self.peer_ids}")
        if not keep or pid in self.peer_ids:
            writer.close(); return
        self.peer_ids.add(pid)
        self.writers.add(writer)
        try:
            for b in self.core.tree.chain_from_genesis():
                if all(tx.txid() in self.core.payload_store for tx in b.txs):
                    pl = {tx.txid(): self.core.payload_store[tx.txid()] for tx in b.txs}
                    await _send(writer, ("fullblock", b.header, b.txs, pl,
                                         b.transfers, b.data_txs))
            while not self._stop.is_set():
                self._handle(await _recv(reader))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.writers.discard(writer); self.peer_ids.discard(pid); writer.close()

    async def _dial(self, host, port):
        for _ in range(60):
            try:
                r, w = await asyncio.open_connection(host, port)
                _dbg(self.core.node_id, f"dialed {host}:{port} OK")
                await self._peer(r, w, dialer=True)
                _dbg(self.core.node_id, f"dial-peer to {host}:{port} ended")
                return
            except (ConnectionError, OSError) as e:
                _dbg(self.core.node_id, f"dial {host}:{port} retry ({e})")
                await asyncio.sleep(1)

    def _handle(self, msg):
        outbox = []
        if _DBG:
            print(f"    [n{self.core.node_id} RECV {msg[0]}]", flush=True)
        if msg[0] == "tx":
            self.core.recv_tx(msg[1], msg[2], outbox)          # (tx, payload)
        elif msg[0] == "xfer":
            self.core.recv_transfer(msg[1], outbox)            # token transfer -> mempool
        elif msg[0] == "dtx":
            self.core.recv_data_tx(msg[1], outbox)             # data lane -> mempool
        elif msg[0] == "block":
            self.core.recv_block(msg[1], msg[2], msg[3], msg[4], outbox)
        elif msg[0] == "fullblock":
            self.core.recv_fullblock(msg[1], msg[2], msg[3], msg[4], msg[5], outbox)
        elif msg[0] == "getblock":
            self.core.serve_block(msg[1], outbox)              # peer needs a full block
        for m in outbox:
            self._bcast(m)

    def _bcast(self, msg):
        for w in list(self.writers):
            asyncio.create_task(self._safe(w, msg))

    async def _safe(self, w, msg):
        try:
            await _send(w, msg)
        except (ConnectionError, OSError):
            self.writers.discard(w)

    async def _loop(self, seconds, settle=12.0):
        loop = asyncio.get_event_loop()
        # BACKPRESSURE FIX: on CPU/CUDA, run training in an executor so the network
        # event loop keeps draining gossip while we train — a fast peer can no
        # longer flood a slow peer into starvation (they stay head-synced because
        # the slow peer installs received blocks *during* its own training). MPS
        # misbehaves off the main thread, so it falls back to blocking training.
        use_executor = self.core.device != "mps"
        end = time.time() + seconds
        while time.time() < end:
            if use_executor:
                hh, weights = self.core.head_snapshot()          # read tree on main thread
                hh, delta, loss = await loop.run_in_executor(     # heavy work off-loop
                    None, self.core.train_from, hh, weights)
            else:
                hh, delta, loss = self.core.train_delta()
            outbox = []
            self.core.submit_delta(hh, delta, outbox)
            rnd = int((time.time() - self.t0) / self.interval)
            if rnd % self.n_total == self.core.node_id:         # rotating leader
                self.core.propose(outbox)
            for m in outbox:
                self._bcast(m)
            h = self.core.tree.blocks[self.core.tree.head].header.height
            print(f"  node {self.core.node_id}  height {h}  inner loss {loss:.3f}", flush=True)
            await asyncio.sleep(self.interval)                  # let gossip flow
        # quiescent settle so in-flight blocks land and heads converge
        for _ in range(int(settle / self.interval) + 1):
            await asyncio.sleep(self.interval)
            outbox = []; self.core.rebroadcast(outbox)
            for m in outbox:
                self._bcast(m)
        self._stop.set()

    async def run(self, seconds):
        server = await asyncio.start_server(self._peer, self.host, self.port)
        async with server:
            await asyncio.sleep(0.5)
            dials = [asyncio.create_task(self._dial(h, p)) for h, p in self.peers]
            await self._loop(seconds)
            for d in dials:
                d.cancel()
        h = self.core.tree.blocks[self.core.tree.head].header.height
        lineage = ">".join(b.hash[:6] for b in self.core.tree.chain_from_genesis())
        print(f"node {self.core.node_id} LINEAGE {lineage}", flush=True)
        print(f"node {self.core.node_id} done — height {h}, head {self.core.tree.head[:16]}, "
              f"seen_tx {len(self.core.seen_tx)} seen_block {len(self.core.seen_block)} "
              f"pending {len(self.core.pending)} peers {len(self.writers)}, "
              f"val loss {self.core.val_loss():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--peers", default="")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--device", default=None)      # cuda|mps|cpu (auto if unset)
    ap.add_argument("--model", default="toy", choices=list(MODEL_PRESETS))
    ap.add_argument("--data", default=None)        # corpus path (default TinyShakespeare)
    ap.add_argument("--genesis", default=None)     # pretrained genesis .npz (all nodes SAME file)
    ap.add_argument("--inner", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--data-contributor", default=None,
                    help="genesis param: address earning the data share (same on ALL nodes)")
    a = ap.parse_args()
    apply_flags(a)
    peers = [(h, int(p)) for h, p in (x.split(":") for x in a.peers.split(",") if x)]
    node = GossipNode(a.id, "0.0.0.0", a.port, peers, a.n,
                      interval=a.interval, t0=a.t0 or None)
    asyncio.run(node.run(a.seconds))


def apply_flags(a):
    """Set module config from CLI flags (shared by gossip and watch mains)."""
    global DEVICE_OVERRIDE, MODEL_CFG, DATA_PATH, GENESIS_FILE, INNER_STEPS, BATCH, \
        DATA_CONTRIBUTOR
    DEVICE_OVERRIDE = getattr(a, "device", None)
    if getattr(a, "data_contributor", None):
        DATA_CONTRIBUTOR = a.data_contributor
    if getattr(a, "model", None):
        MODEL_CFG = MODEL_PRESETS[a.model]
    if getattr(a, "data", None):
        DATA_PATH = a.data
    if getattr(a, "genesis", None):
        GENESIS_FILE = a.genesis
    if getattr(a, "inner", None):
        INNER_STEPS = a.inner
    if getattr(a, "batch", None):
        BATCH = a.batch


if __name__ == "__main__":
    main()
