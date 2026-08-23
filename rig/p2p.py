"""Peer-to-peer gossip: nodes agree on a chain with no coordinator (§5, §3).

The LAN node still has one coordinator dictating blocks. A real network has
none — every node holds a BlockTree, gossips signed txs and blocks to peers,
proposes blocks itself, and resolves disagreement by heaviest-valid-chain fork
choice. This module builds that as a deterministic in-process simulator (so the
distributed-systems properties are testable) driving real training.

Properties the simulator lets us test:
  * a tx injected at one node reaches every node (gossip flood);
  * all nodes converge on the same head (consensus without a coordinator);
  * the model trains across the gossip network;
  * a network partition produces two forks that HEAL to the heavier chain when
    the partition lifts (Nakamoto fork choice doing its job).

Bodies travel with their tx in the sim (the DA layer is modelled as gossip);
`rig/blockchain.py` still validates every block from first principles on arrival.
"""

from dataclasses import dataclass, field

import numpy as np

from .blockchain import BlockTree, ValidationError, build_block
from .chain import beacon, dequantize, quantize, state_root
from .crypto import BackpropTx, Key, delta_hash
from .model import TinyTransformer

MODEL = TinyTransformer()
INNER_STEPS = 4
SHARD_BATCH = 32
EVAL_BATCH = 96
INCLUDE_K = 4
LR = 0.3

# rev 5 provenance: sim nodes train on the founding corpus, so every tree seeds
# the same genesis data entry (a consensus parameter — identical on all nodes)
# and every delta names it.
from .token import address as _address  # noqa: E402  (import after constants)

P2P_FOUNDER = _address(Key.generate(b"p2p-sim-founder-0000000000000000").pub)


@dataclass
class GossipNode:
    node_id: int
    key: Key
    tree: BlockTree
    peers: list = field(default_factory=list)          # list[GossipNode]
    mempool: dict = field(default_factory=dict)         # txid -> (tx, body)
    seen_tx: set = field(default_factory=set)
    seen_block: set = field(default_factory=set)
    orphans: dict = field(default_factory=dict)         # parent_hash -> [blocks]

    # -- tx production & gossip --------------------------------------------
    def make_tx(self):
        """Train against this node's current head and sign a delta tx."""
        h = self.tree.blocks[self.tree.head].header.height
        w = dequantize(self.tree.head_state())
        rng = np.random.default_rng(int(beacon(h, f"n{self.node_id}").integers(1 << 30)))
        v = w.copy()
        for _ in range(INNER_STEPS):
            v = MODEL.train_step(v, MODEL.sample_batch(rng, SHARD_BATCH), lr=LR, steps=1)
        body = quantize(v - w)
        ptr = f"da://{self.key.pub[:8]}/{h}/{self.node_id}"
        tx = BackpropTx(miner=self.key.pub, base_height=h,
                        delta_hash=delta_hash(body.tobytes()), da_pointer=ptr,
                        data_refs=["genesis"]).signed(self.key)
        return tx, body

    def submit_own_tx(self, outbox):
        tx, body = self.make_tx()
        if tx.txid() not in self.seen_tx:
            self.seen_tx.add(tx.txid())
            self.mempool[tx.txid()] = (tx, body)
            outbox.append(("tx", tx, body))

    def recv_tx(self, tx, body, outbox):
        if tx.txid() in self.seen_tx or not tx.verify():
            return
        if delta_hash(body.tobytes()) != tx.delta_hash:
            return
        self.seen_tx.add(tx.txid())
        self.mempool[tx.txid()] = (tx, body)
        outbox.append(("tx", tx, body))                # forward (flood)

    # -- block production & gossip -----------------------------------------
    def propose(self, outbox):
        head = self.tree.head
        hh = self.tree.blocks[head].header.height
        w_base = dequantize(self.tree.head_state())
        eb = MODEL.sample_batch(beacon(hh, "eval"), EVAL_BATCH)
        base_loss = MODEL.loss(w_base, eb)
        cands = []
        for txid, (tx, body) in self.mempool.items():
            if tx.base_height != hh:                    # only txs built on this head
                continue
            score = base_loss - MODEL.loss(w_base + dequantize(body), eb)
            if score > 0:
                cands.append((score, tx, body))
        if not cands:
            return None
        cands.sort(key=lambda t: (-t[0], t[1].txid()))
        chosen = cands[:INCLUDE_K]
        accepted = [c[1] for c in chosen]
        bodies = {c[1].da_pointer: c[2] for c in chosen}
        works = {c[1].txid(): c[0] for c in chosen}
        block = build_block(self.tree, head, accepted, bodies, works, self.key)
        if self.tree.add_block(block):
            self.seen_block.add(block.hash)
            self._prune(block)
            outbox.append(("block", block))
        return block

    def recv_block(self, block, outbox):
        if block.hash in self.seen_block:
            return
        try:
            became_head = self.tree.add_block(block)
        except ValidationError as e:
            if "orphan" in str(e):                      # buffer until parent arrives
                self.orphans.setdefault(block.header.prev_hash, []).append(block)
            return
        self.seen_block.add(block.hash)
        self._prune(block)
        outbox.append(("block", block))                # forward
        # attach any orphans now that this block landed
        for child in self.orphans.pop(block.hash, []):
            self.recv_block(child, outbox)

    def _prune(self, block):
        for tx in block.txs:
            self.mempool.pop(tx.txid(), None)

    def rebroadcast_chain(self, outbox):
        """Re-announce our whole chain to peers (genesis-first so parents precede
        children). This is how a healed partition reconciles: a peer ignores
        blocks it already has (seen_block) and adopts the fork it was missing,
        after which heaviest-chain fork choice picks one winner everywhere."""
        for b in self.tree.chain_from_genesis():
            outbox.append(("block", b))


class Network:
    """Deterministic gossip fabric. Messages sent in step t arrive at step t+1
    (one-hop latency), so gossip floods outward a hop per step."""

    def __init__(self, n_nodes=5, topology="ring", seed=0, genesis_dim=None):
        genesis_dim = genesis_dim or MODEL.param_count
        w0 = quantize(MODEL.init(np.random.default_rng(seed)))
        self.nodes = []
        for i in range(n_nodes):
            key = Key.generate(f"node{i}".encode().ljust(32, b"0"))
            self.nodes.append(GossipNode(
                i, key, BlockTree(w0.copy(), data_contributor=P2P_FOUNDER)))
        self._wire(topology)
        self.inflight = {i: [] for i in range(n_nodes)}   # node -> [(kind, payload)]
        self.partition = None

    def _wire(self, topology):
        n = len(self.nodes)
        for i, node in enumerate(self.nodes):
            if topology == "ring":
                node.peers = [self.nodes[(i - 1) % n], self.nodes[(i + 1) % n]]
            elif topology == "full":
                node.peers = [self.nodes[j] for j in range(n) if j != i]
            else:
                raise ValueError(topology)

    def set_partition(self, groups):
        """groups: list of node-id sets that can talk only within the set."""
        self.partition = groups

    def _can_talk(self, a, b):
        if self.partition is None:
            return True
        for g in self.partition:
            if a in g and b in g:
                return True
        return False

    def inject_tx(self, node_id, tx, body):
        """Seed a tx at one node's inbox (as if a peer sent it)."""
        self.inflight[node_id].append(("tx", tx, body))

    def step(self, proposers=None, sync=False, produce=True):
        n = len(self.nodes)
        # 1. deliver in-flight, collect each node's outgoing
        outbox = {i: [] for i in range(n)}
        for i, node in enumerate(self.nodes):
            for kind, *payload in self.inflight[i]:
                if kind == "tx":
                    node.recv_tx(payload[0], payload[1], outbox[i])
                else:
                    node.recv_block(payload[0], outbox[i])
        # 2. sync (reconcile after heal), or make txs + propose
        if sync:
            for i, node in enumerate(self.nodes):
                node.rebroadcast_chain(outbox[i])
        else:
            if produce:
                for i, node in enumerate(self.nodes):
                    node.submit_own_tx(outbox[i])
            for i in (proposers if proposers is not None else [self._leader()]):
                self.nodes[i].propose(outbox[i])
        # 3. queue outgoing to peers (arrives next step), respecting partitions
        nxt = {i: [] for i in range(n)}
        for i, node in enumerate(self.nodes):
            for msg in outbox[i]:
                for peer in node.peers:
                    if self._can_talk(i, peer.node_id):
                        nxt[peer.node_id].append(msg)
        self.inflight = nxt

    _round = 0

    def _leader(self):
        self._round += 1
        return self._round % len(self.nodes)

    def heads(self):
        return [node.tree.head for node in self.nodes]

    def converged(self):
        return len(set(self.heads())) == 1

    def head_height(self):
        return max(self.nodes[i].tree.blocks[self.nodes[i].tree.head].header.height
                   for i in range(len(self.nodes)))

    def accuracy(self):
        test = MODEL.sample_batch(np.random.default_rng(123456), 200)
        # accuracy of the (agreed) head state, taken from node 0's view
        return MODEL.accuracy(dequantize(self.nodes[0].tree.head_state()), test)


def run_gossip(n_nodes=5, steps=60, topology="ring", seed=0, verbose=True):
    net = Network(n_nodes=n_nodes, topology=topology, seed=seed)
    for t in range(steps):
        net.step()
        if verbose and (t % 10 == 0 or t == steps - 1):
            heights = [net.nodes[i].tree.blocks[net.nodes[i].tree.head].header.height
                       for i in range(n_nodes)]
            print(f"step {t:>3}  heights {heights}  converged {net.converged()}  "
                  f"acc {net.accuracy():.3f}", flush=True)
    return net


def _heads_state(net):
    return set(net.nodes[i].tree.blocks[net.nodes[i].tree.head].header.state_root
               for i in range(len(net.nodes)))


def main():
    print("=" * 70)
    print("  SESTRIAN — gossip network: consensus with no coordinator")
    print("=" * 70)
    net = Network(n_nodes=6, topology="full", seed=0)
    print("\n6 nodes, no coordinator. Each trains on its head, gossips signed")
    print("txs + blocks, and follows the heaviest valid chain.\n")
    for _ in range(18):
        net.step()
    for _ in range(6):
        net.step(proposers=[])                       # quiesce and settle
    print(f"  after production+settle: single head "
          f"{len(set(net.heads())) == 1}, height {net.head_height()}, "
          f"model acc {net.accuracy():.3f}")

    print("\n  → partition the network into {0,1,2} | {3,4,5}; each side mines on")
    print("    its own, producing two competing forks…")
    net.set_partition([{0, 1, 2}, {3, 4, 5}])
    for _ in range(12):
        net.step(proposers=[0, 3])
    a = net.nodes[0].tree.blocks[net.nodes[0].tree.head].header
    b = net.nodes[3].tree.blocks[net.nodes[3].tree.head].header
    print(f"    fork A {a.block_hash()[:10]} (ht {a.height})  ≠  "
          f"fork B {b.block_hash()[:10]} (ht {b.height})")

    print("\n  → heal the partition; nodes reconcile by gossiping their chains…")
    net.set_partition(None)
    for _ in range(8):
        net.step(sync=True)
    print(f"    after heal: single head {len(set(net.heads())) == 1}, "
          f"one agreed history {len(_heads_state(net)) == 1}, "
          f"height {net.head_height()}, acc {net.accuracy():.3f}")
    print("=" * 70)
    ok = len(set(net.heads())) == 1 and len(_heads_state(net)) == 1
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
