"""Async socket gossip over real TCP: coordinator-free consensus (§5).

Loopback (real sockets, one event loop) so the test needs no second machine;
the same code ran cross-machine (Mac + a second machine) — see docs/internal/distributed-systems.md."""

from rig.chain import state_root
from rig.gossip_net import run_cluster


def test_async_gossip_reaches_consensus_over_sockets():
    nodes, trees = run_cluster(n=3, seconds=6, base_port=9740, seed=0, interval=0.4)
    heights = [n.head_height() for n in nodes]
    roots = set(n.head_root() for n in nodes)
    assert len(roots) == 1                      # one agreed history, no coordinator
    assert min(heights) >= 3                     # blocks were actually produced
    # every node's winning chain replays to its committed head state
    for n in nodes:
        t = n.core.tree
        assert state_root(t.replay_head()) == t.blocks[t.head].header.state_root


def test_async_gossip_trains_the_model():
    import numpy as np
    from rig.chain import dequantize
    from rig.p2p import MODEL
    nodes, _ = run_cluster(n=3, seconds=6, base_port=9760, seed=0, interval=0.4)
    acc = MODEL.accuracy(dequantize(nodes[0].core.tree.head_state()),
                         MODEL.sample_batch(np.random.default_rng(123456), 200))
    assert acc > 0.5
