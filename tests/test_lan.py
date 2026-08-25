"""LAN node: coordinator + miners over real TCP produce a correct, replayable chain.

Uses loopback (127.0.0.1) so the test needs no second machine; the same code
path runs cross-machine (verified by hand against a second machine — see docs/internal/lan.md).
"""

import socket
import threading
import time

from rig.chain import dequantize, state_root
from rig.lan import MODEL, run_coordinator, run_miner


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_coordinator_and_miners_over_tcp():
    port = _free_port()
    result = {}

    def coord():
        result["chain"] = run_coordinator(port, n_miners=2, blocks=12, seed=7,
                                          host="127.0.0.1")

    t = threading.Thread(target=coord, daemon=True)
    t.start()
    time.sleep(0.4)                                  # let it bind/listen
    miners = [threading.Thread(target=run_miner, args=("127.0.0.1", port, i),
                               daemon=True) for i in range(2)]
    for m in miners:
        m.start()
    t.join(timeout=30)

    chain = result["chain"]
    assert chain is not None and chain.height == 12
    assert state_root(chain.replay()) == chain.blocks[-1].root   # replayable
    assert MODEL.accuracy(dequantize(chain.w_int),
                          MODEL.sample_batch(__import__("numpy").random.default_rng(1006),
                                             200)) > 0.5          # it trained
