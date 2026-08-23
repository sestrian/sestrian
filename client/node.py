"""Production client node — real GPT trained across machines through the chain.

A coordinator holds the head weights; miners connect, sync the head, train the
real GPT on their own GPU for H inner steps, sign the pseudo-gradient delta
(Ed25519), and submit it. The coordinator verifies each signature and applies
the chain's deterministic aggregation — so deltas from a CUDA 2080 Ti and an
Apple-MPS M3 land in one bit-exact chain state. The model trained is real; the
data is real; only the aggregation is fixed-point (the consensus boundary, §6.3).

This is the round-robin/coordinator form (the gossip form is rig/gossip_net.py);
it is the simplest thing that proves real cross-GPU distributed training over the
network. Both sides must agree on MODEL_CFG (they exchange weight vectors).

  # coordinator (any machine):
  python -m client.node coordinator --port 9800 --miners 2 --rounds 20

  # each miner (point --host at the coordinator's IP):
  python -m client.node miner --host 100.x.y.z --port 9800 --id 0
"""

import argparse
import socket
import time

import numpy as np

from rig.chain import dequantize, quantize, state_root, trimmed_mean_int
from rig.crypto import BackpropTx, Key, delta_hash
from rig.protocol import recv_msg, send_msg
from .compress import Compressor, decompress, payload_bytes
from .data import ByteData
from .gpt import GPTConfig, build
from .trainer import DiLoCoMiner, flat_params, set_flat_params

# shared architecture — coordinator and miners MUST match (they trade weights)
MODEL_CFG = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128)
INNER_STEPS = 15
BATCH = 32
KEEP_FRAC = 0.02        # top-k delta compression (~50x on the wire)


def _new_model():
    return build(MODEL_CFG)


# --------------------------------------------------------------------------
# Coordinator
# --------------------------------------------------------------------------
def run_coordinator(port, n_miners, rounds, host="0.0.0.0", seed=0):
    model, device = _new_model()
    data = ByteData(block_size=MODEL_CFG.block_size, device=device)
    base = quantize(flat_params(model))               # genesis head (int64)
    n_params = base.size

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(n_miners)
    print(f"coordinator on {host}:{port} — real GPT {model.num_params()/1e6:.1f}M "
          f"params on {device}; waiting for {n_miners} miners…", flush=True)
    conns = {}
    while len(conns) < n_miners:
        c, addr = srv.accept()
        hello = recv_msg(c)
        conns[hello["miner_id"]] = (c, hello["pub"])
        print(f"  miner {hello['miner_id']} joined from {addr[0]} "
              f"({len(conns)}/{n_miners})", flush=True)

    def val():
        set_flat_params(model, dequantize(base)); return data.estimate_loss(model, iters=8)["val"]

    print(f"genesis val loss {val():.3f}", flush=True)
    t0 = time.time()
    for r in range(rounds):
        w_bytes = dequantize(base).astype(np.float32).tobytes()   # send head (fp32)
        for mid, (c, _) in conns.items():
            send_msg(c, {"type": "train", "weights": w_bytes, "height": r})
        deltas, wire = [], 0
        for mid, (c, pub) in conns.items():                       # synchronous barrier
            m = recv_msg(c)
            d = decompress(m["payload"])                          # compressed on the wire
            wire += payload_bytes(m["payload"])
            tx = BackpropTx(miner=pub, base_height=r,
                            delta_hash=delta_hash(d.tobytes()), da_pointer=f"da://{r}/{mid}")
            tx.sig = m["sig"]
            if tx.verify() and delta_hash(d.tobytes()) == tx.delta_hash:
                deltas.append(d)                                  # signed + intact
            else:
                print(f"  ! rejected delta from miner {mid} (bad signature)", flush=True)
        if deltas:
            base = base + trimmed_mean_int(deltas)
        if r % 4 == 0 or r == rounds - 1:
            raw = n_params * 8 * n_miners
            print(f"  round {r:>2}  val loss {val():.3f}  deltas {len(deltas)}/{n_miners}  "
                  f"root {state_root(base)[:10]}  wire {wire/1e3:.0f}KB (vs {raw/1e6:.0f}MB "
                  f"raw, {raw/max(1,wire):.0f}x)  ({time.time()-t0:.0f}s)", flush=True)

    for c, _ in conns.values():
        send_msg(c, {"type": "stop"}); c.close()
    srv.close()
    print(f"\ndone: real {model.num_params()/1e6:.1f}M GPT trained across "
          f"{n_miners} miners; final val loss {val():.3f}", flush=True)
    return base


# --------------------------------------------------------------------------
# Miner
# --------------------------------------------------------------------------
def run_miner(host, port, miner_id, seed=0):
    model, device = _new_model()
    data = ByteData(block_size=MODEL_CFG.block_size, device=device)
    miner = DiLoCoMiner(model, data, device)
    key = Key.generate(f"miner-{miner_id}".encode().ljust(32, b"0"))
    sock = None
    for attempt in range(60):                       # coordinator may still be starting
        try:
            sock = socket.create_connection((host, port), timeout=5)
            break
        except (ConnectionError, OSError):
            time.sleep(1)
    if sock is None:
        raise ConnectionError(f"could not reach coordinator {host}:{port}")
    send_msg(sock, {"type": "hello", "miner_id": miner_id, "pub": key.pub})
    print(f"miner {miner_id} connected to {host}:{port} — real GPT on {device}", flush=True)
    comp = Compressor(keep_frac=KEEP_FRAC)               # top-k + error feedback
    n = 0
    try:
        while True:
            msg = recv_msg(sock)
            if msg["type"] == "stop":
                break
            w = np.frombuffer(msg["weights"], dtype=np.float32).astype(np.float64)
            set_flat_params(model, w)                            # sync head
            delta, loss = miner.inner_train(INNER_STEPS, BATCH, seed=seed * 1000 + n)
            payload = comp.compress(dequantize(delta))           # compress for the wire
            dense = decompress(payload)                          # what the chain commits to
            sig = key.sign(BackpropTx(miner=key.pub, base_height=msg["height"],
                                      delta_hash=delta_hash(dense.tobytes()),
                                      da_pointer=f"da://{msg['height']}/{miner_id}"
                                      ).signing_bytes())
            send_msg(sock, {"type": "delta", "miner_id": miner_id,
                            "payload": payload, "sig": sig})
            n += 1
            print(f"  miner {miner_id} round {n}: inner loss {loss:.3f}", flush=True)
    finally:
        sock.close()
    print(f"miner {miner_id} done ({n} rounds)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="role", required=True)
    c = sub.add_parser("coordinator")
    c.add_argument("--port", type=int, default=9800)
    c.add_argument("--miners", type=int, default=2)
    c.add_argument("--rounds", type=int, default=20)
    m = sub.add_parser("miner")
    m.add_argument("--host", required=True); m.add_argument("--port", type=int, default=9800)
    m.add_argument("--id", type=int, required=True)
    a = ap.parse_args()
    if a.role == "coordinator":
        run_coordinator(a.port, a.miners, a.rounds)
    else:
        run_miner(a.host, a.port, a.id)


if __name__ == "__main__":
    main()
