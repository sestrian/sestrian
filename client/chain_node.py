"""Integrated real client — the mechanisms folded into real GPU training.

This is the production client with the distributed-systems layer wired in around
a real PyTorch model. Per round:

  * the threshold-BLS beacon (rig/beacon.py, DKG committee) produces round
    randomness → it ELECTS the leader and ASSIGNS each miner a corpus shard, so
    no miner picks its own data (§6.2, §7.2);
  * each miner trains the real GPT on its assigned shard on its own GPU, then
    ERASURE-CODES the pseudo-gradient delta (rig/da.py, now MB-fast) and sends
    the shards + a signed BackpropTx carrying the DA root;
  * the coordinator SAMPLES each delta's shards for availability (a withholder
    who sends < k shards is rejected), reconstructs the delta from any k, checks
    the signature and the hash, prices admission with the WRITE-PRICE homeostat,
    and STAKES/SLASHES via the stake ledger;
  * accepted deltas are aggregated deterministically and committed in a
    HASH-LINKED block (rig/blockchain.py) whose header commits the new weights
    state root; miners earn from the block.

The beacon here is produced by a dealt committee at the coordinator (the shares
live in one place in this coordinator form); the gossip form (client/gossip.py)
distributes it. Everything else is the real mechanism on the real model.
"""

import argparse
import socket
import time

import numpy as np

from rig import beacon as bcn
from rig import da
from rig.blockchain import BlockTree, build_block
from rig.chain import dequantize, quantize, state_root, trimmed_mean_int
from rig.crypto import BackpropTx, Key, delta_hash
from rig.dkg import run_dkg
from rig.economics import StakeLedger, WritePriceController
from rig.protocol import recv_msg, send_msg
from .data import ByteData
from .gpt import GPTConfig, build
from .trainer import DiLoCoMiner, flat_params, set_flat_params

MODEL_CFG = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128)
INNER_STEPS = 15
BATCH = 32
DA_K, DA_N = 3, 6          # erasure: any 3 of 6 shards reconstruct a delta
DA_SAMPLES = 3
STAKE_BOND = 100.0
DATA_SHARDS = 8


def _new_model():
    return build(MODEL_CFG)


def run_coordinator(port, n_miners, rounds, host="0.0.0.0", seed=0):
    model, device = _new_model()
    data = ByteData(block_size=MODEL_CFG.block_size, device=device)
    keys = run_dkg(max(3, n_miners + 1), min(3, max(2, n_miners)))  # beacon committee
    tree = BlockTree(quantize(flat_params(model)))
    stake = StakeLedger()
    writeprice = WritePriceController(target_rate=max(1, n_miners))
    print(f"coordinator on {host}:{port} — real GPT {model.num_params()/1e6:.1f}M on "
          f"{device}; beacon {keys.t}-of-{keys.n}; waiting for {n_miners} miners…", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(n_miners)
    conns = {}
    while len(conns) < n_miners:
        c, addr = srv.accept()
        h = recv_msg(c)
        conns[h["miner_id"]] = (c, h["pub"])
        stake.stake(h["pub"], STAKE_BOND)
        print(f"  miner {h['miner_id']} joined from {addr[0]} — staked {STAKE_BOND} "
              f"({len(conns)}/{n_miners})", flush=True)

    def val(base_int):
        set_flat_params(model, dequantize(base_int)); return data.estimate_loss(model, iters=8)["val"]

    print(f"genesis val loss {val(tree.head_state()):.3f}", flush=True)
    t0 = time.time()
    for r in range(rounds):
        base = tree.head_state()
        # 1. beacon → leader + per-miner data-shard assignment
        gsig, _ = bcn.produce(keys, r + 1, list(range(1, keys.t + 1)))
        beacon_hex = bcn.randomness(gsig).hex()
        leader = int(bcn.beacon_rng(gsig, "leader").integers(n_miners))
        assign_rng = bcn.beacon_rng(gsig, "assign")
        assign = {mid: int(assign_rng.integers(DATA_SHARDS)) for mid in conns}

        w_bytes = dequantize(base).astype(np.float32).tobytes()
        for mid, (c, _) in conns.items():
            send_msg(c, {"type": "train", "weights": w_bytes, "height": r,
                         "shard": [assign[mid], DATA_SHARDS]})

        # 2. collect: verify sig, check DA availability via Merkle proofs,
        #    reconstruct from any k shards, price/stake
        accepted, bodies, works, admitted = [], {}, {}, 0
        for mid, (c, pub) in conns.items():
            m = recv_msg(c)
            tx = BackpropTx(miner=pub, base_height=r,
                            delta_hash=m["delta_hash"], da_pointer=m["da_pointer"])
            tx.sig = m["sig"]
            if not tx.verify():
                stake.slash(pub, "invalid signature", "coordinator")
                print(f"  ! miner {mid}: bad signature — slashed", flush=True)
                continue
            root = bytes.fromhex(m["root_hex"])
            if da.da_pointer(root) != tx.da_pointer:            # root must match the signed pointer
                continue
            valid = {i: sb for i, (sb, pf) in m["served"].items()
                     if da.verify_shard(sb, i, pf, root)}       # each shard proven vs root
            if len(valid) < DA_K:                               # withholding → unrecoverable
                stake.slash(pub, "DA withholding", "coordinator")
                print(f"  ! miner {mid}: DA unavailable ({len(valid)}/{DA_K}) — slashed", flush=True)
                continue
            delta = np.frombuffer(da.reconstruct(valid, DA_K, m["orig_len"]),
                                  dtype=np.int64).copy()
            if delta_hash(delta.tobytes()) != tx.delta_hash:
                continue
            if stake.staked.get(pub, 0.0) < writeprice.price:   # write-price gate
                continue
            accepted.append(tx)
            bodies[tx.da_pointer] = delta
            works[tx.txid()] = 1.0
            admitted += 1
        writeprice.observe(admitted); writeprice.maybe_retarget()

        # 3. aggregate + commit a hash-linked block; reward included miners
        block = build_block(tree, tree.head, accepted, bodies, works,
                            proposer=conns[leader][1] if leader in conns else "coordinator")
        tree.add_block(block)
        for tx in accepted:
            stake.reward(tx.miner, 1.0)

        if r % 3 == 0 or r == rounds - 1:
            print(f"  round {r:>2}  val {val(tree.head_state()):.3f}  admitted {admitted}/"
                  f"{n_miners}  leader m{leader}  beacon {beacon_hex[:8]}  "
                  f"writeprice {writeprice.price:.1f}  root {tree.head[:10]}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    for c, _ in conns.values():
        send_msg(c, {"type": "stop"}); c.close()
    srv.close()
    replay_ok = state_root(tree.replay_head()) == tree.blocks[tree.head].header.state_root
    print(f"\ndone: real GPT trained through the full stack (beacon+DA+stake+blocks). "
          f"final val {val(tree.head_state()):.3f}, replay bit-exact {replay_ok}", flush=True)
    return tree


def run_miner(host, port, miner_id, seed=0, withhold=False):
    model, device = _new_model()
    data = ByteData(block_size=MODEL_CFG.block_size, device=device)
    miner = DiLoCoMiner(model, data, device)
    key = Key.generate(f"miner-{miner_id}".encode().ljust(32, b"0"))
    sock = None
    for _ in range(60):
        try:
            sock = socket.create_connection((host, port), timeout=5); break
        except (ConnectionError, OSError):
            time.sleep(1)
    send_msg(sock, {"type": "hello", "miner_id": miner_id, "pub": key.pub})
    print(f"miner {miner_id} connected — real GPT on {device}", flush=True)
    n = 0
    try:
        while True:
            msg = recv_msg(sock)
            if msg["type"] == "stop":
                break
            set_flat_params(model, np.frombuffer(msg["weights"], dtype=np.float32).astype(np.float64))
            delta, loss = miner.inner_train(INNER_STEPS, BATCH,
                                            seed=seed * 1000 + n, shard=tuple(msg["shard"]))
            body = delta.tobytes()
            blob = da.disperse(body, DA_K, DA_N)          # erasure-code + Merkle
            # honest miner serves all shards with proofs; a withholder serves none
            served = {} if withhold else {i: (blob.shards[i], blob.proof(i))
                                          for i in range(DA_N)}
            tx = BackpropTx(miner=key.pub, base_height=msg["height"],
                            delta_hash=delta_hash(body), da_pointer=da.da_pointer(blob.root))
            send_msg(sock, {"type": "delta", "miner_id": miner_id, "sig": key.sign(tx.signing_bytes()),
                            "delta_hash": tx.delta_hash, "da_pointer": tx.da_pointer,
                            "root_hex": blob.root.hex(), "orig_len": blob.orig_len,
                            "served": served})
            n += 1
            print(f"  miner {miner_id} round {n}: shard {msg['shard'][0]} inner loss {loss:.3f}", flush=True)
    finally:
        sock.close()
    print(f"miner {miner_id} done ({n} rounds)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="role", required=True)
    c = sub.add_parser("coordinator")
    c.add_argument("--port", type=int, default=9810); c.add_argument("--miners", type=int, default=2)
    c.add_argument("--rounds", type=int, default=15)
    m = sub.add_parser("miner")
    m.add_argument("--host", required=True); m.add_argument("--port", type=int, default=9810)
    m.add_argument("--id", type=int, required=True)
    m.add_argument("--withhold", action="store_true")
    a = ap.parse_args()
    if a.role == "coordinator":
        run_coordinator(a.port, a.miners, a.rounds)
    else:
        run_miner(a.host, a.port, a.id, withhold=a.withhold)


if __name__ == "__main__":
    main()
