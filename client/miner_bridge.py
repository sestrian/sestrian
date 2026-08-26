"""The PyTorch side of the trainer bridge — pure compute, zero consensus.

Connects to a local sestrian-node (Rust), receives the head state once, then
per training round: trains the real model for N inner steps on its own GPU and
returns the COMPRESSED quantized pseudo-gradient. The node signs, gossips, and
settles it; when the head advances the node sends a sparse state diff so this
process stays synced without ever re-downloading the model.

  python -m client.miner_bridge --node-port 7999 --model small \
      --data data/stories_train.txt --inner 300 --batch 32 [--device cuda]

Frames: [u32 BE length][bytes]; JSON control messages; the initial state
arrives as a raw i64-LE frame after a {"bin_next": true} header.
"""

import argparse
import base64
import json
import socket
import secrets as _secrets
import select
import struct
from collections import deque
import time

import numpy as np

from rig.chain import dequantize, quantize
from .compress import Compressor, compress as topk_compress
from .data import ByteData
from .gossip import MODEL_PRESETS, build_preset
from .moe import ChainLayout, MoEGPT, MoEGPTConfig
from .trainer import DiLoCoMiner, set_flat_params

KEEP_FRAC = 0.02


def _send(sock, obj: dict):
    raw = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _send_bin(sock, raw: bytes):
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _recv(sock) -> bytes:
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            raise ConnectionError("node closed")
        hdr += chunk
    n = struct.unpack(">I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("node closed")
        buf += chunk
    return bytes(buf)


def _payload_json(payload: dict) -> dict:
    return {"n": payload["n"],
            "idx": base64.b64encode(payload["idx"]).decode(),
            "val": base64.b64encode(payload["val"]).decode()}


def _sparse_dense(sp: dict) -> np.ndarray:
    out = np.zeros(sp["n"], dtype=np.int64)
    idx = np.frombuffer(base64.b64decode(sp["idx"]), dtype="<u4")
    val = np.frombuffer(base64.b64decode(sp["val"]), dtype="<i8")
    out[idx.astype(np.int64)] = val
    return out


def _sparse_dense_local(payload: dict) -> np.ndarray:
    """Densify a LOCAL (un-b64'd) compressor payload — for the claim set."""
    from .compress import decompress
    return decompress(payload)



def _serve_generate(sock, msg, model, device, height, state, gen):
    """Answer one chat request. Shared by the main loop and the mid-round poll
    so both paths cannot drift apart.

    `gen` is a dedicated torch.Generator: sampling must never draw from the
    global RNG, because this can now run BETWEEN training steps and a training
    round has to stay reproducible for a validator.
    """
    import torch
    if state is None:
        _send(sock, {"t": "generated", "height": -1,
                     "text": "(model not yet synced)"})
        return
    raw = str(msg.get("prompt", " ")).encode("utf-8")
    raw = raw[-(model.cfg.block_size - 1):] or b" "
    n_new = min(int(msg.get("n", 120)), 240)
    idx = torch.tensor([list(raw)], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model.generate(idx, n_new, temperature=0.85, generator=gen)
    # Restore whatever mode we interrupted. Hard-coding train() here would be a
    # bug on a --serve-only bridge, which is never in train mode.
    model.train(was_training)
    text = bytes(out[0].tolist()[len(raw):]).decode("utf-8", errors="replace")
    _send(sock, {"t": "generated", "height": height, "text": text})
    print(f"h{height}: generated {n_new} bytes", flush=True)


def run(a):
    # The architecture comes from the CHAIN: the node names its model preset
    # in the first state message, and the trainer builds from that. --model
    # is an override for local/custom networks; a bare `miner_bridge` against
    # any network just works. Construction is deferred until that message.
    cfg = model = layout = miner = data = None
    device = a.device
    is_moe = False
    comp = Compressor(keep_frac=KEEP_FRAC)

    def to_torch(chain_vec):
        return layout.torch_of(chain_vec) if layout is not None else chain_vec

    def to_chain(torch_vec):
        return layout.chain_of(torch_vec) if layout is not None else torch_vec

    def build_from(name, epl=None):
        nonlocal cfg, model, layout, miner, data, device, is_moe
        cfg = MODEL_PRESETS[name]
        is_moe = isinstance(cfg, MoEGPTConfig)
        model, device = build_preset(name, device=a.device,
                                     experts_per_layer=epl)
        layout = ChainLayout(model) if is_moe else None
        data = ByteData(path=a.data, block_size=cfg.block_size, device=device) \
            if a.data else ByteData(block_size=cfg.block_size, device=device)
        miner = DiLoCoMiner(model, data, device)
        print(f"miner bridge: {name}, {model.num_params()/1e6:.1f}M params on "
              f"{device}" + (f", {layout.n_pages()} pages" if layout else ""),
              flush=True)

    while True:                                     # reconnect loop
        try:
            sock = socket.create_connection(("127.0.0.1", a.node_port), timeout=10)
        except OSError:
            time.sleep(2)
            continue
        sock.settimeout(None)
        try:
            _send(sock, {"t": "hello"})
            state = None                            # int64 chain state (our copy)
            height = -1
            step_secs = 0.0                         # measured per-inner-step cost
            # Sampling RNG for chat, isolated from the global stream so serving
            # a request mid-round cannot perturb a training round's batches.
            chat_gen = torch.Generator()
            chat_gen.manual_seed(_secrets.randbits(63))
            deferred = deque()                      # messages read mid-round
            while True:
                msg = json.loads(_recv(sock)) if not deferred else deferred.popleft()
                t = msg.get("t")
                if model is None and t != "state":
                    continue        # nothing exists until the first state
                if t == "state":
                    raw = _recv(sock)               # the raw i64 frame (CHAIN order)
                    state = np.frombuffer(raw, dtype="<i8").copy()
                    height = int(msg["height"])
                    epl = msg.get("experts_per_layer")
                    if model is None:
                        # first contact: the node names the network's model
                        name = a.model or msg.get("model")
                        if not name:
                            raise SystemExit(
                                "node did not name its model (old node?) — "
                                "pass --model explicitly")
                        if a.model and msg.get("model") \
                                and a.model != msg.get("model"):
                            print(f"WARNING: --model {a.model} overrides the "
                                  f"node's {msg.get('model')}", flush=True)
                        build_from(name, epl=list(epl) if epl else None)
                    # v1: the node tells us the model shape; rebuild if the
                    # chain grew while we were away (ragged expert counts)
                    rebuild = (layout is not None and epl is not None
                               and list(epl) != model.experts_per_layer)
                    if rebuild:
                        model, _ = build_preset(a.model or msg.get("model"),
                                                device=device,
                                                experts_per_layer=list(epl))
                        miner.model = model
                        layout = ChainLayout(model)
                        comp.residual = None
                        print(f"rebuilt model for experts_per_layer={epl}",
                              flush=True)
                    if layout is not None and state.size != layout.n:
                        _send(sock, {"t": "resync"})
                        state = None
                        continue
                    set_flat_params(model, dequantize(to_torch(state)))
                    print(f"synced full state @ h{height} "
                          f"({state.size/1e6:.1f}M params)", flush=True)
                elif t == "advance":
                    if state is None or int(msg.get("dim", state.size)) != state.size:
                        _send(sock, {"t": "resync"})
                        continue
                    state = state + _sparse_dense(msg["sparse"])
                    height = int(msg["height"])
                    set_flat_params(model, dequantize(to_torch(state)))
                elif t == "grow":
                    # v1 GROWTH EVENT: the chain appended an expert page. The
                    # node sends the page's deterministic init as a raw i64
                    # frame; we append it to our state, instantiate the expert,
                    # and rebuild the layout. (Dense presets never see this.)
                    raw = _recv(sock)
                    page = np.frombuffer(raw, dtype="<i8").copy()
                    if state is None or layout is None:
                        _send(sock, {"t": "resync"})
                        continue
                    info = msg["page"]
                    model.add_expert(int(info["layer"]), dequantize(page))
                    layout = ChainLayout(model)
                    state = np.concatenate([state, page])
                    height = int(msg["height"])
                    comp.residual = None            # dimensions changed
                    if state.size != int(msg["new_dim"]) or state.size != layout.n:
                        _send(sock, {"t": "resync"})
                        state = None
                        continue
                    set_flat_params(model, dequantize(to_torch(state)))
                    print(f"GROWTH @ h{height}: +expert layer {info['layer']} "
                          f"-> {state.size/1e6:.1f}M params", flush=True)
                elif t == "train":
                    want_h = int(msg["height"])
                    if a.serve_only:
                        continue                # this bridge only generates
                    if state is None or want_h != height:
                        _send(sock, {"t": "resync"})
                        continue
                    # AUTO-FIT (the slow-GPU fix): a delta is includable only at
                    # base_height == head, so a round that overruns the block
                    # interval is dropped as stale and earns NOTHING. Fit the
                    # inner steps to the node's budget using the measured
                    # per-step cost, so any GPU contributes something every
                    # round instead of everything-or-nothing.
                    steps = a.inner
                    budget = float(msg.get("budget_s", 0) or 0)
                    if budget > 0 and step_secs > 0:
                        steps = int(max(1, min(a.inner, budget / step_secs)))
                        if steps < a.inner:
                            print(f"auto-fit: {steps}/{a.inner} inner steps to "
                                  f"fit {budget:.0f}s budget "
                                  f"({step_secs*1000:.0f}ms/step)", flush=True)
                    t_start = time.time()

                    # Answer chat BETWEEN inner steps instead of after the whole
                    # round. A round is ~24 steps of ~2s, so a request that
                    # arrives mid-round waited up to ~45s; now it waits one step.
                    #
                    # Reading the socket here is only safe for `generate`, which
                    # is a single self-contained frame. Anything else — `state`
                    # in particular — is followed by a second binary frame, so
                    # consuming it here would leave that payload in the stream
                    # and the next JSON read would land on raw bytes. So: defer
                    # the message untouched and STOP polling for the rest of the
                    # round, letting the main loop take it in order.
                    poll = {"on": True}

                    def _between():
                        if not poll["on"]:
                            return
                        r, _, _ = select.select([sock], [], [], 0)
                        if not r:
                            return
                        m = json.loads(_recv(sock))
                        if m.get("t") == "generate":
                            _serve_generate(sock, m, model, device, height,
                                            state, chat_gen)
                        else:
                            deferred.append(m)
                            poll["on"] = False

                    delta_int, loss = miner.inner_train(
                        steps, a.batch, seed=int(msg.get("seed", 0)),
                        between_steps=_between)
                    elapsed = time.time() - t_start
                    # EMA of per-step cost (first measurement seeds it outright)
                    obs = elapsed / max(1, steps)
                    step_secs = obs if step_secs <= 0 else 0.7 * step_secs + 0.3 * obs
                    # v1: the delta goes on the wire in CHAIN order, zeroed
                    # outside the pages we may claim (frozen pages reject txs),
                    # with the claim set attached and the compressor keeping at
                    # least the node's work-quota floor (min_nnz).
                    chain_delta = to_chain(delta_int)
                    min_nnz = int(msg.get("min_nnz", 0))
                    max_nnz = int(msg.get("max_nnz", 0))
                    quota_4dp = int(msg.get("quota_4dp", 0))
                    if layout is not None:
                        active = msg.get("active_pages")
                        active = list(range(layout.n_pages())) if active is None \
                            else [int(p) for p in active]
                        # PROTOCOL v2 — CLAIM PLANNING under the delta envelope:
                        # the payload never grows with quota, so a rising quota
                        # shrinks the claimable span. Budget the claim in params
                        # (max_nnz * 1e6 / quota), rank active pages by the
                        # gradient mass this round actually produced, and claim
                        # greedily — the miner SPECIALIZES on the experts its
                        # data teaches best instead of spraying the whole model.
                        if max_nnz and quota_4dp:
                            budget = max_nnz * 1_000_000 // quota_4dp
                            spans = {p: layout.page_span(p) for p in active}
                            mass = {p: int(np.abs(
                                        chain_delta[spans[p][0]:spans[p][1]]
                                    ).sum()) for p in active}
                            ranked = sorted(active,
                                            key=lambda p: (-mass[p], p))
                            claim, used = [], 0
                            for pg in ranked:
                                plen = spans[pg][1] - spans[pg][0]
                                if used + plen <= budget and mass[pg] > 0:
                                    claim.append(pg)
                                    used += plen
                            if not claim:            # degenerate round: claim
                                claim = [ranked[0]]  # the strongest page anyway
                            claim.sort()
                            min_nnz = used * quota_4dp // 1_000_000
                        else:
                            claim = active
                        chain_delta = layout.zero_outside(chain_delta, claim)
                        if comp.residual is not None and \
                                comp.residual.shape[0] == layout.n:
                            # error feedback must not resurrect coordinates the
                            # tx couldn't claim (frozen or outside this claim)
                            comp.residual = layout.zero_outside(
                                comp.residual, claim)
                        payload = comp.compress(dequantize(chain_delta),
                                                min_keep=min_nnz,
                                                max_keep=max_nnz)
                        pages = sorted(claim)
                    else:
                        payload = comp.compress(dequantize(chain_delta),
                                                min_keep=min_nnz,
                                                max_keep=max_nnz)
                        pages = [0]
                    # inner_train mutated the model; restore chain state so the
                    # next round trains from the agreed head, not our drift
                    set_flat_params(model, dequantize(to_torch(state)))
                    _send(sock, {"t": "delta", "height": want_h, "loss": loss,
                                 "pages": pages,
                                 "payload": _payload_json(payload)})
                    print(f"h{want_h}: trained {steps}x{a.batch} in "
                          f"{elapsed:.0f}s, loss {loss:.3f}", flush=True)
                elif t == "eval":
                    # rev 7 DELTA SCORING: measure each candidate delta's loss
                    # improvement on a HELD-OUT batch (val split, seeded from the
                    # block context so the shard is not miner-chosen). The node
                    # commits the returned micro-nat scores in its block; reward
                    # weighting follows them. Scores are per-proposer claims —
                    # bonded and challengeable — so float nondeterminism between
                    # GPUs never touches consensus.
                    import torch
                    want_h = int(msg.get("height", -1))
                    if state is None or want_h != height:
                        _send(sock, {"t": "scores", "height": want_h,
                                     "scores": {}})
                        continue
                    # NOISE FLOOR: a single val batch's measurement error is
                    # the same magnitude as a delta's true per-round improvement
                    # (measured live: one delta, two GPUs, 464 vs 3020 u-nats),
                    # so single-batch scores were coin flips clamped at zero.
                    # Evaluate over several seeded batches; noise ~ 1/sqrt(n).
                    EVAL_BATCHES = 4
                    seed0 = int(msg.get("seed", 0))
                    batches = []
                    for bi in range(EVAL_BATCHES):
                        gen = torch.Generator().manual_seed(seed0 + bi)
                        batches.append(data.get_batch("val", a.batch,
                                                      generator=gen))
                    xb, yb = batches[0]      # routing capture uses batch 0
                    scores, sketches = {}, {}
                    model.eval()
                    with torch.no_grad():
                        import torch.nn.functional as F

                        def per_token_loss_all():
                            outs = []
                            for bx, by in batches:
                                logits, _ = model(bx, by)
                                outs.append(F.cross_entropy(
                                    logits.view(-1, logits.size(-1)),
                                    by.reshape(-1), reduction="none",
                                ).view(by.shape))
                            return outs

                        def per_token_loss():
                            logits, _ = model(xb, yb)
                            return F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                yb.reshape(-1), reduction="none",
                            ).view(yb.shape)

                        # CLAIM-AWARE SCORING: capture which experts each
                        # held-out token routes through (baseline model), so a
                        # delta is measured on the tokens that actually use the
                        # experts it claims. A specialized delta is no longer
                        # invisible to an evaluator whose data routes elsewhere
                        # — the failure that zeroed half the committed scores
                        # and blocked organic growth (found live).
                        routed = {}
                        hooks = []
                        if layout is not None:
                            for li, blk in enumerate(model.blocks):
                                def mk(li):
                                    def h(mod, inp, out):
                                        w = mod._gates(inp[0])
                                        routed[li] = (w > 0)
                                    return h
                                hooks.append(
                                    blk.moe.register_forward_hook(mk(li)))
                        base_all = per_token_loss_all()
                        for h in hooks:
                            h.remove()
                        base_tok = base_all[0]
                        base = float(sum(t.mean() for t in base_all)
                                     / len(base_all))

                        def token_mask(pages):
                            if layout is None:
                                return None
                            claimed = [layout.experts[p - 1]
                                       for p in pages if p >= 1]
                            if not claimed:
                                return None          # backbone-only: all tokens
                            m = torch.zeros_like(yb, dtype=torch.bool)
                            for (li, e) in claimed:
                                r = routed.get(li)
                                if r is not None and e < r.shape[-1]:
                                    m |= r[..., e]
                            # too few routed tokens = noise, not signal
                            if int(m.sum()) < 16:
                                return None
                            return m

                        # LEAVE-ONE-OUT (v3 design, stage 1): apply ALL
                        # candidates, then score each as aggregate-with minus
                        # aggregate-without, on its claim-routed tokens. The
                        # aggregate carries k-times one delta's signal (noise-
                        # robust) and duplicated gradients earn ~nothing —
                        # redundancy is priced at zero, which is the incentive
                        # that makes miners seek DIFFERENT data.
                        deltas = msg.get("deltas", [])
                        dense_by_tx = {d["txid"]: _sparse_dense(d["sparse"])
                                       for d in deltas}
                        agg_all = state.copy()
                        for dd in dense_by_tx.values():
                            agg_all = agg_all + dd
                        set_flat_params(model, dequantize(to_torch(agg_all)))
                        all_tok = per_token_loss_all()
                        for d in deltas:
                            without = agg_all - dense_by_tx[d["txid"]]
                            set_flat_params(model,
                                            dequantize(to_torch(without)))
                            wo_all = per_token_loss_all()
                            m = token_mask([int(p) for p in d.get("pages", [])])
                            if m is None:
                                wm = float(sum(t.mean() for t in wo_all)
                                           / len(wo_all))
                                am = float(sum(t.mean() for t in all_tok)
                                           / len(all_tok))
                                imp = wm - am
                            else:
                                b0 = float(wo_all[0][m].mean()) \
                                    - float(all_tok[0][m].mean())
                                rest_w = sum(float(t.mean())
                                             for t in wo_all[1:])
                                rest_a = sum(float(t.mean())
                                             for t in all_tok[1:])
                                n = len(all_tok)
                                imp = (b0 + (rest_w - rest_a)) / n
                            scores[d["txid"]] = max(0, int(round(imp * 1e6)))
                            # rev 8: the delta's integer influence sketch —
                            # exactly recomputable from the DA body by anyone
                            from rig.sketch import sketch_sparse
                            sp = d["sparse"]
                            idx = np.frombuffer(base64.b64decode(sp["idx"]),
                                                dtype="<u4")
                            val = np.frombuffer(base64.b64decode(sp["val"]),
                                                dtype="<i8")
                            sketches[d["txid"]] = sketch_sparse(
                                idx.tolist(), val.tolist())
                        set_flat_params(model, dequantize(to_torch(state)))
                    model.train()
                    _send(sock, {"t": "scores", "height": height,
                                 "scores": scores, "sketches": sketches})
                    print(f"h{height}: scored+sketched {len(scores)} deltas "
                          f"(base loss {base:.3f})", flush=True)
                elif t == "generate":
                    # serve chat from the chain-synced model (works on any
                    # bridge; a --produce-less node makes this a pure server)
                    _serve_generate(sock, msg, model, device, height, state, chat_gen)
        except (ConnectionError, OSError) as e:
            print(f"bridge disconnected ({e}); retrying…", flush=True)
            time.sleep(2)
        finally:
            sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-port", type=int, default=7999)
    ap.add_argument("--model", default=None, choices=list(MODEL_PRESETS),
                help="override the model the node names (local nets)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--serve-only", action="store_true",
                    help="only answer generate requests; never submit deltas")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
