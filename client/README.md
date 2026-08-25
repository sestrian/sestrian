# Sestrian client — train the model, earn from it

This is the real thing (not the numpy mechanism-proof in `../rig/`): a client
that trains an actual PyTorch language model on your own GPU and contributes to
a shared chain. It runs on **CUDA (Nvidia)**, **Apple MPS (M-series Mac)**, or
**CPU** — auto-detected.

Verified: a real GPT trained on real text across an M3 (MPS) and an RTX 2080 Ti
(CUDA) at the same time, over the network, with signed deltas deterministically
aggregated (val loss 5.6 → 2.5).

## What it does

Each round your client:
1. syncs the current model weights from the coordinator,
2. trains the model locally for a few steps on your GPU (real backprop on real
   text — this is the "inner loop", and it's yours, unconstrained),
3. computes the weight change (the *pseudo-gradient delta*), signs it with your
   key, and submits it,
4. the coordinator verifies the signature and folds your delta into the chain by
   deterministic fixed-point aggregation — so your CUDA delta and someone's MPS
   delta land in one bit-exact chain state.

## Files

| file | role |
|---|---|
| `gpt.py` | the real model — a small nanoGPT (byte-level, no tokenizer to ship) |
| `data.py` | real text (TinyShakespeare, auto-downloaded on first run) |
| `trainer.py` | the DiLoCo bridge — GPU training → quantised delta → chain |
| `node.py` | the runnable coordinator + miner over TCP |

## Run it (two machines, or two terminals)

Coordinator (any machine — its IP is what miners dial):
```bash
python -m client.node coordinator --port 9800 --miners 2 --rounds 20
```

Each miner (point `--host` at the coordinator's IP; use `0`, `1`, … for `--id`):
```bash
python -m client.node miner --host <coordinator-ip> --port 9800 --id 0
```

The model architecture (`MODEL_CFG` in `node.py`) must match on both sides —
they exchange weight vectors. Bump `n_layer` / `n_embd` to train a bigger model
(bounded by your GPU's memory).

See [`../docs/joining.md`](../docs/joining.md) for a clean
setup (the first-user target).
