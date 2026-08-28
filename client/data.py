"""Real text data for the language model — byte-level, no tokenizer to ship.

Byte-level LM (vocab 256) so a volunteer needs no tokenizer files: the corpus is
raw UTF-8 bytes. The default corpus is TinyShakespeare (public domain), but any
text file works; a client points at whatever data its channel provides.
"""

import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(HERE, "data", "shakespeare.txt")
_CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
               "data/tinyshakespeare/input.txt")


def _ensure_corpus(path: str):
    """Fetch the default public-domain corpus on first run so the client just works."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(_CORPUS_URL, path)


class ByteData:
    def __init__(self, path: str = DEFAULT_CORPUS, block_size: int = 128,
                 device: str = "cpu", val_frac: float = 0.1):
        if path == DEFAULT_CORPUS:
            _ensure_corpus(path)
        # MEMORY-MAP the corpus — never load it. The old path was
        # f.read() + torch.from_numpy(...copy()): TWO full copies in RAM, so an
        # 18GB corpus cost ~36GB resident per miner (measured 31GB on the CUDA
        # trainer) and set a ~2x-corpus RAM floor on anyone joining. A memmap
        # costs address space only; the OS pages in the handful of 128-byte
        # windows a batch touches and evicts them under pressure. get_batch
        # copies each sampled window into the batch tensor, so nothing torch
        # trains on aliases the mapping.
        data = np.memmap(path, dtype=np.uint8, mode="c")  # copy-on-write: torch-safe, still lazily paged
        n = len(data)
        cut = int(n * (1 - val_frac))
        self.train = torch.from_numpy(data[:cut])
        self.val = torch.from_numpy(data[cut:])
        self.block_size = block_size
        self.device = device
        self.n_bytes = n

    def get_batch(self, split: str, batch_size: int, generator=None, shard=None):
        """Sample a batch. `shard=(id, total)` restricts sampling to the beacon-
        assigned slice of the corpus, so a miner trains only on its assigned data
        (§6.2 — no self-selected data)."""
        src = self.train if split == "train" else self.val
        lo, hi = 0, len(src) - self.block_size - 1
        if shard is not None:
            sid, total = shard
            span = (hi) // total
            lo = sid * span
            hi = lo + span
        ix = lo + torch.randint(hi - lo, (batch_size,), generator=generator)
        x = torch.stack([src[i:i + self.block_size] for i in ix]).long()
        y = torch.stack([src[i + 1:i + 1 + self.block_size] for i in ix]).long()
        if self.device == "cuda":                       # pinned async copy — CUDA only
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model, batch_size=32, iters=20):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(iters):
                x, y = self.get_batch(split, batch_size)
                _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out


def decode(bytes_tensor) -> str:
    return bytes(int(b) for b in bytes_tensor.tolist()).decode("utf-8", errors="replace")
