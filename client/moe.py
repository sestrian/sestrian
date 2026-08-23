"""Real (PyTorch) mixture-of-experts GPT — the protocol-v1 network model.

The chain's state layout (rig/model_state.py) is the source of truth:

    [ backbone | expert(layer 0, e 0) | expert(0,1) | … | grown experts ]

Each expert page is that expert's parameters in module order — fc.weight,
fc.bias, proj.weight, proj.bias — which is exactly the consensus page-init
layout (W1 row-major, b1, W2, b2). The backbone page is every non-expert
parameter in named_parameters() order. `ChainLayout` computes the permutation
between torch's flat order (client/trainer.flat_params) and the chain order;
consensus never sees torch order.

v1 architecture invariants:
  * positions are ROTARY (RoPE, reusing client/gpt.CausalSelfAttention) — no
    learned position table, per the whitepaper invariant; context is a runtime
    choice, not a weight shape.
  * the router is preallocated to E_MAX columns per layer (a consensus genesis
    parameter), so growth events add expert pages WITHOUT touching backbone
    shapes. Routing masks non-instantiated columns to -inf.
  * different layers may hold different expert counts (growth is per-layer,
    round-robin), so the model is built from an explicit per-layer count.
  * `add_expert(layer, init)` appends one expert whose weights come from the
    chain's deterministic page-init — the client-side half of a growth event.

Sharding note (v1): the old `mask_delta`/`shard_aggregate` are GONE. Under
protocol v1 a delta TX carries an explicit page-claim set and consensus
aggregates per page over actual claimants (rig/chain.paged_transition), so
"zero a coordinate you don't hold" is no longer a transport hack that dilutes
aggregation — it is the validated tx format itself.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import CausalSelfAttention, GPTConfig, apply_genesis, pick_device


@dataclass
class MoEGPTConfig:
    vocab_size: int = 256          # byte-level, no tokenizer
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    n_experts: int = 4             # experts per layer at genesis
    e_max: int = 8                 # router columns preallocated (growth headroom)
    top_k: int = 2
    dropout: float = 0.0

    @property
    def d_ff(self):
        return 4 * self.n_embd

    def attn_cfg(self) -> GPTConfig:
        return GPTConfig(vocab_size=self.vocab_size, block_size=self.block_size,
                         n_layer=self.n_layer, n_head=self.n_head,
                         n_embd=self.n_embd, dropout=self.dropout)


class Expert(nn.Module):
    """One FFN expert: the unit of sharding AND the chain's growth unit. Its
    parameter order (fc.weight, fc.bias, proj.weight, proj.bias) IS the
    consensus expert-page layout — do not reorder."""

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

    @torch.no_grad()
    def load_page(self, page_f64: np.ndarray):
        """Load this expert from a chain page (dequantized float64, chain page
        layout: W1 row-major, b1, W2, b2)."""
        d, f = self.fc.in_features, self.fc.out_features
        w1, rest = page_f64[:d * f], page_f64[d * f:]
        b1, rest = rest[:f], rest[f:]
        w2, b2 = rest[:f * d], rest[f * d:]
        self.fc.weight.copy_(torch.from_numpy(w1.reshape(f, d)).to(self.fc.weight))
        self.fc.bias.copy_(torch.from_numpy(b1).to(self.fc.bias))
        self.proj.weight.copy_(torch.from_numpy(w2.reshape(d, f)).to(self.proj.weight))
        self.proj.bias.copy_(torch.from_numpy(b2).to(self.proj.bias))


class MoEFeedForward(nn.Module):
    """Top-k routed mixture over the layer's INSTANTIATED experts. The router is
    E_MAX wide (consensus backbone shape); columns without an expert — and any
    the caller masks (frozen-for-training is a delta-level concern, but serving
    can also exclude) — are -inf before the topk, so they get exactly 0 gate."""

    def __init__(self, cfg: MoEGPTConfig, n_experts: int):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.n_embd, cfg.e_max, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(n_experts)])

    def _gates(self, x):
        logits = self.router(x)                                   # (..., E_MAX)
        n = len(self.experts)
        if n < self.cfg.e_max:
            fill = torch.full_like(logits[..., n:], float("-inf"))
            logits = torch.cat([logits[..., :n], fill], dim=-1)
        k = min(self.cfg.top_k, n)
        topv, topi = logits.topk(k, dim=-1)
        w = torch.zeros_like(logits).scatter_(-1, topi, F.softmax(topv, dim=-1))
        return w                                                  # k non-zeros/token

    def forward(self, x):
        w = self._gates(x)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                # dense in training…
            we = w[..., e:e + 1]
            if torch.any(we > 0):
                out = out + we * expert(x)
        return out

    @torch.no_grad()
    def serve(self, x, held=None):
        """Sparse serving: evaluate ONLY the experts a token routes to (and that
        this node holds, if `held` is given). Returns (output, experts_touched)."""
        w = self._gates(x)
        out = torch.zeros_like(x)
        touched = set()
        for e, expert in enumerate(self.experts):
            we = w[..., e:e + 1]
            if torch.any(we > 0) and (held is None or e in held):
                out = out + we * expert(x)
                touched.add(e)
        return out, touched


class MoEBlock(nn.Module):
    def __init__(self, cfg: MoEGPTConfig, n_experts: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.attn_cfg())           # RoPE, fused
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.moe = MoEFeedForward(cfg, n_experts)                 # experts live here

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class MoEGPT(nn.Module):
    """A byte-level RoPE GPT whose FFNs are mixtures of experts. Ordinary
    nn.Module, so it plugs into client/trainer.py (flat_params/DiLoCo)
    unchanged; ChainLayout maps it onto the chain's page layout."""

    def __init__(self, cfg: MoEGPTConfig, experts_per_layer: list[int] | None = None):
        super().__init__()
        self.cfg = cfg
        epl = list(experts_per_layer if experts_per_layer is not None
                   else [cfg.n_experts] * cfg.n_layer)
        assert len(epl) == cfg.n_layer and all(0 < e <= cfg.e_max for e in epl)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # no position embedding — positions are rotary (RoPE), inside attention
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MoEBlock(cfg, e) for e in epl])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    @property
    def experts_per_layer(self) -> list[int]:
        return [len(b.moe.experts) for b in self.blocks]

    def forward(self, idx, targets=None):
        x = self.drop(self.tok(idx))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0):
        for _ in range(n_new):
            idx_c = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_c)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def add_expert(self, layer: int, page_f64: np.ndarray | None = None) -> int:
        """The client half of a growth event: append one expert to `layer`,
        loading its weights from the chain's deterministic page init. Returns
        the new expert's index within the layer. The caller must rebuild its
        ChainLayout afterwards — the flat dimensions changed."""
        moe = self.blocks[layer].moe
        assert len(moe.experts) < self.cfg.e_max, "router headroom exhausted"
        ex = Expert(self.cfg).to(next(self.parameters()).device)
        if page_f64 is not None:
            ex.load_page(page_f64)
        moe.experts.append(ex)
        return len(moe.experts) - 1

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_moe(cfg: MoEGPTConfig = None, device: str = None, seed: int = None,
              experts_per_layer: list[int] | None = None):
    cfg = cfg or MoEGPTConfig()
    device = device or pick_device()
    model = MoEGPT(cfg, experts_per_layer).to(device)
    if seed is not None:                                          # shared genesis
        apply_genesis(model, seed)
    return model, device


# --------------------------------------------------------------------------
# ChainLayout — the torch↔chain permutation (consensus page table is truth)
# --------------------------------------------------------------------------
class ChainLayout:
    """Maps between torch's flat parameter vector (client/trainer.flat_params —
    named_parameters() order, experts interleaved inside their layers) and the
    CHAIN order (backbone first, then expert pages in (layer, expert) order).

    chain_vec = torch_vec[to_chain];  torch_vec = chain_vec[to_torch]

    Also exposes the consensus page table view this model implies: page 0 =
    backbone span [0, backbone_params); page 1.. = expert pages in (layer,
    expert) order, each of expert_page_len — matching rig/model_state.py
    exactly (assert against the node's ModelState before trusting a sync)."""

    def __init__(self, model: MoEGPT):
        self.n = model.num_params()
        d, f = model.cfg.n_embd, model.cfg.d_ff
        self.expert_page_len = d * f + f + f * d + d
        offs, i = [], 0
        for name, p in model.named_parameters():
            offs.append((name, i, i + p.numel()))
            i += p.numel()
        assert i == self.n

        def expert_key(name):
            parts = name.split(".")
            if "experts" in parts and "blocks" in parts:
                return (int(parts[parts.index("blocks") + 1]),
                        int(parts[parts.index("experts") + 1]))
            return None

        backbone_runs = [(s, e) for name, s, e in offs if expert_key(name) is None]
        expert_runs: dict = {}
        for name, s, e in offs:
            k = expert_key(name)
            if k is not None:
                expert_runs.setdefault(k, []).append((s, e))
        self.backbone_params = sum(e - s for s, e in backbone_runs)
        self.experts = sorted(expert_runs)                 # [(layer, e), ...]
        for k in self.experts:
            assert sum(e - s for s, e in expert_runs[k]) == self.expert_page_len

        # to_chain: torch indices in chain order
        pieces = [np.arange(s, e) for s, e in backbone_runs]
        self.page_spans = [(0, self.backbone_params)]
        pos = self.backbone_params
        for k in self.experts:
            for s, e in expert_runs[k]:
                pieces.append(np.arange(s, e))
            self.page_spans.append((pos, pos + self.expert_page_len))
            pos += self.expert_page_len
        self.to_chain = np.concatenate(pieces)
        assert self.to_chain.size == self.n
        self.to_torch = np.empty(self.n, dtype=np.int64)
        self.to_torch[self.to_chain] = np.arange(self.n)

    # ---- conversions -----------------------------------------------------
    def chain_of(self, torch_vec: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(torch_vec[self.to_chain])

    def torch_of(self, chain_vec: np.ndarray) -> np.ndarray:
        # to_torch[torch_index] = chain_position, so this inverts chain_of
        return np.ascontiguousarray(chain_vec[self.to_torch])

    # ---- page queries (chain-order coordinates) --------------------------
    def n_pages(self) -> int:
        return len(self.page_spans)

    def page_span(self, page_id: int) -> tuple[int, int]:
        return self.page_spans[page_id]

    def page_of_expert(self, layer: int, e: int) -> int:
        return 1 + self.experts.index((layer, e))

    def pages_touched(self, chain_delta: np.ndarray) -> list[int]:
        """Which pages a (chain-order) delta actually touches — the claim set."""
        out = []
        for pid, (s, e) in enumerate(self.page_spans):
            if np.any(chain_delta[s:e] != 0):
                out.append(pid)
        return out

    def zero_outside(self, chain_delta: np.ndarray, pages: list[int]) -> np.ndarray:
        """Enforce the v1 tx rule client-side: zero everything outside the
        claimed pages (so the tx can never be rejected for stray coordinates)."""
        keep = np.zeros(self.n, dtype=bool)
        for p in pages:
            s, e = self.page_spans[p]
            keep[s:e] = True
        out = chain_delta.copy()
        out[~keep] = 0
        return out


def load_fraction(layout: ChainLayout, held_pages: list[int]) -> float:
    """Fraction of parameters a node loads holding `held_pages` (page 0 =
    backbone, which every trainer needs) — the memory win of sharding."""
    total = sum(layout.page_span(p)[1] - layout.page_span(p)[0]
                for p in held_pages)
    return float(total) / layout.n
