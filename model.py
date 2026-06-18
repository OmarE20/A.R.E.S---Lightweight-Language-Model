"""ARES model: a decoder-only (GPT-style) transformer, built from scratch.

Every core component is implemented by hand. In particular, multi-head
self-attention is written out explicitly — Q/K/V projections, the multi-head
reshape/split, scaled dot-product attention, a causal mask, and the output
projection. We deliberately do NOT use any prebuilt transformer/attention
building block:

  * NO torch.nn.Transformer / TransformerEncoder / TransformerDecoder / *Layer
  * NO nn.MultiheadAttention
  * NO HuggingFace transformers

We DO use PyTorch primitives, which is expected and allowed: nn.Linear,
nn.Embedding, nn.LayerNorm, nn.Dropout, and functional ops (F.softmax,
F.cross_entropy, F.gelu).

Architecture (pre-norm GPT):
    block(x):  x = x + attn(LN(x))
               x = x + ffn(LN(x))
    final:     logits = OutputHead(LN(stack of blocks(embeddings)))
The output head is weight-tied with the token embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-head causal self-attention (hand-built)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with a causal mask.

    Shapes use B=batch, T=time/sequence length, C=d_model, nh=n_heads,
    hd=head dim (C // nh).
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads

        # One Linear produces Q, K and V together (3*C outputs), then we split.
        # This is exactly three separate Q/K/V projections fused for efficiency.
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        # Output projection that mixes the concatenated heads back to C.
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout

        # Lower-triangular causal mask, registered as a (non-learned) buffer so it
        # moves with .to(device) and is saved/loaded with the module. Shape
        # (1, 1, block, block) to broadcast over (B, nh, T, T) attention scores.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V and split along the channel dim.
        qkv = self.qkv_proj(x)                  # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=2)  # each (B, T, C)

        # Reshape each into heads: (B, T, C) -> (B, nh, T, hd). Splitting the
        # channels into nh groups is the "multi-head" mechanism — each head
        # attends in its own hd-dimensional subspace.
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention, written out explicitly.
        # scores[b,h,i,j] = (q_i . k_j) / sqrt(hd)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B,nh,T,T)

        # Causal mask: position i may only attend to positions j <= i. We set the
        # upper triangle (future) to -inf so softmax assigns it zero weight.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values: (B,nh,T,T) @ (B,nh,T,hd) -> (B,nh,T,hd)
        y = att @ v
        # Re-assemble heads: (B,nh,T,hd) -> (B,T,C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection + residual dropout.
        y = self.resid_dropout(self.out_proj(y))
        return y


# ---------------------------------------------------------------------------
# Position-wise feed-forward network (hand-built)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Two linear layers with a GELU nonlinearity, hidden dim 4 * d_model."""

    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer block: pre-norm + residual around attn and ffn
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """One transformer block in pre-norm configuration.

        x = x + attn(LN1(x))
        x = x + ffn (LN2(x))

    Pre-norm (LayerNorm *before* each sub-layer, with the residual added to the
    un-normalized stream) keeps a clean identity path from input to output,
    which makes deep transformers far more stable to train than the original
    post-norm design. This matches modern GPT practice.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (optional alternative to learned)
# ---------------------------------------------------------------------------
def sinusoidal_position_encoding(block_size: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encodings from Vaswani et al. (2017).

    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
    Returned shape: (block_size, d_model).
    """
    pos = torch.arange(block_size, dtype=torch.float).unsqueeze(1)        # (T,1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(block_size, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size is not None, "cfg.vocab_size must be set (build data first)"
        self.cfg = cfg
        self.block_size = cfg.block_size

        # Token embedding: id -> d_model vector.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Positional encoding: learned table (default) or fixed sinusoidal buffer.
        self.pos_encoding = cfg.pos_encoding
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        elif cfg.pos_encoding == "sinusoidal":
            self.register_buffer(
                "pos_table", sinusoidal_position_encoding(cfg.block_size, cfg.d_model)
            )
        else:
            raise ValueError(f"Unknown pos_encoding: {cfg.pos_encoding!r}")

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)

        # Output head: project d_model -> vocab logits.
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share the (vocab x d_model) matrix between the input
        # embedding and the output projection. This couples "what a token means
        # as input" with "how the model scores it as output", saves vocab*d_model
        # parameters, and typically improves generalization on small corpora.
        if cfg.tie_weights:
            self.head.weight = self.token_emb.weight

        # Initialize weights (GPT-2 style).
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 trick): keeps the variance
        # of the residual stream from growing with depth.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_encoding == "learned":
            # Subtract the positional table; token emb is tied to the head so it
            # is genuinely used in the forward compute and we keep it counted.
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits, loss).

        logits: (B, T, vocab_size). loss is None unless targets are given.
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )

        tok = self.token_emb(idx)  # (B, T, C)
        if self.pos_encoding == "learned":
            pos = torch.arange(T, device=idx.device)
            pos_emb = self.pos_emb(pos)            # (T, C)
        else:
            pos_emb = self.pos_table[:T]           # (T, C)
        x = self.drop(tok + pos_emb)               # broadcast add over batch

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten time into the batch dim for cross-entropy over next tokens.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def configure_optimizers(self, cfg):
        """AdamW with weight decay on 2D params (matmuls/embeddings) only.

        Biases and LayerNorm gains are 1D and excluded from weight decay — the
        standard GPT recipe. Returns a torch.optim.AdamW.
        """
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 generator: torch.Generator | None = None):
        """Autoregressively sample `max_new_tokens` continuations.

        At each step we crop the running context to the last `block_size` tokens
        (the model has no memory beyond that), take the logits at the final
        position, apply temperature, optional top-k filtering, softmax, and
        sample one token.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]            # crop context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                       # (B, vocab) last step

            if temperature <= 0:
                # Greedy / argmax decoding (temperature -> 0 limit).
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    # Mask everything below the k-th best logit to -inf.
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)

            idx = torch.cat([idx, idx_next], dim=1)
        return idx


if __name__ == "__main__":
    # Smoke check: build a tiny model, run a forward pass, print shapes/params.
    from config import tiny_config

    cfg = tiny_config(vocab_size=65)
    model = GPT(cfg)
    print(f"params: {model.num_params():,}")
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, x)
    print("logits", tuple(logits.shape), "loss", float(loss))
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    print("forward OK")
