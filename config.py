"""Hyperparameter configuration for ARES.

Everything that defines a run lives here in a single dataclass so it can be
logged verbatim and serialized into checkpoints (making runs reproducible and
resumable). The defaults match the project spec: a deliberately small GPT-style
decoder meant to be tuned and to train on a single GPU/CPU.

The model-architecture fields (vocab_size, d_model, n_heads, n_layers,
block_size, dropout) are saved into each checkpoint so `generate.py` can rebuild
the exact same network without guessing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # ---- Tokenizer / data --------------------------------------------------
    # tokenizer: "char" (from-scratch char-level) or "bpe" (tiktoken gpt2 BPE).
    tokenizer: str = "char"
    dataset: str = "tinyshakespeare"  # logical name of the corpus
    data_dir: str = "data"            # where the corpus text file is cached

    # vocab_size is resolved at runtime from the tokenizer (None until then).
    vocab_size: int | None = None

    # ---- Model architecture (spec defaults) --------------------------------
    d_model: int = 256       # embedding / residual stream width
    n_heads: int = 4         # number of attention heads (d_model must divide)
    n_layers: int = 4        # number of stacked transformer blocks
    block_size: int = 128    # context length (max sequence the model sees)
    dropout: float = 0.1     # dropout prob on embeddings, attention, FFN, resid
    bias: bool = True        # use bias terms in Linear/LayerNorm layers
    pos_encoding: str = "learned"  # "learned" (default) or "sinusoidal"
    tie_weights: bool = True       # weight-tie token embedding & output head

    # ---- Optimization ------------------------------------------------------
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1     # AdamW weight decay (applied to 2D params only)
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0        # max grad norm (<=0 disables clipping)

    max_iters: int = 5000         # total training iterations
    warmup_iters: int = 100       # linear LR warmup steps
    lr_decay: bool = True         # cosine-decay the LR after warmup
    min_lr: float = 3e-5          # floor for the cosine schedule

    # ---- Logging / eval / checkpointing ------------------------------------
    eval_interval: int = 250      # iters between val-loss estimates
    eval_iters: int = 50          # batches averaged per val-loss estimate
    log_interval: int = 50        # iters between train-loss prints
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"
    always_save_checkpoint: bool = False  # if True, save every eval, not just best

    # ---- Runtime -----------------------------------------------------------
    device: str = "auto"   # "auto" -> cuda if available else cpu
    seed: int = 1337
    compile: bool = False  # torch.compile (off by default for portability)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # ---- Serialization helpers --------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# A tiny config used by the test-suite and quick smoke checks. Small enough to
# run a forward/backward pass in milliseconds on CPU.
def tiny_config(**overrides) -> Config:
    cfg = Config(
        tokenizer="char",
        d_model=64,
        n_heads=4,
        n_layers=2,
        block_size=32,
        dropout=0.0,
        batch_size=8,
        max_iters=50,
        warmup_iters=0,
        lr_decay=False,
        eval_interval=10,
        eval_iters=5,
        log_interval=10,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# Default config instance importable as `from config import default_config`.
default_config = Config()
