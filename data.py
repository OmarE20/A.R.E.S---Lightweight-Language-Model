"""Data pipeline for ARES.

Responsibilities:
  * download a small plaintext corpus (TinyShakespeare) if absent,
  * tokenize it with either a from-scratch character-level tokenizer or
    tiktoken's GPT-2 BPE,
  * encode the full corpus to a 1D tensor of token ids,
  * carve a ~90/10 train/val split,
  * sample random contiguous (input, target) batches where targets are inputs
    shifted by one position.

The two tokenizers expose the same `encode`/`decode`/`vocab_size` interface so
the rest of the codebase is agnostic to which one is in use. The char-level
tokenizer is included as an "even more from scratch" option to compare against
BPE, per the spec.
"""

from __future__ import annotations

import os
import pickle
import urllib.request

import numpy as np
import torch

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


# ---------------------------------------------------------------------------
# Corpus download
# ---------------------------------------------------------------------------
def download_corpus(data_dir: str, dataset: str = "tinyshakespeare") -> str:
    """Ensure the corpus text file exists locally; return its path.

    Swapping corpora is as easy as dropping a `<dataset>.txt` into `data_dir`;
    only TinyShakespeare is auto-downloaded.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{dataset}.txt")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    if dataset != "tinyshakespeare":
        raise FileNotFoundError(
            f"Corpus '{path}' not found and only 'tinyshakespeare' can be "
            f"auto-downloaded. Place a plaintext file there to use a custom corpus."
        )

    print(f"Downloading TinyShakespeare to {path} ...")
    urllib.request.urlretrieve(TINYSHAKESPEARE_URL, path)
    print(f"Done ({os.path.getsize(path)} bytes).")
    return path


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
class CharTokenizer:
    """A minimal, fully from-scratch character-level tokenizer.

    The vocabulary is the sorted set of unique characters in the corpus; ids are
    just indices into that list. No subword merges, no external library.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        # Unknown chars (not seen at fit time) are skipped; for a single-corpus
        # setup the encoder is always fit on the same text it encodes.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # ---- (de)serialization so generate.py can rebuild the exact vocab ------
    def state(self) -> dict:
        return {"type": "char", "itos": self.itos}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        obj = cls.__new__(cls)
        obj.itos = {int(k): v for k, v in state["itos"].items()}
        obj.stoi = {v: k for k, v in obj.itos.items()}
        obj.vocab_size = len(obj.itos)
        return obj


class BPETokenizer:
    """Thin wrapper over tiktoken's GPT-2 BPE.

    We use tiktoken's *encoding* (the byte-pair merge tables / algorithm) but
    none of HuggingFace's tokenizer classes — this is allowed by the spec, which
    explicitly names tiktoken as part of the stack.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken

        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, s: str) -> list[int]:
        return self.enc.encode(s)

    def decode(self, ids) -> str:
        return self.enc.decode([int(i) for i in ids])

    def state(self) -> dict:
        return {"type": "bpe", "encoding_name": self.encoding_name}

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        return cls(encoding_name=state.get("encoding_name", "gpt2"))


def build_tokenizer(kind: str, text: str | None = None):
    """Factory: build a tokenizer of the requested kind."""
    if kind == "char":
        if text is None:
            raise ValueError("char tokenizer needs the corpus text to fit its vocab")
        return CharTokenizer(text)
    if kind == "bpe":
        return BPETokenizer()
    raise ValueError(f"Unknown tokenizer kind: {kind!r} (expected 'char' or 'bpe')")


def load_tokenizer_from_state(state: dict):
    if state["type"] == "char":
        return CharTokenizer.from_state(state)
    if state["type"] == "bpe":
        return BPETokenizer.from_state(state)
    raise ValueError(f"Unknown tokenizer state type: {state['type']!r}")


# ---------------------------------------------------------------------------
# Dataset: encode corpus -> ids, split, and batch
# ---------------------------------------------------------------------------
class Dataset:
    """Holds the encoded corpus split into train/val and serves random batches."""

    def __init__(self, cfg, split_ratio: float = 0.9, verbose: bool = True):
        path = download_corpus(cfg.data_dir, cfg.dataset)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = build_tokenizer(cfg.tokenizer, text)
        cfg.vocab_size = self.tokenizer.vocab_size  # resolve vocab into config

        # Encode the entire corpus to a 1D tensor of token ids.
        ids = np.array(self.tokenizer.encode(text), dtype=np.int64)
        data = torch.from_numpy(ids)

        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.block_size = cfg.block_size
        self.vocab_size = self.tokenizer.vocab_size

        if verbose:
            print(
                f"corpus: {len(text):,} chars -> {len(data):,} tokens "
                f"({cfg.tokenizer}, vocab={self.vocab_size}); "
                f"train={len(self.train_data):,} val={len(self.val_data):,}"
            )

    def get_batch(self, split: str, batch_size: int, device: str = "cpu",
                  generator: torch.Generator | None = None):
        """Sample a batch of (inputs x, targets y).

        x = tokens [i : i+block_size]; y = tokens [i+1 : i+block_size+1] so that
        position t in y is the next-token label for position t in x.
        """
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Corpus split '{split}' too small ({len(data)} tokens) for "
                f"block_size={self.block_size}."
            )
        ix = torch.randint(max_start, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        if device.startswith("cuda"):
            # pin + async copy is a small speedup on GPU; harmless on CPU path.
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def save_tokenizer(tokenizer, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(tokenizer.state(), f)


def load_tokenizer(path: str):
    with open(path, "rb") as f:
        return load_tokenizer_from_state(pickle.load(f))


if __name__ == "__main__":
    # Quick self-check: load data and print a batch's shapes + the shift property.
    from config import Config

    cfg = Config(tokenizer="char", block_size=16)
    ds = Dataset(cfg)
    xb, yb = ds.get_batch("train", batch_size=4)
    print("x shape", tuple(xb.shape), "y shape", tuple(yb.shape))
    # Verify targets are inputs shifted by one (y[:, :-1] == x[:, 1:]).
    assert torch.equal(xb[:, 1:], yb[:, :-1]), "targets must be inputs shifted by one"
    print("shift check OK")
    print("decoded x[0]:", repr(ds.tokenizer.decode(xb[0].tolist())))
