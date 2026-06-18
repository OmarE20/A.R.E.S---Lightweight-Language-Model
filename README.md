# ARES — A lightweight decoder-only LLM, built from scratch in PyTorch

ARES is a small **GPT-style (decoder-only) transformer language model** implemented
from first principles. Every core component — multi-head causal self-attention,
the feed-forward network, pre-norm transformer blocks, positional encodings,
weight-tied output head — is **hand-written**. It trains on a modest text corpus
(TinyShakespeare by default) on a single GPU or even CPU/Colab, and generates
locally coherent text.

The point of the project is understanding: it exercises the full architecture and
training/inference lifecycle of a modern LLM at small scale, with no black boxes.

References: Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and
Vaswani et al., ["Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762).

---

## The from-scratch rule (the whole point)

**No prebuilt transformer or attention building blocks, and no HuggingFace.**
Specifically *not* used anywhere in the code:

- `torch.nn.Transformer`, `nn.TransformerEncoder`/`Decoder`/`*Layer`
- `nn.MultiheadAttention`
- the HuggingFace `transformers` library / its model & tokenizer classes

Multi-head self-attention is written out by hand: explicit Q/K/V projections,
the multi-head reshape/split, scaled dot-product attention, a causal mask, and
the output projection.

PyTorch **primitives are used and expected**: `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout`, and functional ops (`F.softmax`, `F.cross_entropy`,
`F.gelu`). The line is: *primitives yes; prebuilt transformer/attention/tokenizer
mechanics no.*

A test (`tests/test_ares.py::test_no_banned_imports_in_source`) enforces this by
scanning the source (ignoring comments/docstrings) for the banned names.

---

## Project layout

```
.
├── model.py        # Transformer, attention, FFN, blocks — all hand-built
├── data.py         # corpus download, tokenizers, encoding, split, batching
├── train.py        # training loop, LR schedule, logging, checkpointing, plots
├── generate.py     # inference / autoregressive text generation
├── config.py       # hyperparameter config (dataclass, serialized into ckpts)
├── tests/          # pytest: causal mask, shapes, overfit, ckpt resume, bans
├── plots/          # saved loss curves
├── checkpoints/    # saved model + optimizer + config + tokenizer
├── requirements.txt
├── Makefile        # install / data / train / generate / test
└── README.md
```

---

## Setup

Requires Python 3.10+.

```bash
# (recommended) create a virtual environment first
python -m pip install -r requirements.txt
```

> **NumPy note:** `torch` and `matplotlib` ship extensions compiled against
> NumPy 1.x, so `requirements.txt` pins `numpy<2`. With NumPy 2.x installed you
> will see *"A module compiled using NumPy 1.x cannot run in NumPy 2.x"* — just
> `pip install "numpy<2"`.

On Windows the `make` targets map to these raw commands (use `py` if that's your
launcher):

| Task | `make` | raw command |
|------|--------|-------------|
| install deps | `make install` | `python -m pip install -r requirements.txt` |
| download corpus | `make data` | `python -c "from config import Config; from data import download_corpus; download_corpus(Config().data_dir)"` |
| train (defaults) | `make train` | `python train.py` |
| smoke train | `make train-smoke` | `python train.py --tokenizer char --max_iters 500 --eval_interval 100 --warmup_iters 50` |
| generate | `make generate` | `python generate.py --prompt "ROMEO:" --temperature 0.8` |
| test | `make test` | `python -m pytest -q` |

### First-run commands (download → train → generate)

```bash
python -m pip install -r requirements.txt
python train.py --tokenizer char --max_iters 500 --eval_interval 100 --eval_iters 20 --warmup_iters 50
python generate.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8
```

The corpus auto-downloads on the first `train.py` run; no manual data step needed.

---

## Architecture summary (the interview surface)

ARES is a stack of `n_layers` identical **pre-norm** transformer blocks operating
on a residual stream of width `d_model`. Token ids → embeddings → +positional
encoding → blocks → final LayerNorm → linear head → vocabulary logits.

### 1. Token & positional embeddings (`model.py`)
- **Token embedding** (`nn.Embedding`): a `vocab_size × d_model` lookup table
  mapping each token id to a dense vector.
- **Positional encoding**: by default a **learned** `block_size × d_model` table
  (`pos_encoding="learned"`), added to the token embeddings so the otherwise
  permutation-invariant attention knows *where* each token is. A **sinusoidal**
  alternative (Vaswani et al.) is implemented in `sinusoidal_position_encoding`
  and selectable via `pos_encoding="sinusoidal"`.

### 2. Multi-head causal self-attention (`CausalSelfAttention`) — hand-built
For input `x` of shape `(B, T, C)` with `C = d_model`:

1. **Q/K/V projection.** A single `nn.Linear(C, 3C)` produces queries, keys and
   values together; we `split` them into three `(B, T, C)` tensors. (Fusing the
   three projections into one matmul is purely an efficiency choice — it is
   mathematically identical to three separate `Q`, `K`, `V` linears.)
2. **Multi-head split.** Each of Q/K/V is reshaped `(B, T, C) → (B, nh, T, hd)`
   with `hd = C // nh`. Splitting the channels into `nh` heads lets each head
   attend in its own subspace and learn a different relation.
3. **Scaled dot-product attention.** `scores = (Q @ Kᵀ) / sqrt(hd)`, shape
   `(B, nh, T, T)`. The `1/sqrt(hd)` scaling keeps the dot products from growing
   with head dimension, which would otherwise push softmax into saturated,
   tiny-gradient regions.
4. **Causal mask.** A lower-triangular mask (registered buffer) sets every
   "future" entry `j > i` to `-inf` *before* softmax, so position `i` can only
   attend to positions `≤ i`. This is what makes the model autoregressive: the
   prediction at each position never sees tokens that come after it. (Tested two
   ways — see below.)
5. **Softmax + weighted sum.** `softmax` over the last dim turns scores into
   weights; `weights @ V` produces each position's context vector.
6. **Re-assemble + output projection.** Heads are concatenated back to `(B, T, C)`
   and passed through a final `nn.Linear(C, C)`. Dropout is applied to the
   attention weights and to the residual output.

### 3. Feed-forward network (`FeedForward`)
Position-wise MLP: `Linear(C → 4C) → GELU → Linear(4C → C) → Dropout`. The 4×
expansion is the standard transformer width; GELU is the nonlinearity (ReLU is a
fine alternative). It mixes information across channels (attention mixes across
positions; the FFN mixes within a position).

### 4. Pre-norm transformer block (`Block`)
```
x = x + attn(LayerNorm(x))      # sub-layer 1
x = x + ffn (LayerNorm(x))      # sub-layer 2
```
- **Pre-norm placement.** LayerNorm is applied *before* each sub-layer, not after.
  This keeps an unobstructed identity (residual) path from input to output, which
  makes deep transformers far more stable to train than the original post-norm
  design — it's standard modern GPT practice.
- **Residual connections** around both sub-layers let gradients flow directly to
  early layers and let each block learn a *delta* to the stream rather than
  having to preserve information itself.

### 5. Final norm + output head, with weight tying
After the blocks, a final LayerNorm, then `nn.Linear(d_model → vocab_size)`
produces logits. The head is **weight-tied** with the token embedding
(`self.head.weight = self.token_emb.weight`):
- it ties "what a token means as input" to "how the model scores it as output",
- it removes a whole `vocab_size × d_model` parameter matrix, and
- it typically improves generalization on small corpora.

### 6. Loss
Training minimizes **cross-entropy** between the logits at each position and the
*next* token (`F.cross_entropy` over the flattened `(B·T, vocab)` logits).

---

## Training (`train.py`)

- **Optimizer:** `AdamW`. Weight decay is applied only to 2D parameters
  (matmuls/embeddings); biases and LayerNorm gains are excluded — the standard
  GPT recipe.
- **Gradient clipping:** global-norm clip at `grad_clip` (default 1.0) guards
  against the occasional exploding-gradient step on a tiny corpus.
- **LR schedule:** linear warmup → cosine decay to `min_lr` (toggle with
  `--lr_decay`).
- **Eval:** train & val loss are estimated by **averaging over `eval_iters`
  batches** so the curve isn't dominated by per-batch noise.
- **Checkpointing:** saves model + optimizer + full config + loss history to
  `checkpoints/ckpt.pt` (best val by default), and the tokenizer to
  `checkpoints/tokenizer.pkl`, so a run can `--resume` and `generate.py` can
  rebuild the exact model.
- **Plot:** a train/val loss curve is written to `plots/loss_curve.png`.
- **Reproducibility:** NumPy + PyTorch RNGs are seeded; batch sampling uses a
  dedicated seeded `torch.Generator`; the full config is printed at the start of
  every run and stored inside each checkpoint.

### Default hyperparameters (`config.py`)
`d_model=256 · n_heads=4 · n_layers=4 · block_size=128 · dropout=0.1 ·
batch_size=32 · learning_rate=3e-4 · optimizer=AdamW`. `vocab_size` is set by the
tokenizer. These are deliberately small and meant to be tuned.

---

## Inference (`generate.py`)

Loads a checkpoint (with its config + tokenizer), encodes a prompt, and
autoregressively samples one token at a time, **cropping the context to the last
`block_size` tokens each step**. Sampling controls:

- `--temperature` — scales logits before softmax. `>1` = more random, `<1` =
  greedier; `0` = deterministic argmax/greedy decoding.
- `--top_k` — optional: restrict sampling to the `k` most likely tokens each step.

```bash
python generate.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8
python generate.py --prompt "To be"  --top_k 40 --temperature 0.9
```

---

## Tokenizers

Both expose the same `encode`/`decode`/`vocab_size` interface; pick with
`--tokenizer`:

- **`char`** (default) — a fully from-scratch character-level tokenizer: vocab is
  the sorted set of unique characters; ids are indices. Tiny vocab (~65 for
  TinyShakespeare), longer sequences, "even more from scratch."
- **`bpe`** — tiktoken's GPT-2 byte-pair encoding (~50k vocab). We use tiktoken's
  *encoding algorithm* (allowed by the spec), not any HF tokenizer class.

---

## Tests

```bash
python -m pytest -q
```

Fast tests (tiny CPU config, well under a second total) covering the load-bearing
logic:

- **forward shapes** are `[batch, block_size, vocab_size]` (full and short seqs);
- **causal mask, no future leakage** — perturbing the last input token leaves all
  earlier positions' logits bit-for-bit unchanged;
- **causal mask, direct** — post-softmax attention weights are exactly zero in the
  upper (future) triangle for every head;
- **overfit a single batch** — loss collapses, proving the train wiring works;
- **checkpoint roundtrip** — a saved checkpoint reloads to identical weights and
  identical outputs;
- **weight tying** — head and embedding share one tensor and the param count drops
  by exactly `vocab_size × d_model`;
- **temperature=0 determinism**;
- **no banned imports** in the source.

---

## Smoke-run results

<!-- SMOKE_RESULTS_START -->
Real run captured for this README — **CPU only**, char tokenizer, 500 iterations
(~137 s on a laptop CPU), default model (`d_model=256, n_heads=4, n_layers=4,
block_size=128`, **3,208,960 params**):

```
python train.py --tokenizer char --max_iters 500 --eval_interval 100 --eval_iters 20 --warmup_iters 50
```

| iter | train loss | val loss | lr |
|-----:|-----------:|---------:|------|
| 0 | 4.2457 | 4.2430 | 6.0e-06 |
| 100 | 2.4981 | 2.5137 | 2.92e-04 |
| 200 | 2.4128 | 2.4250 | 2.32e-04 |
| 300 | 2.3213 | 2.3480 | 1.42e-04 |
| 400 | 2.2717 | 2.2919 | 6.16e-05 |
| 500 | 2.2440 | **2.2608** | 3.0e-05 |

**Val loss decreases monotonically** from 4.24 → 2.26 (≈ from random over a
65-symbol vocab toward a real character model). Train and val track closely, so
no meaningful overfitting yet at this length.

Loss curve (`plots/loss_curve.png`):

![ARES smoke-run loss curve](plots/loss_curve.png)

Sample generation from that checkpoint (`temperature=0.8`, `--seed 1337`):

```
python generate.py --prompt "ROMEO:" --max_new_tokens 280 --temperature 0.8 --seed 1337
```

```
ROMEO:
O slcherer cones ury thist thit thize tar pror thind ariat.


CCARINIO:
If thy le nout's ve het he sseare
Mear, t athou tave ard sbest theat thof t ive.

OFWAULEEV:
He rigaty, heant he wand hear.
```

After only 500 CPU iterations the model has already learned the *form* of the
data — `NAME:` speaker tags, blank lines between speeches, capitalization,
word-like spacing — but not real words or meaning. **A longer run is left to you**
(e.g. `python train.py` with the defaults, or add `--tokenizer bpe`) and produces
markedly more coherent text.
<!-- SMOKE_RESULTS_END -->

> **Quality is bounded by scale.** This is a small model trained briefly on ~1MB
> of text. Character-level output learns Shakespearean *texture* — line breaks,
> `NAME:` speaker tags, plausible word shapes — but not long-range meaning. A
> longer run with the default config (and/or the BPE tokenizer) improves
> coherence substantially; that longer run is intentionally left to you (see
> punch list).

---

## Efficiency notes (weight tying & gradient clipping)

- **Weight tying** removes a `vocab_size × d_model` matrix. For the char model
  (`vocab≈65, d_model=256`) that's ~17k params — small; for the BPE model
  (`vocab=50257, d_model=256`) it's ~12.9M params, a *large* fraction of the
  model. The test asserts the exact reduction. See
  `test_weight_tying_shares_tensor_and_saves_params`.
- **Gradient clipping** (max-norm 1.0) caps the global gradient norm so a single
  noisy batch can't blow up the weights — cheap insurance that matters more on a
  small, repetitive corpus. Disable with `--grad_clip 0` to observe the
  difference.

---

## Reproducibility

Every run seeds NumPy + PyTorch (`--seed`, default 1337), samples batches from a
dedicated seeded generator, prints its full config at startup, and embeds that
config in the checkpoint. Re-running with the same flags reproduces the run;
`generate.py` with a fixed `--seed` reproduces a sample.

---

## Future work (documented, intentionally out of scope for this pass)

- A **full training run** to strong coherence (longer wall-clock; the smoke run
  here just proves the pipeline).
- **top-p (nucleus) sampling** beyond the basic top-k already implemented.
- **KV caching** for faster generation.
- **Ablations:** effect of `n_layers` / `block_size` on val loss.
- A full **char-vs-BPE** comparison (needs multiple full runs).
- **Mixed-precision** training.
