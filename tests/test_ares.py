"""Fast tests for the load-bearing logic in ARES.

Everything runs on a tiny CPU config in well under a second:
  * forward-pass output shapes are [batch, block, vocab]
  * the causal mask genuinely prevents attending to future positions
    (changing a future token can't alter earlier positions' outputs)
  * a single batch can be overfit -> loss goes down (training wiring works)
  * a saved checkpoint reloads and resumes with identical weights
  * weight tying actually shares one tensor and trims the param count
  * no banned transformer/attention classes are used anywhere in the source
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import tiny_config  # noqa: E402
from model import GPT, CausalSelfAttention  # noqa: E402


def _model(**overrides):
    cfg = tiny_config(vocab_size=37, **overrides)
    torch.manual_seed(0)
    return GPT(cfg), cfg


# ---------------------------------------------------------------------------
def test_forward_shapes():
    model, cfg = _model()
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_forward_accepts_short_sequences():
    model, cfg = _model()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size // 2))
    logits, _ = model(x)
    assert logits.shape == (2, cfg.block_size // 2, cfg.vocab_size)


# ---------------------------------------------------------------------------
def test_causal_mask_no_future_leakage():
    """Output at position t must not depend on tokens at positions > t.

    We feed a sequence, then change the LAST token and confirm every earlier
    position's logits are unchanged. If the mask leaked, they would shift.
    """
    model, cfg = _model()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # perturb the future token
        perturbed, _ = model(x2)
    # All positions except the last must be bit-for-bit identical.
    assert torch.allclose(base[0, :-1], perturbed[0, :-1], atol=1e-6)
    # The last position is allowed to (and should) change.
    assert not torch.allclose(base[0, -1], perturbed[0, -1])


def test_attention_scores_are_lower_triangular():
    """Directly inspect the post-softmax attention weights: the upper triangle
    (future) must be exactly zero for every head."""
    cfg = tiny_config(vocab_size=37)
    attn = CausalSelfAttention(cfg)
    attn.eval()
    captured = {}

    import torch.nn.functional as F

    orig_softmax = F.softmax

    def spy(t, dim=-1, **kw):
        out = orig_softmax(t, dim=dim, **kw)
        captured["w"] = out.detach()
        return out

    F.softmax = spy
    try:
        x = torch.randn(1, cfg.block_size, cfg.d_model)
        attn(x)
    finally:
        F.softmax = orig_softmax

    w = captured["w"][0]  # (n_heads, T, T)
    T = cfg.block_size
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert torch.all(w[:, upper] == 0), "attention leaked into future positions"


# ---------------------------------------------------------------------------
def test_overfit_single_batch_loss_decreases():
    """The full train wiring should drive loss down on one repeated batch."""
    # Higher LR so a 2-layer model memorizes one batch of random targets fast.
    model, cfg = _model(dropout=0.0, learning_rate=1e-2)
    model.train()
    opt = model.configure_optimizers(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))

    _, first = model(x, y)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    _, last = model(x, y)
    # Should overfit hard: loss collapses well below half the starting value.
    assert last.item() < first.item() * 0.5, (first.item(), last.item())


# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    model, cfg = _model()
    opt = model.configure_optimizers(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(x, x)
    loss.backward()
    opt.step()

    path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "config": cfg.to_dict(), "iter_num": 7,
                "best_val_loss": 1.23, "history": {}}, path)

    ckpt = torch.load(path, weights_only=False)
    from config import Config
    cfg2 = Config.from_dict(ckpt["config"])
    model2 = GPT(cfg2)
    model2.load_state_dict(ckpt["model"])
    assert ckpt["iter_num"] == 7
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(),
                                  model2.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)

    # Outputs match after reload (eval mode to disable dropout).
    model.eval(); model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(x)[0], model2(x)[0], atol=1e-6)


# ---------------------------------------------------------------------------
def test_weight_tying_shares_tensor_and_saves_params():
    tied, _ = _model(tie_weights=True)
    untied, _ = _model(tie_weights=False)
    # Tied: the head weight IS the embedding weight (same storage).
    assert tied.head.weight.data_ptr() == tied.token_emb.weight.data_ptr()
    assert untied.head.weight.data_ptr() != untied.token_emb.weight.data_ptr()
    # Tying removes one (vocab x d_model) matrix worth of parameters.
    assert untied.num_params() - tied.num_params() == 37 * tied.cfg.d_model


def test_temperature_zero_is_deterministic():
    """temperature=0 (argmax) generation must be reproducible run-to-run."""
    model, cfg = _model()
    model.eval()
    idx = torch.zeros(1, 1, dtype=torch.long)
    a = model.generate(idx.clone(), max_new_tokens=10, temperature=0.0)
    b = model.generate(idx.clone(), max_new_tokens=10, temperature=0.0)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
def test_no_banned_imports_in_source():
    """Guard the project's central constraint: no prebuilt transformer/attention
    blocks and no HuggingFace."""
    import glob
    import io
    import re
    import tokenize

    banned = [
        r"nn\.Transformer\b",
        r"nn\.TransformerEncoder",
        r"nn\.TransformerDecoder",
        r"TransformerEncoderLayer",
        r"TransformerDecoderLayer",
        r"nn\.MultiheadAttention",
        r"MultiheadAttention",
        r"\bfrom\s+transformers\b",
        r"\bimport\s+transformers\b",
    ]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    srcs = glob.glob(os.path.join(root, "*.py"))
    assert srcs, "no source files found"

    def strip_strings_and_comments(code: str) -> str:
        # Tokenize and drop COMMENT + STRING tokens so docstrings/prose that
        # mention the banned names (e.g. model.py's module docstring) don't
        # trip the check — only real executable code is scanned.
        out = []
        toks = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in toks:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
        return " ".join(out)

    for src in srcs:
        with open(src, "r", encoding="utf-8") as fh:
            code = fh.read()
        code_only = strip_strings_and_comments(code)
        for pat in banned:
            assert not re.search(pat, code_only), f"banned {pat} in {src}"
