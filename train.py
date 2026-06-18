"""Training loop for ARES.

Implements next-token cross-entropy training with AdamW, gradient clipping, a
warmup+cosine LR schedule, periodic *multi-batch* val-loss estimation (to reduce
noise), checkpointing (model + optimizer + config + tokenizer, so runs resume),
and a Matplotlib loss curve saved to plots/.

Run:
    python train.py                      # defaults from config.py
    python train.py --max_iters 500 --tokenizer char
    python train.py --resume             # continue from latest checkpoint
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from config import Config
from data import Dataset, save_tokenizer
from model import GPT


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay to min_lr
# ---------------------------------------------------------------------------
def get_lr(it: int, cfg: Config) -> float:
    if not cfg.lr_decay:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / max(1, cfg.warmup_iters)
    if it >= cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Multi-batch val/train loss estimate (averaged to de-noise)
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, dataset, cfg, device, generator):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = dataset.get_batch(split, cfg.batch_size, device, generator)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, cfg, iter_num, best_val, history):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val,
            "history": history,  # for resuming the loss-curve plot
        },
        path,
    )


def plot_losses(history, out_path, title):
    import matplotlib

    matplotlib.use("Agg")  # headless backend; no display needed
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history["iters"], history["train"], label="train loss")
    plt.plot(history["iters"], history["val"], label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def add_cli_overrides(parser: argparse.ArgumentParser) -> None:
    """Expose every Config field as an optional CLI flag (typed)."""
    defaults = Config()
    for f, fdef in Config.__dataclass_fields__.items():  # type: ignore[attr-defined]
        cur = getattr(defaults, f)
        if isinstance(cur, bool):
            parser.add_argument(f"--{f}", type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=None)
        elif isinstance(cur, int) and not isinstance(cur, bool):
            parser.add_argument(f"--{f}", type=int, default=None)
        elif isinstance(cur, float):
            parser.add_argument(f"--{f}", type=float, default=None)
        else:
            parser.add_argument(f"--{f}", type=str, default=None)


def build_config_from_args(args) -> Config:
    cfg = Config()
    for f in Config.__dataclass_fields__:  # type: ignore[attr-defined]
        v = getattr(args, f, None)
        if v is not None:
            setattr(cfg, f, v)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train ARES")
    parser.add_argument("--resume", action="store_true", help="resume from latest ckpt")
    add_cli_overrides(parser)
    args = parser.parse_args()

    cfg = build_config_from_args(args)
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)

    # Dedicated RNG for batch sampling so seeding is deterministic & isolated.
    gen = torch.Generator().manual_seed(cfg.seed)

    # ---- Data (also resolves cfg.vocab_size) -------------------------------
    dataset = Dataset(cfg)
    save_tokenizer(dataset.tokenizer, os.path.join(cfg.checkpoint_dir, "tokenizer.pkl"))

    # ---- Model + optimizer -------------------------------------------------
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizers(cfg)

    iter_num = 0
    best_val = float("inf")
    history = {"iters": [], "train": [], "val": []}

    ckpt_path = os.path.join(cfg.checkpoint_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        iter_num = ckpt["iter_num"]
        best_val = ckpt["best_val_loss"]
        history = ckpt.get("history", history)

    print("=" * 60)
    print("ARES training run")
    print(f"device={device}  params={model.num_params():,}")
    print(cfg)
    print("=" * 60)

    model.train()
    t0 = time.time()
    running_loss = None

    while iter_num <= cfg.max_iters:
        # Set this step's learning rate.
        lr = get_lr(iter_num, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---- Periodic evaluation + checkpoint ------------------------------
        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss(model, dataset, cfg, device, gen)
            history["iters"].append(iter_num)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(
                f"iter {iter_num:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | "
                f"{time.time() - t0:.1f}s"
            )
            if losses["val"] < best_val or cfg.always_save_checkpoint:
                best_val = min(best_val, losses["val"])
                save_checkpoint(ckpt_path, model, optimizer, cfg, iter_num, best_val, history)

        if iter_num == cfg.max_iters:
            break

        # ---- One optimization step -----------------------------------------
        xb, yb = dataset.get_batch("train", cfg.batch_size, device, gen)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping: rescale grads so their global L2 norm <= grad_clip.
        # Guards against the occasional exploding-gradient step on small data.
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        running_loss = loss.item() if running_loss is None else (
            0.9 * running_loss + 0.1 * loss.item()
        )
        if iter_num % cfg.log_interval == 0:
            print(f"  step {iter_num:5d} | loss {loss.item():.4f} "
                  f"| ema {running_loss:.4f}")

        iter_num += 1

    # ---- Final artifacts ---------------------------------------------------
    plot_path = os.path.join(cfg.plot_dir, "loss_curve.png")
    if history["iters"]:
        plot_losses(history, plot_path, f"ARES loss ({cfg.tokenizer}, "
                                        f"{cfg.n_layers}L/{cfg.d_model}d)")
        print(f"Saved loss curve -> {plot_path}")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
