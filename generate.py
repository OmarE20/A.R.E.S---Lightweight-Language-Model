"""Inference / text generation for ARES.

Loads a checkpoint (which carries its own config and tokenizer), encodes a
prompt, autoregressively samples a continuation with temperature (and optional
top-k) control, and prints the decoded text.

Run:
    python generate.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8
    python generate.py --prompt "To be" --top_k 40 --temperature 0.9
"""

from __future__ import annotations

import argparse
import os

import torch

from config import Config
from data import load_tokenizer
from model import GPT


def load_model(ckpt_path: str, device: str):
    # weights_only=False: our checkpoints carry the config dict + loss history,
    # not just tensors. Only load checkpoints you trust/produced.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def main():
    parser = argparse.ArgumentParser(description="Generate text with ARES")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.pkl")
    parser.add_argument("--prompt", type=str, default="\n",
                        help="text prompt to condition on")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help=">0 sample; lower=greedier; 0=argmax")
    parser.add_argument("--top_k", type=int, default=None,
                        help="restrict sampling to the k most likely tokens")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint}. Train first: `python train.py`."
        )

    model, cfg, _ = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    ids = tokenizer.encode(args.prompt)
    if len(ids) == 0:
        ids = tokenizer.encode("\n")  # fall back so we always have a seed token
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    print(f"# model: {cfg.n_layers}L {cfg.d_model}d {cfg.n_heads}h | "
          f"tokenizer={cfg.tokenizer} | temp={args.temperature} top_k={args.top_k}")
    for s in range(args.num_samples):
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            generator=gen if device == "cpu" else None,
        )
        text = tokenizer.decode(out[0].tolist())
        print("=" * 60)
        print(f"sample {s + 1}/{args.num_samples}")
        print("=" * 60)
        print(text)


if __name__ == "__main__":
    main()
