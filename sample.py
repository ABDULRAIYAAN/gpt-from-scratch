"""
Generate text from a trained checkpoint.

Usage:
    python sample.py
    python sample.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
"""

import argparse

import torch

from data import pick_device
from model import GPT, GPTConfig
from tokenizer import load_tokenizer


def main():
    p = argparse.ArgumentParser(description="Sample from a trained GPT.")
    p.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0, help="<1 = safer, >1 = wilder")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    tok = load_tokenizer(ckpt)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from step {ckpt['step']} (val loss {ckpt['val_loss']:.4f})\n")

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
        print(tok.decode(out[0].tolist()))
        if i < args.num_samples - 1:
            print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
