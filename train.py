"""
Train the GPT on a text file (Tiny Shakespeare by default).

Usage:
    python train.py                      # full model from the video (use a GPU)
    python train.py --config small       # small model that trains on a laptop CPU
    python train.py --data my_text.txt   # train on your own text
    python train.py --config small --tokenizer bpe --bpe-vocab-size 512   # subword tokens
    python train.py --max-iters 2000 --device cuda
"""

import argparse
import math
import os
import time
from dataclasses import asdict

import torch

from data import DEFAULT_PATH, get_batch, load_dataset, pick_device
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer

# Two presets. "full" matches the hyperparameters in the video (~10.8M params).
PRESETS = {
    "full": dict(
        batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2,
        max_iters=5000, eval_interval=500, eval_iters=200, learning_rate=3e-4,
    ),
    "small": dict(
        batch_size=32, block_size=64, n_embd=128, n_head=4, n_layer=4, dropout=0.1,
        max_iters=3000, eval_interval=250, eval_iters=50, learning_rate=1e-3,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a GPT from scratch.")
    p.add_argument("--config", choices=PRESETS.keys(), default="full")
    p.add_argument("--data", default=DEFAULT_PATH, help="path to a UTF-8 text file")
    p.add_argument("--out", default="checkpoints", help="directory for ckpt.pt and samples")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                   help="char: one token per character; bpe: byte-pair encoding learned from the data")
    p.add_argument("--bpe-vocab-size", type=int, default=512, help="vocabulary size for --tokenizer bpe")
    # Any preset value can be overridden from the command line.
    for k, v in PRESETS["full"].items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    """Average loss over many batches, so the number we print is not noisy."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            X, Y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    cfg = dict(PRESETS[args.config])
    for k in cfg:
        override = getattr(args, k)
        if override is not None:
            cfg[k] = override

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # ---- data ----
    text = load_dataset(args.data)
    if args.tokenizer == "bpe":
        print(f"learning {args.bpe_vocab_size - 256} BPE merges ...")
        tok = BPETokenizer.train(text, args.bpe_vocab_size)
    else:
        tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% train, last 10% validation
    splits = {"train": data[:n], "val": data[n:]}
    print(
        f"dataset: {len(text):,} chars -> {len(data):,} tokens ({len(text) / len(data):.2f} chars/token) | "
        f"vocab size: {tok.vocab_size} | tokenizer: {args.tokenizer} | device: {device}"
    )

    # ---- model ----
    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=cfg["dropout"],
    )
    model = GPT(model_cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters | config: {args.config}")
    print(f"random-guess loss would be ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    # ---- training loop ----
    best_val = float("inf")
    history = []
    t0 = time.time()
    for it in range(cfg["max_iters"] + 1):
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
            losses = estimate_loss(model, splits, cfg, device)
            elapsed = time.time() - t0
            print(
                f"step {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed / 60:.1f} min"
            )
            history.append({"step": it, **losses})
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "tokenizer": tok.state(),
                        # app.py and export_web.py read the character list directly
                        **({"chars": tok.chars} if args.tokenizer == "char" else {}),
                        "step": it,
                        "val_loss": best_val,
                        "history": history,
                    },
                    os.path.join(args.out, "ckpt.pt"),
                )
            if it == cfg["max_iters"]:
                break

        xb, yb = get_batch(splits["train"], cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min | best val loss {best_val:.4f}")
    print(f"checkpoint saved to {os.path.join(args.out, 'ckpt.pt')}")

    # ---- sample from the trained model ----
    model.eval()
    start = tok.encode("\n") if "\n" in text else [0]  # start of a new line when the text has one
    context = torch.tensor([start], dtype=torch.long, device=device)
    sample = tok.decode(model.generate(context, max_new_tokens=1000)[0].tolist())
    with open(os.path.join(args.out, "sample.txt"), "w", encoding="utf-8") as f:
        f.write(sample)
    print("\n--- sample ---")
    print(sample[:600])


if __name__ == "__main__":
    main()
