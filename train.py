"""
Train the GPT on a text file (Tiny Shakespeare by default).

Usage:
    python train.py                      # full model from the video (use a GPU)
    python train.py --config small       # small model that trains on a laptop CPU
    python train.py --data my_text.txt   # train on your own text
    python train.py --config small --tokenizer bpe --bpe-vocab-size 512   # subword tokens
    python train.py --max-iters 2000 --device cuda
    python train.py --config small --lr-schedule cosine --grad-clip 1.0   # warmup, cosine decay, clipping
    python train.py --config small --tie-weights                          # one matrix for embedding and output

Two checkpoints are written to --out: ckpt.pt holds the step with the best estimated validation
loss, final.pt the weights after the last step. When training ends, both are scored on the whole
validation split (the same exact number evaluate.py prints), and the score is saved in each
checkpoint under "val_full". --no-full-eval skips that, e.g. for a very large validation text.
"""

import argparse
import copy
import math
import os
import time
from dataclasses import asdict

import torch

from data import DEFAULT_PATH, get_batch, load_dataset, pick_device, split_text
from evaluate import evaluate_text
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
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="constant: the same learning rate every step; cosine: linear warmup, then cosine decay")
    p.add_argument("--warmup-iters", type=int, default=100,
                   help="steps of linear warmup for --lr-schedule cosine")
    p.add_argument("--min-lr", type=float, default=None,
                   help="learning rate at the last step for --lr-schedule cosine (default: learning rate / 10)")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="clip the gradient's global norm to this value before each step (0 = off)")
    p.add_argument("--tie-weights", action="store_true",
                   help="share one matrix between the token embedding and the output layer")
    p.add_argument("--no-full-eval", action="store_true",
                   help="don't score ckpt.pt and final.pt on the whole validation split at the end")
    # Any preset value can be overridden from the command line.
    for k, v in PRESETS["full"].items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    return p.parse_args()


def get_lr(it, max_iters, learning_rate, schedule="constant", warmup_iters=100, min_lr=None):
    """Learning rate for training step `it` (0 .. max_iters - 1).

    "constant" returns learning_rate every step, which is how the committed checkpoint was trained.
    "cosine" ramps up linearly over the first warmup_iters steps (from learning_rate / warmup_iters
    to learning_rate), then follows half a cosine down to min_lr at the last step:

        lr = min_lr + 0.5 * (1 + cos(pi * progress)) * (learning_rate - min_lr)

    where progress goes from 0 just after warmup to 1 at step max_iters - 1. Warmup keeps the
    first, largest updates from the randomly initialised model small while AdamW's running averages
    settle; the decay takes smaller steps near the end, so the weights settle into a minimum
    instead of bouncing around it.
    """
    if schedule == "constant":
        return learning_rate
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule {schedule!r}")
    if min_lr is None:
        min_lr = learning_rate / 10
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    decay_steps = max(1, max_iters - 1 - warmup_iters)
    progress = min(1.0, (it - warmup_iters) / decay_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


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


def full_validation(model, model_cfg, tok, val_text, out_dir, best_step, last_step, device):
    """Score final.pt and ckpt.pt on every token of val_text and save the score in each file.

    The losses printed during training are estimates over eval_iters random batches, too noisy to
    tell apart two steps a few thousandths of a nat apart, which is how ckpt.pt sometimes keeps an
    earlier step than the last one. evaluate_text scores each validation token exactly once with
    back-to-back block_size windows, so this is the same exact number `python evaluate.py` gives,
    without a separate run. `model` holds the last step's weights; the picked step is loaded back
    from ckpt.pt only when it is a different step.

    Each checkpoint gets a "val_full" entry: {"loss", "bits_per_char", "chars", "stride"}.
    Returns {"final": ..., "best": ...} with the step and the evaluate_text result for each.
    """
    final = {"step": last_step, **evaluate_text(model, tok, val_text)}
    if best_step == last_step:
        best = final  # the same weights, so the same score
    else:
        # A copy rather than GPT(model_cfg): building a new model would draw its random initial
        # weights and move the random state on, which would change the sample written after this.
        picked = copy.deepcopy(model)
        picked.load_state_dict(torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)["model"])
        best = {"step": best_step, **evaluate_text(picked, tok, val_text)}
    for name, r in (("final.pt", final), ("ckpt.pt", best)):
        path = os.path.join(out_dir, name)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt["val_full"] = {
            "loss": r["loss"],                    # nats per token
            "bits_per_char": r["bits_per_char"],  # comparable across tokenizers
            "chars": r["chars"],
            "stride": model_cfg.block_size,       # evaluate.py's default window step
        }
        torch.save(ckpt, path)
    return {"final": final, "best": best}


def full_validation_line(full, n_chars):
    """One line for the training log: the exact validation score of the last and the picked step."""
    f, b = full["final"], full["best"]
    line = (f"full validation split ({n_chars:,} chars): last step {f['step']} "
            f"{f['bits_per_char']:.3f} bits/char (loss {f['loss']:.4f})")
    if b["step"] == f["step"]:
        return line + " | ckpt.pt is the same step"
    return line + (f" | picked step {b['step']} {b['bits_per_char']:.3f} bits/char (loss {b['loss']:.4f})"
                   f" | last - picked {f['bits_per_char'] - b['bits_per_char']:+.3f}")


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
    # Split the text itself (first 90% train, last 10% validation) before tokenizing, so every
    # tokenizer is validated on exactly the same characters and evaluate.py can compare them.
    # For characters this is the same split as before; BPE merges are learned from train text only.
    train_text, val_text = split_text(text)
    if args.tokenizer == "bpe":
        print(f"learning {args.bpe_vocab_size - 256} BPE merges ...")
        tok = BPETokenizer.train(train_text, args.bpe_vocab_size)
    else:
        tok = CharTokenizer.from_text(text)
    splits = {
        "train": torch.tensor(tok.encode(train_text), dtype=torch.long),
        "val": torch.tensor(tok.encode(val_text), dtype=torch.long),
    }
    n_tokens = len(splits["train"]) + len(splits["val"])
    print(
        f"dataset: {len(text):,} chars -> {n_tokens:,} tokens ({len(text) / n_tokens:.2f} chars/token) | "
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
        tie_weights=args.tie_weights,
    )
    model = GPT(model_cfg).to(device)
    print(
        f"model: {model.num_params() / 1e6:.2f}M parameters | config: {args.config}"
        + (" | embedding and output weights tied" if args.tie_weights else "")
    )
    print(f"random-guess loss would be ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    schedule = dict(schedule=args.lr_schedule, warmup_iters=args.warmup_iters, min_lr=args.min_lr)
    if args.lr_schedule != "constant" or args.grad_clip > 0:
        print(
            f"lr schedule: {args.lr_schedule} (peak {cfg['learning_rate']:g}"
            + (f", {args.warmup_iters} warmup steps, final {get_lr(cfg['max_iters'] - 1, cfg['max_iters'], cfg['learning_rate'], **schedule):g}"
               if args.lr_schedule == "cosine" else "")
            + f") | grad clip: {args.grad_clip if args.grad_clip > 0 else 'off'}"
        )

    # ---- training loop ----
    if device == "cuda":
        hardware = torch.cuda.get_device_name()
    elif device == "cpu":
        hardware = f"{os.cpu_count()}-core CPU"
    else:
        hardware = device.upper()

    def save_checkpoint(path, step, val_loss, elapsed):
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": asdict(model_cfg),
                "tokenizer": tok.state(),
                # the character list on its own too, as checkpoints from before BPE support stored it
                **({"chars": tok.chars} if args.tokenizer == "char" else {}),
                "step": step,
                "val_loss": val_loss,       # the estimated val loss at this step
                "history": history,
                # how long and where it trained, for the browser demo's spec table
                "max_iters": cfg["max_iters"],
                "minutes": elapsed / 60,
                "hardware": hardware,
                # the optimisation settings, so runs with different schedules can be told apart
                "learning_rate": cfg["learning_rate"],
                "lr_schedule": args.lr_schedule,
                "warmup_iters": args.warmup_iters if args.lr_schedule == "cosine" else 0,
                "grad_clip": args.grad_clip,
                # the random seed, so seeds.py can tell repeat runs of one recipe apart
                "seed": args.seed,
            },
            path,
        )

    best_val = float("inf")
    best_step = 0
    history = []
    grad_norms = []  # global gradient norm (before clipping) of each step since the last eval
    t0 = time.time()
    for it in range(cfg["max_iters"] + 1):
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
            losses = estimate_loss(model, splits, cfg, device)
            elapsed = time.time() - t0
            extra = ""
            if args.lr_schedule != "constant" and it > 0:
                extra = f" | lr {optimizer.param_groups[0]['lr']:.2e}"  # the rate the last step used
            if grad_norms:
                extra += f" | grad norm {sum(grad_norms) / len(grad_norms):.2f}"
            print(
                f"step {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed / 60:.1f} min{extra}"
            )
            history.append({"step": it, **losses})
            grad_norms = []
            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = it
                save_checkpoint(os.path.join(args.out, "ckpt.pt"), it, best_val, elapsed)
            if it == cfg["max_iters"]:
                # Also keep the weights from the last step. ckpt.pt is picked by an estimate of the
                # validation loss over eval_iters random batches, which can't reliably tell apart
                # two evals 0.005 nats apart, so it sometimes keeps an earlier step. With a decaying
                # learning rate the last step should be the better model; evaluate.py can score
                # both on the whole validation split (seeds.py --pick both does it for a sweep).
                save_checkpoint(os.path.join(args.out, "final.pt"), it, losses["val"], elapsed)
                break

        for group in optimizer.param_groups:
            group["lr"] = get_lr(it, cfg["max_iters"], cfg["learning_rate"], **schedule)
        xb, yb = get_batch(splits["train"], cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            # Rescale every gradient by the same factor so their combined L2 norm (all parameters
            # flattened into one vector) is at most grad_clip; the direction is unchanged.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None])
            )
        grad_norms.append(norm.item())
        optimizer.step()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min | best val loss {best_val:.4f} (step {best_step})")
    print(f"checkpoint saved to {os.path.join(args.out, 'ckpt.pt')} | last step saved to {os.path.join(args.out, 'final.pt')}")

    # ---- score both checkpoints on the whole validation split ----
    if not args.no_full_eval:
        full = full_validation(model, model_cfg, tok, val_text, args.out, best_step, cfg["max_iters"], device)
        print(full_validation_line(full, len(val_text)))

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
