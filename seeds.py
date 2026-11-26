"""
Train one recipe with several random seeds and measure how far the result moves.

One training run is one draw. The seed picks the starting weights, the order of the random
training batches and the dropout masks, so the same settings with a different seed give a
slightly different model. A 0.01 bits-per-character gap between two runs only means
something if it is bigger than that seed-to-seed spread.

This script trains the same settings once per seed (by running train.py), scores every
checkpoint on the full validation split with evaluate.py, and prints each run plus the mean,
the sample standard deviation and the range of bits per character.

Usage:
    # everything after "--" goes to train.py; --seed, --out and --data are set here
    python seeds.py --seeds 1 2 3 --out checkpoints/seeds/cosine -- --config small --lr-schedule cosine --max-iters 5000

    # summarise checkpoints that are already trained (e.g. the same recipe from earlier runs)
    python seeds.py --ckpts checkpoints/seeds/cosine/seed1/ckpt.pt checkpoints/cosine/ckpt.pt

    # score the best-estimated step of each run (ckpt.pt) as well as the last step (final.pt)
    python seeds.py --pick both --ckpts checkpoints/seeds/cosine/seed*/ckpt.pt

train.py keeps two checkpoints per run: ckpt.pt, the step whose validation loss looked best on a
small random sample, and final.pt, the weights after the last step. --pick chooses which one is
scored; "both" scores the two and prints how much the choice moved each run. The default is
"final": in every run so far where the two differed, the last step was the better model, and
scoring it keeps the noisy pick out of the comparison. A run trained before final.pt existed only
has ckpt.pt, so that is scored instead, with a note.

Seeds that finished training are not trained again unless --retrain is given, and a seed that
was cut off part-way carries on from its last evaluation (train.py --resume), so an interrupted
sweep can be restarted with the same command.
"""

import argparse
import math
import os
import subprocess
import sys

import torch

from data import DEFAULT_PATH, load_dataset, pick_device, split_text
from evaluate import evaluate_text, load_checkpoint
from train import RESUME_FILE

ROOT = os.path.dirname(os.path.abspath(__file__))
# train.py arguments this script sets itself, so they must not be passed after "--"
RESERVED = ("--seed", "--out", "--data", "--resume", "--stop-at")


def spread(values: list[float]) -> dict:
    """Mean, sample standard deviation (n - 1 in the denominator) and range of values.

    The standard deviation needs at least two runs; with one it is None.
    """
    n = len(values)
    if n == 0:
        raise ValueError("need at least one value")
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "std": std, "min": min(values), "max": max(values)}


def train_seed(seed: int, out_dir: str, train_args: list[str], data: str, retrain: bool = False) -> str:
    """Run train.py with this seed into out_dir and return the checkpoint path.

    train.py's output is printed as it arrives and also saved to out_dir/train_log.txt.

    A run is finished when final.pt exists. train.py writes ckpt.pt at the very first
    evaluation, so ckpt.pt alone doesn't mean the run is done: if resume.pt is there too, the run
    was cut off and is continued from it, and its log is appended to. ckpt.pt with neither file
    is a run from before train.py kept the last step, which is used as it is.
    """
    ckpt = os.path.join(out_dir, "ckpt.pt")
    resume = os.path.join(out_dir, RESUME_FILE)
    finished = os.path.exists(os.path.join(out_dir, "final.pt")) or (
        os.path.exists(ckpt) and not os.path.exists(resume))
    if finished and not retrain:
        print(f"seed {seed}: {ckpt} already exists, not training it again")
        return ckpt
    resuming = os.path.exists(resume) and not retrain
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "train.py"), *train_args,
           "--seed", str(seed), "--out", out_dir, "--data", data]
    if not retrain:
        cmd.append("--resume")  # carries on from resume.pt if a run was cut off, else starts at step 0
    print(f"seed {seed}: {'resuming the interrupted run: ' if resuming else ''}{' '.join(cmd[2:])}")
    with open(os.path.join(out_dir, "train_log.txt"), "a" if resuming else "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        if proc.wait() != 0:
            raise RuntimeError(f"train.py failed for seed {seed} (exit code {proc.returncode})")
    return ckpt


def score(paths: list[str], val_text: str, device: str, stride: int | None = None) -> list[dict]:
    """Bits per character of each checkpoint on val_text, plus the run details it saved."""
    runs = []
    for path in paths:
        model, tok = load_checkpoint(path, device)
        r = evaluate_text(model, tok, val_text, stride)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        runs.append({
            "path": path,
            "seed": saved.get("seed"),   # checkpoints from before this script don't store it
            "step": saved.get("step"),   # the step these weights are from
            **r,
        })
    return runs


def report(runs: list[dict]) -> str:
    """Markdown table of the runs, then the spread of bits per character and loss per token."""
    lines = ["| Checkpoint | Seed | Step | Loss/token | Bits/char |", "|---|---|---|---|---|"]
    for r in runs:
        seed = "?" if r["seed"] is None else r["seed"]
        lines.append(f"| `{r['path']}` | {seed} | {r['step']} | {r['loss']:.4f} | {r['bits_per_char']:.3f} |")
    lines.append("")
    for key, name, digits in (("bits_per_char", "bits/char", 3), ("loss", "loss/token", 4)):
        s = spread([r[key] for r in runs])
        std = "n/a (one run)" if s["std"] is None else f"{s['std']:.{digits}f}"
        lines.append(
            f"{name} over {s['n']} run{'s' if s['n'] > 1 else ''}: mean {s['mean']:.{digits}f} | std {std} | "
            f"range {s['min']:.{digits}f} - {s['max']:.{digits}f} ({s['max'] - s['min']:.{digits}f})"
        )
    return "\n".join(lines)


def final_path(ckpt: str) -> str:
    """The final.pt that train.py writes next to ckpt.pt (the weights after the last step)."""
    path = os.path.join(os.path.dirname(ckpt), "final.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no final.pt next to {ckpt}; it was trained before train.py kept the last step, "
            f"so --pick both can't compare it (--pick best scores its ckpt.pt)"
        )
    return path


def final_or_ckpt(ckpt: str) -> str:
    """final.pt next to ckpt if there is one, else ckpt itself (a run from before final.pt existed)."""
    try:
        return final_path(ckpt)
    except FileNotFoundError:
        print(f"note: no final.pt next to {ckpt} (trained before train.py kept the last step), scoring it instead")
        return ckpt


def compare(best: list[dict], final: list[dict]) -> str:
    """Markdown table of best-estimated step against last step for each run, then the paired difference.

    The two lists hold the same runs in the same order. A negative difference means the last
    step scores fewer bits per character than the step train.py picked.
    """
    lines = ["| Run | Seed | Picked step | Bits/char (picked) | Bits/char (last step) | Last - picked |",
             "|---|---|---|---|---|---|"]
    diffs = []
    for b, f in zip(best, final, strict=True):
        d = f["bits_per_char"] - b["bits_per_char"]
        diffs.append(d)
        seed = "?" if b["seed"] is None else b["seed"]
        lines.append(f"| `{os.path.dirname(b['path']) or '.'}` | {seed} | {b['step']} | {b['bits_per_char']:.3f} | "
                     f"{f['bits_per_char']:.3f} | {d:+.3f} |")
    s = spread(diffs)
    std = "n/a (one run)" if s["std"] is None else f"{s['std']:.3f}"
    better = sum(d < 0 for d in diffs)
    same = sum(b["step"] == f["step"] for b, f in zip(best, final))
    lines += ["", f"last - picked over {s['n']} run{'s' if s['n'] > 1 else ''}: mean {s['mean']:+.3f} | std {std} | "
                  f"last step better in {better}, picked step better in {s['n'] - better - same}, "
                  f"the same step in {same}"]
    return "\n".join(lines)


def main():
    argv = sys.argv[1:]
    # Everything after a bare "--" is passed through to train.py untouched.
    train_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    own_args = argv[: argv.index("--")] if "--" in argv else argv

    p = argparse.ArgumentParser(description="Seed-to-seed spread of bits per character for one training recipe.")
    p.add_argument("--seeds", type=int, nargs="+", help="train once per seed (train.py's default seed is 1337)")
    p.add_argument("--out", default="checkpoints/seeds", help="seed N trains into <out>/seed<N>/")
    p.add_argument("--ckpts", nargs="+", default=[], help="already-trained checkpoints to include")
    p.add_argument("--retrain", action="store_true", help="train a seed again even if its checkpoint exists")
    p.add_argument("--data", default=DEFAULT_PATH, help="text to train on and validate with")
    p.add_argument("--stride", type=int, default=None, help="evaluate.py window step (default: block_size)")
    p.add_argument("--device", default="auto", help="device for scoring (train.py has its own --device)")
    p.add_argument("--pick", choices=["final", "best", "both"], default="final",
                   help="score final.pt (last step, the default), ckpt.pt (best estimated val loss) or both")
    args = p.parse_args(own_args)

    if not args.seeds and not args.ckpts:
        p.error("give --seeds to train, --ckpts to summarise, or both")
    for a in train_args:
        if a.split("=")[0] in RESERVED:
            p.error(f"{a.split('=')[0]} is set by seeds.py; pass it before the '--' instead")

    paths = [train_seed(s, os.path.join(args.out, f"seed{s}"), train_args, args.data, args.retrain)
             for s in (args.seeds or [])]
    paths += args.ckpts

    if args.pick == "final":
        finals = [final_or_ckpt(path) for path in paths]
    elif args.pick == "both":
        finals = [final_path(path) for path in paths]  # comparing needs both files for every run
    _, val_text = split_text(load_dataset(args.data))  # the same characters train.py validates on
    device = pick_device(args.device)
    print(f"\nscoring {len(paths)} run(s) on the validation split ({len(val_text):,} characters)\n")
    if args.pick == "best":
        print(report(score(paths, val_text, device, args.stride)))
    elif args.pick == "final":
        print(report(score(finals, val_text, device, args.stride)))
    else:
        best, final = score(paths, val_text, device, args.stride), score(finals, val_text, device, args.stride)
        print("Best estimated step (ckpt.pt):\n")
        print(report(best))
        print("\nLast step (final.pt):\n")
        print(report(final))
        print("\nPicked step against last step:\n")
        print(compare(best, final))


if __name__ == "__main__":
    main()
