"""
Build the browser demo (docs/index.html) from a trained checkpoint.

The page embeds the weights as float16 and runs the model in JavaScript
(web/engine.js), so it works as a static site: open it locally or host
the docs/ folder on GitHub Pages.

Usage:
    python export_web.py
    python export_web.py --ckpt checkpoints/ckpt.pt --out docs/index.html
"""

import argparse
import base64
import json
import os

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "ckpt.pt"))
    p.add_argument("--out", default=os.path.join(HERE, "docs", "index.html"))
    p.add_argument("--prompt", default="ROMEO:\n", help="prompt for the sample shown on load")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = {k: v for k, v in ckpt["model"].items() if not k.endswith("tril")}
    tensors = {}
    for k, v in state.items():
        a = v.float().numpy().astype(np.float16)
        tensors[k] = {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}
    weights = {
        "config": ckpt["model_config"],
        "chars": ckpt["chars"],
        "step": ckpt["step"],
        "val_loss": ckpt["val_loss"],
        "tensors": tensors,
    }

    # Sample shown before the visitor presses Write, ending on a complete speech.
    torch.manual_seed(args.seed)
    tok = CharTokenizer(ckpt["chars"])
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    text = tok.decode(model.generate(idx, 420, temperature=0.8)[0].tolist())[len(args.prompt):]
    if "\n\n" in text:
        text = text[: text.rindex("\n\n")]
    sample = {"prompt": args.prompt, "text": text}

    def embed(obj):
        return json.dumps(obj).replace("</", "<\\/")

    with open(os.path.join(HERE, "web", "template.html"), encoding="utf-8") as f:
        page = f.read()
    with open(os.path.join(HERE, "web", "engine.js"), encoding="utf-8") as f:
        engine = f.read()
    page = page.replace("__WEIGHTS__", embed(weights)).replace("__SAMPLE__", embed(sample)).replace("__ENGINE__", engine)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
