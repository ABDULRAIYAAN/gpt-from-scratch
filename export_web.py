"""
Build the browser demo (docs/index.html) from a trained checkpoint.

The page embeds the weights as float16 and the tokenizer state, and runs
the model in JavaScript (web/engine.js, with web/tokenizer.js), so it works
as a static site: open it locally or host the docs/ folder on GitHub Pages.
Character-level and BPE checkpoints both work, and the page's wording and spec
table are filled in for whichever one it is (characters or tokens).

Usage:
    python export_web.py
    python export_web.py --ckpt checkpoints/ckpt.pt --out docs/index.html
    python export_web.py --ckpt checkpoints/bpe/ckpt.pt --out docs/bpe.html
"""

import argparse
import base64
import json
import os
from dataclasses import asdict

import numpy as np
import torch

from data import DEFAULT_PATH, load_dataset, split_text
from evaluate import evaluate_text
from model import GPT, GPTConfig
from tokenizer import load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
COMMITTED_CKPT = os.path.join(HERE, "checkpoints", "ckpt.pt")
# The committed checkpoint was saved before train.py recorded how long it trained and on what.
COMMITTED_TRAINING = "5,000 steps, 10 min, 2-core CPU"

# Reference numbers the page compares against (see the README's results tables).
BIGRAM_VAL_LOSS = 2.49        # bigram.py, nats per character
CHAR_BITS_PER_CHAR = 2.327    # character model, evaluate.py on the full validation split
GPT3_PARAMS = 175e9
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def weights_blob(model: GPT, tokenizer: dict, step: int, val_loss: float) -> dict:
    """Pack a model's weights as float16 base64 tensors for web/engine.js.

    tokenizer is the state tokenizer.py saves ({"type": "char", "chars": [...]} or
    {"type": "bpe", "merges": [...]}); web/tokenizer.js rebuilds the same tokenizer from it.
    """
    tensors = {}
    for k, v in model.state_dict().items():
        if k.endswith("tril"):  # the causal mask is rebuilt in JS, no need to ship it
            continue
        if k == "lm_head.weight" and model.config.tie_weights:
            continue  # the same matrix as token_embedding_table.weight; engine.js reuses that one
        a = v.detach().float().cpu().numpy().astype(np.float16)
        tensors[k] = {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}
    return {
        "config": asdict(model.config),
        "tokenizer": tokenizer,
        "step": step,
        "val_loss": val_loss,
        "tensors": tensors,
    }


def training_summary(ckpt: dict) -> str:
    """'5,000 steps, 11 min, 2-core CPU' from what train.py saves in a checkpoint.

    minutes is the time when the checkpoint was saved, so it only describes the whole run
    when the best step was the last one; otherwise it is left out.
    """
    last = ckpt["history"][-1]["step"] if ckpt.get("history") else ckpt["step"]
    steps = ckpt.get("max_iters", last)
    parts = [f"{steps:,} steps"]
    if "minutes" in ckpt and ckpt["step"] == steps:
        parts.append(f"{ckpt['minutes']:.0f} min")
    if "hardware" in ckpt:
        parts.append(ckpt["hardware"])
    return ", ".join(parts)


def page_text(model: GPT, tok_state: dict, ckpt: dict, training: str, val_stats: dict | None) -> dict:
    """The page's wording and spec table for this checkpoint, keyed by template placeholder.

    The character model's text is exactly what the page always said; a BPE model talks
    about tokens instead, and gets its context in characters and its bits per character
    from val_stats (evaluate_text on the validation split) when the dataset is available.
    """
    cfg = model.config
    bpe = tok_state["type"] == "bpe"
    unit = "token" if bpe else "character"
    T, V = cfg.block_size, cfg.vocab_size
    n_params = model.num_params()
    approx_chars = f"about {round(T * val_stats['chars_per_token'])} characters" if bpe and val_stats else None

    if bpe:
        n_merges = len(tok_state["merges"])
        lede = ("Give it the start of a scene and it writes the rest, one token at a time: "
                "a token is a common piece of text, such as \u201c the\u201d or \u201cing\u201d. "
                "Watch which tokens it is weighing up, and which earlier tokens it is paying "
                "attention to, or pick the next token yourself.")
        window = f"The {T} tokens it can see" + (f", {approx_chars}" if approx_chars else "") + "."
        memory = f"{T} tokens" + (f" ({approx_chars})" if approx_chars else "")
        vocab_note = (f"<strong>Reads bytes:</strong> it can take in any text, emoji included, but "
                      f"it only ever saw the characters in the plays, so it rarely writes anything else.")
        each_step = (f"Each step, the model looks at the last {T} tokens and scores all {V:,} tokens it knows: "
                     f"the 256 possible bytes, plus {n_merges} pieces such as \u201c the\u201d that byte-pair "
                     f"encoding learned by merging the most common pairs in the plays.")
    else:
        lede = ("Give it the start of a scene and it writes the rest, one character at a time. "
                "Watch which letters it is weighing up, and which earlier characters it is paying "
                "attention to, or pick the next letter yourself.")
        window = f"The {T} characters it can see."
        memory = f"{T} characters"
        vocab_note = (f"<strong>Only knows {V} characters:</strong> most digits, symbols like @ # + / "
                      f"and emoji are dropped.")
        each_step = f"Each step, the model looks at the last {T} characters and scores all {V} characters it knows."

    specs = [
        ("Parameters", f"{n_params:,}"),
        ("Architecture", f"{cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd}-wide"),
        ("Context", f"{T} tokens, ~{round(T * val_stats['chars_per_token'])} characters"
         if bpe and val_stats else f"{T} {unit}s"),
        ("Vocabulary", f"{V:,} {unit}s"),
        ("Training text", "1.1M characters"),
        ("Training", training),
    ]
    if bpe:
        specs.append(("Validation loss", f"{ckpt['val_loss']:.2f} per token"))
        if val_stats:
            specs.append(("Bits per character", f"{val_stats['bits_per_char']:.2f} (characters {CHAR_BITS_PER_CHAR:.2f})"))
    else:
        specs.append(("Validation loss", f"{ckpt['val_loss']:.2f} (bigram {BIGRAM_VAL_LOSS:.2f})"))

    return {
        "unit": unit,
        "lede": lede,
        "main_attr": ' class="tokens"' if bpe else "",
        "next_button": "Next token" if bpe else "Next letter",
        "lens_sub": (f"{window} Brighter means it paid more attention to that {unit} for this guess, "
                     f"averaged over all {cfg.n_layer * cfg.n_head} attention heads."),
        "gpt3_ratio": f"{round(GPT3_PARAMS / n_params, -3):,.0f}",
        "memory": memory,
        "vocab_note": vocab_note,
        "each_step": each_step,
        "n_blocks": NUMBER_WORDS[cfg.n_layer] if cfg.n_layer < len(NUMBER_WORDS) else str(cfg.n_layer),
        "specs": "\n".join(f"        <dt>{k}</dt><dd>{v}</dd>" for k, v in specs),
    }


def fill_template(page: str, fields: dict) -> str:
    """Put the page_text() wording into web/template.html's {{placeholders}}."""
    for key, value in fields.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        raise ValueError(f"unfilled placeholder in web/template.html near {page[page.index('{{'):][:40]!r}")
    return page


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=COMMITTED_CKPT)
    p.add_argument("--out", default=os.path.join(HERE, "docs", "index.html"))
    p.add_argument("--prompt", default="ROMEO:\n", help="prompt for the sample shown on load")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--data", default=DEFAULT_PATH, help="training text, for a BPE model's bits per character")
    p.add_argument("--training", default=None,
                   help='the spec table\'s training row, e.g. "5,000 steps, 10 min, 2-core CPU" '
                        "(default: built from what train.py saved in the checkpoint)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tok = load_tokenizer(ckpt)  # also handles checkpoints that only stored the character list

    # Seed before building the model: construction draws random init weights, so the
    # seed has to come first for the sample below to be the same on every rebuild.
    torch.manual_seed(args.seed)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])  # converts old per-head checkpoints
    model.eval()

    # Export what the loaded model holds, so the page always gets the current layout
    # (fused sa.qkv.weight) even when the checkpoint was saved in the old one.
    weights = weights_blob(model, tok.state(), ckpt["step"], ckpt["val_loss"])

    # Sample shown before the visitor presses Write, ending on a complete speech.
    # Both tokenizers decode back to exactly the prompt, so slicing it off by length works.
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    text = tok.decode(model.generate(idx, 420, temperature=0.8)[0].tolist())[len(args.prompt):]
    if "\n\n" in text:
        text = text[: text.rindex("\n\n")]
    sample = {"prompt": args.prompt, "text": text}

    # A BPE page states its context in characters and its bits per character, measured on
    # the validation split. Skipped (with a note) if the text can't be read or downloaded.
    val_stats = None
    if tok.state()["type"] == "bpe":
        try:
            val_stats = evaluate_text(model, tok, split_text(load_dataset(args.data))[1])
            print(f"validation split: {val_stats['chars_per_token']:.2f} chars/token, "
                  f"{val_stats['bits_per_char']:.3f} bits/char")
        except OSError as e:
            print(f"skipping bits per character, could not read the dataset: {e}")
    training = args.training
    if training is None:
        old = "minutes" not in ckpt and os.path.abspath(args.ckpt) == COMMITTED_CKPT
        training = COMMITTED_TRAINING if old else training_summary(ckpt)
    text_fields = page_text(model, tok.state(), ckpt, training, val_stats)

    def embed(obj):
        return json.dumps(obj).replace("</", "<\\/")

    with open(os.path.join(HERE, "web", "template.html"), encoding="utf-8") as f:
        page = f.read()
    with open(os.path.join(HERE, "web", "tokenizer.js"), encoding="utf-8") as f:
        tokenizer_js = f.read()
    with open(os.path.join(HERE, "web", "engine.js"), encoding="utf-8") as f:
        engine = f.read()
    # Wording first, while the page is only the template; then the scripts; the weights and
    # sample are data and must not be searched for placeholders.
    page = fill_template(page, text_fields)
    page = page.replace("__TOKENIZER__", tokenizer_js).replace("__ENGINE__", engine)
    page = page.replace("__WEIGHTS__", embed(weights)).replace("__SAMPLE__", embed(sample))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
