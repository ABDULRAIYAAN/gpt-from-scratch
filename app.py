"""
Gradio web app for a trained checkpoint. Runs locally or on a free Hugging Face Space.

Character and BPE checkpoints both work: the tokenizer is rebuilt from the checkpoint,
and generated tokens go through a streaming decoder, so a character whose bytes are
split across two BPE tokens is only shown once it is complete.

Usage:
    pip install gradio
    python app.py                                    # committed character model
    python app.py --ckpt checkpoints/bpe/ckpt.pt     # a BPE model
    GPT_CKPT=checkpoints/bpe/ckpt.pt python app.py   # the same, e.g. on a Space
    # then open http://127.0.0.1:7860
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer, load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "ckpt.pt")
YIELD_EVERY = 8  # send the page an update every 8 new characters


def load(path: str):
    """Model (in eval mode) and tokenizer from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location="cpu")
    tok = load_tokenizer(ckpt)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok


def clean_prompt(tok, prompt: str) -> str:
    """The prompt as text the tokenizer can encode.

    The character model only knows the characters in the plays, so anything else is dropped.
    BPE starts from raw bytes and can encode any text, so the prompt is kept as typed.
    An empty prompt becomes a newline, the start of a fresh line of script.
    """
    if isinstance(tok, CharTokenizer):
        prompt = "".join(c for c in prompt if c in tok.stoi)
    return prompt or "\n"


def write(model: GPT, tok, prompt: str, length: int, temperature: float):
    """Yield the prompt plus the text written so far, until `length` new characters exist.

    Length is counted in characters for both tokenizers, so the slider means the same thing
    whichever model is loaded; a BPE model needs about half as many steps to get there.
    """
    clean = clean_prompt(tok, prompt)
    idx = torch.tensor([tok.encode(clean)], dtype=torch.long)  # (1, T)
    stream = tok.decoder()
    text, written, last_yield = clean, 0, 0
    # Always finishes: at worst a character takes 4 byte tokens (or comes out as U+FFFD).
    while written < length:
        idx = model.generate(idx, 1, temperature=temperature)  # (1, T + 1)
        piece = stream.push(idx[0, -1].item())  # "" while a multi-byte character is unfinished
        text += piece
        written += len(piece)
        if written - last_yield >= YIELD_EVERY:
            last_yield = written
            yield text
    yield text + stream.flush()


def describe(model: GPT, tok) -> str:
    """The Markdown blurb at the top of the page, worked out from the loaded model."""
    n = model.num_params()
    size = f"{n / 1e6:.2f}M" if n >= 100_000 else f"{n / 1e3:.0f}K"  # 816,705 -> "0.82M"
    if isinstance(tok, CharTokenizer):
        how = "one character at a time"
    else:
        how = (f"one token at a time, using {tok.vocab_size:,} byte-pair encoding tokens: "
               f"common pieces of text such as “ the” or “ing”")
    return ("# Tiny Shakespeare GPT\n"
            f"A {size}-parameter Transformer built from scratch in PyTorch and trained on "
            f"Shakespeare's plays. Give it the start of a scene and it writes the rest, {how}.")


def build_demo(model: GPT, tok):
    import gradio as gr  # only needed to serve the page, so the functions above work without it

    def on_click(prompt: str, length: float, temperature: float):
        yield from write(model, tok, prompt, int(length), temperature)

    with gr.Blocks(title="Tiny Shakespeare GPT") as demo:
        gr.Markdown(describe(model, tok))
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Opening lines", value="ROMEO:\n", lines=4)
                length = gr.Slider(100, 2000, value=600, step=100, label="Length (characters)")
                temperature = gr.Slider(0.3, 1.5, value=0.8, step=0.1, label="Temperature")
                go = gr.Button("Write the scene")
            with gr.Column(scale=2):
                output = gr.Textbox(label="Script", lines=24)
        go.click(on_click, [prompt, length, temperature], output, api_name="write")
    return demo


def main():
    p = argparse.ArgumentParser(description="Serve a trained GPT in a Gradio page.")
    p.add_argument("--ckpt", default=os.environ.get("GPT_CKPT", DEFAULT_CKPT))
    args = p.parse_args()
    build_demo(*load(args.ckpt)).launch()


if __name__ == "__main__":
    main()
