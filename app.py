"""
Gradio web app for the trained model. Runs locally or on a free Hugging Face Space.

Usage:
    pip install gradio
    python app.py            # then open http://127.0.0.1:7860
"""

import os

import gradio as gr
import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "ckpt.pt")

ckpt = torch.load(CKPT, map_location="cpu")
tok = CharTokenizer(ckpt["chars"])
model = GPT(GPTConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()


def write(prompt: str, length: int, temperature: float):
    """Stream generated text back to the page as it is written."""
    clean = "".join(c for c in prompt if c in tok.stoi) or "\n"
    idx = torch.tensor([tok.encode(clean)], dtype=torch.long)
    text = clean
    for _ in range(int(length)):
        idx = model.generate(idx, 1, temperature=temperature)
        text += tok.itos[idx[0, -1].item()]
        if len(text) % 8 == 0:
            yield text
    yield text


with gr.Blocks(title="Tiny Shakespeare GPT") as demo:
    gr.Markdown(
        "# Tiny Shakespeare GPT\n"
        "A 0.82M-parameter Transformer built from scratch in PyTorch and trained on "
        "Shakespeare's plays. Give it the start of a scene and it writes the rest, "
        "one character at a time."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Opening lines", value="ROMEO:\n", lines=4)
            length = gr.Slider(100, 2000, value=600, step=100, label="Length (characters)")
            temperature = gr.Slider(0.3, 1.5, value=0.8, step=0.1, label="Temperature")
            go = gr.Button("Write the scene")
        with gr.Column(scale=2):
            output = gr.Textbox(label="Script", lines=24)
    go.click(write, [prompt, length, temperature], output)

if __name__ == "__main__":
    demo.launch()
