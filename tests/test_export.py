"""Check the wording export_web.py puts into the demo page for each tokenizer."""

import os
import re

import torch

from export_web import COMMITTED_CKPT, COMMITTED_TRAINING, fill_template, page_text, training_summary
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = open(os.path.join(ROOT, "web", "template.html"), encoding="utf-8").read()

TEXT = ("ROMEO:\nBut, soft! what light through yonder window breaks?\n"
        "It is the east, and Juliet is the sun.\n\n") * 4


def committed():
    ckpt = torch.load(COMMITTED_CKPT, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model, load_tokenizer(ckpt), ckpt


def bpe_model():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_embd=16, n_head=2, n_layer=3, dropout=0.0)
    ckpt = {"step": 5000, "val_loss": 2.8381, "max_iters": 5000, "minutes": 11.4, "hardware": "2-core CPU",
            "history": [{"step": 5000}]}
    return GPT(config), tok, ckpt


STATS = {"chars_per_token": 1.88, "bits_per_char": 2.1771}


def test_every_placeholder_is_filled():
    """page_text() fills exactly the placeholders the template has, for both tokenizers."""
    in_template = set(re.findall(r"\{\{(\w+)\}\}", TEMPLATE))
    model, tok, ckpt = committed()
    assert set(page_text(model, tok.state(), ckpt, COMMITTED_TRAINING, None)) == in_template
    model, tok, ckpt = bpe_model()
    for stats in (STATS, None):
        page = fill_template(TEMPLATE, page_text(model, tok.state(), ckpt, training_summary(ckpt), stats))
        assert "{{" not in page


def test_character_page_keeps_its_wording():
    """The committed model's page says exactly what it always did."""
    model, tok, ckpt = committed()
    page = fill_template(TEMPLATE, page_text(model, tok.state(), ckpt, COMMITTED_TRAINING, None))
    for text in [
        "one character at a time. Watch which letters it is weighing up",
        '<button type="button" id="one">Next letter</button>',
        "The 64 characters it can see. Brighter means it paid more attention to that character "
        "for this guess, averaged over all 16 attention heads.",
        "about 214,000 times smaller than GPT-3",
        "or remember more than 64 characters.",
        "<strong>Only knows 65 characters:</strong> most digits, symbols like @ # + / and emoji are dropped.",
        "Each step, the model looks at the last 64 characters and scores all 65 characters it knows.",
        "Inside, four Transformer blocks let every character attend to the characters before it.",
        "<dt>Parameters</dt><dd>816,705</dd>",
        "<dt>Context</dt><dd>64 characters</dd>",
        "<dt>Training</dt><dd>5,000 steps, 10 min, 2-core CPU</dd>",
        "<dt>Validation loss</dt><dd>1.60 (bigram 2.49)</dd>",
        "<main>",
    ]:
        assert text in page, text
    body = page.split("<body>")[1].split("<script")[0]
    assert "token" not in body.lower()


def test_bpe_page_talks_about_tokens():
    model, tok, ckpt = bpe_model()
    page = fill_template(TEMPLATE, page_text(model, tok.state(), ckpt, training_summary(ckpt), STATS))
    for text in [
        '<main class="tokens">',
        '<button type="button" id="one">Next token</button>',
        "The model's top guesses for the next token",
        "The 64 tokens it can see, about 120 characters. Brighter means it paid more attention "
        "to that token for this guess, averaged over all 6 attention heads.",
        "or remember more than 64 tokens (about 120 characters).",
        f"scores all 300 tokens it knows: the 256 possible bytes, plus {len(tok.merges)} pieces",
        "Inside, three Transformer blocks let every token attend to the tokens before it.",
        "<dt>Context</dt><dd>64 tokens, ~120 characters</dd>",
        "<dt>Vocabulary</dt><dd>300 tokens</dd>",
        "<dt>Training</dt><dd>5,000 steps, 11 min, 2-core CPU</dd>",
        "<dt>Validation loss</dt><dd>2.84 per token</dd>",
        "<dt>Bits per character</dt><dd>2.18 (characters 2.33)</dd>",
    ]:
        assert text in page, text
    # nothing left over from the character model's wording
    html = page.split("<body>")[1].split("<script")[0]
    for old in ["Next letter", "next character", "65 characters", "bigram", "every character attend"]:
        assert old not in html, old


def test_bpe_page_without_the_dataset():
    """With no validation stats, the page leaves out the numbers that need them."""
    model, tok, ckpt = bpe_model()
    page = fill_template(TEMPLATE, page_text(model, tok.state(), ckpt, training_summary(ckpt), None))
    assert "The 64 tokens it can see. Brighter" in page
    assert "<dt>Context</dt><dd>64 tokens</dd>" in page
    assert "Bits per character" not in page and "about None" not in page


def test_training_summary():
    ckpt = {"step": 5000, "max_iters": 5000, "minutes": 11.4, "hardware": "2-core CPU"}
    assert training_summary(ckpt) == "5,000 steps, 11 min, 2-core CPU"
    # saved before the last step: the minutes don't cover the whole run, so leave them out
    assert training_summary(dict(ckpt, step=4500)) == "5,000 steps, 2-core CPU"
    # older checkpoints: total steps from the history, nothing else known
    assert training_summary({"step": 4500, "history": [{"step": 0}, {"step": 4500}]}) == "4,500 steps"
