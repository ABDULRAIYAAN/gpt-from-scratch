"""Check the Gradio app's generation loop with character and BPE checkpoints, without Gradio."""

import random
import sys

import pytest
import torch

import app
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer

TEXT = ("ROMEO:\nBut, soft! what light through yonder window breaks?\n"
        "It is the east, and Juliet is the sun. Café à la crème — \U0001F339\n\n") * 4


@pytest.fixture(scope="module")
def committed():
    return app.load(app.DEFAULT_CKPT)


def tiny_bpe():
    torch.manual_seed(0)
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_embd=16, n_head=2, n_layer=2, dropout=0.0)
    return GPT(config).eval(), tok


def test_app_imports_without_gradio():
    """The generation code can be imported and tested without Gradio installed."""
    assert "gradio" not in sys.modules


def test_clean_prompt():
    chars = CharTokenizer.from_text("ROMEO:\n abc")
    assert app.clean_prompt(chars, "ROMEO: \U0001F339@#") == "ROMEO: "
    assert app.clean_prompt(chars, "\U0001F339") == "\n"
    assert app.clean_prompt(chars, "") == "\n"
    _, bpe = tiny_bpe()
    assert app.clean_prompt(bpe, "ROMEO: \U0001F339@#") == "ROMEO: \U0001F339@#"  # bytes, so nothing is dropped
    assert app.clean_prompt(bpe, "") == "\n"


@pytest.mark.parametrize("make", ["char", "bpe"])
def test_stream_decoder_matches_decode(make):
    """Pushing ids one at a time and flushing gives exactly decode(ids), even for broken UTF-8."""
    tok = CharTokenizer.from_text(TEXT) if make == "char" else tiny_bpe()[1]
    rng = random.Random(0)
    for _ in range(300):
        ids = [rng.randrange(tok.vocab_size) for _ in range(rng.randrange(0, 40))]
        d = tok.decoder()
        assert "".join(d.push(i) for i in ids) + d.flush() == tok.decode(ids)


def test_stream_decoder_holds_back_split_characters():
    """A character split over several byte tokens comes out whole, never as U+FFFD."""
    tok = BPETokenizer([])  # bytes only, so every multi-byte character is split
    d = tok.decoder()
    pieces = [d.push(i) for i in tok.encode("a\U0001F339é")]
    assert pieces == ["a", "", "", "", "\U0001F339", "", "é"]
    assert d.flush() == ""


def test_character_app_writes_what_it_always_did(committed):
    """Same seed, same text as the old character-only loop: one token per character."""
    model, tok = committed
    torch.manual_seed(1)
    *updates, final = app.write(model, tok, "ROMEO:\n", 200, 0.8)

    torch.manual_seed(1)
    idx = torch.tensor([tok.encode("ROMEO:\n")], dtype=torch.long)
    old = "ROMEO:\n"
    for _ in range(200):
        idx = model.generate(idx, 1, temperature=0.8)
        old += tok.itos[idx[0, -1].item()]
    assert final == old
    assert all(final.startswith(u) for u in updates)
    assert [len(u) - 7 for u in updates] == list(range(8, 201, 8))


def test_bpe_app_writes_decoded_tokens():
    """A BPE model's output is the prompt plus decode() of the sampled tokens, at least `length` characters."""
    model, tok = tiny_bpe()
    prompt = "Juliet \U0001F339"
    torch.manual_seed(2)
    *updates, final = app.write(model, tok, prompt, 120, 1.0)
    assert final.startswith(prompt) and len(final) - len(prompt) >= 120
    assert all(final.startswith(u) for u in updates)

    # The same seed draws the same tokens, so one of their prefixes must decode to the output.
    torch.manual_seed(2)
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    new = model.generate(idx, 4 * 120, temperature=1.0)[0, idx.size(1):].tolist()
    assert any(prompt + tok.decode(new[:n]) == final for n in range(1, len(new) + 1))


def test_describe(committed):
    model, tok = committed
    assert "0.82M-parameter" in app.describe(model, tok)
    assert "one character at a time" in app.describe(model, tok)
    model, tok = tiny_bpe()
    text = app.describe(model, tok)
    assert "one token at a time" in text and "300 byte-pair encoding tokens" in text
    assert "character at a time" not in text
