import math
import os

import pytest
import torch
from torch.nn import functional as F

from data import DEFAULT_PATH, split_text
from evaluate import evaluate_text, load_checkpoint, total_nll
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

TINY = GPTConfig(vocab_size=11, block_size=8, n_embd=16, n_head=2, n_layer=2, dropout=0.0)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(TINY).eval()


def random_ids(n, seed=0):
    return torch.randint(TINY.vocab_size, (n,), generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def nll_one_at_a_time(model, ids):
    """Reference: predict each token from up to block_size tokens before it, one forward pass each."""
    T, total = model.config.block_size, 0.0
    for t in range(1, len(ids)):
        context = ids[max(0, t - T) : t].unsqueeze(0)
        logits, _ = model(context)
        total -= F.log_softmax(logits[0, -1], dim=-1)[ids[t]].item()
    return total


@torch.no_grad()
def nll_back_to_back(model, ids):
    """Reference for the default stride: consecutive non-overlapping windows, every position scored."""
    T, total = model.config.block_size, 0.0
    for begin in range(0, len(ids) - 1, T):
        x = ids[begin : begin + T][: len(ids) - 1 - begin]
        y = ids[begin + 1 : begin + 1 + len(x)]
        logits, _ = model(x.unsqueeze(0))
        total += F.cross_entropy(logits[0], y, reduction="sum").item()
    return total


@pytest.mark.parametrize("n", [2, 5, 8, 9, 30, 41])
@pytest.mark.parametrize("stride", [1, 3, 8])
def test_every_token_is_scored_once(n, stride):
    nll, count = total_nll(tiny_model(), random_ids(n), stride=stride, batch_size=4)
    assert count == n - 1 and nll > 0


@pytest.mark.parametrize("n", [5, 8, 30, 41])
def test_default_stride_matches_back_to_back_windows(n):
    model, ids = tiny_model(), random_ids(n, seed=n)
    nll, _ = total_nll(model, ids, batch_size=3)
    assert nll == pytest.approx(nll_back_to_back(model, ids), rel=1e-5)


@pytest.mark.parametrize("n", [5, 8, 30, 41])
def test_stride_one_gives_every_token_its_full_context(n):
    model, ids = tiny_model(), random_ids(n, seed=n)
    nll, _ = total_nll(model, ids, stride=1, batch_size=5)
    assert nll == pytest.approx(nll_one_at_a_time(model, ids), rel=1e-5)


def test_stride_must_fit_in_the_context_window():
    model, ids = tiny_model(), random_ids(50)
    with pytest.raises(ValueError):
        total_nll(model, ids, stride=0)
    with pytest.raises(ValueError):
        total_nll(model, ids, stride=TINY.block_size + 1)


def test_uniform_model_scores_log2_vocab_bits_per_char():
    # A model whose output layer is all zeros gives every character probability 1/65.
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    tok = CharTokenizer(sorted(set(text)) + [chr(0x100 + i) for i in range(65 - len(set(text)))])
    model = GPT(GPTConfig(vocab_size=65, block_size=16, n_embd=16, n_head=2, n_layer=1, dropout=0.0))
    torch.nn.init.zeros_(model.lm_head.weight)
    torch.nn.init.zeros_(model.lm_head.bias)
    r = evaluate_text(model, tok, text)
    assert r["tokens"] == r["chars"] == len(text) - 1
    assert r["bits_per_char"] == pytest.approx(math.log2(65), rel=1e-6)
    assert r["bits_per_token"] == pytest.approx(r["bits_per_char"])


def test_bits_per_char_divides_by_characters_not_tokens():
    text = "the cat and the hat, then the other thing. " * 20
    tok = BPETokenizer.train(text, 300)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_embd=16, n_head=2, n_layer=1, dropout=0.0))
    r = evaluate_text(model, tok, text)
    ids = tok.encode(text)
    assert r["tokens"] == len(ids) - 1
    assert r["chars"] == len(text) - len(tok.decode(ids[:1]))
    assert r["chars_per_token"] > 1.5  # the merges do compress this text
    assert r["bits_per_char"] == pytest.approx(r["bits_per_token"] * r["tokens"] / r["chars"])


def test_evaluation_leaves_the_model_mode_alone():
    model = tiny_model().train()
    total_nll(model, random_ids(20))
    assert model.training


def test_split_text_matches_the_old_token_split_for_characters():
    # train.py used to split the encoded tokens; for characters, splitting the text is identical.
    text = "".join(chr(97 + (i * 7) % 26) for i in range(1001))
    tok = CharTokenizer.from_text(text)
    data = tok.encode(text)
    n = int(0.9 * len(data))
    train_text, val_text = split_text(text)
    assert tok.encode(train_text) == data[:n] and tok.encode(val_text) == data[n:]
    assert train_text + val_text == text


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(DEFAULT_PATH)),
                    reason="trained checkpoint or dataset not present")
def test_committed_checkpoint_on_the_full_validation_split():
    model, tok = load_checkpoint(CKPT, "cpu")
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        _, val_text = split_text(f.read())
    r = evaluate_text(model, tok, val_text)
    # For characters, bits per character is just the loss in bits.
    assert r["bits_per_char"] == pytest.approx(r["loss"] / math.log(2))
    # Close to the val loss saved at training time, which was estimated from random batches.
    assert r["loss"] == pytest.approx(torch.load(CKPT, map_location="cpu")["val_loss"], abs=0.05)
