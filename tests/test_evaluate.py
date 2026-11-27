import math
import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from data import DEFAULT_PATH, split_text
from evaluate import context_buckets, evaluate_text, load_checkpoint, loss_by_context, nll_by_context, total_nll
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


# --- loss by context length -------------------------------------------------------------


@torch.no_grad()
def nll_per_context_reference(model, ids, context_of):
    """Score target t from exactly context_of(t) tokens before it, one forward pass each,
    and add the nats up by context length."""
    T = model.config.block_size
    nll, count = torch.zeros(T, dtype=torch.float64), torch.zeros(T, dtype=torch.long)
    for t in range(1, len(ids)):
        c = context_of(t)
        logits, _ = model(ids[t - c : t].unsqueeze(0))
        nll[c - 1] -= F.log_softmax(logits[0, -1], dim=-1)[ids[t]].item()
        count[c - 1] += 1
    return nll, count


@pytest.mark.parametrize("n", [2, 5, 9, 30, 41])
@pytest.mark.parametrize("stride", [1, 3, 8])
def test_loss_by_context_adds_up_to_the_total(n, stride):
    model, ids = tiny_model(), random_ids(n, seed=n)
    nll, count = nll_by_context(model, ids, stride=stride, batch_size=4)
    total, scored = total_nll(model, ids, stride=stride, batch_size=4)
    assert nll.shape == count.shape == (TINY.block_size,)
    assert count.sum().item() == scored == n - 1
    assert nll.sum().item() == pytest.approx(total, rel=1e-6)


@pytest.mark.parametrize("n", [5, 8, 30, 41])
def test_back_to_back_windows_give_each_position_its_own_context(n):
    # With stride = block_size, target t sits at position (t-1) % T of its window.
    model, ids = tiny_model(), random_ids(n, seed=n)
    T = TINY.block_size
    nll, count = nll_by_context(model, ids, batch_size=3)
    ref_nll, ref_count = nll_per_context_reference(model, ids, lambda t: (t - 1) % T + 1)
    assert torch.equal(count, ref_count)
    assert torch.allclose(nll, ref_nll, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n", [5, 30])
def test_stride_one_gives_each_target_all_the_context_it_can(n):
    model, ids = tiny_model(), random_ids(n, seed=n)
    T = TINY.block_size
    nll, count = nll_by_context(model, ids, stride=1, batch_size=5)
    ref_nll, ref_count = nll_per_context_reference(model, ids, lambda t: min(t, T))
    assert torch.equal(count, ref_count)
    assert torch.allclose(nll, ref_nll, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("T", [1, 2, 5, 8, 64, 100, 256])
def test_context_buckets_cover_every_length_once(T):
    buckets = context_buckets(T)
    lengths = [c for lo, hi in buckets for c in range(lo, hi + 1)]
    assert lengths == list(range(1, T + 1))
    assert [hi for _, hi in buckets] == [min(2 ** k, T) for k in range(len(buckets))]  # 1, 2, 4, 8, ... T


def test_loss_by_context_rows_average_to_evaluate_text():
    text = "".join(chr(97 + (i * 7 + i // 5) % 11) for i in range(300))
    tok = CharTokenizer.from_text(text)
    model = tiny_model()
    for stride in (None, 3):
        rows = loss_by_context(model, tok, text, stride=stride)
        r = evaluate_text(model, tok, text, stride=stride)
        assert sum(row["targets"] for row in rows) == r["tokens"]
        assert sum(row["share"] for row in rows) == pytest.approx(1.0)
        assert all(row["targets"] > 0 for row in rows)  # empty buckets are left out
        avg = sum(row["loss"] * row["targets"] for row in rows) / r["tokens"]
        assert avg == pytest.approx(r["loss"], rel=1e-6)
        assert all(row["bits_per_token"] == pytest.approx(row["loss"] / math.log(2)) for row in rows)


def test_loss_by_context_leaves_the_model_mode_alone():
    model = tiny_model().train()
    nll_by_context(model, random_ids(20))
    assert model.training


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(DEFAULT_PATH)),
                    reason="trained checkpoint or dataset not present")
def test_committed_checkpoint_does_better_with_more_context():
    model, tok = load_checkpoint(CKPT, "cpu")
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        _, val_text = split_text(f.read())
    rows = loss_by_context(model, tok, val_text)
    bits = [row["bits_per_token"] for row in rows]
    assert [row["context"] for row in rows] == context_buckets(64)
    # One character of context is worth far less than 33-64; the first few buckets fall steadily.
    assert bits[0] > bits[1] > bits[2] > bits[3] > bits[-1]
    assert bits[0] - bits[-1] > 1.0

    # The command line prints the same table.
    out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluate.py"), CKPT, "--by-context",
                          "--device", "cpu"], capture_output=True, text=True, check=True, cwd=ROOT).stdout
    assert "by context length (chars the prediction could see)" in out
    assert f"| 33-64 | {rows[-1]['targets']:,} |" in out
    assert f"{rows[0]['bits_per_token']:.3f}" in out
