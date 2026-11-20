"""The key/value cache: same logits as the full forward pass, and the same window rule as web/engine.js."""

import os
import shutil

import pytest
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig, KVCache
from test_engine import DIGITS, fp16_model, run_engine
from export_web import weights_blob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

TINY = GPTConfig(vocab_size=11, block_size=16, n_embd=24, n_head=3, n_layer=2, dropout=0.0)


def tiny_model(seed=0, config=TINY):
    torch.manual_seed(seed)
    model = GPT(config).eval()
    with torch.no_grad():  # larger weights than the init, so attention is far from uniform
        for p in model.parameters():
            p.normal_(0, 0.3)
    return model


def random_ids(T, B=2, seed=0, vocab=TINY.vocab_size):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(vocab, (B, T), generator=g)


def test_cached_chunks_match_one_full_pass():
    """Feeding the window in pieces through a cache gives the same logits as all at once."""
    model = tiny_model()
    idx = random_ids(TINY.block_size)  # (2, 16)
    with torch.no_grad():
        full, _ = model(idx)
        cache = KVCache(TINY.n_layer)
        pieces = [model(idx[:, a:b], cache=cache)[0] for a, b in [(0, 5), (5, 6), (6, 9), (9, 10), (10, 16)]]
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=0, atol=1e-5)
    assert len(cache) == TINY.block_size
    assert cache.k[1].shape == (2, TINY.n_head, TINY.block_size, TINY.n_embd // TINY.n_head)


def test_cache_refuses_to_overflow_the_window():
    model = tiny_model()
    cache = KVCache(TINY.n_layer)
    with torch.no_grad():
        model(random_ids(TINY.block_size), cache=cache)
        with pytest.raises(AssertionError, match="block_size"):
            model(random_ids(1), cache=cache)


def test_cached_generate_is_identical_within_the_window():
    """Until the window fills, the cache changes the cost of sampling, not the samples."""
    model = tiny_model(seed=1)
    prompt = random_ids(4, seed=1)
    for top_k in (None, 3):
        torch.manual_seed(5)
        plain = model.generate(prompt, TINY.block_size - 4, temperature=0.9, top_k=top_k)
        torch.manual_seed(5)
        cached = model.generate(prompt, TINY.block_size - 4, temperature=0.9, top_k=top_k, use_cache=True)
        assert torch.equal(cached, plain)
    assert not torch.equal(plain[0], plain[1])  # the two rows really are sampled separately


def reference_context(seq_len, block):
    """Which tokens next_logits looks at after each prefix, written out the simple way.

    Mirrors session() in web/engine.js: start from the last block tokens of the prompt,
    add one token per step, and when the window is full start again from the newest block/2.
    """
    keep = block // 2
    start, length = max(0, 1 - block), 1  # the prompt here is the first token
    contexts = [(start, 1)]
    for end in range(2, seq_len + 1):
        if length >= block:
            start, length = end - keep, keep
        else:
            length += 1
        contexts.append((start, end))
    return contexts


def test_next_logits_follows_the_half_window_rule():
    """Past the window, each step's logits equal a plain forward pass over the context the rule picks."""
    model = tiny_model(seed=2)
    seq = random_ids(50, seed=2)
    cache = KVCache(TINY.n_layer)
    starts = []
    with torch.no_grad():
        for (start, end) in reference_context(50, TINY.block_size):
            got = model.next_logits(seq[:, :end], cache)
            want = model(seq[:, start:end])[0][:, -1]
            torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
            assert 1 <= len(cache) <= TINY.block_size and cache.seen == end
            starts.append(start)
    assert len(set(starts)) > 3  # the cache was rebuilt several times
    # every context after the first window holds between block/2 and block tokens
    assert all(TINY.block_size // 2 <= e - s <= TINY.block_size for s, e in reference_context(50, 16)[16:])


def test_next_logits_with_a_prompt_longer_than_the_window():
    model = tiny_model(seed=3)
    seq = random_ids(TINY.block_size + 7, seed=3)
    cache = KVCache(TINY.n_layer)
    with torch.no_grad():
        first = model.next_logits(seq[:, :-1], cache)  # 22 tokens: only the last 16 fit
        torch.testing.assert_close(first, model(seq[:, -TINY.block_size - 1:-1])[0][:, -1], rtol=0, atol=1e-5)
        second = model.next_logits(seq, cache)  # the window was full: restart on the newest half
        torch.testing.assert_close(second, model(seq[:, -TINY.block_size // 2:])[0][:, -1], rtol=0, atol=1e-5)
        with pytest.raises(AssertionError, match="one new token"):
            model.next_logits(seq, cache)  # nothing new since the last call


def test_cached_generate_past_the_window():
    """Long cached samples are the tokens the half-window rule gives, sampled one by one."""
    model = tiny_model(seed=4)
    prompt = random_ids(3, B=1, seed=4)
    torch.manual_seed(9)
    cached = model.generate(prompt, 45, use_cache=True)
    torch.manual_seed(9)
    seq = prompt
    with torch.no_grad():
        for start, end in reference_context(48, TINY.block_size)[2:-1]:
            logits = model(seq[:, start:end])[0][:, -1]
            seq = torch.cat((seq, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
    assert torch.equal(cached, seq)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_pytorch_cache_matches_the_browser_session(tmp_path):
    """The PyTorch cache and web/engine.js agree step by step, through two refills of the window."""
    config = GPTConfig(vocab_size=10, block_size=12, n_embd=24, n_head=3, n_layer=2, dropout=0.0)
    model = fp16_model(config, seed=6)
    blob = weights_blob(model, DIGITS.state(), step=0, val_loss=0.0)
    seq = random_ids(30, B=1, seed=6, vocab=10)[0].tolist()
    res = run_engine(blob, [[0]], tmp_path, streams=[seq])

    cache = KVCache(config.n_layer)
    with torch.no_grad():
        probs = torch.stack([F.softmax(model.next_logits(torch.tensor([seq[:n]]), cache)[0], dim=-1)
                             for n in range(1, len(seq) + 1)])
    torch.testing.assert_close(res["sessionProbs"][0], probs, rtol=0, atol=1e-5)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_committed_checkpoint_samples_the_same_with_the_cache():
    """For the first window the trained model writes exactly the same text with or without the cache."""
    ckpt = torch.load(CKPT, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"])).eval()
    model.load_state_dict(ckpt["model"])
    prompt = torch.zeros((1, 1), dtype=torch.long)
    n = model.config.block_size - 1
    torch.manual_seed(0)
    plain = model.generate(prompt, n, temperature=0.8)
    torch.manual_seed(0)
    assert torch.equal(model.generate(prompt, n, temperature=0.8, use_cache=True), plain)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_sample_script_takes_the_cache_flag():
    """sample.py --kv-cache runs, and writes the same first window of text as without it."""
    import subprocess
    import sys

    def run(*extra):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "sample.py"), "--tokens", "200", "--seed", "3", *extra],
                             check=True, capture_output=True, text=True, cwd=ROOT).stdout
        return out.split("\n", 2)[2]  # drop the "loaded checkpoint" line and the blank after it

    plain, cached = run(), run("--kv-cache")
    assert len(cached) == len(plain) == 202  # the newline prompt, 200 characters, then print's newline
    assert cached[:64] == plain[:64]  # the 1-token prompt and its first 63 samples fill one window
    assert cached != plain  # after that the context differs, so the text does too
