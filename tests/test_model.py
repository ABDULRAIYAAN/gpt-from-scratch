import os

import pytest
import torch

from legacy import LegacyGPT, split_state_dict
from model import GPT, GPTConfig, MultiHeadAttention, upgrade_state_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

TINY = GPTConfig(vocab_size=11, block_size=16, n_embd=24, n_head=3, n_layer=2, dropout=0.0)


def random_ids(config, B=2, T=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(config.vocab_size, (B, T or config.block_size), generator=g)


def test_attention_is_one_fused_linear():
    sa = MultiHeadAttention(TINY)
    assert sa.qkv.weight.shape == (3 * TINY.n_embd, TINY.n_embd)
    assert sa.qkv.bias is None
    keys = set(GPT(TINY).state_dict())
    assert "blocks.0.sa.qkv.weight" in keys
    assert not any(".heads." in k or k.endswith("tril") for k in keys)


def test_fused_matches_per_head_loop():
    torch.manual_seed(0)
    old = LegacyGPT(TINY).eval()
    new = GPT(TINY).eval()
    new.load_state_dict(old.state_dict())  # old per-head layout, converted on load

    idx = random_ids(TINY)
    with torch.no_grad():
        old_logits, old_loss = old(idx, idx)
        new_logits, new_loss = new(idx, idx)
    torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=1e-5)
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=1e-6)


def test_upgrade_round_trip_and_idempotent():
    torch.manual_seed(1)
    fused = GPT(TINY).state_dict()
    legacy = split_state_dict(fused, TINY.n_head, TINY.n_layer)
    upgraded = upgrade_state_dict(legacy, TINY.n_head, TINY.n_layer)
    assert upgraded.keys() == fused.keys()
    for k in fused:
        assert torch.equal(upgraded[k], fused[k]), k
    # Already-fused state dicts pass through untouched.
    again = upgrade_state_dict(fused, TINY.n_head, TINY.n_layer)
    assert all(torch.equal(again[k], fused[k]) for k in fused)


def test_upgrade_does_not_modify_the_callers_dict():
    legacy = LegacyGPT(TINY).state_dict()
    before = set(legacy)
    GPT(TINY).load_state_dict(legacy)
    assert set(legacy) == before


def test_new_checkpoints_round_trip(tmp_path):
    torch.manual_seed(2)
    a = GPT(TINY).eval()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": a.state_dict()}, path)
    b = GPT(TINY).eval()
    b.load_state_dict(torch.load(path, map_location="cpu")["model"])
    idx = random_ids(TINY)
    with torch.no_grad():
        torch.testing.assert_close(b(idx)[0], a(idx)[0], rtol=0, atol=0)


def test_causal_mask_hides_the_future():
    torch.manual_seed(3)
    model = GPT(TINY).eval()
    idx = random_ids(TINY, B=1)
    changed = idx.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % TINY.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10], rtol=0, atol=0)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_shorter_than_block_and_generate():
    torch.manual_seed(4)
    model = GPT(TINY).eval()
    idx = random_ids(TINY, B=3, T=5)
    logits, loss = model(idx)
    assert logits.shape == (3, 5, TINY.vocab_size) and loss is None
    out = model.generate(idx, max_new_tokens=TINY.block_size + 4, top_k=5)
    assert out.shape == (3, 5 + TINY.block_size + 4)


def test_parameter_count_unchanged():
    # Fusing q/k/v only regroups weights: the count must match the per-head model.
    assert GPT(TINY).num_params() == LegacyGPT(TINY).num_params()


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_committed_checkpoint_still_loads():
    ckpt = torch.load(CKPT, map_location="cpu")
    assert any(".heads." in k for k in ckpt["model"]), "expected the original per-head checkpoint"
    config = GPTConfig(**ckpt["model_config"])

    new = GPT(config).eval()
    new.load_state_dict(ckpt["model"])  # strict: every key must be accounted for
    old = LegacyGPT(config).eval()
    old.load_state_dict(ckpt["model"])

    assert new.num_params() == 816_705
    idx = random_ids(config, B=4)
    with torch.no_grad():
        torch.testing.assert_close(new(idx)[0], old(idx)[0], rtol=0, atol=1e-5)
