"""Weight tying: one matrix shared by the token embedding and the output layer."""

import io
import os
import shutil
import subprocess
import sys

import pytest
import torch

from export_web import weights_blob
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer
from test_engine import BPE_TEXT, fp16_model, run_engine, torch_logits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

SMALL = dict(vocab_size=65, block_size=64, n_embd=128, n_head=4, n_layer=4, dropout=0.0)
TINY = dict(vocab_size=11, block_size=12, n_embd=16, n_head=2, n_layer=2, dropout=0.0)


def test_tied_model_shares_one_matrix():
    model = GPT(GPTConfig(**TINY, tie_weights=True))
    assert model.lm_head.weight is model.token_embedding_table.weight
    assert model.lm_head.weight.data_ptr() == model.token_embedding_table.weight.data_ptr()
    assert model.lm_head.bias is not None  # the bias is not shared
    untied = GPT(GPTConfig(**TINY))
    assert untied.lm_head.weight is not untied.token_embedding_table.weight


def test_tying_saves_vocab_times_embedding_parameters():
    untied = GPT(GPTConfig(**SMALL)).num_params()
    tied = GPT(GPTConfig(**SMALL, tie_weights=True)).num_params()
    assert untied == 816_705  # the committed small model
    assert untied - tied == 65 * 128
    bpe = dict(SMALL, vocab_size=512)
    assert GPT(GPTConfig(**bpe)).num_params() - GPT(GPTConfig(**bpe, tie_weights=True)).num_params() == 512 * 128


def test_tied_model_is_an_untied_model_with_equal_matrices():
    """Same logits as an untied model whose output weight is a copy of the embedding."""
    torch.manual_seed(0)
    tied = GPT(GPTConfig(**TINY, tie_weights=True)).eval()
    untied = GPT(GPTConfig(**TINY)).eval()
    sd = tied.state_dict()
    sd["lm_head.weight"] = sd["token_embedding_table.weight"].clone()
    untied.load_state_dict(sd)
    idx = torch.randint(0, TINY["vocab_size"], (3, 12))
    torch.testing.assert_close(tied(idx)[0], untied(idx)[0], rtol=0, atol=0)


def test_shared_gradient_is_the_sum_of_both_uses():
    """Backprop adds the embedding's gradient and the output layer's gradient into one tensor."""
    torch.manual_seed(1)
    tied = GPT(GPTConfig(**TINY, tie_weights=True))
    untied = GPT(GPTConfig(**TINY))
    sd = tied.state_dict()
    sd["lm_head.weight"] = sd["token_embedding_table.weight"].clone()
    untied.load_state_dict(sd)
    idx = torch.randint(0, TINY["vocab_size"], (4, 12))
    targets = torch.randint(0, TINY["vocab_size"], (4, 12))
    tied(idx, targets)[1].backward()
    untied(idx, targets)[1].backward()
    expected = untied.token_embedding_table.weight.grad + untied.lm_head.weight.grad
    torch.testing.assert_close(tied.token_embedding_table.weight.grad, expected)
    # Tokens that never appear in the input still get a gradient, through the output layer.
    unused = sorted(set(range(TINY["vocab_size"])) - set(idx.flatten().tolist()))
    if unused:
        assert untied.token_embedding_table.weight.grad[unused].abs().sum() == 0
        assert tied.token_embedding_table.weight.grad[unused].abs().sum() > 0


def test_optimizer_step_keeps_the_tie():
    torch.manual_seed(2)
    model = GPT(GPTConfig(**TINY, tie_weights=True))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
    before = model.token_embedding_table.weight.detach().clone()
    idx = torch.randint(0, TINY["vocab_size"], (2, 12))
    model(idx, idx)[1].backward()
    opt.step()
    assert not torch.equal(model.token_embedding_table.weight, before)
    assert model.lm_head.weight is model.token_embedding_table.weight


def test_tied_checkpoint_round_trip():
    """A saved tied model loads back tied, with the same outputs."""
    torch.manual_seed(3)
    model = GPT(GPTConfig(**TINY, tie_weights=True)).eval()
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "model_config": vars(model.config)}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)
    loaded = GPT(GPTConfig(**ckpt["model_config"])).eval()
    loaded.load_state_dict(ckpt["model"])
    assert loaded.lm_head.weight is loaded.token_embedding_table.weight
    idx = torch.randint(0, TINY["vocab_size"], (2, 12))
    torch.testing.assert_close(loaded(idx)[0], model(idx)[0], rtol=0, atol=0)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_committed_checkpoint_stays_untied():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert "tie_weights" not in ckpt["model_config"]  # saved before the option existed
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    assert model.lm_head.weight is not model.token_embedding_table.weight
    assert model.num_params() == 816_705


def test_export_ships_the_tied_matrix_once():
    tok = CharTokenizer([chr(ord("a") + i) for i in range(TINY["vocab_size"])])
    tied = weights_blob(GPT(GPTConfig(**TINY, tie_weights=True)), tok.state(), step=0, val_loss=0.0)
    untied = weights_blob(GPT(GPTConfig(**TINY)), tok.state(), step=0, val_loss=0.0)
    assert "lm_head.weight" not in tied["tensors"] and "lm_head.bias" in tied["tensors"]
    assert "lm_head.weight" in untied["tensors"]
    assert tied["config"]["tie_weights"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_engine_runs_a_tied_char_model(tmp_path):
    tok = CharTokenizer([str(i) for i in range(10)] + [" "])
    model = fp16_model(GPTConfig(**TINY, tie_weights=True), seed=4)
    blob = weights_blob(model, tok.state(), step=0, val_loss=0.0)
    seqs = [[1], [3, 1, 4, 1, 5, 9, 2, 6], list(range(11)) + [10]]  # the last one fills the window
    res = run_engine(blob, seqs, tmp_path)
    torch.testing.assert_close(res["logits"], torch_logits(model, seqs), rtol=0, atol=1e-4)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_engine_runs_a_tied_bpe_model(tmp_path):
    tok = BPETokenizer.train(BPE_TEXT, vocab_size=300)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=24, n_embd=16, n_head=2, n_layer=2,
                       dropout=0.0, tie_weights=True)
    model = fp16_model(config, seed=5)
    blob = weights_blob(model, tok.state(), step=0, val_loss=0.0)
    seqs = [tok.encode("JULIET:\nO Romeo"), list(range(276, 300))]
    res = run_engine(blob, seqs, tmp_path)
    torch.testing.assert_close(res["logits"], torch_logits(model, seqs), rtol=0, atol=1e-4)


def test_train_tie_weights_flag(tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("To be, or not to be, that is the question.\n" * 40, encoding="utf-8")
    out = tmp_path / "run"
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"), "--config", "small", "--data", str(data),
        "--out", str(out), "--device", "cpu", "--max-iters", "4", "--eval-interval", "2",
        "--eval-iters", "2", "--batch-size", "4", "--block-size", "8", "--n-embd", "16",
        "--n-head", "2", "--n-layer", "1", "--tie-weights",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "embedding and output weights tied" in result.stdout

    ckpt = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["model_config"]["tie_weights"] is True
    sd = ckpt["model"]
    assert sd["lm_head.weight"].data_ptr() == sd["token_embedding_table.weight"].data_ptr()  # stored once
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(sd)
    assert model.lm_head.weight is model.token_embedding_table.weight
