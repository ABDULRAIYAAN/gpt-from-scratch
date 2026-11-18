"""Check web/engine.js against PyTorch, for both weight layouts and both tokenizers."""

import base64
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from export_web import weights_blob
from legacy import split_state_dict
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer, load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def fp16_model(config, seed):
    """A random model whose weights are exactly representable in float16, like the export."""
    torch.manual_seed(seed)
    model = GPT(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
            p.copy_(p.half().float())
    return model


def run_engine(blob, seqs, tmp_path, texts=()):
    """Everything run_engine.js reports, with the logits as a tensor."""
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(blob))
    out = subprocess.run(
        ["node", os.path.join(HERE, "run_engine.js"), str(path), json.dumps(seqs), json.dumps(list(texts))],
        check=True, capture_output=True, text=True,
    )
    res = json.loads(out.stdout)
    res["logits"] = torch.tensor(res["logits"])
    return res


def torch_logits(model, seqs):
    with torch.no_grad():
        return torch.stack([model(torch.tensor([s]))[0][0, -1] for s in seqs])


def legacy_blob(blob, model):
    """The same weights in the layout older exports used (one q/k/v per head)."""
    cfg = model.config
    sd = split_state_dict(model.state_dict(), cfg.n_head, cfg.n_layer)
    tensors = {}
    for k, v in sd.items():
        a = v.float().numpy().astype(np.float16)
        tensors[k] = {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}
    return dict(blob, tensors=tensors)


SEQS = [[1], [3, 1, 4, 1, 5, 9, 2, 6], list(range(10)) * 2]  # last one fills the 20-token window


DIGITS = CharTokenizer([str(i) for i in range(10)])


@pytest.mark.parametrize("layout", ["fused", "legacy", "chars-only"])
def test_engine_matches_pytorch(layout, tmp_path):
    config = GPTConfig(vocab_size=10, block_size=20, n_embd=24, n_head=3, n_layer=2, dropout=0.0)
    model = fp16_model(config, seed=0)
    blob = weights_blob(model, DIGITS.state(), step=0, val_loss=0.0)
    assert ("blocks.0.sa.qkv.weight" in blob["tensors"]) and not any(".heads." in k for k in blob["tensors"])
    if layout == "legacy":
        blob = legacy_blob(blob, model)
    if layout == "chars-only":  # exports from before BPE support had no tokenizer state
        blob = {k: v for k, v in blob.items() if k != "tokenizer"}
        blob["chars"] = DIGITS.chars
    res = run_engine(blob, SEQS, tmp_path, texts=["3141", "2 7x1"])
    torch.testing.assert_close(res["logits"], torch_logits(model, SEQS), rtol=0, atol=1e-4)
    assert res["tokenizer"] == "char"
    assert res["encoded"] == [DIGITS.encode("3141"), DIGITS.encode("271")]  # unknown characters are dropped
    assert res["streamed"] == [DIGITS.decode(s) for s in SEQS]


BPE_TEXT = (
    "ROMEO:\nBut, soft! what light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n\n"
    "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?\n"
) * 3 + "Caf\u00e9 na\u00efve \u4e16\u754c \U0001f3ad\U0001f3ad"


def test_engine_runs_a_bpe_model(tmp_path):
    """A BPE checkpoint: same logits as PyTorch, same ids as tokenizer.py, and text streams back whole."""
    tok = BPETokenizer.train(BPE_TEXT, vocab_size=300)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=24, n_embd=16, n_head=2, n_layer=2, dropout=0.0)
    model = fp16_model(config, seed=1)
    blob = weights_blob(model, tok.state(), step=0, val_loss=0.0)

    prompt_ids = tok.encode("JULIET:\nO Romeo")
    emoji_ids = tok.encode("na\u00efve \U0001f3ad \u4e16\u754c")  # multi-byte characters split over tokens
    seqs = [prompt_ids, list(range(276, 300)), emoji_ids]  # the second fills the 24-token window
    texts = ["ROMEO:\nWhat light", "\U0001f3ad caf\u00e9\tq", ""]
    res = run_engine(blob, seqs, tmp_path, texts=texts)

    torch.testing.assert_close(res["logits"], torch_logits(model, seqs), rtol=0, atol=1e-4)
    assert res["tokenizer"] == "bpe"
    assert res["encoded"] == [tok.encode(t) for t in texts]
    assert res["start"] == tok.encode("\n")  # an empty prompt starts from a newline, not byte 0
    assert res["streamed"] == [tok.decode(s) for s in seqs]
    assert "\ufffd" not in res["streamed"][2]
    assert any(not is_utf8(tok.vocab[i]) for i in emoji_ids)  # so the streaming really was tested


def is_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_engine_on_committed_checkpoint(tmp_path):
    ckpt = torch.load(CKPT, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"])).eval()
    model.load_state_dict(ckpt["model"])
    blob = weights_blob(model, load_tokenizer(ckpt).state(), ckpt["step"], ckpt["val_loss"])
    with torch.no_grad():  # compare against the float16 weights the page actually ships
        for p in model.parameters():
            p.copy_(p.half().float())
    seqs = [[0], [ckpt["chars"].index(c) for c in "ROMEO:\nWhat light"]]
    res = run_engine(blob, seqs, tmp_path, texts=["ROMEO:\nWhat light"])
    torch.testing.assert_close(res["logits"], torch_logits(model, seqs), rtol=0, atol=1e-4)
    assert res["encoded"] == [seqs[1]]
    assert res["start"] == [0]  # the newline, as before


def page_json(page, element_id):
    start = page.index(f'id="{element_id}">') + len(f'id="{element_id}">')
    return json.loads(page[start : page.index("</script>", start)].replace("<\\/", "</"))


def test_export_builds_a_fused_page(tmp_path):
    out = tmp_path / "index.html"
    subprocess.run([sys.executable, os.path.join(ROOT, "export_web.py"), "--out", str(out)], check=True, cwd=ROOT)
    page = out.read_text(encoding="utf-8")
    weights = page_json(page, "weights")
    assert "blocks.0.sa.qkv.weight" in weights["tensors"]
    assert not any(".heads." in k for k in weights["tensors"])
    assert weights["tokenizer"]["type"] == "char" and len(weights["tokenizer"]["chars"]) == 65
    for name in ["__WEIGHTS__", "__SAMPLE__", "__ENGINE__", "__TOKENIZER__"]:
        assert name not in page
    assert "function createTokenizer" in page and "function createGPT" in page
    # the committed demo is what the exporter builds today, byte for byte
    with open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8") as f:
        assert page == f.read()


def test_export_builds_a_bpe_page(tmp_path):
    """export_web.py takes a BPE checkpoint and writes a page whose tokenizer and sample match it."""
    tok = BPETokenizer.train(BPE_TEXT, vocab_size=300)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_embd=16, n_head=2, n_layer=1, dropout=0.0)
    model = GPT(config)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "model_config": config.__dict__, "tokenizer": tok.state(),
                "step": 7, "val_loss": 5.0}, ckpt_path)
    out = tmp_path / "bpe.html"
    data = tmp_path / "text.txt"
    data.write_text(BPE_TEXT * 4, encoding="utf-8")
    subprocess.run([sys.executable, os.path.join(ROOT, "export_web.py"), "--ckpt", str(ckpt_path), "--out", str(out),
                    "--data", str(data)], check=True, cwd=ROOT, capture_output=True)
    page = out.read_text(encoding="utf-8")
    weights, sample = page_json(page, "weights"), page_json(page, "sample")
    assert weights["tokenizer"] == tok.state()
    assert weights["config"]["vocab_size"] == tok.vocab_size and weights["step"] == 7
    assert sample["prompt"] == "ROMEO:\n" and isinstance(sample["text"], str)
    assert "__TOKENIZER__" not in page and "__ENGINE__" not in page
    # the wording is for tokens, with bits per character measured on the --data text
    assert '<main class="tokens">' in page and "Next token" in page
    assert "<dt>Bits per character</dt>" in page and "{{" not in page
