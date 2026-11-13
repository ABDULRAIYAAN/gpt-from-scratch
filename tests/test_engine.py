"""Check web/engine.js against PyTorch, for both the fused and the old per-head weight layout."""

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


def run_engine(blob, seqs, tmp_path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(blob))
    out = subprocess.run(
        ["node", os.path.join(HERE, "run_engine.js"), str(path), json.dumps(seqs)],
        check=True, capture_output=True, text=True,
    )
    return torch.tensor(json.loads(out.stdout))


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


@pytest.mark.parametrize("layout", ["fused", "legacy"])
def test_engine_matches_pytorch(layout, tmp_path):
    config = GPTConfig(vocab_size=10, block_size=20, n_embd=24, n_head=3, n_layer=2, dropout=0.0)
    model = fp16_model(config, seed=0)
    blob = weights_blob(model, [str(i) for i in range(10)], step=0, val_loss=0.0)
    assert ("blocks.0.sa.qkv.weight" in blob["tensors"]) and not any(".heads." in k for k in blob["tensors"])
    if layout == "legacy":
        blob = legacy_blob(blob, model)
    js = run_engine(blob, SEQS, tmp_path)
    torch.testing.assert_close(js, torch_logits(model, SEQS), rtol=0, atol=1e-4)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_engine_on_committed_checkpoint(tmp_path):
    ckpt = torch.load(CKPT, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"])).eval()
    model.load_state_dict(ckpt["model"])
    blob = weights_blob(model, ckpt["chars"], ckpt["step"], ckpt["val_loss"])
    with torch.no_grad():  # compare against the float16 weights the page actually ships
        for p in model.parameters():
            p.copy_(p.half().float())
    seqs = [[0], [ckpt["chars"].index(c) for c in "ROMEO:\nWhat light"]]
    js = run_engine(blob, seqs, tmp_path)
    torch.testing.assert_close(js, torch_logits(model, seqs), rtol=0, atol=1e-4)


def test_export_builds_a_fused_page(tmp_path):
    out = tmp_path / "index.html"
    subprocess.run([sys.executable, os.path.join(ROOT, "export_web.py"), "--out", str(out)], check=True, cwd=ROOT)
    page = out.read_text(encoding="utf-8")
    start = page.index('id="weights">') + len('id="weights">')
    weights = json.loads(page[start : page.index("</script>", start)].replace("<\\/", "</"))
    assert "blocks.0.sa.qkv.weight" in weights["tensors"]
    assert not any(".heads." in k for k in weights["tensors"])
    assert "__WEIGHTS__" not in page and "__SAMPLE__" not in page and "__ENGINE__" not in page
