import math
import os
import subprocess
import sys

import pytest
import torch

from train import get_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_constant_schedule_is_the_learning_rate_every_step():
    # The committed checkpoint was trained this way, so the default must not change it.
    assert all(get_lr(it, 5000, 1e-3) == 1e-3 for it in range(5000))


def test_cosine_warmup_ramps_linearly_up_to_the_peak():
    lrs = [get_lr(it, 5000, 1e-3, "cosine", warmup_iters=100) for it in range(100)]
    assert lrs[0] == pytest.approx(1e-5)  # the first step is learning_rate / warmup_iters, not 0
    assert lrs[-1] == pytest.approx(1e-3)
    steps = [b - a for a, b in zip(lrs, lrs[1:])]
    assert all(s == pytest.approx(1e-5) for s in steps)


def test_cosine_decays_from_the_peak_to_min_lr_at_the_last_step():
    max_iters, warmup = 5000, 100
    lrs = [get_lr(it, max_iters, 1e-3, "cosine", warmup_iters=warmup, min_lr=1e-4) for it in range(max_iters)]
    assert lrs[warmup] == pytest.approx(1e-3)  # no jump where warmup ends
    assert lrs[-1] == pytest.approx(1e-4)
    decay = lrs[warmup:]
    assert all(b <= a for a, b in zip(decay, decay[1:]))
    # Halfway through the decay, cos(pi / 2) = 0, so the rate is halfway between peak and floor.
    mid = warmup + (max_iters - 1 - warmup) // 2
    assert lrs[mid] == pytest.approx(0.5 * (1e-3 + 1e-4), rel=1e-3)
    # The formula, at an arbitrary step.
    it = 1234
    progress = (it - warmup) / (max_iters - 1 - warmup)
    assert lrs[it] == pytest.approx(1e-4 + 0.5 * (1 + math.cos(math.pi * progress)) * 9e-4)


def test_cosine_defaults_and_edge_cases():
    # min_lr defaults to a tenth of the peak.
    assert get_lr(999, 1000, 3e-4, "cosine") == pytest.approx(3e-5)
    # Without warmup the first step uses the full rate.
    assert get_lr(0, 1000, 3e-4, "cosine", warmup_iters=0) == pytest.approx(3e-4)
    # Steps past the end stay at the floor instead of coming back up the cosine.
    assert get_lr(5000, 1000, 3e-4, "cosine") == pytest.approx(3e-5)
    # Warmup longer than the run is just a ramp.
    assert get_lr(9, 10, 1e-3, "cosine", warmup_iters=100) == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        get_lr(0, 10, 1e-3, "linear")


def test_train_runs_with_warmup_cosine_and_clipping(tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("To be, or not to be, that is the question.\n" * 40, encoding="utf-8")
    out = tmp_path / "run"
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"), "--config", "small", "--data", str(data),
        "--out", str(out), "--device", "cpu", "--max-iters", "6", "--eval-interval", "3",
        "--eval-iters", "2", "--batch-size", "4", "--block-size", "8", "--n-embd", "16",
        "--n-head", "2", "--n-layer", "1", "--lr-schedule", "cosine", "--warmup-iters", "2",
        "--grad-clip", "0.5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "lr schedule: cosine" in result.stdout and "grad clip: 0.5" in result.stdout
    # Steps 3 and 6 report the rate the previous step used and the mean gradient norm.
    step3 = next(line for line in result.stdout.splitlines() if line.startswith("step     3"))
    assert f"lr {get_lr(2, 6, 1e-3, 'cosine', warmup_iters=2):.2e}" in step3
    assert "grad norm" in step3

    ckpt = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["lr_schedule"] == "cosine"
    assert ckpt["warmup_iters"] == 2
    assert ckpt["grad_clip"] == 0.5
    assert ckpt["learning_rate"] == 1e-3


@pytest.mark.parametrize("lr, best_is_last", [(0.05, True), (0.3, False)])
def test_train_keeps_the_best_estimated_step_and_the_last_step(tmp_path, lr, best_is_last):
    # At 0.05 the tiny model keeps improving, so the last step is also the best one. At 0.3 the
    # rate is far too high and every step makes it worse, so ckpt.pt keeps an earlier step.
    data = tmp_path / "tiny.txt"
    data.write_text("To be, or not to be, that is the question.\n" * 40, encoding="utf-8")
    out = tmp_path / "run"
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"), "--config", "small", "--data", str(data),
        "--out", str(out), "--device", "cpu", "--max-iters", "12", "--eval-interval", "3",
        "--eval-iters", "2", "--batch-size", "4", "--block-size", "8", "--n-embd", "16",
        "--n-head", "2", "--n-layer", "1", "--learning-rate", str(lr),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "final.pt" in result.stdout

    best = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    final = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    history = final["history"]
    assert [h["step"] for h in history] == [0, 3, 6, 9, 12]
    # final.pt is the last step with that step's estimate; ckpt.pt is the step with the lowest estimate.
    assert final["step"] == 12 and final["val_loss"] == history[-1]["val"]
    lowest = min(history, key=lambda h: h["val"])
    assert best["step"] == lowest["step"] and best["val_loss"] == lowest["val"]
    assert f"best val loss {lowest['val']:.4f} (step {lowest['step']})" in result.stdout
    # The same weights exactly when the last step was the best one, different weights otherwise.
    assert (best["step"] == final["step"]) == best_is_last
    same = all(torch.equal(best["model"][k], final["model"][k]) for k in final["model"])
    assert same == best_is_last
    # Both carry everything the other scripts need to rebuild the model.
    for ckpt in (best, final):
        assert ckpt["model_config"] == final["model_config"] and ckpt["tokenizer"] == final["tokenizer"]
        assert ckpt["seed"] == 1337 and ckpt["max_iters"] == 12
