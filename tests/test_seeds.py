import os
import statistics
import subprocess
import sys

import pytest
import torch

from seeds import compare, final_path, report, spread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

# train.py settings for a run that takes a second or two
TINY_TRAIN = [
    "--config", "small", "--device", "cpu", "--max-iters", "4", "--eval-interval", "2",
    "--eval-iters", "2", "--batch-size", "4", "--block-size", "8", "--n-embd", "16",
    "--n-head", "2", "--n-layer", "1",
]


def tiny_text(tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("To be, or not to be, that is the question.\n" * 40, encoding="utf-8")
    return data


def run(args, cwd, script="seeds.py"):
    return subprocess.run([sys.executable, os.path.join(ROOT, script), *args],
                          capture_output=True, text=True, cwd=cwd, timeout=300)


def test_spread_matches_the_statistics_module():
    values = [2.273, 2.281, 2.266, 2.290]
    s = spread(values)
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(statistics.mean(values))
    assert s["std"] == pytest.approx(statistics.stdev(values))  # sample std, n - 1
    assert (s["min"], s["max"]) == (2.266, 2.290)


def test_spread_of_one_run_has_no_std_and_of_none_is_an_error():
    assert spread([2.3]) == {"n": 1, "mean": 2.3, "std": None, "min": 2.3, "max": 2.3}
    with pytest.raises(ValueError):
        spread([])


def test_report_lists_every_run_and_the_summary():
    runs = [
        {"path": "a/ckpt.pt", "seed": 1, "step": 5000, "loss": 1.5, "bits_per_char": 2.20},
        {"path": "b/ckpt.pt", "seed": None, "step": 4500, "loss": 1.6, "bits_per_char": 2.30},
    ]
    text = report(runs)
    assert "| `a/ckpt.pt` | 1 | 5000 | 1.5000 | 2.200 |" in text
    assert "| `b/ckpt.pt` | ? | 4500 | 1.6000 | 2.300 |" in text  # old checkpoints store no seed
    assert "bits/char over 2 runs: mean 2.250 | std 0.071 | range 2.200 - 2.300 (0.100)" in text
    assert "loss/token over 2 runs: mean 1.5500" in text
    assert "std n/a (one run)" in report(runs[:1])


def test_the_same_seed_trains_the_same_model_and_a_new_seed_does_not(tmp_path):
    # The whole point of comparing seeds: the seed must be the only thing that changes a run.
    data = tiny_text(tmp_path)
    for name, seed in (("a", 7), ("b", 7), ("c", 8)):
        result = run([*TINY_TRAIN, "--data", str(data), "--out", str(tmp_path / name), "--seed", str(seed)],
                     tmp_path, script="train.py")
        assert result.returncode == 0, result.stderr
    a, b, c = (torch.load(tmp_path / n / "ckpt.pt", map_location="cpu", weights_only=False) for n in "abc")
    assert a["seed"] == 7 and c["seed"] == 8
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert not all(torch.equal(a["model"][k], c["model"][k]) for k in a["model"])


def test_seeds_trains_each_seed_scores_it_and_resumes(tmp_path):
    data = tiny_text(tmp_path)
    out = tmp_path / "sweep"
    result = run(["--seeds", "3", "4", "--out", str(out), "--data", str(data), "--device", "cpu",
                  "--", *TINY_TRAIN], tmp_path)
    assert result.returncode == 0, result.stderr
    for seed in (3, 4):
        ckpt = torch.load(out / f"seed{seed}" / "ckpt.pt", map_location="cpu", weights_only=False)
        assert ckpt["seed"] == seed
        log = (out / f"seed{seed}" / "train_log.txt").read_text(encoding="utf-8")
        assert "done in" in log and "best val loss" in log
        assert f"| {seed} |" in result.stdout
    assert "bits/char over 2 runs: mean" in result.stdout

    # Running the same command again reuses the checkpoints instead of training again.
    mtime = os.path.getmtime(out / "seed3" / "ckpt.pt")
    again = run(["--seeds", "3", "--out", str(out), "--data", str(data), "--device", "cpu",
                 "--", *TINY_TRAIN], tmp_path)
    assert again.returncode == 0, again.stderr
    assert "already exists, not training it again" in again.stdout
    assert os.path.getmtime(out / "seed3" / "ckpt.pt") == mtime


def test_seeds_refuses_train_args_it_sets_itself(tmp_path):
    for bad in (["--seed", "5"], ["--out=elsewhere"], ["--data", "x.txt"]):
        result = run(["--seeds", "1", "--", *TINY_TRAIN, *bad], tmp_path)
        assert result.returncode != 0
        assert "is set by seeds.py" in result.stderr
    assert run([], tmp_path).returncode != 0  # nothing to train or score


def test_seeds_summarises_the_committed_checkpoint(tmp_path):
    # --ckpts alone trains nothing; the committed checkpoint predates the seed field.
    data = os.path.join(ROOT, "data", "input.txt")
    if not os.path.exists(data):
        pytest.skip("Tiny Shakespeare not downloaded")
    result = run(["--ckpts", CKPT, "--device", "cpu"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "| ? | " in result.stdout
    assert "| 2.327 |" in result.stdout  # the README's number for this checkpoint
    assert "bits/char over 1 run: mean 2.327 | std n/a (one run)" in result.stdout


def test_compare_pairs_each_run_with_its_last_step():
    best = [
        {"path": "a/ckpt.pt", "seed": 1, "step": 4500, "bits_per_char": 2.283},
        {"path": "b/ckpt.pt", "seed": 2, "step": 5000, "bits_per_char": 2.264},
        {"path": "c/ckpt.pt", "seed": None, "step": 4000, "bits_per_char": 2.300},
    ]
    final = [
        {"path": "a/final.pt", "seed": 1, "step": 5000, "bits_per_char": 2.275},
        {"path": "b/final.pt", "seed": 2, "step": 5000, "bits_per_char": 2.264},
        {"path": "c/final.pt", "seed": None, "step": 5000, "bits_per_char": 2.304},
    ]
    text = compare(best, final)
    assert "| `a` | 1 | 4500 | 2.283 | 2.275 | -0.008 |" in text
    assert "| `b` | 2 | 5000 | 2.264 | 2.264 | +0.000 |" in text
    assert "| `c` | ? | 4000 | 2.300 | 2.304 | +0.004 |" in text
    diffs = [-0.008, 0.0, 0.004]
    assert f"mean {statistics.mean(diffs):+.3f} | std {statistics.stdev(diffs):.3f}" in text
    assert "last step better in 1, picked step better in 1, the same step in 1" in text
    with pytest.raises(ValueError):
        compare(best, final[:2])  # the two lists must be the same runs


def test_final_path_needs_a_final_checkpoint(tmp_path):
    (tmp_path / "ckpt.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="only --pick best"):
        final_path(str(tmp_path / "ckpt.pt"))
    (tmp_path / "final.pt").write_bytes(b"")
    assert final_path(str(tmp_path / "ckpt.pt")) == str(tmp_path / "final.pt")
    with pytest.raises(FileNotFoundError):
        final_path(CKPT)  # the committed checkpoint was trained before final.pt existed


def test_seeds_scores_the_last_step_too(tmp_path):
    data = tiny_text(tmp_path)
    out = tmp_path / "sweep"
    result = run(["--seeds", "5", "6", "--out", str(out), "--data", str(data), "--device", "cpu",
                  "--pick", "both", "--", *TINY_TRAIN], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Best estimated step (ckpt.pt)" in result.stdout and "Last step (final.pt)" in result.stdout
    for seed in (5, 6):
        assert f"seed{seed}/final.pt` | {seed} | 4 |" in result.stdout  # TINY_TRAIN runs 4 steps
    assert "last - picked over 2 runs: mean" in result.stdout

    final_only = run(["--ckpts", str(out / "seed5" / "ckpt.pt"), "--data", str(data), "--device", "cpu",
                      "--pick", "final"], tmp_path)
    assert final_only.returncode == 0, final_only.stderr
    assert "seed5/final.pt` | 5 | 4 |" in final_only.stdout and "ckpt.pt`" not in final_only.stdout
