import os
import subprocess
import sys

import pytest
import torch

from train import RESUME_FILE, settings_mismatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A run that takes a second or two. The small preset's dropout (0.1) stays on, so the resumed run
# only matches if the dropout masks, and not just the batches, pick up where they left off.
TINY = [
    "--config", "small", "--device", "cpu", "--max-iters", "12", "--eval-interval", "3",
    "--eval-iters", "2", "--batch-size", "4", "--block-size", "8", "--n-embd", "16",
    "--n-head", "2", "--n-layer", "1",
]


def tiny_text(tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("To be, or not to be, that is the question.\n" * 40, encoding="utf-8")
    return data


def train(args, cwd, script="train.py"):
    result = subprocess.run([sys.executable, os.path.join(ROOT, script), *args],
                            capture_output=True, text=True, cwd=cwd, timeout=300)
    return result


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("extra", [
    [],                                                                        # the default recipe
    ["--lr-schedule", "cosine", "--warmup-iters", "2", "--grad-clip", "0.5"],  # the schedule's step count matters
    ["--tokenizer", "bpe", "--bpe-vocab-size", "270", "--tie-weights"],        # merges and the shared matrix
])
def test_a_resumed_run_ends_exactly_like_an_uninterrupted_one(tmp_path, extra):
    data = tiny_text(tmp_path)
    base = [*TINY, "--data", str(data), *extra]
    whole = train([*base, "--out", str(tmp_path / "whole")], tmp_path)
    assert whole.returncode == 0, whole.stderr

    # Cut the second run off after the step-6 evaluation, then carry on with --resume.
    cut = tmp_path / "cut"
    first = train([*base, "--out", str(cut), "--stop-at", "5"], tmp_path)
    assert first.returncode == 0, first.stderr
    assert "stopping at step 6" in first.stdout
    assert (cut / RESUME_FILE).exists() and not (cut / "final.pt").exists()
    state = load(cut / RESUME_FILE)
    assert state["step"] == 6 and [h["step"] for h in state["history"]] == [0, 3, 6]

    second = train([*base, "--out", str(cut), "--resume"], tmp_path)
    assert second.returncode == 0, second.stderr
    assert "resuming from step 6 of 12" in second.stdout
    # The step-6 eval isn't run again; the rest of the log is the same as the uninterrupted run's.
    assert "step     6 |" not in second.stdout
    def losses(stdout, step):  # "step     9 | train loss ... | val loss ...", without the minutes
        return next(l for l in stdout.splitlines() if l.startswith(f"step {step:5d}")).split(" | ")[:3]
    for step in (9, 12):
        assert losses(second.stdout, step) == losses(whole.stdout, step)
    assert not (cut / RESUME_FILE).exists()  # removed once the run is complete

    for name in ("ckpt.pt", "final.pt"):
        a, b = load(tmp_path / "whole" / name), load(cut / name)
        assert same_weights(a["model"], b["model"])
        assert (a["step"], a["val_loss"], a["history"]) == (b["step"], b["val_loss"], b["history"])
        assert a["val_full"] == b["val_full"] and a["tokenizer"] == b["tokenizer"]
    # The sample at the end draws from the same random state too.
    assert (cut / "sample.txt").read_text(encoding="utf-8") == \
        (tmp_path / "whole" / "sample.txt").read_text(encoding="utf-8")


def test_resume_state_holds_what_the_loop_needs(tmp_path):
    data = tiny_text(tmp_path)
    out = tmp_path / "run"
    result = train([*TINY, "--data", str(data), "--out", str(out), "--stop-at", "3"], tmp_path)
    assert result.returncode == 0, result.stderr
    state = load(out / RESUME_FILE)
    assert state["step"] == 3
    assert set(state) >= {"model", "optimizer", "rng", "best_val", "best_step", "history", "elapsed", "settings"}
    # AdamW has taken 3 steps, and its running averages are saved alongside the weights.
    adam = next(iter(state["optimizer"]["state"].values()))
    assert int(adam["step"]) == 3 and {"exp_avg", "exp_avg_sq"} <= set(adam)
    assert state["settings"]["max_iters"] == 12 and state["settings"]["seed"] == 1337
    assert not (out / (RESUME_FILE + ".tmp")).exists()  # written to a temp file, then renamed


def test_resume_refuses_a_run_with_different_settings(tmp_path):
    data = tiny_text(tmp_path)
    out = tmp_path / "run"
    assert train([*TINY, "--data", str(data), "--out", str(out), "--stop-at", "3"], tmp_path).returncode == 0
    for change, setting in ((["--learning-rate", "0.01"], "learning_rate"), (["--seed", "2"], "seed"),
                            (["--max-iters", "24"], "max_iters")):
        result = train([*TINY, "--data", str(data), "--out", str(out), "--resume", *change], tmp_path)
        assert result.returncode != 0
        assert f"  {setting}:" in result.stderr and "different settings" in result.stderr
    # A different text is refused too, wherever it lives.
    other = tmp_path / "other.txt"
    other.write_text("Now is the winter of our discontent.\n" * 40, encoding="utf-8")
    result = train([*TINY, "--data", str(other), "--out", str(out), "--resume"], tmp_path)
    assert result.returncode != 0 and "data_sha256" in result.stderr
    # The same text copied somewhere else is fine.
    moved = tmp_path / "moved.txt"
    moved.write_bytes(data.read_bytes())
    result = train([*TINY, "--data", str(moved), "--out", str(out), "--resume"], tmp_path)
    assert result.returncode == 0, result.stderr


def test_resume_with_nothing_to_resume_starts_from_step_zero(tmp_path):
    data = tiny_text(tmp_path)
    plain = train([*TINY, "--data", str(data), "--out", str(tmp_path / "plain")], tmp_path)
    fresh = train([*TINY, "--data", str(data), "--out", str(tmp_path / "fresh"), "--resume"], tmp_path)
    assert plain.returncode == 0 and fresh.returncode == 0, fresh.stderr
    assert "starting from step 0" in fresh.stdout
    assert same_weights(load(tmp_path / "plain" / "final.pt")["model"], load(tmp_path / "fresh" / "final.pt")["model"])


def test_settings_mismatch_lists_each_difference():
    assert settings_mismatch({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []
    assert settings_mismatch({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == [
        f"  b: 2 in {RESUME_FILE}, 3 now", f"  c: None in {RESUME_FILE}, 4 now"]


def test_seeds_carries_on_a_seed_that_was_cut_off(tmp_path):
    data = tiny_text(tmp_path)
    seed_dir = tmp_path / "sweep" / "seed4"
    # Seed 4 was cut off after its step-6 evaluation: ckpt.pt and resume.pt exist, final.pt doesn't.
    cut = train([*TINY, "--data", str(data), "--out", str(seed_dir), "--seed", "4", "--stop-at", "6"], tmp_path)
    assert cut.returncode == 0, cut.stderr
    (seed_dir / "train_log.txt").write_text(cut.stdout, encoding="utf-8")
    result = train(["--seeds", "4", "--out", str(tmp_path / "sweep"), "--data", str(data), "--device", "cpu",
                    "--", *TINY], tmp_path, script="seeds.py")
    assert result.returncode == 0, result.stderr
    assert "seed 4: resuming the interrupted run" in result.stdout
    assert "already exists" not in result.stdout
    assert "seed4/final.pt` | 4 | 12 |" in result.stdout  # scored after all 12 steps, not at step 6
    # The log keeps the first part and adds the rest.
    log = (seed_dir / "train_log.txt").read_text(encoding="utf-8")
    assert "stopping at step 6" in log and "resuming from step 6 of 12" in log and "done in" in log
    # The finished seed matches one trained in one go.
    whole = train([*TINY, "--data", str(data), "--out", str(tmp_path / "whole"), "--seed", "4"], tmp_path)
    assert whole.returncode == 0
    assert same_weights(load(seed_dir / "final.pt")["model"], load(tmp_path / "whole" / "final.pt")["model"])

    # Now it is finished, the same command doesn't train it again.
    again = train(["--seeds", "4", "--out", str(tmp_path / "sweep"), "--data", str(data), "--device", "cpu",
                   "--", *TINY], tmp_path, script="seeds.py")
    assert again.returncode == 0 and "already exists, not training it again" in again.stdout


def test_seeds_refuses_resume_flags_it_sets_itself(tmp_path):
    for bad in (["--resume"], ["--stop-at", "3"]):
        result = train(["--seeds", "1", "--", *TINY, *bad], tmp_path, script="seeds.py")
        assert result.returncode != 0
