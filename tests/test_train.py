"""Infrastructure tests: config loading, logging, checkpoint/resume equivalence, divergence handling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from resnet.data import Split
from resnet.train import Config, train


def tiny_data(n: int = 256) -> dict[str, Split]:
    g = torch.Generator().manual_seed(123)
    mk = lambda k: Split(torch.randn(k, 3, 32, 32, generator=g), torch.randint(0, 10, (k,), generator=g))
    return {"train": mk(n), "val": mk(64), "test": mk(64)}


def cfg(tmp: Path, **kw) -> Config:
    base = dict(name="t", depth=8, epochs=4, batch_size=64, device="cpu", out_dir=str(tmp), seed=0)
    return Config(**{**base, **kw})


def test_all_configs_load():
    for p in Path("configs").glob("*.yaml"):
        c = Config.from_yaml(p)
        assert c.name == p.stem, f"{p}: name must match the file name (it decides the run directory)"


def test_unknown_config_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nlearning_rate: 0.1\n")  # typo'd key must not be silently ignored
    with pytest.raises(ValueError, match="learning_rate"):
        Config.from_yaml(p)


def test_run_writes_log_and_summary(tmp_path):
    data = tiny_data()
    s = train(cfg(tmp_path), data, verbose=False)
    run = tmp_path / "t" / "seed0"
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3, 4]
    # Schedule: milestones at 50% and 75% of 4 epochs -> lr 0.1, 0.1, 0.01, 0.001 (logged before the step)
    assert [float(r["lr"]) for r in rows] == pytest.approx([0.1, 0.1, 0.01, 0.001])
    assert json.loads((run / "final.json").read_text())["epochs_run"] == 4
    assert s["diverged"] is False


def test_resume_gives_identical_result(tmp_path):
    """Kill a run after 2 of 4 epochs, resume it, and compare with an uninterrupted run: the weights must match
    exactly. That requires checkpointing the model, optimizer momentum, scheduler AND the data RNG."""
    data = tiny_data()
    train(cfg(tmp_path / "a"), data, verbose=False)
    run_b = tmp_path / "b" / "t" / "seed0"
    c = cfg(tmp_path / "b")
    _train_partial(c, data, stop_after=2)
    assert not (run_b / "final.json").exists()
    train(c, data, verbose=False)  # resumes at epoch 2
    a = torch.load(tmp_path / "a" / "t" / "seed0" / "last.pt", weights_only=False)["model"]
    b = torch.load(run_b / "last.pt", weights_only=False)["model"]
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    assert len(list(csv.DictReader((run_b / "log.csv").open()))) == 4


def _train_partial(c: Config, data, stop_after: int) -> None:
    """Run `c` but crash right after epoch `stop_after` has been checkpointed."""
    import resnet.train as T

    orig = T.evaluate
    calls = {"n": 0}

    def evaluate_then_maybe_crash(model, split, bs=1000):
        calls["n"] += 1
        # evaluate() is called twice per epoch (val, test); crash at the start of epoch stop_after+1's eval
        if calls["n"] > 2 * stop_after:
            raise KeyboardInterrupt
        return orig(model, split, bs)

    T.evaluate = evaluate_then_maybe_crash
    try:
        T.train(c, data, verbose=False)
    except KeyboardInterrupt:
        pass
    finally:
        T.evaluate = orig


def test_finished_run_is_skipped(tmp_path):
    data = tiny_data()
    train(cfg(tmp_path, epochs=1), data, verbose=False)
    mtime = (tmp_path / "t" / "seed0" / "last.pt").stat().st_mtime
    train(cfg(tmp_path, epochs=1), data, verbose=False)
    assert (tmp_path / "t" / "seed0" / "last.pt").stat().st_mtime == mtime


def test_divergence_is_recorded(tmp_path):
    s = train(cfg(tmp_path, bn=False, lr=1e4, epochs=2), tiny_data(), verbose=False)
    assert s["diverged"] is True
