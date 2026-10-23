"""The analysis script on synthetic run directories: partial results, diverged runs, and the figures."""

from __future__ import annotations

import csv
import json

import numpy as np

import scripts.analyze as A
from resnet.train import LOG_FIELDS


def fake_run(root, name, seed, epochs=10, final_err=0.1, diverged=False):
    d = root / name / f"seed{seed}"
    d.mkdir(parents=True)
    with (d / "log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        w.writeheader()
        for e in range(1, epochs + 1):
            acc = 1 - final_err - 0.5 / e
            w.writerow({k: 1.0 for k in LOG_FIELDS} | {"epoch": e, "train_acc": acc + 0.05, "test_acc": acc, "val_acc": acc})
    final = {"name": name, "seed": seed, "diverged": diverged, "epochs_run": epochs}
    if not diverged:
        final |= {"test_acc": acc, "train_acc": acc + 0.05, "val_acc": acc, "train_loss": 0.1, "test_loss": 0.3}
    (d / "final.json").write_text(json.dumps(final))


def test_analyze_end_to_end(tmp_path, monkeypatch):
    runs = tmp_path / "runs" / "e10"
    for s in range(3):
        fake_run(runs, "resnet20", s, final_err=0.08 + 0.01 * s)
        fake_run(runs, "plain56", s, final_err=0.12)
    fake_run(runs, "resnet20_nobn", 0, epochs=3, diverged=True)
    fake_run(runs, "resnet20_nobn", 1, epochs=0, diverged=True)
    (runs / "resnet20_nobn" / "seed1" / "log.csv").unlink()  # diverged in epoch 1: no log.csv at all
    fake_run(runs, "resnet20_nowd", 0)
    (runs / "resnet20_adam" / "seed0").mkdir(parents=True)  # unfinished: no final.json -> ignored
    monkeypatch.setattr(A, "OUT", tmp_path / "results")
    monkeypatch.setattr("sys.argv", ["analyze", "--runs", str(runs)])
    A.main()
    t = (tmp_path / "results" / "e10" / "table.md").read_text()
    # resnet20 final test error = final_err + 0.5/10: 13, 14, 15 % -> mean 14.00, sample std 1.00
    assert "| ResNet-20 (baseline: SGD+mom., BN, wd 1e-4) | 3 | 10 | 14.00 ± 1.00 |" in t
    assert "(1 seed)" in t and "| 2/2 |" in t and "Adam" not in t
    for f in ("degradation.png", "bn.png", "wd.png", "optim.png"):
        assert (tmp_path / "results" / "e10" / f).exists()
