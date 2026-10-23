"""Collect runs/*/seed*/log.csv into the results tables and figures.

    python -m scripts.analyze --runs runs/e60          # writes results/e60/
    python -m scripts.analyze --runs runs/e182         # writes results/e182/

Works with partial results: a variant with fewer than 3 finished seeds is
reported with its seed count, and unfinished runs are left out (their curves
would end early and bias the mean).

Every number is the FINAL epoch's value, never the best epoch. Choosing the
best epoch by test accuracy would be selecting on the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results")
ORDER = ["resnet20", "plain20", "resnet56", "plain56", "resnet20_nobn", "resnet20_nobn_lr001", "resnet20_nowd",
         "resnet20_adam", "resnet20_adamw"]
LABEL = {
    "resnet20": "ResNet-20 (baseline: SGD+mom., BN, wd 1e-4)", "plain20": "Plain-20", "resnet56": "ResNet-56",
    "plain56": "Plain-56", "resnet20_nobn": "ResNet-20, no BN, lr 0.1", "resnet20_nobn_lr001": "ResNet-20, no BN, lr 0.01",
    "resnet20_nowd": "ResNet-20, no weight decay", "resnet20_adam": "ResNet-20, Adam (L2 1e-4)",
    "resnet20_adamw": "ResNet-20, AdamW (wd 1e-2)",
}


def load_runs(root: Path) -> dict[str, list[dict]]:
    """{variant: [ {seed, final, log: {column: np.array}} , ...]} for finished runs only."""
    runs: dict[str, list[dict]] = {}
    for f in sorted(root.glob("*/seed*/final.json")):
        final = json.loads(f.read_text())
        # A run that diverges in its first epoch never writes log.csv.
        log_csv = f.parent / "log.csv"
        rows = list(csv.DictReader(log_csv.open())) if log_csv.exists() else []
        log = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}
        runs.setdefault(f.parent.parent.name, []).append({"seed": final["seed"], "final": final, "log": log})
    return runs


def stack(rs: list[dict], key: str) -> np.ndarray:
    """(n_seeds, n_epochs) array; only runs of the most common length (i.e. same budget, not diverged)."""
    lengths = [len(r["log"].get(key, [])) for r in rs]
    n = max(set(lengths), key=lengths.count)
    return np.array([r["log"][key] for r in rs if len(r["log"].get(key, [])) == n])


def mean_std(a: np.ndarray) -> str:
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else f"{a.mean():.2f} (1 seed)"


def table(runs: dict[str, list[dict]]) -> str:
    lines = ["| variant | seeds | epochs | test error (%) | train error (%) | train loss | val error (%) | diverged |",
             "|---|---|---|---|---|---|---|---|"]
    for name in [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER)):
        rs = runs[name]
        ok = [r for r in rs if not r["final"]["diverged"]]
        div = len(rs) - len(ok)
        if not ok:
            lines.append(f"| {LABEL.get(name, name)} | {len(rs)} | — | — | — | — | — | {div}/{len(rs)} |")
            continue
        get = lambda k: np.array([r["final"][k] for r in ok])
        epochs = sorted({r["final"]["epochs_run"] for r in ok})
        lines.append(f"| {LABEL.get(name, name)} | {len(ok)} | {'/'.join(map(str, epochs))} | "
                     f"{mean_std(100 * (1 - get('test_acc')))} | {mean_std(100 * (1 - get('train_acc')))} | "
                     f"{mean_std(get('train_loss'))} | {mean_std(100 * (1 - get('val_acc')))} | {div}/{len(rs)} |")
    return "\n".join(lines)


def curve(ax, rs: list[dict], key: str, transform=lambda a: a, **kw) -> None:
    """Mean over seeds as a line, min-max over seeds as a band."""
    a = transform(stack(rs, key))
    if a.size == 0:
        return
    ep = np.arange(1, a.shape[1] + 1)
    line, = ax.plot(ep, a.mean(0), **kw)
    if len(a) > 1:
        ax.fill_between(ep, a.min(0), a.max(0), color=line.get_color(), alpha=0.15)


err = lambda a: 100 * (1 - a)


def fig_degradation(runs: dict[str, list[dict]]) -> None:
    """Like He et al. Fig. 1/6: plain nets on the left, ResNets on the right; thin dashed = training error,
    bold = test error."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, kind in zip(axes, ("plain", "resnet")):
        for depth, c in ((20, "C0"), (56, "C3")):
            name = f"{kind}{depth}"
            if name not in runs:
                continue
            curve(ax, runs[name], "train_acc", err, c=c, ls="--", lw=1, label=f"{name}: train")
            curve(ax, runs[name], "test_acc", err, c=c, lw=2, label=f"{name}: test")
        ax.set(title="Plain networks" if kind == "plain" else "Residual networks", xlabel="epoch", ylim=(0, 25))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error (%)")
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png", dpi=150)
    plt.close(fig)


def fig_compare(runs: dict[str, list[dict]], names: list[str], fname: str, title: str, extra: str | None = None) -> None:
    """Three standard panels (test error, train loss, val error), plus an optional extra logged quantity."""
    present = [n for n in names if n in runs and any(not r["final"]["diverged"] for r in runs[n])]
    if not present:
        return
    panels = [("test_acc", "test error (%)", err, None), ("train_loss", "training loss (augmented, train mode)", None, "log"),
              ("grad_norm", "‖∂L/∂W‖, first conv (epoch mean)", None, "log")]
    if extra:
        panels.append((extra, "‖conv weights‖₂", None, None))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4))
    for ax, (key, ylabel, tf, yscale) in zip(axes, panels):
        for i, n in enumerate(present):
            ok = [r for r in runs[n] if not r["final"]["diverged"]]
            curve(ax, ok, key, tf or (lambda a: a), c=f"C{i}", label=LABEL.get(n, n))
        ax.set(xlabel="epoch", ylabel=ylabel)
        if yscale:
            ax.set_yscale(yscale)
        if key == "test_acc":
            ax.set_ylim(5, 30)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="a budget directory, e.g. runs/e60")
    args = ap.parse_args()
    global OUT
    OUT = OUT / Path(args.runs).name
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.runs))
    if not runs:
        raise SystemExit(f"no finished runs under {args.runs}/")
    t = table(runs)
    (OUT / "table.md").write_text(t + "\n")
    print(t)
    fig_degradation(runs)
    fig_compare(runs, ["resnet20", "resnet20_nobn", "resnet20_nobn_lr001"], "bn.png", "BatchNorm on vs off")
    fig_compare(runs, ["resnet20", "resnet20_nowd"], "wd.png", "Weight decay on vs off", extra="weight_norm")
    fig_compare(runs, ["resnet20", "resnet20_adam", "resnet20_adamw"], "optim.png", "SGD+momentum vs Adam vs AdamW",
                extra="weight_norm")
    print(f"wrote {OUT}/table.md and figures")


if __name__ == "__main__":
    main()
