"""Run every training run of one ablation (or all of them), 3 seeds each, skipping finished runs.

    python -m scripts.run_ablation degradation                 # paper budget (182 epochs) -> runs/e182/
    python -m scripts.run_ablation all --epochs 60             # reduced budget, same schedule shape -> runs/e60/
    python -m scripts.run_ablation all --shard 0/2 --device cuda:0   # Kaggle 2xT4: run half on GPU 0 ...
    python -m scripts.run_ablation all --shard 1/2 --device cuda:1   # ... and the other half on GPU 1
    python -m scripts.run_ablation all --list                  # print the run list and estimated cost

Each run is resumable (see resnet/train.py), so after a disconnected session
you re-run the same command. Runs go to runs/e<epochs>/, so different budgets
never mix in one table.

The baseline (resnet20) belongs to every ablation, but it is one set of runs,
shared. Each comparison changes exactly ONE thing relative to it:

    degradation   plain20, resnet20, plain56, resnet56   (shortcuts x depth)
    bn            resnet20  vs  resnet20_nobn (lr 0.1)  and  resnet20_nobn_lr001 (lr 0.01)
    wd            resnet20  vs  resnet20_nowd
    optim         resnet20 (SGD+momentum)  vs  resnet20_adam  vs  resnet20_adamw
"""

from __future__ import annotations

import argparse

from resnet.data import load_splits
from resnet.train import Config, pick_device, train

ABLATIONS = {
    "degradation": ["resnet20", "plain20", "resnet56", "plain56"],
    "bn": ["resnet20", "resnet20_nobn", "resnet20_nobn_lr001"],
    "wd": ["resnet20", "resnet20_nowd"],
    "optim": ["resnet20", "resnet20_adam", "resnet20_adamw"],
}
SEEDS = (0, 1, 2)
# Relative cost per run, in ResNet-20 units (conv FLOPs scale with depth: 56/20 = 2.8).
COST = {20: 1.0, 56: 2.8}


def run_list(ablation: str, seeds: tuple[int, ...]) -> list[tuple[str, int]]:
    names = ABLATIONS[ablation] if ablation != "all" else list(dict.fromkeys(n for v in ABLATIONS.values() for n in v))
    # Seed-major order: after the first pass, every variant has one finished seed, so partial
    # results are already comparable if the compute runs out.
    return [(n, s) for s in seeds for n in names]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ablation", choices=[*ABLATIONS, "all"])
    ap.add_argument("--epochs", type=int, default=None, help="override the 182-epoch paper budget")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", default=None, help="default: runs/e<epochs>")
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th job starting at i (for multiple GPUs)")
    ap.add_argument("--list", action="store_true", help="print the jobs and their cost, then exit")
    args = ap.parse_args()

    i, k = map(int, args.shard.split("/"))
    epochs = args.epochs or Config.epochs
    out_dir = args.out_dir or f"runs/e{epochs}"
    jobs = run_list(args.ablation, tuple(args.seeds))[i::k]
    cfgs = [Config.from_yaml(f"configs/{n}.yaml", seed=s, epochs=epochs, device=args.device,
                             out_dir=out_dir) for n, s in jobs]
    cost = sum(COST[c.depth] * c.epochs for c in cfgs)
    print(f"{len(cfgs)} runs, {cost:.0f} ResNet-20-epochs "
          f"(~{cost * 10 / 3600:.1f} GPU-hours at ~10 s/epoch on a T4)")
    if args.list:
        for c in cfgs:
            done = (c.run_dir / "final.json").exists()
            print(f"  {'done' if done else 'todo'}  {c.name:22s} seed {c.seed}  {c.epochs} epochs")
        return

    data = load_splits(pick_device(args.device))  # load CIFAR once, share it across runs
    for c in cfgs:
        train(c, data)


if __name__ == "__main__":
    main()
