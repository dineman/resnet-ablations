"""Train one run from a YAML config; any config field can be overridden on the command line.

    python -m scripts.train --config configs/resnet20.yaml --seed 1
    python -m scripts.train --config configs/plain56.yaml --epochs 60 --device cuda

Re-running the same command resumes an interrupted run, and skips one that is finished.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from resnet.train import Config, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    for f in fields(Config):
        if f.name == "milestones":
            ap.add_argument("--milestones", type=float, nargs="+")
        elif f.type in ("bool", bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes", "on"))
        else:
            ap.add_argument(f"--{f.name}", type={"int": int, "float": float}.get(f.type, str))
    args = vars(ap.parse_args())
    cfg = Config.from_yaml(args.pop("config"), **args)
    print(train(cfg))


if __name__ == "__main__":
    main()
