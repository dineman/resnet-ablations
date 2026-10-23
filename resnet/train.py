"""One training run: config -> model, optimizer, schedule -> per-epoch CSV log, checkpoints, final JSON.

A run lives in ``runs/<name>/seed<k>/``:

    config.yaml   the exact config used (written first, so a run is always traceable)
    log.csv       one row per epoch (appended, so it survives a resume)
    last.pt       model + optimizer + scheduler + RNG state, written every epoch
    final.json    written only when the run completes; its presence means "done"

If ``last.pt`` exists and ``final.json`` doesn't, the run resumes from the
last finished epoch. This matters on Colab/Kaggle, where sessions get killed:
re-running the same command picks up where it stopped.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from resnet.data import Split, augment, load_splits
from resnet.models import CifarNet, num_params


@dataclass
class Config:
    """Everything that defines a run. Defaults are He et al. 2015, Section 4.2."""

    name: str = "resnet20"
    depth: int = 20
    residual: bool = True
    bn: bool = True
    optimizer: str = "sgd"  # sgd | adam | adamw
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # Paper: divide the lr by 10 at 32k and 48k iterations and stop at 64k. With
    # 45k images and batch 128 that is 352 steps/epoch, so epochs 91 and 136 of 182,
    # i.e. 50% and 75% of training. We keep the *fractions*, so a reduced budget
    # (e.g. --epochs 60) gets the same shape of schedule.
    epochs: int = 182
    milestones: tuple[float, ...] = (0.5, 0.75)
    gamma: float = 0.1
    batch_size: int = 128
    augment: bool = True
    seed: int = 0
    device: str = "auto"  # auto | cuda | mps | cpu
    amp: bool = True  # mixed precision, used on CUDA only
    out_dir: str = "runs"
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
        d = yaml.safe_load(Path(path).read_text()) or {}
        d.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if "milestones" in d:
            d["milestones"] = tuple(d["milestones"])
        return cls(**d)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name / f"seed{self.seed}"


def pick_device(want: str) -> torch.device:
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Note what weight_decay *means* for each optimizer.

    SGD and Adam: L2 regularization, i.e. wd * theta is added to the gradient.
      In Adam this term then gets divided by sqrt(v), like the rest of the
      gradient, so parameters with large gradients are barely decayed.
    AdamW: decoupled decay, theta <- theta - lr * wd * theta, applied outside
      the adaptive update, so every parameter shrinks at the same rate.
    """
    p = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(p, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


@torch.no_grad()
def evaluate(model: torch.nn.Module, split: Split, bs: int = 1000) -> tuple[float, float]:
    """(mean loss, accuracy) in eval mode, i.e. BN uses its running statistics."""
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(split.y), bs):
        x, y = split.x[i : i + bs], split.y[i : i + bs]
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / len(split.y), correct / len(split.y)


LOG_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc",
              "grad_norm", "weight_norm", "seconds"]


def conv_weight_norm(model: torch.nn.Module) -> float:
    """L2 norm of all conv weights. With BN after every conv, the network's function doesn't depend on this
    scale, but the *effective* learning rate does (~ lr / norm^2), which is how weight decay acts here."""
    return math.sqrt(sum(m.weight.detach().float().pow(2).sum().item()
                         for m in model.modules() if isinstance(m, torch.nn.Conv2d)))


def train(cfg: Config, data: dict[str, Split] | None = None, verbose: bool = True) -> dict:
    """Run (or resume) one training run. Returns the final summary dict.

    `data` can be passed in to avoid reloading CIFAR for every run in a sweep.
    """
    out = cfg.run_dir
    if (out / "final.json").exists():
        if verbose:
            print(f"[skip] {out} already finished")
        return json.loads((out / "final.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**asdict(cfg), "milestones": list(cfg.milestones)}, sort_keys=False))

    device = pick_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    data = data or load_splits(device)
    train_x, train_y = data["train"].x, data["train"].y
    n = len(train_y)
    steps_per_epoch = n // cfg.batch_size  # drop the last partial batch: every step has the same BN batch size

    # Seed control. torch.manual_seed sets the weight init. A separate generator on the
    # data device drives shuffling and augmentation, so its state can be checkpointed.
    torch.manual_seed(cfg.seed)
    model = CifarNet(cfg.depth, cfg.residual, cfg.bn).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = make_optimizer(cfg, model)
    milestones = [round(f * cfg.epochs) for f in cfg.milestones]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, cfg.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    start_epoch = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["gen"].cpu() if device.type != "cpu" else ck["gen"])
        start_epoch = ck["epoch"]
        if verbose:
            print(f"[resume] {out} from epoch {start_epoch}")
    elif (out / "log.csv").exists():
        (out / "log.csv").unlink()  # a log without a checkpoint is from a run that died in epoch 1

    first_conv = model.conv.weight
    if verbose and start_epoch == 0:
        print(f"[start] {out}  params={num_params(model):,}  device={device}  steps/epoch={steps_per_epoch}")

    diverged = False
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=device, generator=gen)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        gnorm_sum = torch.zeros((), device=device)
        gnorm_n = torch.zeros((), device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * cfg.batch_size : (s + 1) * cfg.batch_size]
            x, y = train_x[idx], train_y[idx]
            if cfg.augment:
                x = augment(x, gen)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # Gradient norm of the FIRST conv layer: in a deep plain net, this is where
            # vanishing gradients show up. Accumulated on-device to avoid a sync per step.
            # Under fp16 the scaler sometimes overflows: the norm is inf and scaler.step skips
            # the update. Those steps are left out, or one of them would make the epoch mean inf.
            g = first_conv.grad.norm()
            ok = torch.isfinite(g)
            gnorm_sum += torch.where(ok, g, torch.zeros_like(g))
            gnorm_n += ok
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.detach()
            correct += (logits.argmax(1) == y).sum()
        train_loss = loss_sum.item() / steps_per_epoch
        if not math.isfinite(train_loss):
            diverged = True
            print(f"[diverged] {out} at epoch {epoch + 1}")
            break
        row = {
            "epoch": epoch + 1,
            "lr": opt.param_groups[0]["lr"],
            "train_loss": train_loss,
            # Running accuracy over the epoch's *augmented* batches in train mode (as in the paper's
            # training-error curves); the weights change during the epoch, so it lags the end-of-epoch model.
            "train_acc": correct.item() / (steps_per_epoch * cfg.batch_size),
        }
        sched.step()
        row["val_loss"], row["val_acc"] = evaluate(model, data["val"])
        # Test metrics are logged every epoch only to draw curves like the paper's Fig. 6.
        # No decision (lr, epoch, variant) is ever made from them.
        row["test_loss"], row["test_acc"] = evaluate(model, data["test"])
        row["grad_norm"] = gnorm_sum.item() / max(gnorm_n.item(), 1)
        row["weight_norm"] = conv_weight_norm(model)
        row["seconds"] = time.time() - t0

        new_file = not (out / "log.csv").exists()
        with (out / "log.csv").open("a", newline="") as f:
            w = csv.DictWriter(f, LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        # Write to a temp file and rename, so a kill during the save can't corrupt the checkpoint.
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "gen": gen.get_state(), "epoch": epoch + 1},
                   out / "last.tmp")
        (out / "last.tmp").replace(ckpt_path)
        if verbose:
            print(f"{cfg.name} s{cfg.seed} ep {epoch + 1:3d}/{cfg.epochs}  lr {row['lr']:.0e}  "
                  f"train {row['train_loss']:.3f}/{100 * row['train_acc']:.1f}%  "
                  f"val {100 * row['val_acc']:.2f}%  test {100 * row['test_acc']:.2f}%  {row['seconds']:.1f}s",
                  flush=True)

    summary = {"name": cfg.name, "seed": cfg.seed, "params": num_params(model), "diverged": diverged,
               "epochs_run": epoch + 1 - diverged, "device": str(device)}
    if not diverged:
        summary.update({k: row[k] for k in ("train_loss", "train_acc", "val_acc", "test_acc", "test_loss")})
    (out / "final.json").write_text(json.dumps(summary, indent=1))
    return summary
