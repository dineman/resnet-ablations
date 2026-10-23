"""Measure seconds per epoch on THIS machine, and print what each budget would cost.

    python -m scripts.benchmark                # ~1-2 min; run it first on Colab/Kaggle

Times 40 real training steps (after 10 warm-up steps) for ResNet-20 and ResNet-56,
with the same code path as training (on-device augmentation, AMP on CUDA).
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from resnet.data import augment
from resnet.models import CifarNet
from resnet.train import Config, make_optimizer, pick_device
from scripts.run_ablation import run_list


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def sec_per_epoch(depth: int, dev: torch.device, steps: int = 40, steps_per_epoch: int = 351) -> float:
    use_amp = dev.type == "cuda"
    model = CifarNet(depth).to(dev)
    if use_amp:
        model = model.to(memory_format=torch.channels_last)
    opt = make_optimizer(Config(), model)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(4096, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (4096,), device=dev)
    for i in range(10 + steps):
        if i == 10:
            sync(dev)
            t = time.time()
        idx = torch.randint(0, 4096, (128,), device=dev, generator=gen)
        xb = augment(x[idx], gen)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            loss = F.cross_entropy(model(xb).float(), y[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    sync(dev)
    # + ~3% for the per-epoch evaluation on 15k images (forward only)
    return (time.time() - t) / steps * steps_per_epoch * 1.03


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpus", type=int, default=1, help="how many GPUs will run shards in parallel")
    args = ap.parse_args()
    dev = pick_device(args.device)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
    t = {d: sec_per_epoch(d, dev) for d in (20, 56)}
    print(f"device: {name}\n  ResNet-20: {t[20]:.1f} s/epoch\n  ResNet-56: {t[56]:.1f} s/epoch")
    jobs = run_list("all", (0, 1, 2))
    depth = {n: Config.from_yaml(f"configs/{n}.yaml").depth for n, _ in jobs}
    print(f"\nall ablations, 3 seeds ({len(jobs)} runs), on {args.gpus} GPU(s) in parallel:")
    for epochs in (30, 60, 91, 182):
        hours = sum(t[depth[n]] for n, _ in jobs) * epochs / 3600
        print(f"  {epochs:3d} epochs: {hours:5.1f} GPU-hours  ->  ~{hours / args.gpus:5.1f} h wall-clock")


if __name__ == "__main__":
    main()
