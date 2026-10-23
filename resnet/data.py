"""CIFAR-10 loading, the train/val split, and on-device augmentation.

Design choice: the whole dataset (45k x 3 x 32 x 32 float32 = 553 MB, or
138 MB as uint8) sits on the training device, and augmentation is done there
as batched tensor ops. A torchvision DataLoader with per-image PIL transforms
is the usual approach, but Colab gives you 2 CPU cores. ResNet-20 on a GPU
consumes images faster than 2 cores can crop and flip them, so the GPU would
sit idle waiting for data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_cifar10(root: Path = RAW) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X_train (50000, 3, 32, 32) uint8, y_train (50000,) int64, X_test (10000, 3, 32, 32), y_test.

    Binary format: each record is 1 label byte followed by 3072 pixel bytes
    (1024 R, 1024 G, 1024 B, each 32x32 row-major), already channels-first.
    """
    d = root / "cifar-10-batches-bin"

    def read(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
        raw = np.concatenate([np.fromfile(d / f, dtype=np.uint8).reshape(-1, 3073) for f in files])
        return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.int64)

    X_tr, y_tr = read([f"data_batch_{i}.bin" for i in range(1, 6)])
    X_te, y_te = read(["test_batch.bin"])
    return X_tr, y_tr, X_te, y_te


@dataclass
class Split:
    """Standardized float32 tensors on one device. x: (N, 3, 32, 32), y: (N,) int64."""

    x: torch.Tensor
    y: torch.Tensor


def load_splits(device: str | torch.device, n_val: int = 5000, split_seed: int = 0) -> dict[str, Split]:
    """45k train / 5k val / 10k test, standardized with per-channel TRAIN statistics.

    This is the split He et al. used for CIFAR (Section 4.2). The val split is
    fixed (split_seed = 0) for every run, so all runs see the same 45k
    training images, whatever their training seed. The val set is used only for
    monitoring and for the one choice we make (the no-BN learning rate). The
    test set is never used to choose anything.
    """
    X, y, Xt, yt = load_cifar10()
    perm = np.random.default_rng(split_seed).permutation(len(X))
    va, tr = perm[:n_val], perm[n_val:]
    Xtr = X[tr].astype(np.float32) / 255.0
    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = Xtr.std(axis=(0, 2, 3), keepdims=True)

    def make(x: np.ndarray, labels: np.ndarray) -> Split:
        x = (x.astype(np.float32) / 255.0 - mean) / std
        return Split(torch.from_numpy(x).to(device), torch.from_numpy(labels).to(device))

    return {"train": make(X[tr], y[tr]), "val": make(X[va], y[va]), "test": make(Xt, yt)}


def augment(x: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Random crop with `pad`-pixel zero padding, plus random horizontal flip, for a whole batch at once.

    x: (N, C, H, W) standardized float. Zero padding of the *standardized* image
    equals padding with the dataset-mean colour, a negligible difference from
    padding the raw image with black.

    gen is a torch.Generator on x's device, so augmentation is reproducible
    from the run seed.
    """
    N, C, H, W = x.shape
    dev = x.device
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    # One (dy, dx) offset in [0, 2*pad] per image. Gather the crops with advanced indexing:
    # rows[n, i] = dy[n] + i, cols[n, j] = dx[n] + j  ->  xp[n, :, rows[n, i], cols[n, j]].
    dy = torch.randint(0, 2 * pad + 1, (N,), device=dev, generator=gen)
    dx = torch.randint(0, 2 * pad + 1, (N,), device=dev, generator=gen)
    ar_h, ar_w = torch.arange(H, device=dev), torch.arange(W, device=dev)
    rows = (dy[:, None] + ar_h)[:, :, None]  # (N, H, 1)
    cols = (dx[:, None] + ar_w)[:, None, :]  # (N, 1, W)
    n = torch.arange(N, device=dev)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[n, rows, cols]  # (N, H, W, C)
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(N, device=dev, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out).contiguous()
