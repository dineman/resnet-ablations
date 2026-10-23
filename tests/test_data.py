"""Augmentation and data-split tests."""

from __future__ import annotations

import pytest
import torch

from resnet.data import RAW, augment, load_splits


def test_augment_shape_and_reproducibility():
    x = torch.randn(16, 3, 32, 32)
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    a, b = augment(x, g1), augment(x, g2)
    assert a.shape == x.shape
    torch.testing.assert_close(a, b)


def test_augment_is_a_shifted_window_possibly_flipped():
    """Every output must equal some 32x32 window of the zero-padded input, either as-is or mirrored."""
    x = torch.randn(64, 3, 32, 32)
    out = augment(x, torch.Generator().manual_seed(1))
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    n_flipped = 0
    for n in range(len(x)):
        found = False
        for dy in range(9):
            for dx in range(9):
                win = xp[n, :, dy : dy + 32, dx : dx + 32]
                if torch.equal(out[n], win) or torch.equal(out[n], win.flip(2)):
                    found = True
                    n_flipped += not torch.equal(out[n], win)
                    break
            if found:
                break
        assert found, f"image {n} is not a crop of its padded input"
    assert 16 < n_flipped < 48  # about half flipped


def test_augment_offsets_cover_full_range():
    """Vertical offsets 0..8 should all occur. (A classic bug: randint(0, 2*pad) never produces offset 8.)

    Mark the top-left and bottom-left pixels. With vertical offset dy, the top marker (padded row 4)
    lands on row 4 - dy, visible for dy in 0..4. The bottom marker (padded row 35) lands on row 35 - dy,
    visible for dy in 4..8. Flips only move columns, so rows identify dy exactly.
    """
    x = torch.zeros(2000, 1, 32, 32)
    x[:, 0, 0, 0] = 1.0
    x[:, 0, 31, 0] = 2.0
    out = augment(x, torch.Generator().manual_seed(0))
    dys = set()
    for img in out[:, 0]:
        for marker, padded_row in ((1.0, 4), (2.0, 35)):
            pos = (img == marker).nonzero()
            if len(pos):
                dys.add(padded_row - pos[0, 0].item())
    assert dys == set(range(9))


@pytest.mark.skipif(not (RAW / "cifar-10-batches-bin").exists(), reason="CIFAR-10 not downloaded")
def test_splits():
    d = load_splits("cpu")
    assert [len(d[k].y) for k in ("train", "val", "test")] == [45000, 5000, 10000]
    # Standardized with TRAIN stats: per-channel mean ~0 and std ~1 on train, only approximately on val.
    torch.testing.assert_close(d["train"].x.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4, rtol=0)
    torch.testing.assert_close(d["train"].x.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-3, rtol=0)
    assert torch.bincount(d["test"].y).tolist() == [1000] * 10
    # Same split every time (split seed fixed)
    torch.testing.assert_close(load_splits("cpu")["val"].x[:10], d["val"].x[:10])
