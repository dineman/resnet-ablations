"""Model tests: shapes per block, identity shortcuts, parameter counts, BN behaviour, and overfitting a tiny batch."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from resnet.models import BasicBlock, CifarNet, ShortcutA, num_params


@pytest.mark.parametrize("c_in,c_out,stride,hw_out", [(16, 16, 1, 32), (16, 32, 2, 16), (32, 64, 2, 8), (64, 64, 1, 8)])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("use_bn", [True, False])
def test_block_shapes(c_in, c_out, stride, hw_out, residual, use_bn):
    hw_in = hw_out * stride
    y = BasicBlock(c_in, c_out, stride, residual, use_bn)(torch.randn(4, c_in, hw_in, hw_in))
    assert y.shape == (4, c_out, hw_out, hw_out)
    assert (y >= 0).all()  # the block ends in a ReLU


@pytest.mark.parametrize("depth", [20, 56])
@pytest.mark.parametrize("residual", [True, False])
def test_network_output_shape(depth, residual):
    assert CifarNet(depth, residual)(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_stage_shapes():
    """After each stage: (16, 32, 32) -> (32, 16, 16) -> (64, 8, 8), as in the paper's table in Sec. 4.2."""
    m = CifarNet(20)
    x = F.relu(m.bn(m.conv(torch.randn(2, 3, 32, 32))))
    shapes = []
    for b in m.blocks:
        x = b(x)
        shapes.append(tuple(x.shape[1:]))
    assert shapes[2] == (16, 32, 32) and shapes[5] == (32, 16, 16) and shapes[8] == (64, 8, 8)


def test_shortcut_is_exact_identity_when_shapes_match():
    s = ShortcutA(16, 16, 1)
    x = torch.randn(3, 16, 8, 8)
    assert s.is_identity
    assert s(x) is x  # not just equal: the very same tensor, no copy, no parameters
    assert sum(p.numel() for p in s.parameters()) == 0


def test_block_with_zero_residual_branch_is_identity():
    """If F(x) = 0 (here: zero the last BN's gamma and beta), a residual block reduces to relu(x) = x for x >= 0.
    This is the "easy to learn the identity" argument from the paper, checked directly."""
    b = BasicBlock(16, 16, 1, residual=True).eval()
    torch.nn.init.zeros_(b.bn2.weight)
    torch.nn.init.zeros_(b.bn2.bias)
    x = torch.rand(2, 16, 8, 8)  # non-negative, like the output of a previous block
    torch.testing.assert_close(b(x), x)
    # A plain block with the same F = 0 outputs all zeros: it has lost the input entirely.
    p = BasicBlock(16, 16, 1, residual=False).eval()
    torch.nn.init.zeros_(p.bn2.weight)
    torch.nn.init.zeros_(p.bn2.bias)
    assert p(x).abs().max() == 0


def test_downsampling_shortcut_option_a():
    """Shape-changing shortcut: subsample by 2, zero-pad channels, so the original channels pass through unchanged."""
    s = ShortcutA(16, 32, 2)
    x = torch.randn(2, 16, 8, 8)
    y = s(x)
    assert y.shape == (2, 32, 4, 4)
    torch.testing.assert_close(y[:, 8:24], x[:, :, ::2, ::2])
    assert y[:, :8].abs().max() == 0 and y[:, 24:].abs().max() == 0


def test_parameter_counts():
    """The paper's table: ResNet-20 has 0.27M and ResNet-56 0.85M parameters. Plain nets are identical
    (option-A shortcuts are parameter-free), so the degradation comparison is at equal capacity."""
    assert num_params(CifarNet(20)) == 269_722
    assert num_params(CifarNet(56)) == 853_018
    assert num_params(CifarNet(20, residual=False)) == num_params(CifarNet(20))
    assert num_params(CifarNet(56, residual=False)) == num_params(CifarNet(56))


def test_no_bn_variant_has_no_batchnorm():
    m = CifarNet(20, use_bn=False)
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.modules())
    assert all(c.bias is not None for c in m.modules() if isinstance(c, torch.nn.Conv2d))


def test_batchnorm_train_vs_eval():
    """Train mode normalizes with the CURRENT batch's statistics, so one image's output depends on the other
    images in its batch. Eval mode uses the running averages, so outputs are per-image deterministic."""
    torch.manual_seed(0)
    m = CifarNet(20)
    x = torch.randn(8, 3, 32, 32)
    other = torch.randn(8, 3, 32, 32) * 3 + 1
    m.train()
    with torch.no_grad():
        a = m(torch.cat([x, other]))[:8]
        b = m(torch.cat([x, -other]))[:8]
    assert not torch.allclose(a, b, atol=1e-4)  # same x, different batch-mates -> different output
    m.eval()
    with torch.no_grad():
        a = m(torch.cat([x, other]))[:8]
        b = m(torch.cat([x, -other]))[:8]
    torch.testing.assert_close(a, b)


def test_running_stats_update_only_in_train_mode():
    m = CifarNet(20)
    before = m.bn.running_mean.clone()
    m.eval()
    with torch.no_grad():
        m(torch.randn(4, 3, 32, 32) + 5)
    torch.testing.assert_close(m.bn.running_mean, before)
    m.train()
    with torch.no_grad():
        m(torch.randn(4, 3, 32, 32) + 5)
    assert not torch.allclose(m.bn.running_mean, before)


@pytest.mark.parametrize("residual", [True, False])
def test_can_overfit_tiny_batch(residual):
    """32 random images with random labels: memorizing them is pure capacity + optimization, no generalization.
    If this fails, something in the forward/backward/optimizer path is broken."""
    torch.manual_seed(0)
    m = CifarNet(20, residual)
    x, y = torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(150):
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    m.eval()  # also in eval mode, i.e. with BN running statistics instead of batch statistics
    with torch.no_grad():
        assert (m(x).argmax(1) == y).all()
