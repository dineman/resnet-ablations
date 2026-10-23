"""ResNet / plain networks for CIFAR-10, following He et al. 2015, Section 4.2.

Architecture (depth = 6n + 2):

    conv3x3(3 -> 16)  -> BN -> ReLU                       32x32,  16 channels
    n blocks, 16 channels                                  32x32
    n blocks, 32 channels (the first has stride 2)         16x16
    n blocks, 64 channels (the first has stride 2)          8x8
    global average pool -> fc(64 -> 10)

Each "basic block" is two 3x3 convolutions:

    y = relu( F(x) + shortcut(x) ),   F(x) = BN(conv(relu(BN(conv(x)))))

n = 3 gives ResNet-20 (19 conv layers + 1 fc), and n = 9 gives ResNet-56.
In the *plain* variant the "+ shortcut(x)" term is removed. Nothing else
changes: same layers, same parameter count (shortcut option A has no
parameters), same init. That is what makes the degradation comparison
controlled.

Shortcut "option A" (the paper's choice for CIFAR): when a block changes the
shape (stride 2, channels doubled), the shortcut subsamples x spatially by 2
and zero-pads the extra channels. It adds no parameters, so plain and residual
networks have exactly the same parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int, use_bn: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_bn else nn.Identity()


class ShortcutA(nn.Module):
    """Parameter-free shortcut: identity, or (subsample by `stride`, zero-pad channels)."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.stride, self.pad = stride, c_out - c_in
        assert self.pad >= 0

    @property
    def is_identity(self) -> bool:
        return self.stride == 1 and self.pad == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return x
        x = x[:, :, :: self.stride, :: self.stride]
        # F.pad pads the last dims first: (W_left, W_right, H_top, H_bottom, C_front, C_back).
        # Split the new channels on both sides, as the original Torch implementation did.
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))


class BasicBlock(nn.Module):
    """Two 3x3 convs with BN, plus an (optional) shortcut.

    Input  (N, c_in, H, W) float32
    Output (N, c_out, H/stride, W/stride)
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1, residual: bool = True, use_bn: bool = True):
        super().__init__()
        # With BN, the conv bias is redundant: BN subtracts the mean, then adds its own beta.
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=not use_bn)
        self.bn1 = _norm(c_out, use_bn)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=not use_bn)
        self.bn2 = _norm(c_out, use_bn)
        self.residual = residual
        self.shortcut = ShortcutA(c_in, c_out, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class CifarNet(nn.Module):
    """ResNet-(6n+2) or its plain counterpart.

    Args:
        depth:    6n + 2, e.g. 20, 32, 44, 56, 110.
        residual: True for ResNet, False for the plain network.
        use_bn:   False removes every BatchNorm (the convs get biases instead).
    Input (N, 3, 32, 32) float32 -> logits (N, num_classes).
    """

    def __init__(self, depth: int = 20, residual: bool = True, use_bn: bool = True, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n + 2"
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=not use_bn)
        self.bn = _norm(16, use_bn)
        blocks, c_in = [], 16
        for stage, c_out in enumerate((16, 32, 64)):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                blocks.append(BasicBlock(c_in, c_out, stride, residual, use_bn))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, num_classes)
        self._init()

    def _init(self) -> None:
        # He et al. 2015 (ICCV, "Delving deep into rectifiers") init, as in the ResNet paper.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))  # global average pool: (N, 64, 8, 8) -> (N, 64)
        return self.fc(x)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
