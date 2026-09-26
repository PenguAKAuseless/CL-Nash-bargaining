"""Reduced ResNet-18 for CIFAR, with GroupNorm instead of BatchNorm.

The architecture uses a 3x3 stem and no max-pool so 32x32 inputs survive to
the final stage. GroupNorm replaces BatchNorm throughout.

BatchNorm breaks the matrix-free reduction used by the bargain
bargain (E6/E7) -- with players' batches concatenated, its statistics are
computed across the union, so a per-player loss no longer depends on that
player's data alone. GroupNorm normalises per-example (over channel groups),
so it has no such cross-example coupling. Used here too (not just E6/E7) so
one trained checkpoint is reusable across experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """The largest group count <= max_groups that divides num_channels."""
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def make_norm_layer(norm: str):
    """norm='group' (default) or 'batch' for the normalization ablation
    asks for, to MEASURE the induced error from concatenated-batch player
    coupling, not to use in place of GroupNorm."""
    if norm == "group":
        return group_norm
    if norm == "batch":
        return lambda num_channels: nn.BatchNorm2d(num_channels)
    raise ValueError(f"unknown norm {norm!r}, expected 'group' or 'batch'")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, norm: str = "group") -> None:
        super().__init__()
        make_norm = make_norm_layer(norm)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.norm1 = make_norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = make_norm(planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                make_norm(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """Reduced ResNet-18, GroupNorm by default (norm='batch' is the E6
    ablation arm only) and a single linear head over all classes."""

    def __init__(self, num_classes: int, nf: int = 20, norm: str = "group") -> None:
        super().__init__()
        self.norm_kind = norm
        make_norm = make_norm_layer(norm)
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf)
        self.norm1 = make_norm(nf)
        self.layer1 = self._make_layer(nf, 2, stride=1)
        self.layer2 = self._make_layer(nf * 2, 2, stride=2)
        self.layer3 = self._make_layer(nf * 4, 2, stride=2)
        self.layer4 = self._make_layer(nf * 8, 2, stride=2)
        self.n_features = nf * 8
        self.linear = nn.Linear(self.n_features, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s, norm=self.norm_kind))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(x))


def make_backbone(
    num_classes: int, device: torch.device, nf: int = 20, norm: str = "group"
) -> ResNet:
    return ResNet(num_classes=num_classes, nf=nf, norm=norm).to(device)


__all__ = ["BasicBlock", "ResNet", "conv3x3", "group_norm", "make_backbone", "make_norm_layer"]
