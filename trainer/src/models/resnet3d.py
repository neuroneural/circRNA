"""A plain 3D ResNet for whole-volume classification.

Model modules own their own defaults. ``default_HPs`` may read the global config, so
data-dependent sizes (channels, classes) come from one place; anything under
``model.params`` in the experiment config overrides it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


def default_HPs(cfg):
    runtime = cfg.get("runtime") or {}
    return OmegaConf.create(
        {
            "in_channels": runtime.get("n_channels") or 1,
            "n_classes": runtime.get("n_classes") or 2,
            "base_channels": 64,     # width of stage 0; stage i is base_channels * 2**i
            "blocks_per_stage": [2, 2, 2, 2],   # torchvision's `layers`; [2,2,2,2] = ResNet-18
            "dropout": 0.0,
        }
    )


class BasicBlock3D(nn.Module):
    """Standard two-conv residual block, 3D."""

    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels, track_running_stats=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels, track_running_stats=True)
        self.dropout = nn.Dropout3d(dropout_p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += residual
        return F.relu(out)


class ResNet3D(nn.Module):
    """Stage i has ``base_channels * 2**i`` width; stage 0 keeps the spatial size,
    every later stage halves it."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        channels = params.base_channels
        self.conv1 = nn.Conv3d(
            params.in_channels, channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        stages, width_in = [], channels
        for stage_idx, n_blocks in enumerate(params.blocks_per_stage):
            width_out = channels * (2 ** stage_idx)
            stride = 1 if stage_idx == 0 else 2
            stages.append(
                self._make_stage(width_in, width_out, n_blocks, stride, params.dropout)
            )
            width_in = width_out
        self.stages = nn.Sequential(*stages)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(width_in, params.n_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _make_stage(in_channels, out_channels, n_blocks, stride, dropout):
        modules = [BasicBlock3D(in_channels, out_channels, stride, dropout)]
        for _ in range(1, n_blocks):
            modules.append(BasicBlock3D(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*modules)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


MODEL = ResNet3D
