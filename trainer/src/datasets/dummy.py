"""Random volumes, for smoke-testing the pipeline without real data."""

import numpy as np
import torch
from omegaconf import OmegaConf

from src.datasets.base import VolumeDataset


def default_HPs(cfg):
    return OmegaConf.create(
        {
            "n_samples": 64,
            "shape": [64, 64, 64],
            "channels": 1,      # >1 stacks that many noise volumes, to exercise a wider model
            "n_classes": 2,
            "signal": 0.3,      # blob amplitude in noise SDs
            "seed": 0,
        }
    )


class DummyDataset(VolumeDataset):
    """Random volumes and labels, for smoke-testing the pipeline without real data.

    Volumes are generated per index from ``seed``, so a sample is identical across
    epochs and ranks. ``signal`` brightens a cube at the centre of the nonzero classes,
    measured in units of the noise SD and not capped at 1: ``0`` leaves nothing to learn
    and AUC sits at chance, the default ``0.3`` is faint, and ``3`` fits to AUC 1.0.

    The signal is a local blob rather than a mean shift because ``safe_normalize``
    rescales every volume on its own min/max, which would erase a global shift.
    """

    def __init__(self, params):
        self.shape = tuple(params.shape)
        self.n_channels = params.channels
        self.signal = params.signal
        self.seed = params.seed
        rng = np.random.default_rng(self.seed)
        self.labels = rng.integers(0, params.n_classes, size=params.n_samples)
        self.ids = [f"dummy_{i:05d}" for i in range(params.n_samples)]

    def load(self, index):
        shape = self.shape if self.n_channels == 1 else (self.n_channels, *self.shape)
        volume = np.random.default_rng(self.seed + index).normal(size=shape)
        if self.signal and self.labels[index]:
            side = max(2, min(self.shape) // 4)
            blob = tuple(slice((n - side) // 2, (n + side) // 2) for n in self.shape)
            volume[(Ellipsis, *blob)] += self.signal * self.labels[index]   # every channel
        return torch.from_numpy(volume)


DATASET = DummyDataset
