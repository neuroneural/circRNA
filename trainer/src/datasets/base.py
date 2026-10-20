"""The dataset contract every loader here shares: sampler, collate, base class.

Concrete datasets live beside this file; ``src/registry.py`` picks between them.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def safe_normalize(img):
    """Unit-interval normalisation that tolerates constant volumes."""
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return torch.zeros_like(img)
    return (img - mn) / (mx - mn)


class SeededBatchSampler:
    """Vendored from ``DBBatchSampler`` in mindfultensors:
    https://github.com/neuroneural/mindfultensors/blob/main/mindfultensors/utils.py


    Passed as ``sampler=`` (not ``batch_sampler=``), which is what makes each yielded
    array arrive at ``__getitem__`` as a single "index".
    """

    def __init__(self, data_source, batch_size=1, seed=None):
        self.data_size = len(data_source)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(None if self.seed is None else self.seed + self.epoch)
        order = rng.permutation(self.data_size)
        return iter(self._batches(order))

    def _batches(self, order):
        """Split an ordering into batches, leaving no batch of one."""
        step = self.batch_size
        batches = [order[i : i + step] for i in range(0, self.data_size, step)]
        # Fix for trailing batch of size 1
        if len(batches) > 1 and len(batches[-1]) == 1:
            tail = np.concatenate(batches[-2:])
            cut = len(tail) // 2
            batches[-2:] = [tail[:cut], tail[cut:]] if cut > 1 else [tail]
        return batches

    def __len__(self):
        # sizes depend only on data_size and batch_size, not on the ordering
        return len(self._batches(np.arange(self.data_size)))


def collate_volumes(results):
    """``[{index: record}]`` -> ``(inputs [B,C,D,H,W], labels [B], indices [B])``.

    ``indices`` are positions in the full dataset, so a prediction row joins back to
    ``dataset.ids`` and to ``split_indices.json``.
    """
    results = results[0]
    inputs = torch.stack([record["input"] for record in results.values()])
    if inputs.ndim == 4:  # a single-channel dataset returned [B,D,H,W]
        inputs = inputs.unsqueeze(1)
    labels = torch.stack([record["label"] for record in results.values()])
    indices = torch.tensor([record["index"] for record in results.values()])
    return inputs, labels.long(), indices.long()


class VolumeDataset(Dataset):
    """Base class. Subclasses set ``labels`` / ``ids`` and implement ``load``."""

    labels: np.ndarray  # one int per sample, used for the stratified CV split
    ids: list           # one readable id per sample, dumped to <split>_ids.txt
    n_channels = 1      # channels per sample; main() sizes the model's input from it

    normalize = staticmethod(safe_normalize)

    def load(self, index):
        """Sample ``index`` as ``[D,H,W]``, or ``[C,D,H,W]`` when it has channels."""
        raise NotImplementedError

    def __len__(self):
        return len(self.labels)

    def prefetch(self, batch):
        """Optional: pull a whole batch at once, where that is cheaper than one at a time."""

    def __getitem__(self, batch):
        self.prefetch(batch)
        results = {}
        for index in batch:
            index = int(index)
            volume = self.load(index).float()
            if volume.ndim not in (3, 4):
                raise ValueError(f"sample {index} is not 3D or 4D: {tuple(volume.shape)}")
            results[index] = {
                "input": self.normalize(volume) if self.normalize else volume,
                "label": torch.tensor(int(self.labels[index])),   # CrossEntropy wants a class index
                "index": index,
            }
        return results

    def subset(self, indices):
        """A view over ``indices``; records keep their index in the full dataset."""
        return _Subset(self, indices)


class _Subset(Dataset):
    def __init__(self, parent, indices):
        self.parent = parent
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, batch):
        return self.parent[[self.indices[int(i)] for i in batch]]
