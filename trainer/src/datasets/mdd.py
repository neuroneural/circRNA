"""MDD DIRECT volumes: labels and QC tiers from one csv, volume paths from another.

``02/mdd_master.csv`` carries the labels and the QC flags, ``03/mdd_direct_volume_paths.csv``
the per-subject file paths; they join on subject id.
"""

import csv
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from src.datasets.base import VolumeDataset, safe_normalize

MODALITIES = {
    "gm": "gm_probseg_3mm_path",
    "wm": "wm_probseg_3mm_path",
    "csf": "csf_probseg_3mm_path",
    "falff": "falff_3mm_path",
    "falff_global": "falff_globalc_3mm_path",
}


def default_HPs(cfg):
    return OmegaConf.create(
        {
            "master_csv": "../docs/02_on_MDD_data/data/mdd_master.csv",   # labels and QC flags
            "paths_csv": "../docs/03_multimod/data/mdd_direct_volume_paths.csv",  # volume paths
            "modalities": ["gm"],        # MODALITIES keys, in channel order
            "filters": ["in_super_clean"],  # csv flags that must all read True
            "ranges": {"TR": 2.0},       # numeric column -> [min, max], or a single value
            "require": ["Diagnosis", "Sex", "Age"],  # columns that must hold a number
            "label_column": "Diagnosis",  # the column `classes` is read from
            "classes": [[0], [1]],       # value lists; a row's class is the list it falls in
            "fill_missing": {},          # column == value pairs that read a blank label as 0
            "normalize": "falff",        # one of NORMALIZE, see load()
            "check_files": True,         # drop subjects whose volume files are missing
        }
    )


NORMALIZE = ("falff", "per_channel", "volume", "none")


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MDDDataset(VolumeDataset):
    """One row per subject; ``modalities`` become channels of the same volume."""

    normalize = None  # handled in load(), per the `normalize` param

    def __init__(self, params):
        self.params = params
        self.n_channels = len(params.modalities)   # one channel per modality

        # validate the config
        unknown = set(params.modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities {sorted(unknown)}; have {sorted(MODALITIES)}")
        if len(params.modalities) > 1 and any("native" in m for m in params.modalities):
            raise ValueError("native modalities sit on their own grids; stack 3mm ones instead")
        if params.normalize not in NORMALIZE:
            raise ValueError(f"normalize must be one of {NORMALIZE}, got {params.normalize!r}")
        ranges = {
            column: (bounds, bounds) if isinstance(bounds, (int, float)) else tuple(bounds)
            for column, bounds in params.ranges.items()
        }

        # read volume paths csv, complete rows only
        columns = [MODALITIES[name] for name in params.modalities]
        with open(params.paths_csv) as handle:
            self.files = {
                row["subject_id"]: [row[column] for column in columns]
                for row in csv.DictReader(handle)
                if all(row.get(column) for column in columns)
            }

        # read master csv: select and get labels
        counts = {}
        self.ids, labels = [], []
        for subject, row in sorted(self.rows(params.master_csv).items()):
            counts["total"] = counts.get("total", 0) + 1
            if not all(row.get(flag) == "True" for flag in params.filters):
                continue
            counts["after filters"] = counts.get("after filters", 0) + 1
            if any(
                (value := as_float(row.get(column))) is None or not low <= value <= high
                for column, (low, high) in ranges.items()
            ):
                continue
            counts["in ranges"] = counts.get("in ranges", 0) + 1
            if any(as_float(row.get(column)) is None for column in params.require):
                continue
            label = self.to_class(row)
            if label is None or subject not in self.files:
                continue
            self.ids.append(subject)
            labels.append(label)
        self.labels = np.array(labels)
        counts["labelled with volumes"] = len(self.ids)

        # drop subjects whose files are not there
        if params.check_files:
            keep = [i for i, s in enumerate(self.ids) if all(map(os.path.isfile, self.files[s]))]
            counts["volumes on disk"] = len(keep)
            self.ids = [self.ids[i] for i in keep]
            self.labels = self.labels[keep]
        if len(self.ids) == 0:
            raise ValueError(f"no subject survived the selection: {counts}")
        print(f"[MDD] {counts}, channels={list(params.modalities)}, classes={self.class_counts()}")

    @staticmethod
    def rows(path):
        with open(path) as handle:
            return {row["id"]: row for row in csv.DictReader(handle)}

    def class_counts(self):
        values, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))

    def to_class(self, row):
        """Parser to extract the class index or impute 0 if conditions are met. 
        Else None.
        """
        value = as_float(row.get(self.params.label_column))
        if value is None and self.params.fill_missing:
            matches = (as_float(row.get(c)) == v for c, v in self.params.fill_missing.items())
            value = 0.0 if all(matches) else None
        for index, values in enumerate(self.params.classes):
            if value in values:
                return index
        return None

    def load(self, index):
        """Channels in config order; `normalize` decides which of them get rescaled."""
        import nibabel as nib

        how = self.params.normalize
        channels = []
        for name, path in zip(self.params.modalities, self.files[self.ids[index]]):
            volume = torch.from_numpy(np.asanyarray(nib.load(path).dataobj)).float().squeeze()
            rescale = how == "per_channel" or (how == "falff" and name.startswith("falff"))
            channels.append(safe_normalize(volume) if rescale else volume)
        stacked = channels[0] if len(channels) == 1 else torch.stack(channels)
        return safe_normalize(stacked) if how == "volume" else stacked


DATASET = MDDDataset
