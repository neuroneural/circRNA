"""Volumes from a MindfulTensors MongoDB: 3D data in ``<collection>.bin``, labels in ``.meta``."""

import io
import os
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from pymongo import MongoClient

from src.datasets.base import VolumeDataset

LZ4_MAGIC = b"\x04\x22\x4d\x18"


def default_HPs(cfg):
    return OmegaConf.create(
        {
            "host": "localhost",              # mongod address
            "host_slurm": None,               # used instead of host inside a SLURM job
            "port": 27017,
            "database": "multimodalSubnetworks",
            "collection": "ukb",              # reads <collection>.bin and <collection>.meta
            "modalities": ["smri"],           # `kind`s in <collection>.bin, in channel order
            "label_field": "gender_encoded",  # one field in <collection>.meta
            "id_field": "id",                 # readable subject id, in both collections
        }
    )


def decode(blob):
    """Chunk bytes -> 3D tensor. Blobs are ours, hence weights_only=False."""
    if blob[:4] == LZ4_MAGIC:
        import lz4.frame

        blob = lz4.frame.decompress(blob)
    return torch.load(io.BytesIO(blob), weights_only=False)


class MongoVolumeDataset(VolumeDataset):
    """One scalar label per subject; ``modalities`` become channels of the same volume.

    ``.bin`` holds each volume as ordered chunks keyed by (id, kind); ``.meta`` holds the
    scalar fields and the `modalities` list.
    """

    def __init__(self, params):
        self.params = params
        self.n_channels = len(params.modalities)   # one channel per modality
        self._client, self._pid, self._cache = None, None, {}

        # select: only subjects carrying every requested modality
        meta = self.collections["meta"]
        wanted = list(params.modalities)
        self.ids = sorted(meta.distinct(params.id_field, {"modalities": {"$all": wanted}}))
        if not self.ids:
            raise ValueError(f"no subject in {params.collection}.meta has all of {wanted}")

        # get labels
        docs = {
            doc[params.id_field]: doc
            for doc in meta.find(
                {params.id_field: {"$in": self.ids}},
                {params.id_field: 1, params.label_field: 1, "_id": 0},
            )
        }
        missing = [i for i in self.ids if params.label_field not in docs.get(i, {})]
        if missing:
            raise ValueError(f"{len(missing)} ids lack {params.label_field!r}, e.g. {missing[:5]}")
        self.labels = np.array([docs[i][params.label_field] for i in self.ids])
        self.close()  # so no live client is inherited by a fork or pickled into a rank

    @property
    def collections(self):
        if self._pid != os.getpid():  # a fresh worker or DDP rank
            host = self.params.host_slurm if os.environ.get("SLURM_JOB_ID") else self.params.host
            self._client = MongoClient(f"mongodb://{host or self.params.host}:{self.params.port}")
            self._pid = os.getpid()
        database = self._client[self.params.database]
        return {kind: database[f"{self.params.collection}.{kind}"] for kind in ("bin", "meta")}

    def close(self):
        if self._client is not None:
            self._client.close()
        self._client, self._pid = None, None

    def prefetch(self, batch):
        """One query per batch, which is what the array-index contract buys us."""
        params = self.params
        wanted = {self.ids[int(i)]: int(i) for i in batch}
        chunks = defaultdict(list)
        for doc in self.collections["bin"].find(
            {params.id_field: {"$in": list(wanted)}, "kind": {"$in": list(params.modalities)}},
            {params.id_field: 1, "kind": 1, "chunk_id": 1, "chunk": 1, "_id": 0},
        ):
            chunks[(doc[params.id_field], doc["kind"])].append(doc)

        self._cache = {}
        for subject, index in wanted.items():
            parts = [chunks.get((subject, kind)) for kind in params.modalities]
            if not all(parts):  # incomplete subject; load() names it
                continue
            volumes = [
                decode(b"".join(c["chunk"] for c in sorted(p, key=chunk_order))) for p in parts
            ]
            self._cache[index] = volumes[0] if len(volumes) == 1 else torch.stack(volumes)

    def load(self, index):
        if index not in self._cache:
            self.prefetch([index])
        if index not in self._cache:
            raise KeyError(
                f"{self.ids[index]!r} lacks chunks for some of {list(self.params.modalities)}"
            )
        return self._cache[index]


def chunk_order(doc):
    return doc["chunk_id"]


DATASET = MongoVolumeDataset
