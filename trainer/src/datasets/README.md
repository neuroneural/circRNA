# Datasets

`data.name` picks a module in this folder; `data.params` overrides its `default_HPs()`.

## The contract

Subclass `VolumeDataset` from `base.py`:

- implement `load(index)` returning `[D,H,W]`, or `[C,D,H,W]` when the sample has channels
- set `labels` (one int per sample, `0..n_classes-1`) and `ids` (one readable id per sample)
- set `n_channels` if it is not 1 — `train.py` sizes `model.params.in_channels` from it
- override `prefetch(batch)` if the backend is cheaper to read a batch at a time

`train.py` derives `in_channels` and `n_classes` from the dataset. Labels must be exactly `0..n_classes-1`; anything else is rejected at startup.

## dummy.py

Random volumes generated for tests. 
- `signal` controls class contrast: the blob's amplitude in noise SDs, no upper bound.
- `channels` and `n_classes` make it stand in for a wider or multiclass dataset.
conf/config.yaml is a worked example that runs on dummy data.
## mongo.py

A MindfulTensors MongoDB: chunked volumes from `<collection>.bin` keyed by `(id, kind)`,
the label and the per-subject `modalities` list from `<collection>.meta`. Listing several
`modalities` stacks them as channels, in the order given, and keeps only subjects carrying
all of them.

`conf/mongo_fbirn.yaml` is a worked example.

## mdd.py

MDD DIRECT based on joining two csvs: `mdd_master.csv` and `mdd_direct_volume_paths.csv`. Selection runs in four stages, each
counted in the `[MDD]` line: `filters` (csv flags that must read True), `ranges` (numeric
columns inside an inclusive `[min, max]`, or equal to a single value), `require` (columns
that must hold a number), and `to_class`, which allows to group multiclass labels into bins.

`modalities` degermine channels. `normalize` decides
what gets rescaled: `falff` (the fALFF channels only — probsegs are already 0-1),
`per_channel`, `volume` (once after stacking), or `none`.

`conf/mdd_direct.yaml` is a worked example.
