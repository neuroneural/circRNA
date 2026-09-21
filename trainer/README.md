# 3D volume classification with cross-validation

Binary classification of whole 3D volumes, stratified k-fold, one model per fold.
Catalyst runner, Hydra config, single- or multi-GPU on SLURM.

## Quick start

```bash
pip install -r requirements.txt
python train.py                                        # random dummy data, no setup
python train.py --config-name=mdd_direct experiment.name=my_run   # a real dataset
python train.py experiment.name=my_run experiment.resume=true   # continue where it stopped
```

`data.name=dummy` generates random volumes so the pipeline runs with no data at all.
`signal` is the class blob's amplitude in units of the noise SD, with no upper bound:
the default 0.3 is faint, and `+data.params.signal=3` is obvious enough to confirm the
training loop really fits (test AUC goes to 1.0).

Any config key can be overridden on the command line, e.g.
`experiment.epochs=100 experiment.batch_size=4 experiment.target_folds=[0]`.

## Layout

```
train.py                  the entry point: CV loop + Catalyst runner
requirements.txt          pinned deps, Catalyst from the neuroneural fork
conf/                     run configs
src/models/               models
src/datasets/             datasets, and their own README
```

## Config sections breakdown

| section | what it sets |
|---|---|
| `data` | `name` (a module in `src/datasets/`) and `params` |
| `experiment` | `name`, `epochs`, `batch_size` (per rank), `cv_folds`, `target_folds`, `valid_ratio`, `selection`, the seeds below, `resume`, `cudnn_benchmark` |
| `model` | `name` (a module in `src/models/`), `init_seed`, `params` |
| `optimizer` | `base_lr` (per rank), and `scale_lr` for `base_lr * sqrt(world_size)` |
| `loader` | `num_workers`, `prefetch_factor`, `persistent_workers`, `prefetches` for `train` and `eval` |
| `paths` | `logdir`, `init_weights` (a checkpoint to start the model from) |
| `wandb` | `project` / `entity` / `name`; off unless you uncomment the logger |
| `runtime` | filled in by `train.py` at startup — fold index, world size, the scaled lr, the resume checkpoint, and the dataset's channel and class counts. Not for hand editing; it is there so the logged config says what ran |
| `stream_log` | config for stdout logger that fills `run.log` |

`experiment.name` is the log folder name *and* the wandb run name, so a run is easy to
trace in either direction. `experiment.selection` decides which epoch counts as best —
the checkpoint that is kept and the metrics that get reported — and defaults to the
lowest `valid/loss`; `experiment.selection.metric=auc experiment.selection.minimize=false`
switches it to the highest valid AUC.

## Logs

Everything lands under `logs/<experiment.name>/`.

| file | what |
|---|---|
| `CV_records.csv` | one row per fold: split sizes, best epoch, valid and **test** metrics, seeds, wall time |
| `summary.json` | mean, std, median and a 95% t interval across folds |
| `config.yaml` | the resolved config that actually ran |
| `run.log` | everything printed, including from spawned DDP ranks |

And per fold, in `fold_<i>/`:

| file | what |
|---|---|
| `train.csv` `valid.csv` | per-epoch `accuracy, auc, loss` |
| `_epoch_.csv` | per-epoch `lr` and `momentum` |
| `fold_summary.json` | best epoch, best metrics, seeds, wall time |
| `test_predictions.csv` | every test sample's `logit_k, prob_k, target`, merged across ranks |
| `split_indices.json` | the train/valid/test split as dataset indices |
| `model.best.pth` | weights from the best epoch |

and the detail, in `fold_<i>/more/`:

| file | what |
|---|---|
| `fold_config.yaml` | the config plus every seed, enough to reproduce this fold |
| `*_ids.txt` | the same split as readable sample ids |
| `raw_preds_<loader>_rank<r>.csv` | every sample's `logit_k, prob_k, target` per epoch, per rank |
| `best_so_far.jsonl` | best epoch log |

Rerunning an experiment name clears the fold directory first — nothing is deleted, the
old artefacts move to `backup_<timestamp>/`. With `experiment.resume=true` it instead continues from an interrupted fold.

## Seeds

Four knobs, one per thing that can vary:

| seed | fixes | null means |
|---|---|---|
| `experiment.cv_seed` | the fold assignment — which samples are each fold's test set | — |
| `experiment.train_val_seed` | the train/valid slice inside a fold | — (it defaults to `${experiment.cv_seed}`) |
| `experiment.sampler_seed` | the data order within an epoch, and the base Catalyst reseeds the global RNG from (dropout, augmentation) | — |
| `model.init_seed` | the model initialisation | a different init each fold |

All four are recorded in `fold_config.yaml`, `fold_summary.json` and `CV_records.csv`.
