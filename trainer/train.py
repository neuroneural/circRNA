import warnings

warnings.filterwarnings("ignore")

import csv
import gc
import glob
import json
import os
import shutil
import time

import hydra
import numpy as np
import torch
import yaml
from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.utils import get_optimizer_momentum, load_checkpoint
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.datasets.base import SeededBatchSampler, collate_volumes
from src.registry import build_dataset, build_model, resolve_params
from src.utils import (
    clear_rundir,
    setup_distributed_port,
    tee_stdout,
    world_size_from_env,
)

setup_distributed_port()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CustomRunner(dl.Runner):
    """Classification trainer, one CV fold per runner.
    """

    def __init__(self, logdir: str, dataset, cfg):
        super().__init__()
        self._logdir = logdir
        self._moredir = os.path.join(logdir, "more")  # detail logs, out of the main view
        os.makedirs(self._moredir, exist_ok=True)
        self.dataset = dataset
        self.labels = np.asarray(dataset.labels)
        self.cfg = cfg
        # what catalyst writes to hparams.json and hands to wandb's run config
        self._hparams = OmegaConf.to_container(cfg, resolve=True)

        # _pred_columns controls the raw prediction logs, whose columns
        # vary with the number of classes
        self.n_classes = cfg.model.params.n_classes
        self._pred_columns = (
            ["epoch", "rank", "index"]
            + [f"logit_{k}" for k in range(self.n_classes)]
            + [f"prob_{k}" for k in range(self.n_classes)]
            + ["target"]
        )

        self._started = time.time()
        self._split_sizes = {}
        self._best = None

    def get_engine(self):
        # Pin the spawn/world size: Catalyst otherwise spawns one process per *physical*
        # GPU via device_count(). See world_size_from_env().
        n_gpus = world_size_from_env()
        if n_gpus > 1:
            return dl.DistributedDataParallelEngine(
                process_group_kwargs={"backend": "nccl"},
                num_node_workers=n_gpus,
                world_size=n_gpus,
            )
        return dl.GPUEngine() if n_gpus == 1 else dl.CPUEngine()

    def get_loggers(self):
        loggers = {"console": dl.ConsoleLogger(), "csv": dl.CSVLogger(logdir=self._logdir)}

        wandb_cfg = self._hparams["wandb"]
        # wandb.project=null (default config) keeps it off
        if wandb_cfg["project"]:
            loggers["wandb"] = dl.WandbLogger(
                project=wandb_cfg["project"],
                entity=wandb_cfg["entity"],
                name=wandb_cfg["name"] or self._hparams["experiment"]["name"],
                log_batch_metrics=False,
            )
        return loggers

    @property
    def num_epochs(self) -> int:
        return self.cfg.experiment.epochs

    @property
    def seed(self) -> int:
        """Base for Catalyst's own reseeding.

        Catalyst calls ``set_global_seed(self.seed + rank + epoch_step)`` in five places:
        setting up the loaders, the components and the callbacks, and at the start of
        every epoch and every loader. The rank term is what stops ranks drawing identical
        augmentations. Within one epoch the value does not change, so the per-loader call
        just rewinds the stream.
        """
        return self.cfg.experiment.sampler_seed

    def on_experiment_start(self, runner):
        torch.backends.cudnn.benchmark = self.cfg.experiment.cudnn_benchmark
        super().on_experiment_start(runner)
        # engine.prepare has sharded it; pull it out of the epoch loop for a single pass
        self.test_loader = self.loaders.pop("test")
        if not self.cfg.runtime.resume_runner:
            return

        # === Resume routines ===
        # reopen CSVLoggers manually to avoid writing headers again
        if "csv" in self.loggers:  # only the main process has loggers
            for key in [*self.loaders, "_epoch_"]:
                path = os.path.join(self._logdir, f"{key}.csv")
                if os.path.isfile(path):
                    self.loggers["csv"].loggers[key] = open(path, "a")

        # recover the best epoch; last line wins
        path = os.path.join(self._moredir, "best_so_far.jsonl")
        lines = open(path).read().splitlines() if os.path.isfile(path) else []
        if lines:
            self._best = json.loads(lines[-1])
            print(f"[Resume] best epoch so far: {self._best['epoch']} ({self._best['score']:.4f})")

    def get_loaders(self):
        experiment, runtime = self.cfg.experiment, self.cfg.runtime
        # stratified CV split of the dataset, then a stratified valid slice off the train part
        cv = StratifiedKFold(
            n_splits=experiment.cv_folds, shuffle=True, random_state=experiment.cv_seed
        )
        train_idx, test_idx = list(cv.split(np.arange(len(self.labels)), self.labels))[
            runtime.fold_idx
        ]
        train_idx, valid_idx = train_test_split(
            train_idx,
            test_size=experiment.valid_ratio,
            stratify=self.labels[train_idx],
            random_state=experiment.train_val_seed,
        )
        splits = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        self._split_sizes = {name: len(idx) for name, idx in splits.items()}
        print(
            f"[Fold {runtime.fold_idx}] "
            + " ".join(f"{k}={v}" for k, v in self._split_sizes.items())
        )

        if self.engine.is_main_process:
            # save the split, as dataset indices and as ids
            payload = {
                "fold_idx": runtime.fold_idx,
                "cv_folds": experiment.cv_folds,
                "cv_seed": experiment.cv_seed,
                "train_val_seed": experiment.train_val_seed,
            }
            for name, indices in splits.items():
                payload[name] = {
                    "indices": [int(i) for i in indices],
                    "labels": [int(self.labels[int(i)]) for i in indices],
                }
                with open(os.path.join(self._moredir, f"{name}_ids.txt"), "w") as handle:
                    handle.writelines(f"{self.dataset.ids[int(i)]}\n" for i in indices)
            with open(os.path.join(self._logdir, "split_indices.json"), "w") as handle:
                json.dump(payload, handle, indent=4)

            # everything needed to reproduce this fold, seeds included
            fold_cfg = {
                "fold_idx": runtime.fold_idx,
                "seeds": {
                    "cv_seed": experiment.cv_seed,
                    "train_val_seed": experiment.train_val_seed,
                    "sampler_seed": experiment.sampler_seed,
                    "model_init_seed": self.cfg.model.init_seed,
                },
                "world_size": self.engine.num_processes,
                "max_lr": runtime.max_lr,
                "batch_size_per_rank": experiment.batch_size,
                "epochs": experiment.epochs,
                "split_sizes": self._split_sizes,
                "config": self._hparams,
            }
            with open(os.path.join(self._moredir, "fold_config.yaml"), "w") as handle:
                yaml.safe_dump(fold_cfg, handle, indent=4, sort_keys=False)

        loaders = {}
        for key, indices in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
            loader_opts = self.cfg.loader["train" if key == "train" else "eval"]
            num_workers = loader_opts.num_workers
            subset = self.dataset.subset(indices)
            # Plain sampler with the same seed on every rank. accelerate's
            # engine.prepare() already shards the DataLoader, so pre-sharding here would
            # shard twice and drop ~(1 - 1/world_size) of the data each epoch.
            kwargs = {
                "sampler": SeededBatchSampler(
                    subset, experiment.batch_size, experiment.sampler_seed
                ),
                "collate_fn": collate_volumes,
                "pin_memory": True,
                "num_workers": num_workers,
            }
            if num_workers > 0:
                kwargs["persistent_workers"] = loader_opts.persistent_workers
                kwargs["prefetch_factor"] = loader_opts.prefetch_factor
            loader = DataLoader(subset, **kwargs)
            # the prefetch wrapper needs CUDA streams; prefetches=0 disables it on CPU
            prefetches = loader_opts.prefetches
            loaders[key] = (
                BatchPrefetchLoaderWrapper(loader, num_prefetches=prefetches)
                if prefetches > 0
                else loader
            )
        return loaders

    def get_model(self):
        # fixed init, with the global RNG stream restored so it is not perturbed
        if self.cfg.model.init_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.cfg.model.init_seed)

        model = build_model(self.cfg)

        if self.cfg.model.init_seed is not None:
            torch.set_rng_state(rng_state)
            print(f"Model initialized with fixed seed {self.cfg.model.init_seed}")
        return model

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, model):
        # Note: OneCycleLR overwrites the LR defined here
        return torch.optim.Adam(model.parameters(), lr=self.cfg.optimizer.base_lr)

    def get_scheduler(self, optimizer):
        return OneCycleLR(
            optimizer,
            max_lr=self.cfg.runtime.max_lr,
            div_factor=100,
            pct_start=0.1,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )

    def get_callbacks(self):
        selection = self.cfg.experiment.selection
        return {
            # model checkpoint configured according to config
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                save_best=True,
                metric_key=selection.metric,
                loader_key="valid",
                minimize=selection.minimize,
                load_best_on_end=False,  # on_experiment_end loads the best epoch itself
                resume_model=self.cfg.paths.init_weights or None,
            ),
            # rolling full-state checkpoint for mid-fold resume
            "state": dl.CheckpointCallback(
                self._logdir,
                topk=1,
                mode="runner",
                save_best=False,
                save_last=False,
                resume_runner=self.cfg.runtime.resume_runner or None,
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        # key juggling: catalyst asserts the key starts with train/valid/infer
        loader_key_buffer = self.loader_key
        self.loader_key = (
            loader_key_buffer if loader_key_buffer.startswith(("train", "valid")) else "infer"
        )
        super().on_loader_start(runner)
        self.loader_key = loader_key_buffer

        # initialize metrics
        self.meters = {
            "loss": metrics.AdditiveValueMetric(compute_on_call=False),
            "accuracy": metrics.AdditiveValueMetric(compute_on_call=False),
            "auc": metrics.AUCMetric(compute_on_call=False),
        }

        # per-sample predictions logs: rawest logs possible, useful for recovering arbitrary metrics
        raw_preds_path = os.path.join(
            self._moredir, f"raw_preds_{self.loader_key}_rank{self.engine.process_index}.csv"
        )
        needs_header = not os.path.isfile(raw_preds_path) or os.path.getsize(raw_preds_path) == 0
        self.raw_preds_file = open(raw_preds_path, "a", newline="")
        self.raw_preds_writer = csv.writer(self.raw_preds_file)
        if needs_header:
            self.raw_preds_writer.writerow(self._pred_columns)

    def on_loader_end(self, runner):
        self.loader_metrics["loss"] = self.meters["loss"].compute()[0]
        self.loader_metrics["accuracy"] = self.meters["accuracy"].compute()[0]
        # compute() gives per_class, micro, macro, weighted
        _, _, macro, weighted = self.meters["auc"].compute()
        self.loader_metrics["auc"], self.loader_metrics["auc_weighted"] = macro, weighted

        if self.engine.is_ddp:
            # Catalyst's own on_loader_end reduces nothing, so the additive meters are
            # still per-rank here and need reduction.
            for key in self.loader_metrics.keys() - {"auc", "auc_weighted"}:  # AUC all-gathers
                value = torch.tensor([self.loader_metrics[key]], device=self.engine.device)
                self.loader_metrics[key] = utils.distributed.mean_reduce(
                    value, self.engine.num_processes
                ).item()

        self.raw_preds_file.close()
        super().on_loader_end(runner)

    def on_epoch_end(self, runner):
        self.epoch_metrics["_epoch_"]["lr"] = self.optimizer.param_groups[0]["lr"]
        self.epoch_metrics["_epoch_"]["momentum"] = get_optimizer_momentum(self.optimizer)

        # track the best epoch for logs; Catalyst's CheckpointCallback breaks on interruptions
        selection = self.cfg.experiment.selection
        score = self.epoch_metrics["valid"][selection.metric]
        if self._best is None or (
            score < self._best["score"] if selection.minimize else score > self._best["score"]
        ):
            self._best = {
                "epoch": int(self.epoch_step),
                "score": float(score),
                "metrics": {k: dict(v) for k, v in self.epoch_metrics.items() if k != "_epoch_"},
            }
            if self.engine.is_main_process:
                # append-only, so whatever kills the run can damage at most the last line
                with open(os.path.join(self._moredir, "best_so_far.jsonl"), "a") as handle:
                    handle.write(json.dumps(self._best) + "\n")
        super().on_epoch_end(runner)

    def on_experiment_end(self, runner):
        # load best checkpoint and run it on test
        best_ckpt = f"model.{self._best['epoch']:04d}.pth"
        self.engine.wait_for_everyone()
        self.engine.unwrap_model(self.model).load_state_dict(
            load_checkpoint(os.path.join(self._logdir, best_ckpt))
        )

        # prepare for test
        # epoch_step stamps the raw_preds rows, so point it at the epoch these
        # weights come from, then put the true epoch count back for epochs_run
        epoch_buffer, self.epoch_step = self.epoch_step, self._best["epoch"]
        self.loader_key, self.loader = "test", self.test_loader

        # run test
        self._run_event("on_loader_start")
        self._run_loader()
        self._run_event("on_loader_end")

        # save logs and restore things
        self.engine.wait_for_everyone()  # every rank has closed its raw_preds file
        self.epoch_step = epoch_buffer
        self._best["metrics"]["test"] = {
            k: float(v) for k, v in sorted(self.loader_metrics.items())
        }

        # final logging and cleanup
        if self.engine.is_main_process:
            # overwrite model.best.pth with the best epoch's weights,
            # vanilla Catalyst gets this wrong
            src = os.path.join(self._logdir, best_ckpt)
            dst = os.path.join(self._logdir, "model.best.pth")
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)

            # combine raw test predictions into a clear file.
            preds = {}
            for path in sorted(glob.glob(os.path.join(self._moredir, "raw_preds_test_rank*.csv"))):
                with open(path) as handle:
                    preds.update({int(row["index"]): row for row in csv.DictReader(handle)})
            with open(os.path.join(self._logdir, "test_predictions.csv"), "w", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=self._pred_columns)
                writer.writeheader()
                writer.writerows(preds[i] for i in sorted(preds))

            # save fold summary
            selection = self.cfg.experiment.selection
            with open(os.path.join(self._logdir, "fold_summary.json"), "w") as handle:
                json.dump(
                    {
                        "fold_idx": self.cfg.runtime.fold_idx,
                        "selected_on": f"{'min' if selection.minimize else 'max'} "
                                       f"valid/{selection.metric}",
                        "best_epoch": self._best["epoch"],
                        "best_checkpoint": best_ckpt,
                        "epochs_run": int(self.epoch_step),
                        "split_sizes": self._split_sizes,
                        "metrics": self._best["metrics"],
                        "seeds": {
                            "cv_seed": self.cfg.experiment.cv_seed,
                            "train_val_seed": self.cfg.experiment.train_val_seed,
                            "sampler_seed": self.cfg.experiment.sampler_seed,
                            "model_init_seed": self.cfg.model.init_seed,
                        },
                        "world_size": self.engine.num_processes,
                        "batch_size_per_rank": self.cfg.experiment.batch_size,
                        "max_lr": self.cfg.runtime.max_lr,
                        "minutes": round((time.time() - self._started) / 60, 2),
                    },
                    handle,
                    indent=4,
                )
            print(
                f"[Fold {self.cfg.runtime.fold_idx}] best epoch {self._best['epoch']} "
                f"(valid/{selection.metric}={self._best['score']:.4f})"
            )
        super().on_experiment_end(runner)

    def handle_batch(self, batch):
        sample, target, index = batch   # target is [B] class indices, index is the dataset row

        if self.model.training:
            y_hat = self.model.forward(sample)
            loss = self.criterion(y_hat, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, target)

        with torch.no_grad():
            probs = torch.softmax(y_hat, dim=1)
            accuracy = (probs.argmax(1) == target).float().mean()
            # AUCMetric scores every class, so it needs one-hot targets
            auc_target = torch.nn.functional.one_hot(target, self.n_classes)
            # Write raw predictions for every sample in the batch, so arbitrary
            # metrics can be computed later.
            self.raw_preds_writer.writerows(
                [self.epoch_step, self.engine.process_index, row, *logits, *scores, label]
                for row, logits, scores, label in zip(
                    index.cpu().numpy().tolist(),
                    y_hat.detach().float().cpu().numpy().tolist(),
                    probs.float().cpu().numpy().tolist(),
                    target.cpu().numpy().tolist(),
                )
            )

        # batch_metrics feeds the tqdm bar; the loader values are saved in the meters
        self.batch_metrics.update({"loss": loss, "accuracy": accuracy})
        self.meters["loss"].update(loss.item(), self.batch_size)
        self.meters["accuracy"].update(accuracy.item(), self.batch_size)
        self.meters["auc"].update(probs, auc_target)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """ Configure experiment, run the CV folds and roll them up."""
    # ===== Initial setup =====
    logdir = os.path.join(cfg.paths.logdir, cfg.experiment.name)
    os.makedirs(logdir, exist_ok=True)
    if cfg.stream_log:
        tee_stdout(os.path.join(logdir, "run.log"))

    # validate config
    experiment = cfg.experiment
    target_folds = list(
        range(experiment.cv_folds) if experiment.target_folds is None else experiment.target_folds
    )
    for ok, message in (
        (experiment.epochs >= 1, "epochs must be at least 1"),
        (experiment.cv_folds >= 2, "cv_folds must be at least 2"),
        (0 < experiment.valid_ratio < 1, "valid_ratio must be between 0 and 1"),
        (experiment.selection.metric in ("loss", "accuracy", "auc", "auc_weighted"),
         "selection.metric: loss, accuracy, auc or auc_weighted"),
        (target_folds and all(0 <= i < experiment.cv_folds for i in target_folds),
         f"target_folds must be a non-empty subset of 0..{experiment.cv_folds - 1}"),
    ):
        if not ok:
            raise ValueError(f"experiment.{message}")

    # ===== Data setup =====
    # build dataset and get data params from it
    dataset = build_dataset(cfg)
    classes, counts = np.unique(dataset.labels, return_counts=True)
    cfg.runtime.n_channels = int(dataset.n_channels)
    cfg.runtime.n_classes = len(classes)
    print(
        f"[Data] {cfg.data.name}: {len(dataset)} samples, {cfg.runtime.n_channels} channel(s), "
        f"class counts: {dict(zip(classes.tolist(), counts.tolist()))}"
    )

    # check that labels are 0..n_classes-1; CrossEntropy breaks if they are not
    if classes.tolist() != list(range(cfg.runtime.n_classes)):
        raise ValueError(f"{cfg.data.name} labels are {classes.tolist()}, not 0..K-1")

    resolve_params("model", cfg)

    # ===== Env-dependent setup =====
    # get the world size (number of GPUs for DDP)
    cfg.runtime.world_size = max(1, world_size_from_env())

    # max_lr is the peak OneCycleLR climbs to, scaled to world size according to Malladi, 2022
    # https://doi.org/10.48550/arXiv.2205.10287
    cfg.runtime.max_lr = cfg.optimizer.base_lr
    if cfg.optimizer.scale_lr:
        cfg.runtime.max_lr *= cfg.runtime.world_size**0.5
    print(
        f"[LR] max_lr={cfg.runtime.max_lr:g} (base={cfg.optimizer.base_lr}, "
        f"world_size={cfg.runtime.world_size})"
    )

    # ===== Resumed experiment routines =====
    # recovers the completed folds and continues from there
    resume = cfg.experiment.resume
    config_path = os.path.join(logdir, "config.yaml")
    if resume:
        done = [
            i for i in target_folds
            if os.path.isfile(os.path.join(logdir, f"fold_{i}", "fold_summary.json"))
        ]
        print(f"[Resume] on; folds already complete: {done}")

        # a resumed fold keeps the old schedule, splits and optimizer state
        for key in ("experiment.epochs", "experiment.batch_size", "experiment.cv_folds",
                    "experiment.cv_seed", "experiment.train_val_seed",
                    "experiment.sampler_seed", "runtime.world_size"):
            was = OmegaConf.select(OmegaConf.load(config_path), key)
            if was is not None and was != OmegaConf.select(cfg, key):
                raise ValueError(
                    f"resume: {key} was {was}, now {OmegaConf.select(cfg, key)}; "
                    "this will mess with OneCycleLR's schedule; to continue from "
                    "this model instead, point paths.init_weights at its checkpoint"
                )

    # pre-flight maintenance: save the config
    with open(config_path, "w") as handle:
        yaml.safe_dump(
            OmegaConf.to_container(cfg, resolve=True), handle, indent=4, sort_keys=False
        )

    # ===== CV loop =====
    rows = []
    for position, fold_idx in enumerate(target_folds, start=1):
        rundir = os.path.join(logdir, f"fold_{fold_idx}")
        if resume and os.path.isfile(os.path.join(rundir, "fold_summary.json")):
            print(f"[Resume] fold {fold_idx} ({position}/{len(target_folds)}) done, skipping")
        else:
            print(f"Starting fold {fold_idx} ({position}/{len(target_folds)})")
            os.makedirs(rundir, exist_ok=True)

            # decide if (and how to) resume
            runner_checkpoints = sorted(glob.glob(os.path.join(rundir, "runner.*.pth")))
            resume_runner = runner_checkpoints[-1] if (resume and runner_checkpoints) else ""
            if resume_runner:  # pick an interrupted fold up where it stopped
                print(f"[Resume] continuing fold {fold_idx} from {os.path.basename(resume_runner)}")
            else:  # fresh fold: drop what an earlier run left here
                clear_rundir(rundir)

            # RUN THE FOLD
            cfg.runtime.fold_idx = fold_idx
            cfg.runtime.resume_runner = resume_runner
            runner = CustomRunner(logdir=rundir, dataset=dataset, cfg=cfg)
            runner.run()

            # GC step
            del runner
            torch.cuda.empty_cache()
            gc.collect()
            for stale in glob.glob(os.path.join(rundir, "runner.*")) + [
                os.path.join(rundir, "more", "best_so_far.jsonl")
            ]:
                if os.path.isfile(stale):
                    os.remove(stale)
            if not os.path.isfile(os.path.join(rundir, "fold_summary.json")):
                print(f"[Warn] fold {fold_idx} produced no fold_summary.json")

        # aggregate fold results from finished logs; rebuilt after every fold for freshness
        rows = []
        for best_path in glob.glob(os.path.join(logdir, "fold_*", "fold_summary.json")):
            with open(best_path) as handle:
                best = json.load(handle)

            valid, test = best["metrics"].get("valid", {}), best["metrics"].get("test", {})
            rows.append(
                {
                    "fold": best["fold_idx"],
                    **{f"n_{name}": size for name, size in best["split_sizes"].items()},
                    "epochs_run": best["epochs_run"],
                    "best_epoch": best["best_epoch"],
                    **{f"valid_{key}": value for key, value in valid.items()},
                    **{f"test_{key}": value for key, value in test.items()},
                    **best["seeds"],
                    "world_size": best["world_size"],
                    "batch_size_per_rank": best["batch_size_per_rank"],
                    "max_lr": best["max_lr"],
                    "minutes": best["minutes"],
                    "best_checkpoint": best["best_checkpoint"],
                }
            )
        rows.sort(key=lambda row: row["fold"])
        with open(os.path.join(logdir, "CV_records.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # === Complete CV summary ===
    if len(rows) < cfg.experiment.cv_folds:
        print(f"[CV] {len(rows)}/{cfg.experiment.cv_folds} folds done; no summary yet")
    else:
        summary = {"n_folds": len(rows)}
        for key in [c for c in rows[0] if c.startswith(("valid_", "test_"))]:
            values = [row[key] for row in rows if row.get(key) is not None]
            if not values:
                continue
            # 95% t interval on the mean across folds
            mean, n = float(np.mean(values)), len(values)
            half = (
                stats.t.ppf(0.975, n - 1) * float(np.std(values, ddof=1)) / np.sqrt(n)
                if n > 1
                else float("nan")
            )
            summary[key] = {
                "mean": mean,
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "ci95": [mean - half, mean + half],
            }
            if key.startswith("test"):
                print(f"[CV] {key}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
        with open(os.path.join(logdir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=4)


if __name__ == "__main__":
    main()
