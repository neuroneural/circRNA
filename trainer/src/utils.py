import csv
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import textwrap
import time

import numpy as np
import torch
from catalyst import dl

_TEE_PROCESS = None  # kept alive for the life of the run


def setup_distributed_port(low=2000, high=2200):
    """Set MASTER_PORT for Catalyst's DDP to a free port.
    """
    if "MASTER_PORT" in os.environ:
        print(f"[Init] MASTER_PORT used by catalyst DDP already set to {os.environ['MASTER_PORT']}")
        return

    draw = random.Random()
    for _ in range(50):
        port = draw.randint(low, high)
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:  # nothing listening
                break
    else:
        raise RuntimeError(f"no free port found in [{low}, {high}] for MASTER_PORT")

    os.environ["MASTER_PORT"] = str(port)
    print(f"[Init] MASTER_PORT used by catalyst DDP set to {port}")


def world_size_from_env():
    """Get # of GPUs available.

    torch.cuda.device_count() can be faulty on some nodes.
    """
    return int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))


def tee_stdout(path):
    """Duplicate stdout into ``path`` at the fd level, so spawned DDP ranks land in it too.

    tqdm writes to stderr and is deliberately left out, which keeps the file readable.
    """
    global _TEE_PROCESS
    sys.stdout.flush()
    try:
        _TEE_PROCESS = subprocess.Popen(["tee", "-a", path], stdin=subprocess.PIPE)
        os.dup2(_TEE_PROCESS.stdin.fileno(), sys.stdout.fileno())
        sys.stdout.reconfigure(line_buffering=True)
    except (OSError, ValueError) as error:
        print(f"[Log] could not tee stdout into {path}: {error}")


def clear_rundir(rundir):
    """Move an earlier run's artefacts into ``backup_<timestamp>/``.

    Catalyst's CSVLogger opens its files "a+", so a rerun of the same experiment name
    would otherwise interleave two runs in one csv.
    """
    if not os.path.isdir(rundir):
        return
    backup = os.path.join(rundir, time.strftime("backup_%Y%m%d_%H%M%S"))
    artefacts = (".csv", ".json", ".jsonl", ".txt", ".yaml", ".pth")
    moved = 0
    for root, dirs, names in os.walk(rundir):
        dirs[:] = [d for d in dirs if not d.startswith("backup_")]  # keep older backups
        for name in sorted(names):
            if not name.endswith(artefacts):
                continue
            destination = os.path.join(backup, os.path.relpath(root, rundir), name)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(os.path.join(root, name), destination)
            moved += 1
    if moved:
        print(f"[Log] moved {moved} file(s) from an earlier run into {backup}")


class ModelCheckpoint(dl.ICheckpointCallback):
    """Best weights per selection metric, plus the full training state for resuming.

    ``model.best.pth`` holds the first metric's best epoch, ``model.best_<metric>.pth``
    the others'; ``state.last.pth`` (with ``save_state``) the weights, optimizer, scheduler,
    epoch and best epochs after the latest epoch. The main process replaces each file
    atomically, so an interruption at any point leaves the previous complete set.
    """

    def __init__(self, logdir, selection, save_state=True, resume_from=None):
        super().__init__()
        self.logdir, self.selection = logdir, list(selection)
        self.save_state, self.resume_from = save_state, resume_from
        self.best = {}  # metric -> {"epoch", "score", "metrics"} of its best epoch so far

    def path(self, metric):
        name = "model.best" if metric == self.selection[0] else f"model.best_{metric}"
        return os.path.join(self.logdir, f"{name}.pth")

    def _save(self, runner, obj, path):
        if runner.engine.is_main_process:  # every rank holds the same weights
            torch.save(obj, path + ".tmp")
            os.replace(path + ".tmp", path)

    def on_experiment_start(self, runner):
        if self.resume_from:
            state = torch.load(self.resume_from, map_location="cpu")
            runner.engine.unwrap_model(runner.model).load_state_dict(state["model"])  # DDP-safe
            runner.optimizer.load_state_dict(state["optimizer"])
            runner.scheduler.load_state_dict(state["scheduler"])
            runner.epoch_step, self.best = state["epoch"], state["best"]
            runner.loaders["train"].set_epoch(runner.epoch_step)  # data order as if never stopped
            print(f"[Resume] from epoch {runner.epoch_step}, best epochs",
                  {k: v["epoch"] for k, v in self.best.items()})

    def on_epoch_end(self, runner):
        model = runner.engine.unwrap_model(runner.model)
        metrics = {k: dict(v) for k, v in runner.epoch_metrics.items() if k != "_epoch_"}
        for metric in self.selection:
            score, best = metrics["valid"][metric], self.best.get(metric, {}).get("score")
            if best is None or (score < best if metric == "loss" else score > best):
                self.best[metric] = {
                    "epoch": runner.epoch_step, "score": score, "metrics": metrics
                }
                self._save(runner, model.state_dict(), self.path(metric))
        if self.save_state:
            state = {
                "model": model.state_dict(),
                "optimizer": runner.optimizer.state_dict(),
                "scheduler": runner.scheduler.state_dict(),
                "epoch": runner.epoch_step,
                "best": self.best,
            }
            self._save(runner, state, os.path.join(self.logdir, "state.last.pth"))


def run_info(summaries, cfg):
    """Plain-text description of one or more folds, for the info panel of curves.png."""
    params, first = cfg["data"]["params"], summaries[0]
    shown = ("modalities", "label_column", "label_field", "classes", "filters", "ranges",
             "n_samples", "shape", "signal")
    lines = [
        f"experiment  {cfg['experiment']['name']}",
        f"folds       {', '.join(str(s['fold_idx']) for s in summaries)} "
        f"of {cfg['experiment']['cv_folds']}",
        f"data        {cfg['data']['name']}: "
        + ", ".join(f"{key}={params[key]}" for key in shown if key in params),
        "split       " + ", ".join(f"{k} {v}" for k, v in first["split_sizes"].items())
        + (f" (fold {first['fold_idx']})" if len(summaries) > 1 else ""),
        f"model       {cfg['model']['name']}, {cfg['runtime']['n_channels']} channel(s)",
        f"training    {first['epochs_run']} epochs, batch {first['batch_size_per_rank']}"
        f" x {first['world_size']} rank(s), max_lr {first['max_lr']:.4g}, "
        f"{np.mean([s['minutes'] for s in summaries]):.0f} min/fold",
        f"best epoch  {', '.join(str(s['best_epoch']) for s in summaries)}, "
        f"{first['selected_on']} (dots)",
    ]

    # metrics at each fold's best epoch, mean±sd over folds
    for name in ("train", "valid", "test"):
        keys = sorted(k for k in first["metrics"].get(name, {}) if k != "auc_weighted")
        values = {k: [s["metrics"][name][k] for s in summaries] for k in keys}
        if values:
            lines.append(f"{name + ' @best':<11} " + "  ".join(
                f"{k} {np.mean(v):.3f}" + (f"±{np.std(v):.3f}" if len(v) > 1 else "")
                for k, v in values.items()
            ))
    lines.append("curves      train blue, valid orange, test green if logged; one line per fold")
    return "\n".join(textwrap.fill(line, 74, subsequent_indent=" " * 12) for line in lines)


def plot_curves(rundirs, cfg, path):
    """Save the training curves of one or more folds as ``path``, one line per fold.

    Reads each fold's csv logs and fold_summary.json, so it works on finished runs too.
    Never raises: a plotting problem must not cost a finished run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        fig, ((lr_ax, metric_ax), (loss_ax, info_ax)) = plt.subplots(2, 2, figsize=(13, 8))
        alpha = 1 if len(rundirs) == 1 else 0.5
        summaries = []
        for i, rundir in enumerate(rundirs):
            # per-epoch logs as {column: array}, and the fold's summary
            logs = {}
            for name in ("train", "valid", "test", "_epoch_"):
                log_path = os.path.join(rundir, f"{name}.csv")
                if not os.path.isfile(log_path):  # test.csv exists only with test_every_epoch
                    continue
                with open(log_path) as handle:
                    rows = list(csv.DictReader(handle))
                logs[name] = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
            with open(os.path.join(rundir, "fold_summary.json")) as handle:
                summaries.append(json.load(handle))
            best, test = summaries[-1]["best_epoch"], summaries[-1]["metrics"].get("test", {})
            hide = "_" if i else ""  # "_" labels stay out of the legend

            # lr; train vs valid: loss, auc (solid), accuracy (dashed)
            lr_ax.plot(logs["_epoch_"]["step"], logs["_epoch_"]["lr"], color="k", alpha=alpha)
            for name, color in (("train", "C0"), ("valid", "C1"), ("test", "C2")):
                if name not in logs:
                    continue
                log = logs[name]
                loss_ax.plot(log["step"], log["loss"], color=color, alpha=alpha, label=hide + name)
                metric_ax.plot(log["step"], log["auc"], color=color, alpha=alpha,
                               label=f"{hide}{name} auc")
                metric_ax.plot(log["step"], log["accuracy"], color=color, alpha=alpha, ls="--",
                               label=f"{hide}{name} accuracy")

            # best epoch: dots on lr and valid loss, test scores
            for ax, log, key in ((lr_ax, logs["_epoch_"], "lr"), (loss_ax, logs["valid"], "loss")):
                ax.plot(best, np.interp(best, log["step"], log[key]), "o", mfc="w", mec="k", ms=5)
            metric_ax.plot(best, test.get("auc", np.nan), "k*", ms=9, label=f"{hide}test auc")
            metric_ax.plot(best, test.get("accuracy", np.nan), "kx", ms=7,
                           label=f"{hide}test accuracy")

        lr_ax.set(title="learning rate (end of epoch)", xlabel="epoch")
        metric_ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        metric_ax.set(title="AUC and accuracy", xlabel="epoch")
        metric_ax.legend(fontsize=8, ncol=3)

        # log-scale loss, ticks 1-2-5 as plain numbers
        loss_ax.set(title="loss", xlabel="epoch", yscale="log")
        loss_ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        loss_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        loss_ax.yaxis.set_minor_formatter(NullFormatter())
        loss_ax.legend()

        info_ax.axis("off")
        info_ax.text(0, 1, run_info(summaries, cfg), va="top", family="monospace", fontsize=9)

        fig.suptitle(cfg["experiment"]["name"])
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as error:  # noqa: BLE001
        print(f"[Plot] {path} skipped: {error}")
