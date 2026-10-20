import os
import random
import shutil
import socket
import subprocess
import sys
import time

import torch

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
    would otherwise interleave two runs in one csv, and a stale model.<epoch>.pth from
    a longer previous run would survive the topk pruning and look like one of ours.
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
