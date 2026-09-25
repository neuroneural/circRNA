#!/bin/bash
# MDD DIRECT volumes, 2x A100, one CV fold per array task.
#
# Resources are 2/8 of an A100 node: 192*2/8 = 48 cores, 1TB*2/8 = 256G.
# arctrddgxa001 is excluded: its A100s are 40GB, the rest are 80GB.
# %4 throttles to 8 concurrent GPUs, the qTRDGPUH per-user cap.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=256g
#SBATCH -p qTRDGPUH
#SBATCH -t 48:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:2
#SBATCH -J mdd_direct
#SBATCH -D /data/users2/ppopov1/circRNA/trainer   # submit from anywhere
#SBATCH --output=/data/users2/ppopov1/_out/%x_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-9%4

set -e
sleep 5s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, fold: $SLURM_ARRAY_TASK_ID" >&2

export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /data/users2/ppopov1/miniconda/bin/activate circrna
echo "Using python from: $(which python)"

# 2 ranks, each with a train and a valid worker pool: 2*(1+10+10) = 42 of 48.
python train.py --config-name=mdd_direct \
    experiment.name=mdd_gm_wm_csf \
    "experiment.target_folds=[$SLURM_ARRAY_TASK_ID]" \
    experiment.epochs=60 \
    experiment.batch_size=8 \
    loader.train.num_workers=10 \
    loader.eval.num_workers=10

sleep 5s
echo "Job $SLURM_JOB_ID completed"
