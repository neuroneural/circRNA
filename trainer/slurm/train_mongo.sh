#!/bin/bash
# fbirn volumes from MongoDB, 2x A100, one CV fold per array task.
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
#SBATCH -J mongo_fbirn
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
export MKL_THREADING_LAYER=GNU   # conda MKL vs torch's OpenMP

source /data/users2/ppopov1/miniconda/bin/activate circrna
echo "Using python from: $(which python)"

# host_slurm in the config points at the mongod reachable from compute nodes.
# Each worker opens its own connection, so keep the pools modest.
# stream_log=False: parallel tasks would interleave run.log
python train.py --config-name=mongo_fbirn \
    experiment.name=fbirn_smri_dwi_falff_gender \
    "experiment.target_folds=[$SLURM_ARRAY_TASK_ID]" \
    experiment.epochs=60 \
    experiment.batch_size=8 \
    loader.train.num_workers=8 \
    loader.eval.num_workers=8 \
    loader.train.prefetch_factor=2 \
    loader.eval.prefetch_factor=2 \
    stream_log=False

sleep 5s
echo "Job $SLURM_JOB_ID completed"
