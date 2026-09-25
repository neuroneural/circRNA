#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:1
#SBATCH -J resnet3d
#SBATCH -D /data/users2/ppopov1/circRNA/trainer   # submit from anywhere
#SBATCH --output=/data/users2/ppopov1/_out/%x_%j.out
#SBATCH -A psy53c17

set -e
sleep 5s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /data/users2/ppopov1/miniconda/bin/activate circrna
echo "Using python from: $(which python)"

# Multi-GPU: raise --gres=gpu:A100:N above; SLURM_GPUS_ON_NODE drives the
# world size, nothing here needs to change.
# One fold per array task: see train_mdd.sh.
# Dummy data, so this runs anywhere. For a real one:
#   --config-name=mdd_direct  or  --config-name=mongo_fbirn
python train.py \
    data.name=dummy \
    +data.params.signal=3 \
    experiment.name=resnet3d_run1 \
    experiment.epochs=60 \
    experiment.batch_size=8

sleep 5s
echo "Job $SLURM_JOB_ID completed"
