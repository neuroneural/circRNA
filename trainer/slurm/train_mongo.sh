#!/bin/bash
# fbirn volumes from MongoDB, 2x A100, all folds in one job.
#
# Resources are 2/8 of an A100 node. The nodes differ: arctrddgxa001 has 256
# cores and 40GB A100s, arctrddgxa002-003 have 192 cores and 80GB ones, so ask
# for the smaller share -- 192*2/8 = 48 cores, 1TB*2/8 = 256G -- and it fits
# either. Nothing pins GPU memory, so assume 40GB when sizing the batch.
# For one fold per task instead, see train_mdd.sh.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=256g
#SBATCH -p qTRDGPUH
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:A100:2
#SBATCH -J mongo_fbirn
#SBATCH -D /data/users2/ppopov1/_circRNA/trainer   # submit from anywhere
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

# host_slurm in the config points at the mongod reachable from compute nodes.
# Each worker opens its own connection, so keep the pools modest.
python train.py --config-name=mongo_fbirn \
    experiment.name=fbirn_smri_gender \
    experiment.epochs=60 \
    experiment.batch_size=8 \
    loader.train.num_workers=8 \
    loader.eval.num_workers=8 \
    loader.train.prefetch_factor=2 \
    loader.eval.prefetch_factor=2

sleep 5s
echo "Job $SLURM_JOB_ID completed"
