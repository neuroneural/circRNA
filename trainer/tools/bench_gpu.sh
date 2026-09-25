#!/bin/bash
# Throughput sweep with tools/bench_gpu.py, 1x A100.
#
# Resources are 1/8 of an A100 node: 192/8 = 24 cores, 1TB/8 = 128G.
# arctrddgxa001 is excluded: its A100s are 40GB, the rest are 80GB.
# Results append to logs/bench_gpu/results.csv, beside the experiments.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH -t 01:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:1
#SBATCH -J bench_gpu
#SBATCH -D /data/users2/ppopov1/circRNA/trainer   # submit from anywhere
#SBATCH --output=/data/users2/ppopov1/_out/%x_%j.out
#SBATCH -A psy53c17

set -e
sleep 5s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

export TMPDIR=/tmp
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /data/users2/ppopov1/miniconda/bin/activate circrna
echo "Using python from: $(which python)"

# eager vs compile, fp32 vs bf16, layout; full 8..512 sweep in results_*.csv
BATCHES="32 64 128"
python -m tools.bench_gpu --shape 64 64 64 --batches $BATCHES
python -m tools.bench_gpu --shape 64 64 64 --batches $BATCHES --amp
python -m tools.bench_gpu --shape 64 64 64 --batches $BATCHES --amp --channels-last
python -m tools.bench_gpu --shape 64 64 64 --batches $BATCHES --compile
python -m tools.bench_gpu --shape 64 64 64 --batches $BATCHES --amp --compile
python -m tools.bench_gpu --shape 64 64 64 --batches $BATCHES --amp --compile --channels-last

sleep 5s
echo "Job $SLURM_JOB_ID completed"
