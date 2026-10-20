#!/bin/bash
#SBATCH --job-name=mdd_sweep
#SBATCH --partition=qTRDGPUL
#SBATCH --account=psy53c17
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# Sweep the three questions the repo was built to answer, then test each one with
# the mixed model rather than by comparing means.
#
#   sbatch scripts/run_sweep.sh
#   SWEEP=site sbatch scripts/run_sweep.sh     # just one block

set -euo pipefail

REPO="${REPO:-$HOME/mdd-multimodal}"
ENV_NAME="${ENV_NAME:-abide}"
SWEEP="${SWEEP:-all}"

mkdir -p "$REPO/logs" "$REPO/runs"
cd "$REPO"

# NB: under SLURM the batch script is copied to /var/spool/slurmd/..., so
# ${BASH_SOURCE[0]} is NOT in the repo. Resolve via $REPO, which we just cd'd to.
# shellcheck source=scripts/_activate.sh
source "$REPO/scripts/_activate.sh" "$ENV_NAME"

run () { echo; echo "### $*"; python -m src.train "$@"; }

# ---- 1. does scanner correction change anything? ---------------------------
if [ "$SWEEP" = "all" ] || [ "$SWEEP" = "site" ]; then
  for M in ignore feature combat random_effect adversarial; do
    run --config conf/experiments/hc_vs_mdd_sfnc.yaml "site.mode=$M"
  done
  python -m src.evaluate 'runs/hc_vs_mdd_sfnc_*.json' --factor site.mode \
      --csv runs/sweep_site.csv
fi

# ---- 2. which modalities earn their place? ---------------------------------
if [ "$SWEEP" = "all" ] || [ "$SWEEP" = "mods" ]; then
  for MODS in "[sFNC]" "[GM]" "[GM,WM,CSF]" "[GM,CSF,FALFF]" "[GM,CSF,FALFF,sFNC]"; do
    run --config conf/experiments/hc_vs_mdd_sfnc.yaml "data.modalities=$MODS"
  done
  # complete-case vs masked, on the same modality set: what do the 999 sFNC-less
  # subjects actually buy us?
  run --config conf/experiments/hc_vs_mdd_sfnc.yaml data.cohort=all \
      data.allow_missing_modalities=true
  run --config conf/experiments/hc_vs_mdd_sfnc.yaml data.cohort=all \
      data.allow_missing_modalities=false
fi

# ---- 3. how should the ~660 HAMD subjects be used? -------------------------
if [ "$SWEEP" = "all" ] || [ "$SWEEP" = "hamd" ]; then
  for M in direct transfer holdout; do
    run --config conf/experiments/hamd_transfer.yaml "label.hamd_mode=$M"
  done
  python -m src.evaluate 'runs/hamd_transfer_*.json' --factor label.hamd_mode \
      --csv runs/sweep_hamd.csv
fi

# ---- 4. fusion mode --------------------------------------------------------
if [ "$SWEEP" = "all" ] || [ "$SWEEP" = "fusion" ]; then
  for F in joint late stack; do
    run --config conf/experiments/hc_vs_mdd_sfnc.yaml "model.fusion.mode=$F"
  done
fi

echo
python -m src.evaluate 'runs/*.json' --csv runs/all_folds.csv
echo "finished: $(date)"
