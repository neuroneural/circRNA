#!/bin/bash
#SBATCH --job-name=mdd_train
#SBATCH --partition=qTRDGPUL
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=mdd_train_%j.out
#SBATCH --error=mdd_train_%j.err

# One experiment.
#   sbatch run_train.sh conf/experiments/hc_vs_mdd_sfnc.yaml
#   sbatch run_train.sh conf/experiments/loso.yaml site.mode=combat

set -euo pipefail
ENV_NAME="${ENV_NAME:-abide}"
CFG="${1:-conf/experiments/hc_vs_mdd_sfnc.yaml}"
# ---------------------------------------------------------------- locate self
# sbatch COPIES this script to /var/spool/slurmd/..., so $0 and BASH_SOURCE are
# useless for finding anything. SLURM_SUBMIT_DIR is the directory you ran sbatch
# from, which is what we actually want. Falls back to $PWD for `bash script.sh`.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# Find a python file whether the layout is flat (everything in one directory) or
# the repo tree (prep/, src/, scripts/). Either works; no need to reorganise.
find_py () {
    local base; base="$(basename "$1")"
    local c
    for c in "$REPO/$1" "$REPO/$base" "$REPO/prep/$base" "$REPO/src/$base" \
             "$REPO/scripts/$base"; do
        if [ -f "$c" ]; then printf '%s\n' "$c"; return 0; fi
    done
    echo "ERROR: cannot find $base" >&2
    echo "  looked under: $REPO (flat, prep/, src/, scripts/)" >&2
    echo "  run from the directory holding the scripts, or set REPO=/path/to/them" >&2
    return 1
}

# --------------------------------------------------------- conda, self-contained
# Deliberately inlined rather than sourced from a helper file: one less path to
# get wrong, and these scripts get copied between directories.
#   - conda is disabled on the login node outside a job, so this happens in here
#   - `.bashrc` gates on SLURM_JOB_ID, so `set -u` makes sourcing it crash
#   - `conda info --base` before conda is on PATH expands to /etc/profile.d/...
#   - many .bashrc files return early for non-interactive shells (a batch job is
#     non-interactive), so sourcing it is tried first but not trusted
set +u
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc" || true; fi
if ! command -v conda >/dev/null 2>&1; then
    _cands="${CONDA_BASE:-}"
    [ -n "${CONDA_EXE:-}" ] && _cands="$_cands $(dirname "$(dirname "$CONDA_EXE")")"
    _cands="$_cands $HOME/miniconda3 $HOME/anaconda3 $HOME/miniforge3 $HOME/mambaforge
            /opt/miniconda3 /opt/anaconda3 /opt/conda /usr/local/miniconda3
            /usr/local/anaconda3 /apps/miniconda3 /apps/anaconda3"
    for _c in $_cands; do
        if [ -f "$_c/etc/profile.d/conda.sh" ]; then source "$_c/etc/profile.d/conda.sh"; break; fi
    done
fi
if command -v conda >/dev/null 2>&1 && ! type conda 2>/dev/null | grep -q function; then
    _b="$(conda info --base 2>/dev/null || true)"
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && source "$_b/etc/profile.d/conda.sh"
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in this job." >&2
    echo "  Run 'which conda' interactively, then:" >&2
    echo "      CONDA_BASE=/dir/two/levels/above sbatch <this script>" >&2
    exit 1
fi
if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "ERROR: could not activate env '$ENV_NAME'. Available:" >&2
    conda env list >&2 || true
    echo "  Override with:  ENV_NAME=<name> sbatch <this script>" >&2
    exit 1
fi
set -u

echo "repo     : $REPO"
echo "conda    : $ENV_NAME    python: $(which python)"

if [ ! -d "$REPO/src" ]; then
    echo "ERROR: no src/ under $REPO — src.train needs the repo tree." >&2
    exit 1
fi

echo "config   : $CFG"
echo "extra    : ${*:2}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m src.train --config "$CFG" "${@:2}"
echo "finished : $(date)"
