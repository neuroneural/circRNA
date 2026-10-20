#!/bin/bash
#SBATCH --job-name=mdd_prep
#SBATCH --partition=qTRDHPC
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=mdd_prep_%j.out
#SBATCH --error=mdd_prep_%j.err

# Build the subject index and extract every modality. Run once.
#   sbatch run_prep.sh

set -euo pipefail
ENV_NAME="${ENV_NAME:-abide}"
ROOT="${ROOT:-/data/users2/ppopov1/datasets/MDD_DIRECT}"
BIDS="${BIDS:-/data/qneuromark/Data/Depression/MDD_DIRECT/Data_BIDS}"
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
        if [ -f "$c" ]; then
            printf '%s\n' "$c"
            echo "  -> using $c" >&2      # so the log shows WHICH file ran
            return 0
        fi
    done
    echo "ERROR: cannot find $base" >&2
    echo "  looked under: $REPO (flat, prep/, src/, scripts/)" >&2
    echo "  run from the directory holding the scripts, or set REPO=/path/to/them" >&2
    return 1
}

# Resolve every python file BEFORE doing any work, in plain
# assignments. `python "$(find_py X)"` does not abort under set -e
# when find_py fails -- and `exit` inside $( ) only exits the
# subshell -- so it would run `python ""` and fail bafflingly,
# possibly hours into a job.
PY_BUILD_INDEX="$(find_py prep/build_index.py)" || exit 1
PY_EXTRACT_SFNC="$(find_py prep/extract_sfnc.py)" || exit 1
PY_EXTRACT_STRUCT="$(find_py prep/extract_struct.py)" || exit 1

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

mkdir -p "$REPO/data"
echo "volumes  : $ROOT"
echo "bids     : $BIDS"
echo "started  : $(date)"

# 1. the spine. Site and group come from the subject ID; --pheno is only
#    needed for HAMD / age / sex.
python "$PY_BUILD_INDEX" \
    --root "$ROOT" --funvol "${FUNVOL:-$BIDS/FunVoluW}" \
    ${PHENO:+--pheno "$PHENO"} \
    ${CLEAN_LIST:+--clean-list "$CLEAN_LIST"} \
    ${SUPER_LIST:+--super-clean-list "$SUPER_LIST"} \
    --out "$REPO/data/mdd_master.csv"

# 2. GM / WM / CSF / voxelwise fALFF, parcellated.
#    pre_empt.pdf p.4 specifies Schaefer 200; subcortical adds the structures
#    Schaefer lacks (hippocampus, amygdala) which matter for MDD.
python "$PY_EXTRACT_STRUCT" \
    --root "$ROOT" --master "$REPO/data/mdd_master.csv" \
    --modalities GM,WM,CSF,FALFF,FALFF_globalC \
    --n-rois "${N_ROIS:-200}" --subcortical "${SUBCORTICAL:-harvard_oxford}" \
    --out "$REPO/data/struct_features.npz"

# 3. sFNC from the GIFT postprocess MAT (joined on clean_row, never file order).
if [ -n "${SFNC_MAT:-}" ]; then
    python "$PY_EXTRACT_SFNC" --mat "$SFNC_MAT" \
        --master "$REPO/data/mdd_master.csv" --with-spectra \
        --out "$REPO/data/sfnc_features.npz"
else
    echo "no SFNC_MAT set — skipping sFNC (set SFNC_MAT=/path/to/postprocess.mat)"
fi

echo "finished : $(date)"
