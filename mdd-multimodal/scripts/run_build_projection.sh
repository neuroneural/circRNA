#!/bin/bash
#SBATCH --job-name=build_proj
#SBATCH --partition=qTRDHPC
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=build_proj_%j.out
#SBATCH --error=build_proj_%j.err

# Fixed parcel -> Neuromark-component projection, so panel A's structural node
# meanings are anatomical rather than fitted.
#
#   sbatch scripts/run_build_projection.sh
#
# N_ROIS and SUBCORTICAL must match what run_prep.sh used for extract_struct.py,
# or the rows of the projection address the wrong parcels.

set -euo pipefail
ENV_NAME="${ENV_NAME:-abide}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

find_py () {
    local base; base="$(basename "$1")"; local c
    for c in "$REPO/$1" "$REPO/$base" "$REPO/prep/$base"; do
        [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }
    done
    echo "ERROR: cannot find $base under $REPO" >&2; return 1
}

set +u
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" || true
if ! command -v conda >/dev/null 2>&1; then
    for c in ${CONDA_BASE:-} "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$c/etc/profile.d/conda.sh" ] && { source "$c/etc/profile.d/conda.sh"; break; }
    done
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found; set CONDA_BASE" >&2; exit 1; }
conda activate "$ENV_NAME" || { echo "ERROR: cannot activate $ENV_NAME" >&2; exit 1; }
set -u

TEMPLATE="${TEMPLATE:-/data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.nii}"
echo "template   : $TEMPLATE"
echo "n_rois     : ${N_ROIS:-200}   subcortical: ${SUBCORTICAL:-harvard_oxford}"
echo "(these MUST match run_prep.sh / extract_struct.py)"

python -u "$(find_py prep/build_projection.py)" \
    --template "$TEMPLATE" \
    --n-rois "${N_ROIS:-200}" \
    --subcortical "${SUBCORTICAL:-harvard_oxford}" \
    --modalities "${MODALITIES:-GM,WM,CSF,FALFF,FALFF_globalC}" \
    --out "${OUT:-$REPO/data/projection.npz}"
