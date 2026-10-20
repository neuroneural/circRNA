#!/bin/bash
#SBATCH --job-name=verify_fnc
#SBATCH --partition=qTRDHPC
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=verify_fnc_%j.out
#SBATCH --error=verify_fnc_%j.err

# Are the sFNC columns in Neuromark template order? If not, every node label is
# wrong and nothing downstream would tell you.
#
#   sbatch scripts/run_verify_fnc.sh                      # uses data/sfnc_features.npz
#   sbatch scripts/run_verify_fnc.sh /path/to/postprocess_results.mat
#
# Fast: one 53x53 average plus 5000 permutations.

set -euo pipefail
ENV_NAME="${ENV_NAME:-abide}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

find_py () {
    local base; base="$(basename "$1")"; local c
    for c in "$REPO/$1" "$REPO/$base" "$REPO/prep/$base" "$REPO/src/$base"; do
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

SFNC="${1:-$REPO/data/sfnc_features.npz}"
DOMAINS="${DOMAINS:-/data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.txt}"

echo "sfnc    : $SFNC"
echo "domains : $DOMAINS"
python -u "$(find_py prep/verify_fnc_order.py)" --sfnc "$SFNC" --domains "$DOMAINS"
