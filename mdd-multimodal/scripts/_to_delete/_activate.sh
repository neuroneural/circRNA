# Activate the conda environment inside a SLURM job. Sourced, not run.
#
# Three things make this fiddly on this cluster, and all three have bitten us:
#
#   1. conda is disabled on the login node outside a job, so activation has to
#      happen in here rather than before sbatch.
#   2. `.bashrc` gates on SLURM_JOB_ID, so `set -u` turns sourcing it into an
#      "unbound variable" crash. Hence the set +u wrapper.
#   3. `conda info --base` fails when conda is not yet on PATH, and
#      `$(conda info --base)/etc/profile.d/conda.sh` then expands to
#      /etc/profile.d/conda.sh -- which does not exist. That is the
#      "conda: command not found" + "No such file or directory" pair.
#
# Also: many .bashrc files begin with an early return for non-interactive
# shells, and a batch script is non-interactive. So sourcing .bashrc is tried
# first because it is what works here, but it is not trusted to be enough.
#
# Usage, from a script in this directory:
#     source "$(dirname "${BASH_SOURCE[0]}")/_activate.sh" "$ENV_NAME"

_env_name="${1:-abide}"

set +u

# 1. the way that works on this cluster
if [ -f "$HOME/.bashrc" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.bashrc" || true
fi

# 2. if that left conda unavailable, find conda.sh directly
if ! command -v conda >/dev/null 2>&1; then
    _candidates="${CONDA_BASE:-}"
    if [ -n "${CONDA_EXE:-}" ]; then
        _candidates="$_candidates $(dirname "$(dirname "$CONDA_EXE")")"
    fi
    _candidates="$_candidates $HOME/miniconda3 $HOME/anaconda3 $HOME/miniforge3
                 $HOME/mambaforge /opt/miniconda3 /opt/anaconda3 /opt/conda
                 /usr/local/miniconda3 /usr/local/anaconda3
                 /apps/miniconda3 /apps/anaconda3"
    for _c in $_candidates; do
        if [ -f "$_c/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$_c/etc/profile.d/conda.sh"
            break
        fi
    done
fi

# 3. conda on PATH but `conda activate` not defined as a shell function
if command -v conda >/dev/null 2>&1 && ! type conda 2>/dev/null | grep -q function; then
    _base="$(conda info --base 2>/dev/null || true)"
    if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$_base/etc/profile.d/conda.sh"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found inside this job." >&2
    echo "  Run 'which conda' in an interactive session and set CONDA_BASE to" >&2
    echo "  the directory two levels above it, e.g.:" >&2
    echo "      CONDA_BASE=/path/to/miniconda3 sbatch scripts/<script>.sh" >&2
    exit 1
fi

if ! conda activate "$_env_name" 2>/dev/null; then
    echo "ERROR: could not activate conda env '$_env_name'." >&2
    echo "  Available:" >&2
    conda env list >&2 || true
    echo "  Override with:  ENV_NAME=<name> sbatch scripts/<script>.sh" >&2
    exit 1
fi

set -u

echo "conda env : $_env_name"
echo "python    : $(which python)"
