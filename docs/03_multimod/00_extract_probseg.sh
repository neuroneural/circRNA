#!/usr/bin/env bash
#
# Extract the volumetric derivatives we need from DIRECT_II_Results.zip into
# one flat directory, keeping the original filenames.
#
#   tissue probability maps   10575 files   ~37 GB
#     <ID>_space-MNI152NLin2009cAsym[_res-2]_label-{GM,WM,CSF}_probseg.nii.gz
#
#   fALFF, nuisance-regressed  3525 files   ~2.9 GB
#     <ID>_fALFF.nii.gz
#
#   fALFF, + global signal reg 3525 files   ~2.9 GB
#     <ID>_fALFF_globalC.nii.gz
#
# 17625 files, ~43 GB total.
#
# In the archive both fALFF variants are named fALFF_<ID>.nii.gz, differing
# only by folder (fALFF_FunVoluWC/ vs fALFF_FunVoluWglobalC/), so extracting
# them flat would silently overwrite one with the other. Both are therefore
# staged and renamed on the way in: the ID moves to the front so they sort
# alongside the probseg files for the same subject, and the globalC set gets
# a _globalC suffix. 7z cannot rename during extraction, hence the staging.
#
# A CSV mapping every archive member to its output filename is written to
# <target>/extraction_manifest.csv.
#
# probseg filenames come in two spellings -- 1491 subjects have no res- entity
# and 2034 have res-2, a cosmetic fMRIPrep version difference (grids and
# affines are identical). Matched by suffix, so both are caught.
#
# The archive is AES-256 encrypted. `unzip` CANNOT read it: it skips every
# entry with "need PK compat. v5.1" and still exits 0, which looks like
# success. 7z is required.
#
# Runs for real only on a compute node (hostname containing 'arctrd'), since
# the target lives on /data/users2 which is not mounted elsewhere. Anywhere
# else it says so, dry-runs against the local mount, and exits.
#
# Password, in order of precedence:
#     --password-file PATH        (prefer this; chmod 600)
#     $DIRECT_ZIP_PASSWORD
#     interactive prompt
# 7z takes the password as a command-line argument, so it is briefly visible
# in `ps` to others on the node. Unavoidable; keep it out of the sbatch file
# regardless, those are world-readable.
#
# Usage:
#     export DIRECT_ZIP_PASSWORD='...'
#     ./00_extract_probseg.sh              # real run on arctrd*
#     ./00_extract_probseg.sh --dry-run    # force a dry run anywhere
#
set -euo pipefail

HOST_TOKEN=arctrd
REMOTE_ROOT=/data/qneuromark/Data/Depression/MDD_DIRECT
LOCAL_ROOT=/Users/ppopov1/_remote_data/MDD_DIRECT
TARGET=/data/users2/ppopov1/datasets/MDD_DIRECT
ARCHIVE=DIRECT_II_Results.zip

FALFF_BASE=Results/Results/FunVolu
FALFF_PLAIN=fALFF_FunVoluWC          # -> <ID>_fALFF.nii.gz
FALFF_GLOBAL=fALFF_FunVoluWglobalC   # -> <ID>_fALFF_globalC.nii.gz
GLOBAL_SUFFIX=_globalC
MANIFEST=extraction_manifest.csv

N_PROBSEG=10575
N_FALFF=3525
N_TOTAL=$((N_PROBSEG + 2 * N_FALFF))

ROOT=""
PW_FILE=""
DRY=0
OVERWRITE=0

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//;$d'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --target)        TARGET="$2"; shift 2 ;;
    --password-file) PW_FILE="$2"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    --overwrite)     OVERWRITE=1; shift ;;
    -h|--help)       usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- 7z binary
SEVENZ=""
for cand in 7z 7zz 7za; do
  if command -v "$cand" >/dev/null 2>&1; then SEVENZ="$cand"; break; fi
done
if [[ -z "$SEVENZ" ]]; then
  echo "no 7z binary found (tried 7z, 7zz, 7za)." >&2
  echo "try: module spider p7zip   /   module load p7zip   /   brew install sevenzip" >&2
  exit 1
fi
[[ "$SEVENZ" == "7za" ]] && echo "warning: using 7za; some builds lack AES-zip support" >&2

# ------------------------------------------------------------- where am I
HOST="$(hostname)"
NODE="${SLURMD_NODENAME:-}"
# lowercase via tr, not ${var,,} -- macOS ships bash 3.2, which lacks it
HOST_LC="$(printf '%s' "$HOST" | tr '[:upper:]' '[:lower:]')"
NODE_LC="$(printf '%s' "$NODE" | tr '[:upper:]' '[:lower:]')"
COMPUTE=0
if [[ "$HOST_LC" == *"$HOST_TOKEN"* || "$NODE_LC" == *"$HOST_TOKEN"* ]]; then
  COMPUTE=1
fi

if [[ $COMPUTE -eq 0 && $DRY -eq 0 ]]; then
  echo "Host '$HOST' does not contain '$HOST_TOKEN', so this is not a compute node."
  echo "$TARGET is not mounted here -- doing a dry run instead."
  echo
  DRY=1
fi

[[ -n "$ROOT" ]] || { if [[ $COMPUTE -eq 1 ]]; then ROOT="$REMOTE_ROOT"; else ROOT="$LOCAL_ROOT"; fi; }
ZIP="$ROOT/$ARCHIVE"
STAGING="$TARGET/.staging_globalC"

echo "host     : $HOST  ($([[ $COMPUTE -eq 1 ]] && echo 'compute node' || echo 'not a compute node'))"
echo "7z       : $(command -v "$SEVENZ")"
echo "archive  : $ZIP"
echo "target   : $TARGET"
echo "mode     : $([[ $DRY -eq 1 ]] && echo 'DRY RUN' || echo 'EXTRACT')"
echo

[[ -f "$ZIP" ]] || { echo "archive not found: $ZIP" >&2; exit 1; }

# ------------------------------------------------------------- inventory
# One listing pass; filenames are not encrypted, so this needs no password.
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT
echo "listing archive (268k entries, takes a moment) ..."
"$SEVENZ" l -ba -slt "$ZIP" | sed -n 's/^Path = //p' > "$LIST"

n_ps=$(grep -c 'probseg\.nii\.gz$'                        "$LIST" || true)
n_fp=$(grep -c "/${FALFF_PLAIN}/fALFF_.*\.nii\.gz$"       "$LIST" || true)
n_fg=$(grep -c "/${FALFF_GLOBAL}/fALFF_.*\.nii\.gz$"      "$LIST" || true)
echo "archive contains:"
for T in GM WM CSF; do
  printf '    probseg %-4s %s\n' "$T" "$(grep -c "_label-${T}_probseg" "$LIST" || true)"
done
printf '    %-22s %s\n' "$FALFF_PLAIN"  "$n_fp"
printf '    %-22s %s\n' "$FALFF_GLOBAL" "$n_fg"
[[ "$n_ps" -eq "$N_PROBSEG" ]] || echo "    NOTE: expected $N_PROBSEG probseg" >&2
[[ "$n_fp" -eq "$N_FALFF"   ]] || echo "    NOTE: expected $N_FALFF $FALFF_PLAIN" >&2
[[ "$n_fg" -eq "$N_FALFF"   ]] || echo "    NOTE: expected $N_FALFF $FALFF_GLOBAL" >&2

# -------------------------------------------------------------- password
if [[ -n "$PW_FILE" ]]; then
  PW="$(<"$PW_FILE")"
elif [[ -n "${DIRECT_ZIP_PASSWORD:-}" ]]; then
  PW="$DIRECT_ZIP_PASSWORD"
elif [[ -t 0 ]]; then
  read -rsp "DIRECT_II_Results.zip password: " PW; echo
else
  echo "no password available (--password-file, \$DIRECT_ZIP_PASSWORD, or a tty)" >&2
  exit 1
fi
PW="${PW%$'\n'}"

# Test one member before committing to 43 GB.
PROBE="$(grep -m1 'probseg\.nii\.gz$' "$LIST")"
echo
echo -n "password : "
if "$SEVENZ" t -p"$PW" "$ZIP" "$PROBE" >/dev/null 2>&1; then
  echo "OK"
else
  echo "FAILED"
  echo "  7z could not decrypt $(basename "$PROBE")" >&2
  exit 1
fi

# -------------------------------------------------------------- manifest
# Derived from the archive listing, so it records the intended mapping rather
# than what actually landed; the counts at the end are the record of success.
write_manifest() {
  local out="$1"
  {
    printf '# generated,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '# archive,%s\n'   "$ZIP"
    printf '# target,%s\n'    "$TARGET"
    printf 'output_file,kind,archive_path\n'
    awk -v FP="$FALFF_PLAIN" -v FG="$FALFF_GLOBAL" -v SUF="$GLOBAL_SUFFIX" '
      /probseg\.nii\.gz$/ {
        n = split($0, a, "/"); b = a[n]
        kind = "probseg"
        if      (b ~ /label-GM_/)  kind = "probseg_GM"
        else if (b ~ /label-WM_/)  kind = "probseg_WM"
        else if (b ~ /label-CSF_/) kind = "probseg_CSF"
        print b "," kind "," $0; next
      }
      /fALFF_.*\.nii\.gz$/ {
        n = split($0, a, "/"); b = a[n]
        sub(/^fALFF_/, "", b); sub(/\.nii\.gz$/, "", b)   # b is now the subject ID
        if      (index($0, "/" FP "/")) print b "_fALFF.nii.gz,fALFF," $0
        else if (index($0, "/" FG "/")) print b "_fALFF" SUF ".nii.gz,fALFF" SUF "," $0
        next
      }
    ' "$LIST" | sort
  } > "$out"
  # No field can contain a comma: DIRECT paths and filenames use only
  # [A-Za-z0-9._/-], so the rows need no quoting.
}

# ------------------------------------------------------------------- run
AO=$([[ $OVERWRITE -eq 1 ]] && echo "-aoa" || echo "-aos")   # overwrite | skip existing

if [[ $DRY -eq 1 ]]; then
  echo
  echo "would run:"
  for T in GM WM CSF; do
    echo "    $SEVENZ e -p'***' -o'$TARGET' '$ZIP' '*_label-${T}_probseg.nii.gz' -r $AO -bsp1"
  done
  for V in "$FALFF_PLAIN" "$FALFF_GLOBAL"; do
    echo "    $SEVENZ e -p'***' -o'$STAGING' '$ZIP' '$FALFF_BASE/$V/*.nii.gz' -aoa -bsp1"
  done
  echo "    then rename staged fALFF_<ID>.nii.gz -> $TARGET/<ID>_fALFF[${GLOBAL_SUFFIX}].nii.gz"
  echo "    and write the mapping to $TARGET/$MANIFEST"
  echo
  echo "sample of the mapping that would be written:"
  MTMP="$(mktemp)"; write_manifest "$MTMP"
  sed -n '4,7p' "$MTMP" | sed 's/^/    /'
  echo "    ... $(( $(wc -l < "$MTMP") - 4 )) more rows"
  rm -f "$MTMP"
  echo
  echo "dry run complete, nothing written"
  exit 0
fi

mkdir -p "$TARGET"

for T in GM WM CSF; do
  echo
  echo "=== probseg $T ==="
  "$SEVENZ" e -p"$PW" -o"$TARGET" "$ZIP" "*_label-${T}_probseg.nii.gz" -r $AO -bsp1
done

# Both fALFF variants are renamed (ID to the front), so both go through
# staging. No -r, and the full archive path in the pattern: the variants share
# basenames in the archive, so recursion would collapse them together.
trap 'rm -f "$LIST"; rm -rf "$STAGING"' EXIT
for V in "$FALFF_PLAIN" "$FALFF_GLOBAL"; do
  if [[ "$V" == "$FALFF_GLOBAL" ]]; then SUF="$GLOBAL_SUFFIX"; else SUF=""; fi
  echo
  echo "=== $V -> <ID>_fALFF${SUF}.nii.gz ==="

  # "*_fALFF.nii.gz" cannot match "*_fALFF_globalC.nii.gz", so one glob does both
  have=$(find "$TARGET" -maxdepth 1 -name "*_fALFF${SUF}.nii.gz" -type f | wc -l)
  if [[ "$have" -eq "$N_FALFF" && $OVERWRITE -eq 0 ]]; then
    echo "all $N_FALFF already present, skipping"
    continue
  fi

  rm -rf "$STAGING"; mkdir -p "$STAGING"
  "$SEVENZ" e -p"$PW" -o"$STAGING" "$ZIP" "$FALFF_BASE/$V/*.nii.gz" -aoa -bsp1
  moved=0
  for f in "$STAGING"/fALFF_*.nii.gz; do
    [[ -e "$f" ]] || continue
    id="$(basename "$f" .nii.gz)"; id="${id#fALFF_}"
    dest="$TARGET/${id}_fALFF${SUF}.nii.gz"
    if [[ $OVERWRITE -eq 1 ]]; then mv -f "$f" "$dest"; else mv -n "$f" "$dest"; fi
    moved=$((moved+1))
  done
  rm -rf "$STAGING"
  echo "renamed and moved $moved files"
done

write_manifest "$TARGET/$MANIFEST"
echo
echo "mapping written to $TARGET/$MANIFEST"

# ---------------------------------------------------------------- verify
echo
n_ps_disk=$(find "$TARGET" -maxdepth 1 -name '*probseg.nii.gz'            -type f | wc -l)
n_fg_disk=$(find "$TARGET" -maxdepth 1 -name "*_fALFF${GLOBAL_SUFFIX}.nii.gz" -type f | wc -l)
n_fp_disk=$(find "$TARGET" -maxdepth 1 -name '*_fALFF.nii.gz'                 -type f | wc -l)
printf '  %-26s %6d  (expected %d)\n' "probseg"       "$n_ps_disk" "$N_PROBSEG"
printf '  %-26s %6d  (expected %d)\n' "$FALFF_PLAIN"  "$n_fp_disk" "$N_FALFF"
printf '  %-26s %6d  (expected %d)\n' "$FALFF_GLOBAL" "$n_fg_disk" "$N_FALFF"
total=$((n_ps_disk + n_fp_disk + n_fg_disk))
echo "  total $total (expected $N_TOTAL)"

# A wrong password or a partial 7z run leaves files that exist and look
# plausibly sized but are not gzip. nibabel would only choke on them later.
BAD=0; CHECKED=0
while IFS= read -r f; do
  CHECKED=$((CHECKED+1))
  if [[ "$(head -c2 "$f" | od -An -tx1 | tr -d ' ')" != "1f8b" ]]; then
    echo "  NOT GZIP: $f" >&2; BAD=$((BAD+1))
  fi
done < <(find "$TARGET" -maxdepth 1 -name '*.nii.gz' -type f)
echo "  gzip header check: $((CHECKED-BAD))/$CHECKED ok"

[[ $BAD -eq 0 && $total -eq $N_TOTAL ]] || exit 1
