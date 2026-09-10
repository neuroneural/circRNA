#!/usr/bin/env bash
#
# Extract the per-subject tissue probability maps (GM, WM, CSF) from
# DIRECT_II_Results.zip, keeping the original filenames:
#
#     <ID>_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
#     <ID>_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz
#
# Both spellings occur -- 1491 subjects have no res- entity, 2034 have res-2.
# It is a cosmetic difference between fMRIPrep versions; grids and affines are
# identical. Members are matched by suffix so both are caught.
# Expect 3525 subjects x 3 tissues = 10575 files, roughly 37 GB.
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
# Note 7z takes the password as a command-line argument, so it is briefly
# visible in `ps` to other users on the node. Unavoidable with 7z; keep the
# password out of the sbatch file regardless, those are world-readable.
#
# Usage:
#     export DIRECT_ZIP_PASSWORD='...'
#     ./00_extract_probseg.sh                 # real run on arctrd*
#     ./00_extract_probseg.sh --dry-run       # force a dry run anywhere
#     ./00_extract_probseg.sh --tissues GM    # one tissue only
#
set -euo pipefail

HOST_TOKEN=arctrd
REMOTE_ROOT=/data/qneuromark/Data/Depression/MDD_DIRECT
LOCAL_ROOT=/Users/ppopov1/_remote_data/MDD_DIRECT
TARGET=/data/users2/ppopov1/datasets/MDD_DIRECT
ARCHIVE=DIRECT_II_Results.zip
EXPECTED=10575

ROOT=""
LAYOUT=flat
TISSUES="GM WM CSF"
PW_FILE=""
DRY=0
OVERWRITE=0

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//;$d'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --target)        TARGET="$2"; shift 2 ;;
    --layout)        LAYOUT="$2"; shift 2 ;;
    --tissues)       TISSUES="${2//,/ }"; shift 2 ;;
    --password-file) PW_FILE="$2"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    --overwrite)     OVERWRITE=1; shift ;;
    -h|--help)       usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$LAYOUT" in
  flat|tissue) ;;
  *) echo "--layout must be 'flat' or 'tissue'" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- 7z binary
SEVENZ=""
for cand in 7z 7zz 7za; do
  if command -v "$cand" >/dev/null 2>&1; then SEVENZ="$cand"; break; fi
done
if [[ -z "$SEVENZ" ]]; then
  echo "no 7z binary found (tried 7z, 7zz, 7za)." >&2
  echo "try: module spider p7zip   /   module load p7zip" >&2
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

echo "host     : $HOST  ($([[ $COMPUTE -eq 1 ]] && echo 'compute node' || echo 'not a compute node'))"
echo "7z       : $(command -v "$SEVENZ")"
echo "archive  : $ZIP"
echo "target   : $TARGET"
echo "layout   : $LAYOUT"
echo "tissues  : $TISSUES"
echo "mode     : $([[ $DRY -eq 1 ]] && echo 'DRY RUN' || echo 'EXTRACT')"
echo

[[ -f "$ZIP" ]] || { echo "archive not found: $ZIP" >&2; exit 1; }

# ------------------------------------------------------------- inventory
# One listing pass; filenames are not encrypted, so this needs no password.
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT
echo "listing archive (268k entries, takes a moment) ..."
"$SEVENZ" l -ba -slt "$ZIP" | sed -n 's/^Path = //p' | grep 'probseg\.nii\.gz$' > "$LIST" || true

total=$(wc -l < "$LIST")
echo "matched $total probseg members"
for T in $TISSUES; do
  printf '    %-4s %s\n' "$T" "$(grep -c "_label-${T}_probseg" "$LIST" || true)"
done
printf '    %-16s %s\n' "res-2"          "$(grep -c '_res-2_' "$LIST" || true)"
printf '    %-16s %s\n' "no res- entity" "$(grep -vc '_res-2_' "$LIST" || true)"
[[ "$total" -eq "$EXPECTED" ]] || echo "    NOTE: expected $EXPECTED across all three tissues" >&2

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

# Test one member before committing to 37 GB.
PROBE="$(head -1 "$LIST")"
echo
echo -n "password : "
if "$SEVENZ" t -p"$PW" "$ZIP" "$PROBE" >/dev/null 2>&1; then
  echo "OK"
else
  echo "FAILED"
  echo "  7z could not decrypt $(basename "$PROBE")" >&2
  exit 1
fi

# ------------------------------------------------------------------- run
AO=$([[ $OVERWRITE -eq 1 ]] && echo "-aoa" || echo "-aos")   # overwrite | skip existing

if [[ $DRY -eq 1 ]]; then
  echo
  echo "would run:"
  for T in $TISSUES; do
    DEST="$TARGET"; [[ "$LAYOUT" == "tissue" ]] && DEST="$TARGET/$T"
    echo "    $SEVENZ e -p'***' -o'$DEST' '$ZIP' '*_label-${T}_probseg.nii.gz' -r $AO -bsp1"
  done
  echo
  echo "dry run complete, nothing written"
  exit 0
fi

for T in $TISSUES; do
  DEST="$TARGET"; [[ "$LAYOUT" == "tissue" ]] && DEST="$TARGET/$T"
  mkdir -p "$DEST"
  echo
  echo "=== $T -> $DEST ==="
  "$SEVENZ" e -p"$PW" -o"$DEST" "$ZIP" "*_label-${T}_probseg.nii.gz" -r $AO -bsp1
done

# ---------------------------------------------------------------- verify
echo
ON_DISK=$(find "$TARGET" -name '*probseg.nii.gz' -type f | wc -l)
echo "$ON_DISK probseg files now under $TARGET"

BAD=0
while IFS= read -r f; do
  # a truncated or undecrypted file will not start with the gzip magic bytes
  if [[ "$(head -c2 "$f" | od -An -tx1 | tr -d ' ')" != "1f8b" ]]; then
    echo "  NOT GZIP: $f" >&2; BAD=$((BAD+1))
  fi
done < <(find "$TARGET" -name '*probseg.nii.gz' -type f)
echo "gzip header check: $((ON_DISK-BAD))/$ON_DISK ok"

if [[ "$TISSUES" == "GM WM CSF" && "$ON_DISK" -ne "$EXPECTED" ]]; then
  echo "WARNING: expected $EXPECTED files" >&2
fi
[[ $BAD -eq 0 ]] || exit 1
