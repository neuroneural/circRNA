#!/usr/bin/env python3
"""
Scan the MDD DIRECT-II tree and emit one CSV row per subject with the paths to
the three volumetric BOLD series plus the acquisition parameters.

Output columns, in order:
    subject_id, TR, n_timepoints,
    fmri_np_path, fmri_smnp_path, fmri_native_path

Series prefixes, from Data_BIDS/Code_Prep/Step2_prep_SPM.m:
    <bare>  fMRIPrep output, MNI152NLin2009cAsym, 97x115x97 @ 2 mm
    Np      + SPM12 normalise to 53x63x52 @ 3 mm, bbox [-78 -112 -70; 78 76 85]
    SmNp    + 6 mm FWHM smoothing
No slice-timing correction was applied at any stage.

For fALFF the convention (DPARSF/DPABI, and DIRECT's own unprefixed
fALFF_FunVoluWC) is to compute on UNSMOOTHED data and smooth the resulting
map, so fmri_np_path is the intended input.

Filenames are matched against the directory listing rather than constructed,
because DIRECT mixes two naming conventions: 1094 subjects use
    <ID>_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii
and 2431 use
    <ID>_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii
The res-2 entity is a newer fMRIPrep spelling out the resolution that older
versions left implicit. Both are 97x115x97 @ 2 mm with identical affines --
the difference is cosmetic, but constructing filenames silently drops 69% of
the cohort.

Usage:
    python3 00_scan_mdd_direct.py --root /path/to/MDD_DIRECT
"""

import argparse
import csv
import os
import sys

BOLD_SUFFIX = "_desc-preproc_bold.nii"
FIELDS = ["subject_id", "TR", "n_timepoints",
          "fmri_np_path", "fmri_smnp_path", "fmri_native_path"]

# consecutive empty subject dirs that mean "the mount died", not "no data"
MAX_EMPTY_STREAK = 20

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data", "mdd_direct_volume_paths.csv")


def read_trinfo(path):
    """{subject_id: (TR, n_timepoints)} from Data_Info/TRInfo.tsv.

    Columns there are: Subject ID, TR, Slice Number, Time Points, Voxel Size.
    """
    if not os.path.exists(path):
        print(f"warn: {path} not found; acquisition columns will be blank",
              file=sys.stderr)
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.reader(fh, delimiter="\t")
        next(rdr, None)
        for row in rdr:
            if not row or not row[0].strip():
                continue
            vals = [c.strip() for c in row[1:5]]
            vals += [""] * (4 - len(vals))
            out[row[0].strip()] = (vals[0], vals[2])  # TR, Time Points
    return out


def classify(names, sid):
    """Sort a subject directory's filenames into the three BOLD series."""
    found = {"np": "", "smnp": "", "native": ""}
    for n in names:
        if not n.endswith(BOLD_SUFFIX):
            continue
        if n.startswith("SmNp" + sid):
            found["smnp"] = n
        elif n.startswith("Np" + sid):
            found["np"] = n
        elif n.startswith(sid):
            found["native"] = n
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="MDD_DIRECT root (contains Data_BIDS/ and Data_Info/)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output CSV (default: {os.path.relpath(DEFAULT_OUT, HERE)})")
    ap.add_argument("--relative", action="store_true",
                    help="store paths relative to --root instead of absolute")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    funvolu = os.path.join(root, "Data_BIDS", "FunVoluW")
    if not os.path.isdir(funvolu):
        sys.exit(f"not found: {funvolu}")

    tr = read_trinfo(os.path.join(root, "Data_Info", "TRInfo.tsv"))

    subjects = sorted(e.name for e in os.scandir(funvolu)
                      if e.name.startswith("IS") and e.is_dir())
    print(f"{len(subjects)} subject directories under FunVoluW", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n_complete = n_no_tr = 0
    incomplete = []

    # A dropped sshfs mount makes os.scandir return an EMPTY LIST rather than
    # raising, so a dead mount silently yields a complete-looking CSV of blank
    # rows. Abort on a run of empty subject directories, and only move the
    # output into place once the whole scan succeeded.
    empty_streak = 0
    tmp_out = args.out + ".partial"

    with open(tmp_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, sid in enumerate(subjects, 1):
            sdir = os.path.join(funvolu, sid)
            # one scandir per subject: ~21k individual stat() calls do not
            # finish over sshfs
            entries = [e.name for e in os.scandir(sdir)]
            if entries:
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= MAX_EMPTY_STREAK:
                    sys.exit(
                        f"\naborting at {sid} ({i}/{len(subjects)}): "
                        f"{MAX_EMPTY_STREAK} consecutive empty subject directories.\n"
                        f"The mount has almost certainly dropped -- os.scandir returns "
                        f"an empty list instead of raising on a dead sshfs mount.\n"
                        f"No CSV was written (partial output left at {tmp_out}). "
                        f"Remount and rerun, or run this on the cluster directly.")
            found = classify(entries, sid)

            row = {f: "" for f in FIELDS}
            row["subject_id"] = sid
            for key, col in (("np", "fmri_np_path"),
                             ("smnp", "fmri_smnp_path"),
                             ("native", "fmri_native_path")):
                if found[key]:
                    p = os.path.join(sdir, found[key])
                    row[col] = os.path.relpath(p, root) if args.relative else p

            if sid in tr:
                row["TR"], row["n_timepoints"] = tr[sid]
            else:
                n_no_tr += 1

            if all(found.values()):
                n_complete += 1
            else:
                incomplete.append((sid, [k for k, v in found.items() if not v]))

            w.writerow(row)
            if i % 250 == 0:
                print(f"  ... {i}/{len(subjects)}", file=sys.stderr, flush=True)

    os.replace(tmp_out, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  all three series present : {n_complete}", file=sys.stderr)
    print(f"  missing a series         : {len(incomplete)}", file=sys.stderr)
    print(f"  no TRInfo row            : {n_no_tr}", file=sys.stderr)
    for sid, miss in incomplete[:10]:
        print(f"    {sid}: missing {', '.join(miss)}", file=sys.stderr)
    if len(incomplete) > 10:
        print(f"    ... and {len(incomplete) - 10} more", file=sys.stderr)


if __name__ == "__main__":
    main()
