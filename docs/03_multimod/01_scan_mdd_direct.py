#!/usr/bin/env python3
"""
Scan the MDD DIRECT-II tree and emit one CSV row per subject with the paths to
the three volumetric BOLD series plus the acquisition parameters.

Output columns, in order:
    subject_id, TR, n_timepoints,
    fmri_np_path, fmri_smnp_path, fmri_native_path,
    falff_c_path, falff_globalc_path,
    gm_probseg_path, wm_probseg_path, csf_probseg_path

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

DIRECT's own volumetric fALFF and the tissue probability maps are not in the
MDD_DIRECT tree at all -- they live inside DIRECT_II_Results.zip and must be
extracted first by 00_extract_probseg.sh, which writes them flat into one
directory with these names:

    <ID>_fALFF.nii.gz                                       nuisance-regressed
    <ID>_fALFF_globalC.nii.gz                               + global signal reg
    <ID>_space-MNI152NLin2009cAsym[_res-2]_label-GM_probseg.nii.gz
    <ID>_space-MNI152NLin2009cAsym[_res-2]_label-WM_probseg.nii.gz
    <ID>_space-MNI152NLin2009cAsym[_res-2]_label-CSF_probseg.nii.gz

Point --derivatives-dir at that directory. Missing derivatives are not an
error; the columns are just left blank, so the manifest can be built before
extraction has run.

Grid warning: DIRECT's fALFF is 97x115x97 @ 2 mm with a FLIPPED x axis, and
the probsegs are template-native 1 mm. Neither matches the 3 mm Np grid the
BOLD columns point at. Resample deliberately before combining them.

Usage:
    python3 01_scan_mdd_direct.py --root /path/to/MDD_DIRECT
    python3 01_scan_mdd_direct.py --root ... --derivatives-dir /path/to/extracted
"""

import argparse
import csv
import os
import sys

BOLD_SUFFIX = "_desc-preproc_bold.nii"
FIELDS = ["subject_id", "TR", "n_timepoints",
          "fmri_np_path", "fmri_smnp_path", "fmri_native_path",
          "falff_c_path", "falff_globalc_path",
          "gm_probseg_path", "wm_probseg_path", "csf_probseg_path"]

DEFAULT_DERIVATIVES = "/data/users2/ppopov1/datasets/MDD_DIRECT"
MANIFEST = "extraction_manifest.csv"

# column -> filename suffix written by 00_extract_probseg.sh. Order matters:
# "_fALFF_globalC.nii.gz" must be tested before "_fALFF.nii.gz" would be, but
# they cannot collide anyway since the globalC name does not end in _fALFF.
DERIV_SUFFIXES = [
    ("falff_globalc_path", "_fALFF_globalC.nii.gz"),
    ("falff_c_path",       "_fALFF.nii.gz"),
    ("gm_probseg_path",    "_label-GM_probseg.nii.gz"),
    ("wm_probseg_path",    "_label-WM_probseg.nii.gz"),
    ("csf_probseg_path",   "_label-CSF_probseg.nii.gz"),
]

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


def index_derivatives(deriv):
    """{column: {subject_id: abspath}} from a single scandir of the flat dir.

    Subject IDs contain no underscore (IS001-1-0001), so the leading
    underscore-delimited token is the ID for every name 00 writes.
    """
    idx = {col: {} for col, _ in DERIV_SUFFIXES}
    if not deriv:
        return idx
    if not os.path.isdir(deriv):
        print(f"warn: derivatives dir not found: {deriv}\n"
              f"      derivative columns will be blank", file=sys.stderr)
        return idx

    n_seen = 0
    for e in os.scandir(deriv):
        if not e.is_file() or not e.name.endswith(".nii.gz"):
            continue
        n_seen += 1
        for col, suffix in DERIV_SUFFIXES:
            if e.name.endswith(suffix):
                idx[col][e.name.split("_", 1)[0]] = os.path.join(deriv, e.name)
                break

    print(f"indexed {n_seen} .nii.gz in {deriv}", file=sys.stderr)
    for col, _ in DERIV_SUFFIXES:
        print(f"    {col:20s} {len(idx[col])}", file=sys.stderr)

    # Cross-check against 00's manifest, which lists what SHOULD be there.
    man = os.path.join(deriv, MANIFEST)
    if os.path.exists(man):
        with open(man, newline="") as fh:
            expected = sum(1 for ln in fh
                           if ln.strip() and not ln.startswith("#")) - 1
        if expected != n_seen:
            print(f"warn: {MANIFEST} lists {expected} outputs but {n_seen} are "
                  f"on disk -- extraction may be incomplete", file=sys.stderr)
    else:
        print(f"note: no {MANIFEST} in {deriv}, skipping the completeness check",
              file=sys.stderr)
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="MDD_DIRECT root (contains Data_BIDS/ and Data_Info/)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output CSV (default: {os.path.relpath(DEFAULT_OUT, HERE)})")
    ap.add_argument("--derivatives-dir", default=DEFAULT_DERIVATIVES,
                    help="where 00_extract_probseg.sh wrote the fALFF and probseg "
                         f"files (default: {DEFAULT_DERIVATIVES})")
    ap.add_argument("--relative", action="store_true",
                    help="store paths relative to --root instead of absolute; "
                         "derivative paths stay absolute, they live outside --root")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    funvolu = os.path.join(root, "Data_BIDS", "FunVoluW")
    if not os.path.isdir(funvolu):
        sys.exit(f"not found: {funvolu}")

    tr = read_trinfo(os.path.join(root, "Data_Info", "TRInfo.tsv"))
    deriv = index_derivatives(args.derivatives_dir)

    subjects = sorted(e.name for e in os.scandir(funvolu)
                      if e.name.startswith("IS") and e.is_dir())
    print(f"{len(subjects)} subject directories under FunVoluW", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n_complete = n_no_tr = 0
    n_deriv = {col: 0 for col, _ in DERIV_SUFFIXES}
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

            # Derivatives live outside --root, so these stay absolute even
            # with --relative; os.path.relpath would emit ../../.. chains.
            for col, _ in DERIV_SUFFIXES:
                if sid in deriv[col]:
                    row[col] = deriv[col][sid]
                    n_deriv[col] += 1

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
    for col, _ in DERIV_SUFFIXES:
        print(f"  {col:24s} : {n_deriv[col]}", file=sys.stderr)
    for sid, miss in incomplete[:10]:
        print(f"    {sid}: missing {', '.join(miss)}", file=sys.stderr)
    if len(incomplete) > 10:
        print(f"    ... and {len(incomplete) - 10} more", file=sys.stderr)


if __name__ == "__main__":
    main()
