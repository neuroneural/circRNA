#!/usr/bin/env python3
"""
Scan the MDD DIRECT-II tree and emit one CSV row per subject with the paths to
the three volumetric BOLD series plus the acquisition parameters.

Output columns, in order:
    subject_id, TR, n_timepoints,
    fmri_np_path, fmri_smnp_path, fmri_native_path,
    falff_c_path, falff_globalc_path,                    <- 2 mm, as DIRECT shipped
    gm_probseg_path, wm_probseg_path, csf_probseg_path,  <- 1 mm, as DIRECT shipped
    falff_3mm_path, falff_globalc_3mm_path,              <- resampled onto the Np grid
    gm_probseg_3mm_path, wm_probseg_3mm_path, csf_probseg_3mm_path

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

Point --derivatives-dir at that directory, and --proc-dir at where
02_resample_to_fmri_grid.py wrote its 3 mm outputs:

    <ID>_fALFF_3mm.nii.gz            <ID>_GM_probseg_3mm.nii.gz
    <ID>_fALFF_globalC_3mm.nii.gz    <ID>_WM_probseg_3mm.nii.gz
                                     <ID>_CSF_probseg_3mm.nii.gz

Only the _3mm_ columns share the fMRI grid; the others are in DIRECT's native
2 mm and 1 mm sampling. Missing directories are not an error -- those columns
are left blank, so the manifest can be built at any stage of the pipeline.

Grid warning: DIRECT's fALFF is 97x115x97 @ 2 mm with a FLIPPED x axis, and
the probsegs are template-native 1 mm. Neither matches the 3 mm Np grid the
BOLD columns point at. Resample deliberately before combining them.

Paths are chosen from the hostname: a node whose name contains "arctrd" gets
the cluster layout, anything else gets the Mac's sshfs mount points. Any of the
three can still be overridden individually.

Usage:
    python3 01_scan_mdd_direct.py                      # paths from the hostname
    python3 01_scan_mdd_direct.py --proc-dir /elsewhere
"""

import argparse
import csv
import os
import socket
import sys

BOLD_SUFFIX = "_desc-preproc_bold.nii"
FIELDS = ["subject_id", "TR", "n_timepoints",
          "fmri_np_path", "fmri_smnp_path", "fmri_native_path",
          "falff_c_path", "falff_globalc_path",
          "gm_probseg_path", "wm_probseg_path", "csf_probseg_path",
          "falff_3mm_path", "falff_globalc_3mm_path",
          "gm_probseg_3mm_path", "wm_probseg_3mm_path", "csf_probseg_3mm_path"]

MANIFEST = "extraction_manifest.csv"

# Paths differ between the cluster and the Mac's sshfs mounts, so pick a
# profile from the hostname rather than making the caller retype them.
HOST_TOKEN = "arctrd"
REMOTE_PATHS = dict(
    root="/data/qneuromark/Data/Depression/MDD_DIRECT",
    deriv="/data/users2/ppopov1/datasets/MDD_DIRECT",
    proc="/data/users2/ppopov1/datasets/MDD_DIRECT_proc")
LOCAL_PATHS = dict(
    root="/Users/ppopov1/_remote_data/MDD_DIRECT",
    deriv="/Users/ppopov1/_remote_data/MDD_extract",
    proc="/Users/ppopov1/_remote_data/MDD_DIRECT_proc")


def on_compute_node():
    names = [socket.gethostname(), os.environ.get("SLURMD_NODENAME", ""),
             os.environ.get("HOSTNAME", "")]
    return any(HOST_TOKEN in n.lower() for n in names if n)

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

# same, for the 3 mm outputs of 02_resample_to_fmri_grid.py
PROC_SUFFIXES = [
    ("falff_globalc_3mm_path", "_fALFF_globalC_3mm.nii.gz"),
    ("falff_3mm_path",         "_fALFF_3mm.nii.gz"),
    ("gm_probseg_3mm_path",    "_GM_probseg_3mm.nii.gz"),
    ("wm_probseg_3mm_path",    "_WM_probseg_3mm.nii.gz"),
    ("csf_probseg_3mm_path",   "_CSF_probseg_3mm.nii.gz"),
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


def index_derivatives(dirpath, suffixes, manifest=None, label="derivatives"):
    """{column: {subject_id: abspath}} from a single scandir of a flat dir.

    Subject IDs contain no underscore (IS001-1-0001), so the leading
    underscore-delimited token is the ID for every name 00 and 02 write.
    """
    idx = {col: {} for col, _ in suffixes}
    if not dirpath:
        return idx
    if not os.path.isdir(dirpath):
        print(f"warn: {label} dir not found: {dirpath}\n"
              f"      those columns will be blank", file=sys.stderr)
        return idx

    n_seen = 0
    for e in os.scandir(dirpath):
        if not e.is_file() or not e.name.endswith(".nii.gz"):
            continue
        n_seen += 1
        for col, suffix in suffixes:
            if e.name.endswith(suffix):
                idx[col][e.name.split("_", 1)[0]] = os.path.join(dirpath, e.name)
                break

    print(f"indexed {n_seen} .nii.gz in {dirpath}", file=sys.stderr)
    for col, _ in suffixes:
        print(f"    {col:24s} {len(idx[col])}", file=sys.stderr)

    # cross-check against the manifest of what should be there
    if manifest:
        man = os.path.join(dirpath, manifest)
        if os.path.exists(man):
            with open(man, newline="") as fh:
                expected = sum(1 for ln in fh
                               if ln.strip() and not ln.startswith("#")) - 1
            if expected != n_seen:
                print(f"warn: {manifest} lists {expected} outputs but {n_seen} are "
                      f"on disk -- extraction may be incomplete", file=sys.stderr)
        else:
            print(f"note: no {manifest} in {dirpath}, skipping the completeness check",
                  file=sys.stderr)
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    compute = on_compute_node()
    d = REMOTE_PATHS if compute else LOCAL_PATHS

    ap.add_argument("--root", default=d["root"],
                    help="MDD_DIRECT root (contains Data_BIDS/ and Data_Info/)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output CSV (default: {os.path.relpath(DEFAULT_OUT, HERE)})")
    ap.add_argument("--derivatives-dir", default=d["deriv"],
                    help="where 00_extract_probseg.sh wrote the fALFF and probseg files")
    ap.add_argument("--proc-dir", default=d["proc"],
                    help="where 02_resample_to_fmri_grid.py wrote the 3 mm volumes")
    ap.add_argument("--relative", action="store_true",
                    help="store paths relative to --root instead of absolute; "
                         "derivative paths stay absolute, they live outside --root")
    args = ap.parse_args()

    host = socket.gethostname()
    print(f"host        : {host} "
          f"({'compute node' if compute else 'not a compute node'})", file=sys.stderr)
    print(f"profile     : {'remote' if compute else 'local sshfs mounts'}", file=sys.stderr)
    print(f"root        : {args.root}", file=sys.stderr)
    print(f"derivatives : {args.derivatives_dir}", file=sys.stderr)
    print(f"proc        : {args.proc_dir}", file=sys.stderr)
    print(f"out         : {args.out}", file=sys.stderr)
    if not compute:
        print("\nnote: off the cluster this scans 3525 subject directories over sshfs,\n"
              "      which is slow and prone to dropping. Prefer running it on a node.",
              file=sys.stderr)
    print(file=sys.stderr)

    root = os.path.abspath(args.root)
    funvolu = os.path.join(root, "Data_BIDS", "FunVoluW")
    if not os.path.isdir(funvolu):
        sys.exit(f"not found: {funvolu}")

    tr = read_trinfo(os.path.join(root, "Data_Info", "TRInfo.tsv"))
    deriv = index_derivatives(args.derivatives_dir, DERIV_SUFFIXES,
                              manifest=MANIFEST, label="extracted 1 mm/2 mm")
    proc = index_derivatives(args.proc_dir, PROC_SUFFIXES, label="resampled 3 mm")

    subjects = sorted(e.name for e in os.scandir(funvolu)
                      if e.name.startswith("IS") and e.is_dir())
    print(f"{len(subjects)} subject directories under FunVoluW", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n_complete = n_no_tr = 0
    n_deriv = {col: 0 for col, _ in DERIV_SUFFIXES + PROC_SUFFIXES}
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
            for src, suffixes in ((deriv, DERIV_SUFFIXES), (proc, PROC_SUFFIXES)):
                for col, _ in suffixes:
                    if sid in src[col]:
                        row[col] = src[col][sid]
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
    for col, _ in DERIV_SUFFIXES + PROC_SUFFIXES:
        print(f"  {col:24s} : {n_deriv[col]}", file=sys.stderr)
    for sid, miss in incomplete[:10]:
        print(f"    {sid}: missing {', '.join(miss)}", file=sys.stderr)
    if len(incomplete) > 10:
        print(f"    ... and {len(incomplete) - 10} more", file=sys.stderr)


if __name__ == "__main__":
    main()
