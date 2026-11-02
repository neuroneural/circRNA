#!/usr/bin/env python3
"""
build_cnmdd_master_csv.py
=========================

Build a master CSV for CNMDD, the 600-subject Chinese MDD set ("MDD_China": sites anding, huaxi,
xinxiang, zhejiang) in the NeuroMark subject list `subjectlist_demo_Neuromark.mat`, one row per
subject: id, site, Diagnosis, Sex, Age and the path of each ICA timecourse file. With --npz, also
pack the timecourses and labels into one .npz per ICA, as ica53_tc_to_npz.py does for DIRECT.

    python build_cnmdd_master_csv.py subjectlist_demo_Neuromark.mat --ica53 <GIFT output folder> [--npz]

Coding follows the DIRECT master CSV: Diagnosis 1 = MDD, 0 = HC; Sex 1 = male, 2 = female.

IDs: `id` is a pseudonymous cnmdd001..cnmdd600, the subject's position in the subject list. The
original IDs (`raw_id`) are written only with --raw-ids: about 36 of them contain what look like
participants' names, and two are each shared by two different people. The fMRI paths, whose folder
names also carry names, are only used to match subjects to ICA files and are never written.

ICA numbering: GIFT names files <prefix>_sub{K:03d}_timecourses_ica_s1_.nii, K = 1-based position
of the subject in the input list GIFT was given. If the folder holds GIFT's
*_ica_parameter_info.mat, K is looked up from the input files listed there; otherwise K is taken
as the position in the NeuroMark subject list, with a warning.

Needs numpy, pandas, scipy (and nibabel for --npz).
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

DX = {1: 1, 2: 0}   # CNMDD 1 depressive / 2 control -> 1 MDD / 0 HC
SEX = {0: 2, 1: 1}  # CNMDD 0 female / 1 male -> 1 male / 2 female


def load_cnmdd(mat_path):
    """CNMDD demographics in subject-list order, and each subject's fMRI path."""
    cn = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["Neuromark_Demo"].CNMDD
    col = {name.split("(")[0]: j for j, name in enumerate(cn.Demographic.definition)}

    def column(name):
        """One demographic column as floats; empty cells are missing."""
        return pd.Series([float(v) if np.size(v) == 1 else np.nan for v in cn.Demographic.value[:, col[name]]])

    df = pd.DataFrame({
        "id": [f"cnmdd{k:03d}" for k in range(1, len(cn.subjlist) + 1)],
        "raw_id": [str(v) for v in cn.Demographic.value[:, col["SubjectID"]]],
        "site": [p.split("Preprocessed_Data/")[1].split("/")[0] for p in cn.subjlist],
        "Diagnosis": column("Diagnosis").map(DX),
        "Sex": column("Gender").map(SEX),
        "Age": column("Age"),
    })
    return df, list(cn.subjlist)


def tail(path):
    """Site/subject/session/func/file: the part of an fMRI path that survives a change of root."""
    return "/".join(str(path).split(",")[0].strip().split("/")[-5:])


def gift_numbers(ica_dir, fmri):
    """1-based GIFT subject number of each fMRI path (None if GIFT did not use it)."""
    for param in glob.glob(os.path.join(ica_dir, "*_ica_parameter_info.mat")):
        try:
            ses = sio.loadmat(param, squeeze_me=True, struct_as_record=False)["sesInfo"]
            number = {tail(np.atleast_1d(f.name)[0]): k for k, f in enumerate(np.atleast_1d(ses.inputFiles), 1)}
        except Exception as e:  # e.g. a MATLAB v7.3 file, which scipy cannot read
            print(f"  WARN: could not read {param}: {e}")
            continue
        found = [number.get(tail(p)) for p in fmri]
        print(f"  {os.path.basename(param)}: {sum(k is not None for k in found)}/{len(fmri)} subjects among its inputs")
        return found

    print(f"  WARN: no readable GIFT parameter file in {ica_dir}; assuming GIFT used the subject-list order")
    return list(range(1, len(fmri) + 1))


def tc_files(ica_dir):
    """GIFT timecourse files in a folder, keyed by subject number."""
    files = {}
    for path in glob.glob(os.path.join(ica_dir, "*_sub*_timecourses_ica_s1_.nii")):
        files[int(re.search(r"_sub(\d+)_timecourses", path).group(1))] = path
    return files


def pack_npz(df, tc_col, out):
    """Timecourses (cropped to the shortest) and labels in one .npz, arrays named like the CSV columns."""
    import nibabel as nib

    rows = df[df[tc_col] != ""].reset_index(drop=True)
    tcs = [np.squeeze(np.asarray(nib.load(p).get_fdata(), dtype=np.float32)) for p in rows[tc_col]]
    t_min = min(len(t) for t in tcs)

    arrays = {tc_col: np.stack([t[:t_min] for t in tcs]),
              "id": rows["id"].to_numpy(str), "site": rows["site"].to_numpy(str)}
    arrays.update({c: rows[c].to_numpy(np.float32) for c in ("Diagnosis", "Sex", "Age")})
    arrays["full"] = rows[["Diagnosis", "Sex", "Age"]].notna().all(axis=1).to_numpy()  # as in the DIRECT npz
    np.savez_compressed(out, **arrays)
    print(f"Wrote {out}: {tc_col} {arrays[tc_col].shape} (cropped to T = {t_min}); "
          f"{arrays['full'].sum()}/{len(rows)} with Diagnosis, Sex and Age")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", help="path to subjectlist_demo_Neuromark.mat")
    ap.add_argument("--ica53", help="GIFT output folder of the 53-component ICA (Neuromark 1.0)")
    ap.add_argument("--ica105", help="GIFT output folder of the 105-component ICA (Neuromark 2.2)")
    ap.add_argument("--out", default="data/cnmdd_master.csv", help="CSV to write (default: %(default)s)")
    ap.add_argument("--npz", action="store_true", help="also write cnmdd_<ica>_tc.npz next to the CSV, per ICA")
    ap.add_argument("--raw-ids", action="store_true", help="also write the original IDs (some contain names)")
    args = ap.parse_args()
    icas = {label: d for label, d in (("ica53", args.ica53), ("ica105", args.ica105)) if d}
    if args.npz and not icas:
        ap.error("--npz needs --ica53 and/or --ica105")

    # demographics, then one timecourse column per ICA
    df, fmri = load_cnmdd(args.mat)
    for label, ica_dir in icas.items():
        files = tc_files(ica_dir)
        df[f"{label}_tc"] = [files.get(k, "") for k in gift_numbers(ica_dir, fmri)]
        print(f"  {label}_tc: {(df[f'{label}_tc'] != '').sum()}/{len(df)} files found")

    if not args.raw_ids:
        df = df.drop(columns="raw_id")
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(df)} rows, Diagnosis {df['Diagnosis'].value_counts().to_dict()} (1 MDD, 0 HC), "
          f"sites {df['site'].value_counts().to_dict()}")

    if args.npz:
        for label in icas:
            pack_npz(df, f"{label}_tc", os.path.join(out_dir, f"cnmdd_{label}_tc.npz"))


if __name__ == "__main__":
    main()
