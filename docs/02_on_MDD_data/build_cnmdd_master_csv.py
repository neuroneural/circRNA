#!/usr/bin/env python3
"""
build_cnmdd_master_csv.py
=========================

Build ONE master CSV for CNMDD ("ChineseMDD": the 600-subject Chinese MDD set of the NeuroMark
subject list; sites anding, huaxi, xinxiang, zhejiang), one row per subject, joining:

  * demographics + diagnosis   (subjectlist_demo_Neuromark.mat, Neuromark_Demo.CNMDD)
  * ICA timecourse path        (the NeuroMark GIFT run in ICA_DIR, file prefix CMDD)

With --npz, also pack the timecourses of the 53 NeuroMark networks and the labels into one .npz,
laid out like DIRECT's mdd_ica53_tc_superclean.npz.

Paths are HARDCODED to the cluster layout in the CONFIG block (or override with --mat / --ica-dir):

    python build_cnmdd_master_csv.py [--npz] [--raw-ids]

The subject list holds the fMRI files that went INTO the ICA; the per-subject timecourses are
GIFT outputs in ICA_DIR.

IMPORTANT — ICA subjects and numbering (verified against the data, 2026-11-02):
  * GIFT ran on 588 of the 600 subjects; the other 12 get a blank `ica_tc`.
  * File K (CMDD_sub{K:03d}_timecourses_ica_s1_.nii) belongs to the K-th input in GIFT's subject
    file CMDDSubject.mat, so rows are matched to it by fMRI path. K is NOT the subject-list
    position: the 12 skipped subjects shift it.
  * Each file is T x 100 (T = 225, 235 or 245; TR = 2 s): every component of the NeuroMark 1.0
    template. The 53 networks, in the order listed in ICN_ORDER (1-based template components),
    line up with DIRECT's ica53 columns: their group-mean FNC correlates r = 0.98 with DIRECT's,
    every component >= 0.94.
  * `gift_row` (0-based) indexes the subject axis of CMDD_postprocess_results.mat
    (fnc_corrs_all, spectra_tc_all; all 100 components).

Coding follows the DIRECT master CSV: Diagnosis 1 = MDD, 0 = HC; Sex 1 = male, 2 = female.

IDs: `id` is a pseudonymous cnmdd001..cnmdd600 (position in the subject list). The original IDs
(`raw_id`) are written only with --raw-ids: about 36 contain what look like participants' names,
and two are each shared by two different people. The fMRI paths, whose folder names also carry
names, are only used for matching and never written.

Needs numpy, pandas, scipy, h5py (CMDDSubject.mat is MATLAB v7.3), and nibabel for --npz.
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — hardcoded cluster paths / options (edit here)
# ══════════════════════════════════════════════════════════════════════════════
MAT = "/data/qneuromark/Results/Data_info/ChineseMDD/subjectlist_demo_Neuromark.mat"
ICA_DIR = "/data/qneuromark/Results/ICA/ChineseMDD"   # NeuroMark GIFT output, prefix CMDD
SUBJECT_FILE = "CMDDSubject.mat"                      # GIFT's inputs, in its file numbering
TC_PATTERN = "CMDD_sub%03d_timecourses_ica_s1_.nii"
OUT = "data/cnmdd_master.csv"
OUT_NPZ = "data/cnmdd_ica53_tc.npz"                   # written with --npz
ICN_ORDER = "data/ICA_correct_order.csv"              # the 53 networks for --npz, in DIRECT's ica53 order
CHECK = True                                          # stat() each timecourse, blank it if missing
# ══════════════════════════════════════════════════════════════════════════════

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


def gift_inputs(subject_file):
    """fMRI file of each GIFT subject, in GIFT's numbering."""
    import h5py

    with h5py.File(subject_file, "r") as f:
        # each entry is a char matrix of 'file.nii,frame' rows, stored as (chars, rows)
        return ["".join(map(chr, f[ref][:, 0])).split(",")[0].strip() for ref in f["files"]["name"][()].ravel()]


def pack_npz(df, out):
    """53-network timecourses (cropped to the shortest) and labels in one .npz, arrays named like DIRECT's."""
    import nibabel as nib

    rows = df[df["ica_tc"] != ""].reset_index(drop=True)
    icn = np.loadtxt(ICN_ORDER, dtype=int) - 1  # one 1-based component per line
    assert icn.shape == (53,), f"{ICN_ORDER} should list 53 components"
    tcs = [np.asarray(nib.load(p).get_fdata(), dtype=np.float32)[:, icn] for p in rows["ica_tc"]]
    t_min = min(len(t) for t in tcs)

    arrays = {"ica53_tc": np.stack([t[:t_min] for t in tcs]),
              "id": rows["id"].to_numpy(str), "site": rows["site"].to_numpy(str)}
    arrays.update({c: rows[c].to_numpy(np.float32) for c in ("Diagnosis", "Sex", "Age")})
    arrays["full"] = rows[["Diagnosis", "Sex", "Age"]].notna().all(axis=1).to_numpy()  # as in the DIRECT npz
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"Wrote {out}: ica53_tc {arrays['ica53_tc'].shape} (cropped to T = {t_min}); "
          f"{arrays['full'].sum()}/{len(rows)} with Diagnosis, Sex and Age")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", default=MAT, help="NeuroMark subject list (default: %(default)s)")
    ap.add_argument("--ica-dir", default=ICA_DIR, help="GIFT output folder (default: %(default)s)")
    ap.add_argument("--npz", action="store_true", help=f"also write {OUT_NPZ}")
    ap.add_argument("--raw-ids", action="store_true", help="also write the original IDs (some contain names)")
    args = ap.parse_args()

    # demographics, then each subject's position K in GIFT's input list
    df, fmri = load_cnmdd(args.mat)
    number = {path: k for k, path in enumerate(gift_inputs(os.path.join(args.ica_dir, SUBJECT_FILE)), 1)}
    k = pd.Series([number.get(p) for p in fmri], dtype="Int64")
    df["gift_row"] = k - 1
    df["ica_tc"] = [os.path.join(args.ica_dir, TC_PATTERN % i) if pd.notna(i) else "" for i in k]
    if CHECK:
        df["ica_tc"] = [p if p and os.path.exists(p) else "" for p in df["ica_tc"]]

    if not args.raw_ids:
        df = df.drop(columns="raw_id")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    df.to_csv(OUT, index=False)
    no_ica = df[k.isna().to_numpy()]
    print(f"Wrote {OUT}: {len(df)} rows, Diagnosis {df['Diagnosis'].value_counts().to_dict()} (1 MDD, 0 HC), "
          f"sites {df['site'].value_counts().to_dict()}")
    print(f"  in the ICA: {k.notna().sum()}/{len(df)}; not in it: Diagnosis {no_ica['Diagnosis'].value_counts().to_dict()}")
    print(f"  ica_tc resolved: {(df['ica_tc'] != '').sum()}/{len(df)} (check_exists={CHECK})")

    if args.npz:
        pack_npz(df, OUT_NPZ)


if __name__ == "__main__":
    main()
