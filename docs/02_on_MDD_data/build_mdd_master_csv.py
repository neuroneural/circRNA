#!/usr/bin/env python3
"""
build_mdd_master_csv.py
=======================

Build ONE "master" CSV for the MDD DIRECT dataset, one row per subject,
keyed by the consortium subject ID (e.g. ``IS001-1-0192``), joining:

  * demographics + clinical scores      (FULL_sites.csv)
  * acquisition parameters              (Data_Info/TRInfo.tsv)
  * diagnostic group decoded from the ID
  * QC-tier membership flags            (Subjlist_clean.txt / Subjlist_super_clean.txt)
  * file paths for every modality       (volume fMRI, ICA timecourses, ICA spatial maps)
  * sFNC/spectra location               (postprocess .mat + row index)

Paths are HARDCODED to the cluster layout in the CONFIG block below — just edit
those constants if needed and run:

    python build_mdd_master_csv.py

Writes TWO files: OUT (curated columns only) and OUT + "_full" suffix
(curated columns PLUS every clinical column from FULL_sites.csv).

IMPORTANT — ICA file numbering (verified against the data):
  The per-subject ICA NIfTIs are named ``MDD_DIRECT_sub{K:03d}_..._ica_s1_.nii``
  where K is the **1-based position of the subject in the subject list that ICA
  used**, NOT the MATLABIndex:
    * neuromark53_clean/                     -> Subjlist_clean.txt        (2,526)
    * results/ica/neuromark105_super_clean/  -> Subjlist_super_clean.txt  (2,426)
  The same positions index the sFNC/spectra arrays inside each postprocess .mat
  (fnc_corrs_all (S,N,N), spectra_tc_all (S,F,N)): columns `clean_row` (53) and
  `super_clean_row` (105), both 0-based.

Needs only pandas. Building paths does not read imaging data; CHECK just stat()s them.
"""

import glob
import os
import re
import sys

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — hardcoded cluster paths / options (edit here)
# ══════════════════════════════════════════════════════════════════════════════
RAW_ROOT = "/data/qneuromark/Data/Depression/MDD_DIRECT"
PREPROC_ROOT = "/data/users3/bbaker/projects/MDD_preproc"

DEMO_CSV = f"{PREPROC_ROOT}/data/MDDD/FULL_sites.csv"
TRINFO = f"{RAW_ROOT}/Data_Info/TRInfo.tsv"
CLEAN_LIST = f"{PREPROC_ROOT}/data/MDDD/Subjlist_clean.txt"          # -> neuromark53_clean
SUPER_LIST = f"{PREPROC_ROOT}/data/MDDD/Subjlist_super_clean.txt"    # -> neuromark105_super_clean
ICA53_DIR = f"{PREPROC_ROOT}/neuromark53_clean"
ICA105_DIR = f"{PREPROC_ROOT}/results/ica/neuromark105_super_clean"

OUT = "mdd_master.csv"   # curated columns; also writes mdd_master_full.csv (+ all FULL_sites clinical cols)
SUBSET = "full"          # "full" (3,525) | "clean" (2,526) | "super_clean" (2,426)
                         # in_clean / in_super_clean columns let you filter tiers from the full CSV
CHECK = True             # stat() each path and blank it if the file is missing
# ══════════════════════════════════════════════════════════════════════════════

# ── Fixed naming conventions ──────────────────────────────────────────────────
ID_RE = re.compile(r"IS\d+-\d+-\d+")
GROUP_LABELS = {1: "MDD", 2: "HC", 3: "Bipolar", 4: "Schizophrenia"}

# BOLD files: prefix + ID + entities. Some sites add an optional `res-2` entity
# before `desc-preproc`, so volumes are resolved by globbing (see vol_file).
VOL_GLOB = "_task-rest_space-MNI152NLin2009cAsym*desc-preproc_bold.nii"
# 3 mm Neuromark-grid volumes only (what ICA used: SmNp = smoothed, Np = unsmoothed).
# The 2 mm full-res base file is intentionally omitted — ICA never used it.
VOL_PREFIXES = {"vol_np_3mm": "Np", "vol_smnp_3mm": "SmNp"}

TC_PATTERN = "MDD_DIRECT_sub%03d_timecourses_ica_s1_.nii"
SM_PATTERN = "MDD_DIRECT_sub%03d_component_ica_s1_.nii"
POSTPROC_NAME = "MDD_DIRECT_postprocess_results.mat"


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_id(sid):
    """IS001-1-0192 -> (site='IS001', group_code=1, group_label='MDD')."""
    m = ID_RE.search(str(sid))
    if not m:
        return (None, None, None)
    parts = m.group(0).split("-")
    site, grp = parts[0], int(parts[1])
    return site, grp, GROUP_LABELS.get(grp, f"group{grp}")


def load_subjlist(path):
    """Ordered list of subject IDs parsed from a Subjlist_*.txt file (paths)."""
    ids = []
    if path and os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                m = ID_RE.search(line)
                if m:
                    ids.append(m.group(0))
    return ids


def detect_id_col(df):
    for c in ("ID", "Subject", "Subject ID", "subject"):
        if c in df.columns:
            return c
    sys.exit("ERROR: could not find an ID column in the demographics CSV.")


def path_or_blank(p):
    if not p:
        return ""
    if CHECK and not os.path.exists(p):
        return ""
    return p


def vol_file(sid, prefix):
    """Resolve a BOLD variant by globbing (handles the optional res-* entity)."""
    d = os.path.join(RAW_ROOT, "Data_BIDS", "FunVoluW", sid)
    hits = glob.glob(os.path.join(d, f"{prefix}{sid}{VOL_GLOB}"))
    # startswith guard so prefix "" does not match the Np/SmNp files
    hits = [h for h in hits if os.path.basename(h).startswith(f"{prefix}{sid}_")]
    return sorted(hits)[0] if hits else ""


# ── Main ────────────────────────────────────────────────────────────────────
def build():
    # ── QC tiers: ordered lists -> 0-based position maps ───────────────────────
    clean_ids = load_subjlist(CLEAN_LIST)
    super_ids = load_subjlist(SUPER_LIST)
    clean_pos = {sid: i for i, sid in enumerate(clean_ids)}    # -> neuromark53_clean
    super_pos = {sid: i for i, sid in enumerate(super_ids)}    # -> neuromark105_super_clean
    clean_set, super_set = set(clean_ids), set(super_ids)

    # ── Choose subject set (rows) ──────────────────────────────────────────────
    if SUBSET == "super_clean":
        subject_ids = super_ids
    elif SUBSET == "clean":
        subject_ids = clean_ids
    else:
        subject_ids = None  # full -> from demographics below
    if SUBSET != "full" and not subject_ids:
        sys.exit(f"ERROR: subject list for '{SUBSET}' is empty/missing.")

    # ── Demographics + acquisition ─────────────────────────────────────────────
    demo = pd.read_csv(DEMO_CSV)
    demo = demo.rename(columns={detect_id_col(demo): "id"})
    demo = demo.loc[:, ~demo.columns.str.startswith("Unnamed")]
    demo["id"] = demo["id"].astype(str).str.strip()

    tr = pd.read_csv(TRINFO, sep="\t").rename(columns={
        "Subject ID": "id", "Slice Number": "n_slices",
        "Time Points": "n_timepoints", "Voxel Size": "voxel_size",
    })
    tr["id"] = tr["id"].astype(str).str.strip()

    if subject_ids is None:
        subject_ids = sorted(set(demo["id"]) | set(tr["id"]))

    df = pd.DataFrame({"id": subject_ids})
    df = df.merge(demo, on="id", how="left").merge(tr, on="id", how="left")

    # ── Derived columns ────────────────────────────────────────────────────────
    dec = df["id"].map(decode_id)
    df["site"] = dec.map(lambda t: t[0])
    df["group_code"] = dec.map(lambda t: t[1])
    df["group_label"] = dec.map(lambda t: t[2])
    df["in_clean"] = df["id"].isin(clean_set)
    df["in_super_clean"] = df["id"].isin(super_set)
    df["clean_row"] = df["id"].map(clean_pos)          # 0-based, indexes 53 postproc
    df["super_clean_row"] = df["id"].map(super_pos)    # 0-based, indexes 105 postproc
    df["matlab_index"] = (pd.to_numeric(df["MATLABIndex"], errors="coerce")
                          if "MATLABIndex" in df.columns else pd.NA)
    df = df.drop(columns=[c for c in ["MATLABIndex"] if c in df.columns])

    # ── File paths ─────────────────────────────────────────────────────────────
    for col, prefix in VOL_PREFIXES.items():
        df[col] = [vol_file(sid, prefix) for sid in df["id"]]

    # ICA folders: index by 1-based position in the folder's own subject list
    for label, d, pos in (("ica53", ICA53_DIR, clean_pos),
                          ("ica105", ICA105_DIR, super_pos)):
        tc, sm = [], []
        for sid in df["id"]:
            p = pos.get(sid)
            if p is None:
                tc.append(""); sm.append(""); continue
            k = p + 1
            tc.append(path_or_blank(os.path.join(d, TC_PATTERN % k)))
            sm.append(path_or_blank(os.path.join(d, SM_PATTERN % k)))
        df[f"{label}_tc"] = tc
        df[f"{label}_sm"] = sm

    # sFNC + spectra live inside these .mat files; index rows with clean_row (53)
    # / super_clean_row (105): arrays fnc_corrs_all (S,N,N), spectra_tc_all (S,F,N)
    df["sfnc_spectra_mat_53"] = os.path.join(ICA53_DIR, POSTPROC_NAME)
    df["sfnc_spectra_mat_105"] = os.path.join(ICA105_DIR, POSTPROC_NAME)

    # ── Column order ────────────────────────────────────────────────────────────
    front = [
        "id", "site", "group_code", "group_label", "matlab_index",
        "Diagnosis", "Sex", "Age", "Education", "HAMDTotal17", "HAMATotal", "HAMD3",
        "TR", "n_slices", "n_timepoints", "voxel_size",
        "in_clean", "in_super_clean", "clean_row", "super_clean_row",
        "vol_np_3mm", "vol_smnp_3mm",
        "ica53_tc", "ica53_sm", "ica105_tc", "ica105_sm",
        "sfnc_spectra_mat_53", "sfnc_spectra_mat_105",
    ]
    front = [c for c in front if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df_full = df[front + rest]       # curated columns + all FULL_sites clinical columns
    df_lean = df[front]              # curated columns only

    # ── Write both versions + summary ──────────────────────────────────────────
    base, ext = os.path.splitext(OUT)
    full_out = f"{base}_full{ext}"
    df_lean.to_csv(OUT, index=False)
    df_full.to_csv(full_out, index=False)
    print(f"Wrote {len(df_lean)} rows -> {OUT} ({df_lean.shape[1]} cols, curated)")
    print(f"Wrote {len(df_full)} rows -> {full_out} ({df_full.shape[1]} cols, + FULL_sites)")
    print(f"  subset={SUBSET}  check_exists={CHECK}")
    print("  group_label counts:", df["group_label"].value_counts(dropna=False).to_dict())
    missing_from_clean = df.loc[df["in_super_clean"] & ~df["in_clean"], "id"].tolist()
    if missing_from_clean:
        print(f"  WARNING: {len(missing_from_clean)} super_clean IDs not in clean list")
    if CHECK:
        for c in ("vol_smnp_3mm", "ica53_tc", "ica105_tc"):
            print(f"  {c:14s} resolved: {(df[c] != '').sum()}/{len(df)}")


if __name__ == "__main__":
    build()
