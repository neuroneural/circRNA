#!/usr/bin/env python3
"""
ica53_tc_to_npz.py
==================

Read the ICA-53 timecourse NIfTIs referenced by the MDD master CSV
(`ica53_tc` column) and pack them, together with selected metadata columns,
into a single compressed .npz. Arrays inside the .npz are named by their CSV
column name (`ica53_tc`, `id`, `Diagnosis`, `Sex`, `Age`, `Education`,
`HAMDTotal17`, `HAMATotal`, `HAMD3`).

Edit the CONFIG block and run:  python ica53_tc_to_npz.py

Timecourses can differ in length; they are cropped to the minimum length across
the loaded set so the result is a uniform `(N, T_min, 53)` array.

Needs numpy, pandas, nibabel.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib

# ── CONFIG ────────────────────────────────────────────────────────────────────
IN_CSV = "mdd_master.csv"          # master CSV produced by build_mdd_master_csv.py
OUT_NPZ = "mdd_ica53_tc_superclean.npz"
TC_COL = "ica53_tc"                # timecourse path column to load
FILTER_COL = "in_super_clean"      # keep only rows where this column is truthy; None = no filter
META_COLS = ["id", "Diagnosis", "Sex", "Age", "Education",
             "HAMDTotal17", "HAMATotal", "HAMD3"]
LIMIT = None                       # int for a quick test (e.g. 20); None = all
# ──────────────────────────────────────────────────────────────────────────────


def load_tc(path):
    """Load one timecourse NIfTI -> float32 (T, 53)."""
    return np.squeeze(np.asarray(nib.load(path).get_fdata(), dtype=np.float32))


def main():
    df = pd.read_csv(IN_CSV)
    if TC_COL not in df.columns:
        raise SystemExit(f"'{TC_COL}' not in {IN_CSV}")

    # restrict to a QC tier (e.g. super_clean) via a boolean flag column
    if FILTER_COL:
        if FILTER_COL not in df.columns:
            raise SystemExit(f"'{FILTER_COL}' not in {IN_CSV}")
        truthy = df[FILTER_COL].map(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes"))
        df = df[truthy].copy()
        print(f"Filtered to {FILTER_COL}: {len(df)} rows")

    # keep only rows that actually reference a timecourse file
    df = df[df[TC_COL].notna() & (df[TC_COL].astype(str).str.strip() != "")].copy()
    if LIMIT:
        df = df.head(LIMIT)
    df = df.reset_index(drop=True)

    tcs, keep = [], []
    for i, p in enumerate(df[TC_COL]):
        try:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            tcs.append(load_tc(p))
            keep.append(i)
        except Exception as e:
            print(f"  skip [{df.iloc[i]['id']}]: {e}")
    df = df.iloc[keep].reset_index(drop=True)
    if not tcs:
        raise SystemExit("No timecourses could be loaded.")

    # crop every timecourse to the minimum length so they stack uniformly
    lengths = [t.shape[0] for t in tcs]
    t_min, t_max = min(lengths), max(lengths)
    n_crop = sum(1 for L in lengths if L > t_min)
    print(f"Timecourse lengths: min={t_min} max={t_max} (N={len(tcs)})")
    if t_max > t_min:
        print(f"  cropping {n_crop} subject(s) to T={t_min} "
              f"(trimming up to {t_max - t_min} timepoints from the end)")
    else:
        print("  all timecourses equal length; no cropping needed")
    tc_arr = np.stack([t[:t_min] for t in tcs]).astype(np.float32)   # (N, t_min, 53)

    # metadata arrays, keyed by CSV column name
    out = {TC_COL: tc_arr}
    for c in META_COLS:
        if c not in df.columns:
            print(f"  WARN: column '{c}' not in CSV — skipped")
            continue
        if c == "id":
            out[c] = df[c].astype(str).to_numpy()
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)

    np.savez_compressed(OUT_NPZ, **out)

    # summary
    print(f"Saved {OUT_NPZ}  (N = {len(df)} subjects)")
    print(f"  {TC_COL}: stacked array {tc_arr.shape} {tc_arr.dtype}")
    for c in META_COLS:
        if c in out:
            print(f"  {c:12s}: {out[c].shape} {out[c].dtype}")


if __name__ == "__main__":
    main()
