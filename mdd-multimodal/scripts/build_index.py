"""Build data/mdd_master.csv -- one row per subject, the spine everything joins on.

This is the only place that knows about the server's directory layout. Run it
once; after that every other script reads the CSV.

Columns written
---------------
  id                 subject ID as it appears in the filenames
  file_index         0-based position in the sorted file listing (3,525 rows)
  group_label        MDD | HC | Bipolar | Schizophrenia
  group_code         1 | 2 | 3 | 4  (empirically confirmed: 1660/1344/207/314)
  site               site / scanner identifier
  age, sex           demographics, where the phenotype file has them
  HAMDTotal17, HAMD3 symptom scores; present for ~660 MDD subjects only
  in_clean           subject survived the ICA-53 QC tier (n = 2,526)
  clean_row          0-based row within that tier -- THE JOIN KEY for the GIFT MATs
  in_super_clean     subject survived the ICA-105 QC tier (n = 2,426)
  super_clean_row    0-based row within that tier

Why clean_row matters
---------------------
The GIFT postprocess MATs carry no subject IDs. Their `subjects` variable is a
1..N counter with no identity in it. Rows are in QC-tier order, so the join is
positional on clean_row / super_clean_row -- not on any index that counts all
3,525 files. Getting this wrong silently shuffles every label. prep/extract_sfnc.py
does the join here and writes real IDs out, so nothing downstream has to know.

    python prep/build_index.py --root /data/users2/ppopov1/datasets/MDD_DIRECT \
        --out data/mdd_master.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

GROUP_LABELS = {1: "MDD", 2: "HC", 3: "Bipolar", 4: "Schizophrenia"}

# DIRECT / REST-meta-MDD subject IDs carry site and group in the ID itself:
#
#     IS023-2-0031
#       |    |   `-- subject number within that site and group
#       |    `------ group: 1 MDD, 2 HC, 3 Bipolar, 4 Schizophrenia
#       `----------- site:  IS001 .. IS023
#
# Verified against the full 3,525-subject extraction manifest: the middle field
# splits 1660 / 1344 / 207 / 314, which matches the group counts confirmed from
# the data, and there are exactly 23 distinct site prefixes. So site and group
# do not need a phenotype file -- only HAMD, age and sex do.
ID_RE = re.compile(r"^(?P<site>[A-Za-z]*\d+)-(?P<group>\d+)-(?P<num>\d+)$")


def parse_id(sid):
    """-> (site, group_code) or (None, None) if the ID is not in DIRECT form."""
    m = ID_RE.match(str(sid).strip())
    if not m:
        return None, None
    return m.group("site"), int(m.group("group"))


def find_subject_files(funvol_dir, pattern=r"([A-Za-z]*\d+-\d+-\d+)"):
    if not os.path.isdir(funvol_dir):
        raise SystemExit(f"not a directory: {funvol_dir}")
    files = sorted(os.listdir(funvol_dir))
    rows = []
    for i, f in enumerate(files):
        m = re.search(pattern, f)
        if not m:
            continue
        rows.append(dict(id=m.group(1), file_index=len(rows), filename=f))
    if not rows:
        raise SystemExit(f"no subject-looking filenames in {funvol_dir}")
    return pd.DataFrame(rows)


def attach_phenotype(idx, pheno_path, id_col=None):
    if not pheno_path or not os.path.exists(pheno_path):
        print(f"  (no phenotype file at {pheno_path} — writing IDs only)")
        for c in ("group_code", "site", "age", "sex", "HAMDTotal17", "HAMD3"):
            idx[c] = np.nan
        return idx
    ph = pd.read_csv(pheno_path) if pheno_path.endswith(".csv") else pd.read_excel(pheno_path)
    if id_col is None:
        cands = [c for c in ph.columns if c.lower() in ("id", "subid", "subject",
                                                        "subject_id", "participant_id")]
        if not cands:
            raise SystemExit(f"cannot find an ID column in {pheno_path}; pass --pheno-id-col")
        id_col = cands[0]
    ph[id_col] = ph[id_col].astype(str)
    idx["id"] = idx["id"].astype(str)
    merged = idx.merge(ph, how="left", left_on="id", right_on=id_col,
                       suffixes=("", "_ph"))
    missed = merged["group_code"].isna().sum() if "group_code" in merged else len(merged)
    print(f"  phenotype join: {len(merged) - missed}/{len(merged)} matched")
    return merged


def attach_tier(df, tier_file, flag_col, row_col):
    """tier_file lists the subject IDs that survived a QC tier, in MAT row order."""
    df[flag_col] = False
    df[row_col] = np.nan
    if not tier_file or not os.path.exists(tier_file):
        print(f"  (no tier list at {tier_file} — {flag_col} left False)")
        return df
    with open(tier_file) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    pos = {s: i for i, s in enumerate(ids)}
    hit = df["id"].astype(str).map(pos)
    df[flag_col] = hit.notna()
    df[row_col] = hit
    print(f"  {flag_col}: {int(df[flag_col].sum())} subjects (list had {len(ids)})")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/users2/ppopov1/datasets/MDD_DIRECT")
    ap.add_argument("--funvol", default=None,
                    help="dir whose filenames define subject order (default ROOT/Data_BIDS/FunVoluW)")
    ap.add_argument("--pheno", default=None,
                    help="phenotype CSV/XLSX for HAMD/age/sex. Site and group come "
                         "from the subject ID and do not need this file.")
    ap.add_argument("--pheno-id-col", default=None)
    ap.add_argument("--clean-list", default=None, help="IDs surviving ICA-53 QC, in MAT row order")
    ap.add_argument("--super-clean-list", default=None, help="IDs surviving ICA-105 QC")
    ap.add_argument("--out", default="data/mdd_master.csv")
    args = ap.parse_args()

    funvol = args.funvol or os.path.join(args.root, "Data_BIDS", "FunVoluW")
    print(f"scanning {funvol}")
    df = find_subject_files(funvol)
    print(f"  {len(df)} subjects")

    # ---- site and group straight from the subject ID -----------------------
    parsed = df["id"].map(parse_id)
    df["site_from_id"] = [p[0] for p in parsed]
    df["group_from_id"] = [p[1] for p in parsed]
    n_parsed = int(df["site_from_id"].notna().sum())
    print(f"  parsed site+group from ID for {n_parsed}/{len(df)} subjects")
    if n_parsed:
        print("    groups: " + ", ".join(
            f"{GROUP_LABELS.get(k, k)} {int(v)}"
            for k, v in df["group_from_id"].value_counts().sort_index().items()))
        print(f"    sites : {df['site_from_id'].nunique()}")

    df = attach_phenotype(df, args.pheno, args.pheno_id_col)

    # The ID is the authority for site and group; the phenotype file fills the
    # rest. If both exist and disagree, that is a real problem -- say so.
    if "group_code" in df and df["group_code"].notna().any():
        both = df["group_from_id"].notna() & df["group_code"].notna()
        clash = int((df.loc[both, "group_from_id"]
                     != pd.to_numeric(df.loc[both, "group_code"], errors="coerce")).sum())
        if clash:
            print(f"  !! {clash} subjects where the ID's group disagrees with the "
                  "phenotype file. Resolve this before using either.")
    df["group_code"] = df["group_from_id"].where(
        df["group_from_id"].notna(),
        pd.to_numeric(df.get("group_code", np.nan), errors="coerce"))
    df["group_label"] = pd.to_numeric(df["group_code"], errors="coerce").map(GROUP_LABELS)

    site_ph = df["site"] if "site" in df else pd.Series(np.nan, index=df.index)
    df["site"] = df["site_from_id"].where(df["site_from_id"].notna(), site_ph)
    df["site"] = df["site"].fillna("unknown").astype(str)

    df = attach_tier(df, args.clean_list, "in_clean", "clean_row")
    df = attach_tier(df, args.super_clean_list, "in_super_clean", "super_clean_row")

    cols = ["id", "file_index", "filename", "group_label", "group_code", "site",
            "age", "sex", "HAMDTotal17", "HAMD3",
            "in_clean", "clean_row", "in_super_clean", "super_clean_row"]
    out = df[[c for c in cols if c in df.columns]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nwrote {args.out}  ({len(out)} rows)")
    if out["group_label"].notna().any():
        print(out["group_label"].value_counts().to_string())
    n_h = out[["HAMDTotal17", "HAMD3"]].notna().all(axis=1).sum() if "HAMD3" in out else 0
    print(f"  with HAMD17+HAMD3: {n_h}")
    # sanity: the tiers must be nested, or the MAT joins are wrong
    if out["in_super_clean"].any() and out["in_clean"].any():
        bad = int((out["in_super_clean"] & ~out["in_clean"]).sum())
        print(f"  tier nesting check: {bad} subjects in super_clean but not clean"
              + ("  <-- PROBLEM" if bad else "  (ok)"))


if __name__ == "__main__":
    main()
