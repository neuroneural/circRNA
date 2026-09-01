"""
============================================================================
feature_eval_method_compare.py
============================================================================

Compares 8 analysis conditions across all clinical targets.

CONDITIONS
----------
  LOPO_Unstable     : single-person holdout, per-fold feature selection
  LOPO_Stable       : single-person holdout, stable features (>= 60% folds)
  TwoPerson_Unstable: two-person holdout,    per-fold feature selection
  TwoPerson_Stable  : two-person holdout,    stable features (>= 60% folds)
  PersonVec_Unstable: person-level mean,     per-fold feature selection
  PersonVec_Stable  : person-level mean,     stable features
  Annot_Unstable    : annotated circRNAs only, per-fold feature selection
  Annot_Stable      : annotated circRNAs only, stable features

METRIC (reported for ALL conditions)
--------------------------------------
  Inner CV AUC per outer fold — the reliable "left panel" metric.
  OOF / pooled AUC is NOT computed (inflates stable conditions due to
  cross-fold leakage: stable features are selected using all persons'
  data, so the held-out person's data indirectly influenced feature choice).

For STABLE conditions (two-stage):
  Stage 1  Outer CV with per-fold inner feature selection.
           Features appearing in >= STABILITY_THRESHOLD of folds are "stable".
  Stage 2  For each outer fold: inner CV AUC with the FIXED stable feature
           set (no per-fold feature search). These AUCs are plotted.

For UNSTABLE conditions:
  Each outer fold: inner CV selects best (method, top_n) → record best AUC.
  Top features by selection frequency are reported.

OUTPUT
------
  method_compare_<ts>.png          boxplot grid, 1 subplot per target
  method_compare_features_<ts>.txt full feature lists (stable + unstable)
  method_compare_summary_<ts>.csv  mean inner AUC per condition x target

DATA SOURCES (DO NOT SWAP)
---------------------------
  circRNA counts : /data/users2/ppopov1/datasets/circRNA  (Popov lab)
  Clinical labels: /home/users/elatash1/Work/circRNA_Analysis/clean_clinical_metadata.csv
============================================================================
"""

import os
import math
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedGroupKFold, LeaveOneGroupOut, LeavePGroupsOut
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ============================================================
# SETTINGS  — edit only this block
# ============================================================

RANDOM_STATE        = 42
FEATURE_COUNTS      = [10, 20, 50, 100]
METHODS             = ["mi", "rf"]
MAX_INNER_FOLDS     = 3
STABILITY_THRESHOLD = 0.60   # feature must appear in >= 60 % of outer folds

TARGETS = ["GAD-7", "Sex", "Age", "IDS-C", "SHAPSC", "TEPS", "YMRS"]

TARGET_THRESHOLDS = {
    "GAD-7":  10,        # >= 10 = clinically significant anxiety
    "IDS-C":  24,        # >= 24 = moderate-to-severe depression
    "YMRS":   13,        # >= 13 = active manic symptoms
    "SHAPSC": "median",
    "TEPS":   "median",
    "Age":    "median",
}

BINARY_MAP = {
    "Sex":      {"Male": 1, "Female": 0},
    "Response": {"Yes":  1, "No":     0},
}

USE_DEMOGRAPHIC_FEATURES = True
DEMOGRAPHIC_FEATURES     = ["Sex", "Age"]

# circRNA counts: Popov lab folder  (DO NOT change to metadata path)
BASE_PATH     = "/data/users2/ppopov1/datasets/circRNA"

# Clinical labels: Lisa's working folder (DO NOT change to circRNA path)
METADATA_PATH = "/home/users/elatash1/Work/circRNA_Analysis/clean_clinical_metadata.csv"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CONDITIONS  (holdout x stability x feature_space)
# ============================================================

CONDITIONS = {
    "LOPO_Unstable":      {"holdout": "single", "stable": False, "person_vec": False, "annot_only": False},
    "LOPO_Stable":        {"holdout": "single", "stable": True,  "person_vec": False, "annot_only": False},
    "TwoPerson_Unstable": {"holdout": "two",    "stable": False, "person_vec": False, "annot_only": False},
    "TwoPerson_Stable":   {"holdout": "two",    "stable": True,  "person_vec": False, "annot_only": False},
    "PersonVec_Unstable": {"holdout": "single", "stable": False, "person_vec": True,  "annot_only": False},
    "PersonVec_Stable":   {"holdout": "single", "stable": True,  "person_vec": True,  "annot_only": False},
    "Annot_Unstable":     {"holdout": "single", "stable": False, "person_vec": False, "annot_only": True},
    "Annot_Stable":       {"holdout": "single", "stable": True,  "person_vec": False, "annot_only": True},
}

# Paired colors: light = unstable, dark = stable
COND_COLORS = {
    "LOPO_Unstable":      "#6BAED6",
    "LOPO_Stable":        "#1A5276",
    "TwoPerson_Unstable": "#FDB863",
    "TwoPerson_Stable":   "#A04000",
    "PersonVec_Unstable": "#74C476",
    "PersonVec_Stable":   "#1A5C1A",
    "Annot_Unstable":     "#C39BD3",
    "Annot_Stable":       "#6C3483",
}


# ============================================================
# CLASSIFIERS
# ============================================================

CLASSIFIER_TEMPLATES = {
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=2000,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=5000,
        solver="saga",
        random_state=RANDOM_STATE,
    ),
}

CLF_NAMES = list(CLASSIFIER_TEMPLATES.keys())


# ============================================================
# DATA LOADING
# ============================================================

_CIRC_CACHE = {}

def _load_circrna_matrix():
    if "circ_ML_df" in _CIRC_CACHE:
        return _CIRC_CACHE["circ_ML_df"]

    raw_path  = os.path.join(BASE_PATH, "01_raw_data")
    qced_path = os.path.join(BASE_PATH, "03_QCed_data")

    pd.read_excel(os.path.join(raw_path,
                               "ERVIN_All-Samples-Manifest_20250401.xlsx"))
    pd.read_csv(os.path.join(qced_path,
                             "202504_ERVIN_filtered_linear_gene_counts.csv"))
    circ_counts_df = pd.read_csv(os.path.join(
        qced_path,
        "circ_counts_post_filtering_circ_linear_and_1cnt_in.25samples_ERVIN_202504.csv"))

    meta_cols    = ["Chr", "Start", "End", "Gene", "JunctionType",
                    "Strand", "Start-End Region", "chr_start_end_strand"]
    subject_cols        = [c for c in circ_counts_df.columns if c not in meta_cols]
    circ_ML_df          = circ_counts_df[subject_cols].T
    circ_ML_df.columns  = circ_counts_df["chr_start_end_strand"]
    circ_ML_df.index.name = "Sample_ID"

    _CIRC_CACHE["circ_ML_df"] = circ_ML_df
    _CIRC_CACHE["gene_map"] = (
        circ_counts_df.set_index("chr_start_end_strand")["Gene"].to_dict()
        if "Gene" in circ_counts_df.columns else {}
    )
    print("circRNA matrix loaded and cached.")
    return circ_ML_df


def _binarize(meta_df, target):
    if target in BINARY_MAP:
        return meta_df[target].map(BINARY_MAP[target])
    vals   = pd.to_numeric(meta_df[target], errors="coerce")
    thresh = TARGET_THRESHOLDS.get(target, "median")
    cutoff = vals.median() if thresh == "median" else thresh
    label  = "median" if thresh == "median" else f">= {cutoff}"
    print(f"  Binarise {target}: {label} -> 0/1")
    y = (vals >= cutoff).astype(float)
    y[vals.isna()] = np.nan
    return y


def load_data(target):
    circ_ML_df = _load_circrna_matrix()
    meta_df    = pd.read_csv(METADATA_PATH).set_index("Sample_ID")

    common     = [s for s in circ_ML_df.index if s in meta_df.index]
    circ_ML_df = circ_ML_df.loc[common]
    meta_df    = meta_df.loc[common]

    y_series = _binarize(meta_df, target)
    keep     = y_series.notna()
    circ_ML_df, meta_df, y_series = (
        circ_ML_df[keep], meta_df[keep], y_series[keep].astype(int)
    )

    X      = circ_ML_df.copy()
    groups = meta_df["URSI"]

    if USE_DEMOGRAPHIC_FEATURES:
        for col in [c for c in DEMOGRAPHIC_FEATURES if c != target]:
            if col not in meta_df.columns:
                continue
            X[col] = (meta_df[col].map(BINARY_MAP[col]).values
                      if col in BINARY_MAP
                      else pd.to_numeric(meta_df[col], errors="coerce").values)

    print(f"  X: {X.shape}  |  y: {y_series.value_counts().to_dict()}"
          f"  |  {groups.nunique()} people")
    return X, y_series, groups


# ============================================================
# PREPROCESSING TRANSFORMS
# ============================================================

def aggregate_persons(X, y, groups):
    """
    Collapse multiple visits per person into one row by taking the mean.
    Labels use majority vote (round of mean) since y is binary 0/1.

    Returns X_agg (1 row per person), y_agg, groups_agg.
    """
    df        = X.copy()
    df["_y_"] = y.values
    df["_g_"] = groups.values

    agg_df    = df.groupby("_g_").mean().reset_index()
    g_agg     = agg_df["_g_"].values
    y_agg     = agg_df["_y_"].round().astype(int).values
    X_agg     = agg_df.drop(columns=["_y_", "_g_"]).values
    feat_cols  = [c for c in df.columns if c not in ("_y_", "_g_")]

    X_agg_df          = pd.DataFrame(X_agg, columns=feat_cols)
    y_agg_s           = pd.Series(y_agg, name=y.name)
    groups_agg_s      = pd.Series(g_agg, name=groups.name)

    return X_agg_df, y_agg_s, groups_agg_s


def filter_annotated(X, gene_map):
    """
    Keep only circRNA features with a known gene annotation
    (not 'not_annotated', not blank, not NaN).
    Demographic columns (Sex, Age) are always kept.
    """
    demo_cols  = [c for c in ["Sex", "Age"] if c in X.columns]
    circ_cols  = [c for c in X.columns if c not in demo_cols]
    keep_circ  = []
    for c in circ_cols:
        gene = gene_map.get(c, "")
        if (isinstance(gene, str)
                and gene.strip()
                and gene.strip().lower() != "not_annotated"):
            keep_circ.append(c)
    kept = keep_circ + demo_cols
    print(f"  Annotated filter: {len(circ_cols)} circRNAs -> "
          f"{len(keep_circ)} kept (+ {len(demo_cols)} demo cols)")
    return X[kept]


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, y_prob):
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def _groups_per_class(y, groups):
    df = pd.DataFrame({"y": list(y), "g": list(groups)})
    return int(df.groupby("y")["g"].nunique().min())


def make_inner_cv(y, groups):
    gpc = _groups_per_class(y, groups)
    k   = min(MAX_INNER_FOLDS, gpc)
    if k < 2 or len(set(groups)) < 3:
        return LeaveOneGroupOut()
    return StratifiedGroupKFold(n_splits=k, shuffle=True,
                                random_state=RANDOM_STATE)


def make_outer_cv(holdout):
    if holdout == "two":
        return LeavePGroupsOut(n_groups=2)
    return LeaveOneGroupOut()


def n_outer_folds(groups, holdout):
    n = groups.nunique()
    if holdout == "two":
        return math.comb(int(n), 2)
    return int(n)


# ============================================================
# FEATURE SELECTION
# ============================================================

def mi_features(X, y, top_n):
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return (pd.Series(mi, index=X.columns)
              .sort_values(ascending=False)
              .head(top_n).index.tolist())


def rf_features(X, y, top_n):
    rf = RandomForestClassifier(n_estimators=200,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    return (pd.Series(rf.feature_importances_, index=X.columns)
              .sort_values(ascending=False)
              .head(top_n).index.tolist())


FEATURE_FUNCS = {"mi": mi_features, "rf": rf_features}


def inner_score(X_tr, y_tr, g_tr, top_n, method):
    """
    Inner (nested) CV AUC for a given (method, top_n) combination.
    Feature selection runs inside each inner fold on inner training data only.
    Returns 0.5 (chance) if no valid inner folds exist.
    """
    inner_cv       = make_inner_cv(y_tr, g_tr)
    proxy          = LogisticRegression(max_iter=5000, solver="saga",
                                        random_state=RANDOM_STATE)
    oof_true, oof_prob = [], []

    for tr_idx, val_idx in inner_cv.split(X_tr, y_tr, g_tr):
        X_in_tr,  y_in_tr  = X_tr.iloc[tr_idx],  y_tr.iloc[tr_idx]
        X_in_val, y_in_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
        if len(set(y_in_tr)) < 2 or len(set(y_in_val)) < 2:
            continue
        feats = FEATURE_FUNCS[method](X_in_tr, y_in_tr, top_n)
        sc    = StandardScaler()
        clf   = clone(proxy)
        clf.fit(sc.fit_transform(X_in_tr[feats]), y_in_tr)
        oof_true.extend(y_in_val.tolist())
        oof_prob.extend(clf.predict_proba(sc.transform(X_in_val[feats]))[:, 1])

    auc = safe_auc(oof_true, oof_prob)
    return 0.5 if np.isnan(auc) else auc


def fixed_inner_score(X_tr, y_tr, g_tr, features):
    """
    Inner CV AUC using a FIXED feature set — no feature selection inside.
    Used in Stage 2 of stable conditions.
    Returns NaN if feature list is empty or no valid inner folds.
    """
    if not features:
        return float("nan")
    available = [f for f in features if f in X_tr.columns]
    if not available:
        return float("nan")

    inner_cv       = make_inner_cv(y_tr, g_tr)
    proxy          = LogisticRegression(max_iter=5000, solver="saga",
                                        random_state=RANDOM_STATE)
    oof_true, oof_prob = [], []

    for tr_idx, val_idx in inner_cv.split(X_tr, y_tr, g_tr):
        X_in_tr,  y_in_tr  = X_tr.iloc[tr_idx],  y_tr.iloc[tr_idx]
        X_in_val, y_in_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
        if len(set(y_in_tr)) < 2 or len(set(y_in_val)) < 2:
            continue
        sc  = StandardScaler()
        clf = clone(proxy)
        clf.fit(sc.fit_transform(X_in_tr[available]), y_in_tr)
        oof_true.extend(y_in_val.tolist())
        oof_prob.extend(clf.predict_proba(sc.transform(X_in_val[available]))[:, 1])

    auc = safe_auc(oof_true, oof_prob)
    return 0.5 if np.isnan(auc) else auc


# ============================================================
# CONDITION RUNNERS
# ============================================================

def run_unstable(X, y, groups, holdout, cond_name):
    """
    Single-stage: per-fold feature selection via inner CV.
    Returns: inner_aucs (list[float]), feat_counter (Counter).
    """
    outer_cv     = make_outer_cv(holdout)
    inner_aucs   = []
    feat_counter = Counter()
    total_folds  = n_outer_folds(groups, holdout)

    print(f"    [{cond_name}] {total_folds} outer folds")

    for fold_i, (train_idx, _) in enumerate(
            outer_cv.split(X, y, groups), 1):

        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        g_tr = groups.iloc[train_idx]

        if len(set(y_tr)) < 2:
            if fold_i <= 5 or fold_i == total_folds:
                print(f"      fold {fold_i}/{total_folds}: skip (single class in train)")
            inner_aucs.append(float("nan"))
            continue

        scores = {(m, n): inner_score(X_tr, y_tr, g_tr, n, m)
                  for m in METHODS for n in FEATURE_COUNTS}
        best_m, best_n = max(scores, key=scores.get)
        best_auc       = scores[(best_m, best_n)]
        inner_aucs.append(best_auc)

        features = FEATURE_FUNCS[best_m](X_tr, y_tr, best_n)
        feat_counter.update(features)

        if fold_i <= 3 or fold_i % 10 == 0 or fold_i == total_folds:
            print(f"      fold {fold_i}/{total_folds}: inner={best_auc:.3f}"
                  f"  [{best_m}@{best_n}]")

    valid = [v for v in inner_aucs if not np.isnan(v)]
    mu    = np.mean(valid) if valid else float("nan")
    sd    = np.std(valid)  if valid else float("nan")
    print(f"    [{cond_name}] mean inner AUC = {mu:.3f} +/- {sd:.3f}"
          f"  ({len(valid)}/{len(inner_aucs)} valid folds)")
    return inner_aucs, feat_counter


def run_stable(X, y, groups, holdout, cond_name):
    """
    Two-stage stable evaluation.

    Stage 1: outer CV with per-fold inner feature selection; discover which
             features appear in >= STABILITY_THRESHOLD fraction of folds.
    Stage 2: for each outer fold, compute inner CV AUC using the FIXED
             stable feature set (plotted metric).

    Returns: stage2_aucs (list[float]), stable_feats (list[str]),
             stage1_counter (Counter).
    """
    outer_cv      = make_outer_cv(holdout)
    folds         = list(outer_cv.split(X, y, groups))
    total_folds   = len(folds)
    min_folds_req = math.ceil(STABILITY_THRESHOLD * total_folds)

    print(f"    [{cond_name}] Stage 1: {total_folds} outer folds, "
          f"stability threshold = {min_folds_req}/{total_folds} "
          f"({STABILITY_THRESHOLD:.0%})")

    # ── Stage 1: discover stable features ───────────────────────────────
    stage1_counter = Counter()
    for fold_i, (train_idx, _) in enumerate(folds, 1):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        g_tr = groups.iloc[train_idx]

        if len(set(y_tr)) < 2:
            continue

        scores = {(m, n): inner_score(X_tr, y_tr, g_tr, n, m)
                  for m in METHODS for n in FEATURE_COUNTS}
        best_m, best_n = max(scores, key=scores.get)
        features       = FEATURE_FUNCS[best_m](X_tr, y_tr, best_n)
        stage1_counter.update(features)

        if fold_i <= 3 or fold_i % 10 == 0 or fold_i == total_folds:
            print(f"      S1 fold {fold_i}/{total_folds}: [{best_m}@{best_n}]")

    stable_feats = [f for f, cnt in stage1_counter.items()
                    if cnt >= min_folds_req]
    print(f"    [{cond_name}] Stage 1 -> {len(stable_feats)} stable features")

    if not stable_feats:
        print(f"    [{cond_name}] No stable features — Stage 2 skipped (NaN)")
        return [float("nan")] * total_folds, [], stage1_counter

    # ── Stage 2: inner CV with fixed stable features ─────────────────────
    print(f"    [{cond_name}] Stage 2: inner CV with {len(stable_feats)} fixed features")
    stage2_aucs = []
    for fold_i, (train_idx, _) in enumerate(folds, 1):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        g_tr = groups.iloc[train_idx]

        if len(set(y_tr)) < 2:
            stage2_aucs.append(float("nan"))
            continue

        auc = fixed_inner_score(X_tr, y_tr, g_tr, stable_feats)
        stage2_aucs.append(auc)

        if fold_i <= 3 or fold_i % 10 == 0 or fold_i == total_folds:
            print(f"      S2 fold {fold_i}/{total_folds}: inner={auc:.3f}")

    valid = [v for v in stage2_aucs if not np.isnan(v)]
    mu    = np.mean(valid) if valid else float("nan")
    print(f"    [{cond_name}] Stage 2 mean inner AUC = {mu:.3f}"
          f"  ({len(valid)}/{total_folds} valid folds)")
    return stage2_aucs, stable_feats, stage1_counter


def run_condition(X, y, groups, cond_cfg, cond_name, gene_map):
    """
    Apply transforms (PersonVec, AnnotOnly) then run stable or unstable eval.
    Returns dict: inner_aucs, stable_feats, feat_counter, n_folds.
    """
    X_use = X.copy()
    y_use = y.copy()
    g_use = groups.copy()

    # PersonVec: collapse visits -> 1 row per person
    if cond_cfg["person_vec"]:
        X_use, y_use, g_use = aggregate_persons(X_use, y_use, g_use)
        print(f"    PersonVec: {X_use.shape[0]} persons x {X_use.shape[1]} features")

    # Annotated only: drop not_annotated circRNAs
    if cond_cfg["annot_only"]:
        X_use = filter_annotated(X_use, gene_map)

    if X_use.shape[1] == 0:
        print(f"    [{cond_name}] No features after filtering — skipped.")
        nf = n_outer_folds(g_use, cond_cfg["holdout"])
        return {
            "inner_aucs":   [float("nan")] * nf,
            "stable_feats": [],
            "feat_counter": Counter(),
            "n_folds":      nf,
        }

    nf = n_outer_folds(g_use, cond_cfg["holdout"])

    if cond_cfg["stable"]:
        aucs, stable_feats, s1_counter = run_stable(
            X_use, y_use, g_use, cond_cfg["holdout"], cond_name)
        return {
            "inner_aucs":   aucs,
            "stable_feats": stable_feats,
            "feat_counter": s1_counter,   # Stage 1 selection frequencies
            "n_folds":      nf,
        }
    else:
        aucs, feat_counter = run_unstable(
            X_use, y_use, g_use, cond_cfg["holdout"], cond_name)
        return {
            "inner_aucs":   aucs,
            "stable_feats": [],
            "feat_counter": feat_counter,
            "n_folds":      nf,
        }


# ============================================================
# FEATURE REPORTING
# ============================================================

def _fmt_feat_block(feat_counter, n_folds, gene_map, top_n, header):
    """Return formatted lines for a feature frequency list."""
    lines = [f"    {header}"]
    if not feat_counter:
        lines.append("      (none)")
        return lines
    for rank, (feat, cnt) in enumerate(feat_counter.most_common(top_n), 1):
        bar  = "X" * cnt + "." * (n_folds - cnt)
        gene = gene_map.get(feat, "N/A")
        freq = cnt / n_folds
        lines.append(
            f"      {rank:<3}  {cnt}/{n_folds}  [{bar}]  "
            f"freq={freq:.2f}  {feat:<40}  Gene: {gene}"
        )
    return lines


def print_and_save_features(target_results, target, gene_map, fh):
    """Print + write to file all feature info for one target."""
    hdr = f"\n{'='*76}\nFEATURES  |  Target: {target}\n{'='*76}"
    print(hdr); fh.write(hdr + "\n")

    for cond_name, res in target_results.items():
        nf           = res["n_folds"]
        stable_feats = res["stable_feats"]
        feat_ctr     = res["feat_counter"]
        is_stable    = "_Stable" in cond_name

        cond_hdr = f"\n  -- {cond_name} --  ({nf} outer folds)"
        print(cond_hdr); fh.write(cond_hdr + "\n")

        valid_aucs = [v for v in res["inner_aucs"] if not np.isnan(v)]
        mu   = np.mean(valid_aucs) if valid_aucs else float("nan")
        sd   = np.std(valid_aucs)  if valid_aucs else float("nan")
        auc_line = (f"  Inner CV AUC: {mu:.3f} +/- {sd:.3f}"
                    f"  ({len(valid_aucs)}/{nf} valid folds)")
        print(auc_line); fh.write(auc_line + "\n")

        # Stable features (Stage 2 fixed set)
        if is_stable:
            if stable_feats:
                sf_hdr = f"\n  STABLE features ({len(stable_feats)} total," \
                         f" chosen in >= {math.ceil(STABILITY_THRESHOLD*nf)}/{nf} folds):"
                print(sf_hdr); fh.write(sf_hdr + "\n")
                for rank, feat in enumerate(stable_feats[:30], 1):
                    gene = gene_map.get(feat, "N/A")
                    cnt  = feat_ctr.get(feat, 0)
                    bar  = "X" * cnt + "." * (nf - cnt)
                    line = (f"    {rank:<3}  {cnt}/{nf}  [{bar}]  "
                            f"{feat:<40}  Gene: {gene}")
                    print(line); fh.write(line + "\n")
            else:
                no_sf = "  STABLE features: (none — no circRNA reached threshold)"
                print(no_sf); fh.write(no_sf + "\n")

        # Top features by Stage 1 selection frequency (both stable and unstable)
        label  = "TOP 20 features by Stage 1 selection freq" if is_stable \
                 else "TOP 20 features by selection frequency"
        lines  = _fmt_feat_block(feat_ctr, nf, gene_map, 20, label)
        for line in lines:
            print(line); fh.write(line + "\n")


# ============================================================
# PLOTTING
# ============================================================

def plot_compare(all_results, targets, output_path):
    """
    One subplot per target. Each subplot has one boxplot per condition,
    showing the distribution of inner CV AUC values across outer folds.
    Paired colors: light = unstable, dark = stable.
    """
    cond_names = list(CONDITIONS.keys())
    n_targets  = len(targets)
    n_conds    = len(cond_names)

    fig_h = max(6 * n_targets, 16)
    fig, axes = plt.subplots(n_targets, 1,
                              figsize=(max(18, n_conds * 2.5), fig_h),
                              squeeze=False)
    fig.patch.set_facecolor("#f5f5f5")

    for row, target in enumerate(targets):
        ax  = axes[row][0]
        ax.set_facecolor("#f9f9f9")

        target_res  = all_results.get(target, {})
        positions   = list(range(1, n_conds + 1))
        tick_labels = []

        for pos, cond_name in zip(positions, cond_names):
            res        = target_res.get(cond_name, {})
            aucs       = res.get("inner_aucs", [])
            valid_aucs = [v for v in aucs if not np.isnan(v)]
            color      = COND_COLORS.get(cond_name, "#888888")

            if not valid_aucs:
                tick_labels.append(f"{cond_name}\n(no data)")
                # Draw empty placeholder
                ax.scatter([pos], [0.5], marker="x", color="#aaa",
                           s=60, zorder=3)
                continue

            bp = ax.boxplot(
                valid_aucs,
                positions=[pos],
                widths=0.65,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2.2),
                whiskerprops=dict(linewidth=1.3, color="#555"),
                capprops=dict(linewidth=1.3, color="#555"),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markersize=3.5, linestyle="none", alpha=0.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.70)

            rng    = np.random.default_rng(pos + row * 100)
            jitter = rng.uniform(-0.18, 0.18, len(valid_aucs))
            ax.scatter(np.full(len(valid_aucs), pos) + jitter, valid_aucs,
                       color=color, alpha=0.65, zorder=5,
                       edgecolors="white", s=35, linewidths=0.6)

            mu = np.mean(valid_aucs)
            sd = np.std(valid_aucs)
            ax.scatter([pos], [mu], marker="*", color="white",
                       s=110, zorder=6, edgecolors=color, linewidths=0.8)
            tick_labels.append(f"{cond_name}\n{mu:.2f}+-{sd:.2f}")

        ax.axhline(0.5, color="crimson", linestyle="--",
                   linewidth=1.3, zorder=3, alpha=0.7, label="Chance (0.50)")

        # Vertical separators between condition pairs
        for sep_x in [2.5, 4.5, 6.5]:
            ax.axvline(sep_x, color="#bbb", linewidth=0.9, linestyle=":")

        # Light background bands for paired conditions
        for pair_i, x_start in enumerate([0.5, 2.5, 4.5, 6.5]):
            ax.axvspan(x_start, x_start + 2,
                       color=["#EAF2FF", "#FFF5E6", "#E8F8E8", "#F5EEF8"][pair_i],
                       alpha=0.25, zorder=0)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Inner CV AUC", fontsize=9)
        ax.set_title(f"Target: {target}", fontsize=11, fontweight="bold", pad=6)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        if row == 0:
            ax.legend(fontsize=8, framealpha=0.8, loc="upper right")

        # Group labels above the axis
        group_labels = ["LOPO", "TwoPerson", "PersonVec", "AnnotOnly"]
        for gi, (gl, cx) in enumerate(zip(group_labels, [1.5, 3.5, 5.5, 7.5])):
            ax.text(cx, 1.04, gl, ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold",
                    color=["#1A5276", "#A04000", "#1A5C1A", "#6C3483"][gi],
                    transform=ax.get_xaxis_transform())

    fig.suptitle(
        "Method Comparison — Inner CV AUC per Outer Fold\n"
        "Light = Unstable  |  Dark = Stable  |  "
        "star = mean, box = IQR, line = median",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout(h_pad=4.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] Comparison plot saved -> {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 76)
    print("METHOD COMPARISON  —  MULTI-TARGET SWEEP")
    print(f"Targets    : {TARGETS}")
    print(f"Conditions : {list(CONDITIONS.keys())}")
    print(f"Methods    : {METHODS}  |  Feature counts: {FEATURE_COUNTS}")
    print(f"Stability  : >= {STABILITY_THRESHOLD:.0%} of outer folds")
    print("=" * 76)

    ts = time.strftime("%Y%m%d_%H%M%S")

    # Pre-load circRNA matrix once
    _ = _load_circrna_matrix()
    gene_map = _CIRC_CACHE.get("gene_map", {})

    all_results  = {}
    summary_rows = []

    feat_txt_path = os.path.join(_SCRIPT_DIR, f"method_compare_features_{ts}.txt")

    with open(feat_txt_path, "w") as feat_fh:
        for target in TARGETS:
            sep = "#" * 76
            print(f"\n{sep}\n# TARGET: {target}\n{sep}")

            try:
                X, y, groups = load_data(target)
            except Exception as exc:
                print(f"  [ERROR] Cannot load data for '{target}': {exc}")
                import traceback; traceback.print_exc()
                continue

            target_results = {}

            for cond_name, cond_cfg in CONDITIONS.items():
                print(f"\n  ---- {cond_name} ----")
                try:
                    res = run_condition(X, y, groups, cond_cfg, cond_name, gene_map)
                    target_results[cond_name] = res

                    valid = [v for v in res["inner_aucs"] if not np.isnan(v)]
                    mu    = np.mean(valid) if valid else float("nan")
                    sd    = np.std(valid)  if valid else float("nan")

                    summary_rows.append({
                        "Target":          target,
                        "Condition":       cond_name,
                        "Mean_Inner_AUC":  round(mu, 4) if not np.isnan(mu) else "",
                        "SD_Inner_AUC":    round(sd, 4) if not np.isnan(sd) else "",
                        "N_valid_folds":   len(valid),
                        "N_total_folds":   res["n_folds"],
                        "N_stable_feats":  len(res.get("stable_feats", [])),
                    })

                except Exception as exc:
                    print(f"  [ERROR] Condition '{cond_name}' failed: {exc}")
                    import traceback; traceback.print_exc()

            all_results[target] = target_results

            try:
                print_and_save_features(target_results, target, gene_map, feat_fh)
            except Exception as exc:
                print(f"  [WARN] Feature reporting failed for {target}: {exc}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY  —  Mean Inner CV AUC per Condition x Target")
    print("=" * 80)

    sum_df = pd.DataFrame(summary_rows)
    if not sum_df.empty:
        try:
            pivot = sum_df.pivot(index="Target", columns="Condition",
                                 values="Mean_Inner_AUC")
            pivot = pivot[list(CONDITIONS.keys())]   # enforce column order
            print(pivot.to_string())
        except Exception:
            print(sum_df.to_string())

    csv_path = os.path.join(_SCRIPT_DIR, f"method_compare_summary_{ts}.csv")
    sum_df.to_csv(csv_path, index=False)
    print(f"\n[csv] Summary saved -> {csv_path}")
    print(f"[txt] Feature lists -> {feat_txt_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    png_path = os.path.join(_SCRIPT_DIR, f"method_compare_{ts}.png")
    try:
        plot_compare(all_results, TARGETS, png_path)
    except Exception as exc:
        print(f"[WARN] Plot failed: {exc}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\nTotal runtime: {h}h {m}m {s}s")
    print(f"Feature lists : {feat_txt_path}")
    print(f"Summary CSV   : {csv_path}")
    print(f"Plot          : {png_path}")
