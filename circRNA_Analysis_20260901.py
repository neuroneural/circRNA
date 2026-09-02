"""
============================================================================
circRNA_polyssifier_lopo_20260906.py
============================================================================

Person-level LOPO comparison of feature selection methods (MI, RF, RFE)
across 9 polyssifier-equivalent classifiers for targets GAD-7, Age, TEPS.

TWO-STAGE DESIGN (mirrors feature_eval_stable_lopo.py):
  Stage 1 — Feature Comparison
    LOPO outer loop (groups = URSI, person-level — NOT sample-level).
    For each fold:
      • For MI and RF: inner person-level CV picks best top_n (10/20/50/100)
      • For RFE:       fixed TOP_N = 50 (inner selection is too slow for RFE
                        with ~2000 features; step=100 features per iteration)
      • Select features on the full training partition (no leakage)
      • Fit all 9 classifiers; record per-fold per-classifier test AUC
      • Accumulate feature-selection frequency across folds
    → Outputs: per-fold AUC table, redesigned 2-panel per-target figures

  Stage 2 — Stable-Feature Evaluation
    Best method per target = highest mean Stage 1 fold AUC.
    Stable features = those appearing in >= STABILITY_THRESHOLD of folds.
    Run a second LOPO with ONLY the stable features (no inner CV needed).
    All 9 classifiers, pooled OOF AUC + 1000-bootstrap 95% CI reported.
    → Output: test-AUC table (all targets × classifiers)

NOTE: Method selection (Step 1 → Step 2) is based on held-out fold AUCs,
which is a mild form of adaptive selection.  Stage 2 AUC is reported per
classifier independently and is fully person-level held-out.

CHANGES from 20260820:
  • plot_boxplots_for_target() replaced with plot_per_target_redesigned():
      Top panel  — Stage 1 inner AUC boxplots (MI vs RF per classifier)
      Bottom panel — Stage 2 bootstrap AUC horizontal boxplots (per clf)
      Uses CORAL→NAVY colormap, x-axis zoomed to data range, AUC labels.
  • Output filenames now pretty_<target>_<ts>.png instead of boxplot_*.png

DATA SOURCES (never swap these):
  circRNA counts  →  /data/users2/ppopov1/datasets/circRNA
  clinical labels →  /home/users/elatash1/Work/circRNA_Analysis/clean_clinical_metadata.csv

OUTPUTS (written next to this script)
  pretty_GAD7_<ts>.png            — redesigned 2-panel figure for GAD-7
  pretty_Age_<ts>.png             — same for Age
  pretty_TEPS_<ts>.png            — same for TEPS
  test_auc_table_<ts>.png         — Stage 2 test AUC heat-table (all 3 targets)
  stage1_results_<ts>.csv         — fold-level AUC records (all targets)
  stage2_summary_<ts>.csv         — Stage 2 pooled OOF AUC per target × classifier
  stable_features_<target>_<ts>.csv
============================================================================
"""

import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE        = 42
FEATURE_COUNTS      = [10, 20, 50, 100]   # top_n candidates for MI & RF inner CV
RFE_TOP_N           = 50                  # fixed top_n for RFE (no inner selection)
RFE_STEP            = 100                 # absolute features removed per RFE iteration
METHODS             = ["mi", "rf", "rfe"]
MAX_INNER_FOLDS     = 3
STABILITY_THRESHOLD = 0.30               # feature must appear in >= 30% of folds

TARGETS = ["GAD-7", "Age", "TEPS"]

TARGET_THRESHOLDS = {
    "GAD-7":  10,       # >= 10 → high anxiety
    "TEPS":   "median", # >= median → high TEPS
    "Age":    "median", # >= median → older
}

BINARY_MAP = {
    "Sex": {"Male": 1, "Female": 0},
}

USE_DEMOGRAPHIC_FEATURES = True
DEMOGRAPHIC_FEATURES     = ["Sex", "Age"]   # added as extra features (if not the target)

BASE_PATH     = "/data/users2/ppopov1/datasets/circRNA"
METADATA_PATH = "/home/users/elatash1/Work/circRNA_Analysis/clean_clinical_metadata.csv"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CLASSIFIERS  (9 polyssifier-equivalent)
# ============================================================

CLF_NAMES = [
    "MLP", "KNN", "SVM", "Linear SVM", "Decision Tree",
    "Random Forest", "Logistic Reg.", "Naive Bayes", "Voting",
]

CLF_SHORT = [
    "MLP", "KNN", "SVM", "LinSVM", "DTree",
    "RF", "LogReg", "NaiveBayes", "Voting",
]

def _make_classifiers():
    """
    Build a fresh dict of estimators each call.
    (sklearn estimators are stateful after fit; never reuse across folds.)
    """
    lr = lambda: LogisticRegression(max_iter=5000, solver="saga",
                                    random_state=RANDOM_STATE)
    mlp = lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000,
                                early_stopping=True, random_state=RANDOM_STATE)
    rf  = lambda: RandomForestClassifier(n_estimators=200,
                                         random_state=RANDOM_STATE, n_jobs=-1)
    return {
        "MLP":           mlp(),
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "SVM":           SVC(kernel="rbf",    probability=True, random_state=RANDOM_STATE),
        "Linear SVM":    SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": rf(),
        "Logistic Reg.": lr(),
        "Naive Bayes":   GaussianNB(),
        "Voting":        VotingClassifier(
            estimators=[("lr", lr()), ("mlp", mlp()), ("rf", rf())],
            voting="soft",
        ),
    }


# ============================================================
# STYLE  (legacy — kept for test-AUC table)
# ============================================================

METHOD_COLORS  = {"mi": "#2E75B6", "rf": "#2E8B57", "rfe": "#8B2252"}
METHOD_LABELS  = {"mi": "Mutual Information (MI)",
                  "rf": "Random Forest Importance (RF)",
                  "rfe": f"Recursive Feature Elimination (RFE, top {RFE_TOP_N})"}
TARGET_COLORS  = {"GAD-7": "#003057", "Age": "#C45911", "TEPS": "#2E8B57"}


# ============================================================
# REDESIGNED PLOT PALETTE
# ============================================================

_NAVY  = "#003057"
_TEAL  = "#007A8A"
_CORAL = "#D95B43"
_LGRAY = "#E8EDF2"
_MGRAY = "#C4CDD6"
_WHITE = "#FFFFFF"

_METHOD_COLOR  = {"mi": _NAVY, "rf": _TEAL}

TARGET_LABELS  = {
    "GAD-7": "GAD-7 Anxiety",
    "Age":   "Age",
    "TEPS":  "TEPS Reward",
}

_DEMO_PATTERN = re.compile(r"^[A-Z][a-z]+$")   # catches "Age", "Sex", etc.


def _auc_color(v, lo=0.45, hi=1.0):
    """Interpolate CORAL (low AUC) → NAVY (high AUC)."""
    t  = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    c0 = mcolors.to_rgb(_CORAL)
    c1 = mcolors.to_rgb(_NAVY)
    return tuple(c0[i] + t * (c1[i] - c0[i]) for i in range(3))


# ============================================================
# DATA LOADING  (identical to feature_eval_stable_lopo.py)
# ============================================================

_CIRC_CACHE: dict = {}

def _load_circrna_matrix():
    if "circ_ML_df" in _CIRC_CACHE:
        return _CIRC_CACHE["circ_ML_df"]

    raw_path  = os.path.join(BASE_PATH, "01_raw_data")
    qced_path = os.path.join(BASE_PATH, "03_QCed_data")

    pd.read_excel(os.path.join(raw_path, "ERVIN_All-Samples-Manifest_20250401.xlsx"))
    pd.read_csv(os.path.join(qced_path,
                              "202504_ERVIN_filtered_linear_gene_counts.csv"))
    circ_counts_df = pd.read_csv(os.path.join(
        qced_path,
        "circ_counts_post_filtering_circ_linear_and_1cnt_in.25samples_ERVIN_202504.csv"))

    meta_cols     = ["Chr", "Start", "End", "Gene", "JunctionType",
                     "Strand", "Start-End Region", "chr_start_end_strand"]
    subject_cols  = [c for c in circ_counts_df.columns if c not in meta_cols]
    circ_ML_df    = circ_counts_df[subject_cols].T
    circ_ML_df.columns   = circ_counts_df["chr_start_end_strand"]
    circ_ML_df.index.name = "Sample_ID"

    _CIRC_CACHE["circ_ML_df"] = circ_ML_df
    _CIRC_CACHE["gene_map"]   = (
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
    print(f"  Binarise {target}: cutoff={'median (' + str(round(cutoff,2)) + ')' if thresh == 'median' else '>= ' + str(cutoff)} → 0/1")
    y = (vals >= cutoff).astype(float)
    y[vals.isna()] = np.nan
    return y


def load_data(target):
    circ_ML_df = _load_circrna_matrix()
    meta_df    = pd.read_csv(METADATA_PATH).set_index("Sample_ID")

    common     = [s for s in circ_ML_df.index if s in meta_df.index]
    circ_ML_df = circ_ML_df.loc[common]
    meta_df    = meta_df.loc[common]

    y_series   = _binarize(meta_df, target)
    keep       = y_series.notna()
    circ_ML_df = circ_ML_df[keep]
    meta_df    = meta_df[keep]
    y_series   = y_series[keep].astype(int)

    X      = circ_ML_df.copy()
    groups = meta_df["URSI"]

    # Optionally append demographic features (exclude the target itself)
    if USE_DEMOGRAPHIC_FEATURES:
        for col in [c for c in DEMOGRAPHIC_FEATURES if c != target]:
            if col not in meta_df.columns:
                continue
            X[col] = (meta_df[col].map(BINARY_MAP[col]).values
                      if col in BINARY_MAP
                      else pd.to_numeric(meta_df[col], errors="coerce").values)

    print(f"  X: {X.shape}  |  y dist: {y_series.value_counts().to_dict()}"
          f"  |  {groups.nunique()} people")
    return X, y_series, groups


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, y_prob):
    arr = list(y_true)
    if len(set(arr)) < 2:
        return float("nan")
    return roc_auc_score(arr, list(y_prob))


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


def bootstrap_auc(y_true, y_prob, n_boot=1000, seed=42):
    rng    = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n      = len(y_true)
    aucs   = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(set(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    return aucs


# ============================================================
# FEATURE SELECTION
# ============================================================

def mi_features(X, y, top_n):
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return (pd.Series(mi, index=X.columns)
              .sort_values(ascending=False)
              .head(top_n).index.tolist())


def rf_features(X, y, top_n):
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                n_jobs=-1)
    rf.fit(X, y)
    return (pd.Series(rf.feature_importances_, index=X.columns)
              .sort_values(ascending=False)
              .head(top_n).index.tolist())


def rfe_features(X, y, top_n=RFE_TOP_N, step=RFE_STEP):
    """
    Logistic-Regression RFE.  step=100 (absolute) keeps runtime reasonable
    on ~2000 features: ceil((n_features - top_n) / 100) LogReg fits.
    """
    if X.shape[1] <= top_n:
        return X.columns.tolist()
    lr  = LogisticRegression(max_iter=5000, solver="saga",
                              random_state=RANDOM_STATE)
    rfe = RFE(lr, n_features_to_select=top_n, step=step)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()


FEATURE_FUNCS = {"mi": mi_features, "rf": rf_features, "rfe": rfe_features}


def inner_score_proxy(X_tr, y_tr, g_tr, top_n, method):
    """
    Evaluate a (method, top_n) combination using inner person-level CV
    with a LogReg proxy — used only for MI and RF to pick the best top_n.
    Returns the pooled inner AUC (0.5 if undeterminable).
    """
    inner_cv = make_inner_cv(y_tr, g_tr)
    proxy    = LogisticRegression(max_iter=5000, solver="saga",
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
        oof_prob.extend(
            clf.predict_proba(sc.transform(X_in_val[feats]))[:, 1])

    auc = safe_auc(oof_true, oof_prob)
    return 0.5 if np.isnan(auc) else auc


def pick_best_top_n(X_tr, y_tr, g_tr, method):
    """
    MI / RF: use inner CV to find optimal top_n.
    RFE:     always return RFE_TOP_N (inner CV would be prohibitively slow).
    Returns (best_top_n, inner_auc_for_that_top_n).
    """
    if method == "rfe":
        return RFE_TOP_N, float("nan")

    scores  = {n: inner_score_proxy(X_tr, y_tr, g_tr, n, method)
               for n in FEATURE_COUNTS}
    best_n  = max(scores, key=scores.get)
    return best_n, scores[best_n]


# ============================================================
# STAGE 1: LOPO FEATURE COMPARISON
# ============================================================

def stage1_lopo(X, y, groups, target):
    """
    Outer LOPO (LeaveOneGroupOut, groups=URSI).

    For each fold × method:
      1. Pick best top_n (MI/RF via inner CV; RFE fixed at RFE_TOP_N)
      2. Select features on the full training partition
      3. Fit all 9 classifiers; evaluate on held-out person
      4. Accumulate feature frequency counter

    Returns
    -------
    fold_records     : list[dict]  — one row per fold × method × classifier
    feature_counters : dict[method → Counter]
    n_folds          : int
    """
    lopo    = LeaveOneGroupOut()
    n_folds = groups.nunique()

    fold_records     = []
    feature_counters = {m: Counter() for m in METHODS}

    print(f"\n[Stage 1] {target} — {n_folds} LOPO folds | "
          f"{len(METHODS)} methods × {len(CLF_NAMES)} classifiers")

    for fold_i, (train_idx, test_idx) in enumerate(
            lopo.split(X, y, groups), 1):

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
        g_tr       = groups.iloc[train_idx]
        held_ursi  = groups.iloc[test_idx].iloc[0]

        # Single-class test folds are uninformative for AUC
        single_class_test = len(set(y_te.tolist())) < 2

        print(f"  Fold {fold_i:>2}/{n_folds} — held: {held_ursi} "
              f"({len(y_te)} sample(s), labels={sorted(set(y_te.tolist()))})"
              + (" [single-class test — AUC=NaN]" if single_class_test else ""))

        for method in METHODS:

            # ── Feature selection ──────────────────────────────────────────
            best_n, inner_auc = pick_best_top_n(X_tr, y_tr, g_tr, method)
            feats             = FEATURE_FUNCS[method](X_tr, y_tr, best_n)
            feature_counters[method].update(feats)

            # ── Scale ──────────────────────────────────────────────────────
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr[feats])
            X_te_s = scaler.transform(X_te[feats])

            # ── Evaluate all 9 classifiers ─────────────────────────────────
            clfs       = _make_classifiers()
            fold_aucs  = {}

            for clf_name in CLF_NAMES:
                auc = float("nan")
                if not single_class_test:
                    try:
                        clf = clfs[clf_name]
                        clf.fit(X_tr_s, y_tr)
                        prob = clf.predict_proba(X_te_s)[:, 1]
                        auc  = safe_auc(y_te.tolist(), prob.tolist())
                    except Exception as e:
                        print(f"      [{method}/{clf_name}] WARN: {e}")

                fold_aucs[clf_name] = auc
                fold_records.append({
                    "Target":     target,
                    "Fold":       fold_i,
                    "HeldURSI":   str(held_ursi),
                    "Method":     method,
                    "BestTopN":   best_n,
                    "InnerAUC":   round(inner_auc, 4) if not np.isnan(inner_auc) else "",
                    "Classifier": clf_name,
                    "FoldAUC":    auc,
                    "NFeats":     len(feats),
                })

            # Quick per-fold summary line
            mlp_a = fold_aucs.get("MLP",           float("nan"))
            rf_a  = fold_aucs.get("Random Forest",  float("nan"))
            lr_a  = fold_aucs.get("Logistic Reg.",  float("nan"))
            print(f"    {method.upper():<5} top_n={best_n:>3} "
                  + (f" inner={inner_auc:.3f}" if not np.isnan(inner_auc) else "        ")
                  + f"  MLP={mlp_a:.3f}  RF={rf_a:.3f}  LR={lr_a:.3f}")

    # Stability summary
    min_count = int(np.ceil(STABILITY_THRESHOLD * n_folds))
    print(f"\n  Stable features (>= {STABILITY_THRESHOLD*100:.0f}% = {min_count}/{n_folds} folds):")
    for method in METHODS:
        n_stable = sum(1 for c in feature_counters[method].values()
                       if c >= min_count)
        print(f"    {method.upper()}: {n_stable} stable features")

    return fold_records, feature_counters, n_folds


# ============================================================
# STAGE 2: STABLE-FEATURE LOPO EVALUATION
# ============================================================

def stage2_evaluate(X, y, groups, stable_features, target, method_label):
    """
    LOPO pass with a FIXED set of stable features — no inner CV needed.
    Returns pooled OOF AUC + bootstrap CI for all 9 classifiers.
    """
    feats = [f for f in stable_features if f in X.columns]
    if not feats:
        print("  [WARN] No stable features found in X — skipping Stage 2.")
        return {n: float("nan") for n in CLF_NAMES}, {n: [] for n in CLF_NAMES}

    lopo      = LeaveOneGroupOut()
    n_folds   = groups.nunique()
    oof_true  = []
    oof_probs = {name: [] for name in CLF_NAMES}

    print(f"\n[Stage 2] {target} ({method_label}) — "
          f"{len(feats)} stable features, {n_folds} folds")

    for fold_i, (train_idx, test_idx) in enumerate(
            lopo.split(X, y, groups), 1):

        X_tr, y_tr = X.iloc[train_idx][feats], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx][feats],  y.iloc[test_idx]
        held_ursi  = groups.iloc[test_idx].iloc[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        oof_true.extend(y_te.tolist())

        clfs = _make_classifiers()
        for name in CLF_NAMES:
            try:
                clf = clfs[name]
                clf.fit(X_tr_s, y_tr)
                prob = clf.predict_proba(X_te_s)[:, 1]
                oof_probs[name].extend(prob.tolist())
            except Exception as e:
                print(f"    [{name}] WARN fold {fold_i}: {e}")
                oof_probs[name].extend([0.5] * len(y_te))

        if fold_i % 5 == 0 or fold_i == n_folds:
            print(f"  Fold {fold_i}/{n_folds} done (held: {held_ursi})")

    print(f"\n  Pooled OOF AUC — {target} | {len(feats)} stable features:")
    pooled_aucs = {}
    boot_aucs   = {}
    for name in CLF_NAMES:
        try:
            pooled_aucs[name] = roc_auc_score(oof_true, oof_probs[name])
            boot_aucs[name]   = bootstrap_auc(oof_true, oof_probs[name])
            ci_lo = np.percentile(boot_aucs[name], 2.5)
            ci_hi = np.percentile(boot_aucs[name], 97.5)
            print(f"   {name:<22} AUC = {pooled_aucs[name]:.4f}  "
                  f"[{ci_lo:.3f}–{ci_hi:.3f}]")
        except Exception as e:
            pooled_aucs[name] = float("nan")
            boot_aucs[name]   = []
            print(f"   {name:<22} ERROR: {e}")

    best = max(pooled_aucs, key=lambda k: pooled_aucs[k]
               if not np.isnan(pooled_aucs[k]) else -1)
    print(f"   → BEST: {best}  AUC = {pooled_aucs[best]:.4f}")
    return pooled_aucs, boot_aucs


# ============================================================
# PLOT 1: REDESIGNED PER-TARGET FIGURE  (replaces plot_boxplots_for_target)
# ============================================================

def plot_per_target_redesigned(
    fold_records,
    boot_aucs_for_target,
    feature_counters,
    best_method,
    target,
    n_folds,
    output_path,
    min_folds=None,
):
    """
    Two-panel redesigned figure for one target.

    Top panel  — Stage 1 inner AUC vertical boxplots (MI vs RF per classifier)
    Bottom panel — Stage 2 bootstrap AUC horizontal boxplots (per classifier)

    Parameters
    ----------
    fold_records : list[dict]
        Stage 1 fold-level records (all_fold_records from main).
    boot_aucs_for_target : dict  {clf_name → list[float]}
        Stage 2 bootstrap distributions; stage2_boot[target].
    feature_counters : dict  {method → Counter}
        Feature selection frequency from Stage 1.
    best_method : str   "mi" | "rf" | "rfe"
    target : str
    n_folds : int
    output_path : str
    min_folds : int or None
        Defaults to ceil(STABILITY_THRESHOLD × n_folds).
    """
    if min_folds is None:
        min_folds = int(np.ceil(STABILITY_THRESHOLD * n_folds))

    # ── Stable circRNA feature count for footer ────────────────────────────
    counter = feature_counters.get(best_method, Counter())
    _demo_set = set(DEMOGRAPHIC_FEATURES)
    n_stable_circrna = sum(
        1 for feat, cnt in counter.items()
        if cnt >= min_folds
        and feat not in _demo_set
        and not _DEMO_PATTERN.match(feat)
    )

    # ── Stage 1 inner AUC data ─────────────────────────────────────────────
    df1 = pd.DataFrame(fold_records)
    df1 = df1[df1["Target"] == target].copy()
    df1["InnerAUC"] = pd.to_numeric(df1["InnerAUC"], errors="coerce")

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor(_WHITE)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 2],
        hspace=0.42,
        left=0.10, right=0.88,
        top=0.91, bottom=0.06,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ═══════════════════════════════════════════════════════════════════════
    # TOP PANEL — Stage 1 inner AUC per method (MI, RF), one value per fold
    # Each box = distribution of InnerAUC across n_folds LOPO folds.
    # InnerAUC is from the feature-selection proxy step (LogReg inner CV),
    # computed once per fold per method — same for all classifiers, so we
    # deduplicate by Fold before plotting.
    # ═══════════════════════════════════════════════════════════════════════
    methods_shown = [m for m in ["mi", "rf"] if m in df1["Method"].unique()]
    n_clfs = len(CLF_NAMES)

    METHOD_XLABELS = {
        "mi": "MI\n(Mutual Information)",
        "rf": "RF\n(Random Forest)",
    }

    boxes_top  = []
    positions_top = list(range(len(methods_shown)))
    colors_top = []

    for method in methods_shown:
        # One InnerAUC per fold (deduplicate — value is classifier-independent)
        sub  = df1[df1["Method"] == method].drop_duplicates(subset=["Fold"])
        vals = sub["InnerAUC"].dropna().tolist()
        boxes_top.append(vals if vals else [float("nan")])
        colors_top.append(_METHOD_COLOR.get(method, "#555555"))

    bp = ax_top.boxplot(
        boxes_top,
        positions=positions_top,
        widths=0.45,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color=_WHITE, linewidth=2.2),
        whiskerprops=dict(linewidth=1.3, color="#555"),
        capprops=dict(linewidth=1.3, color="#555"),
        flierprops=dict(marker="o", markersize=4, linestyle="none",
                        alpha=0.50),
    )
    for patch, color in zip(bp["boxes"], colors_top):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    # Jitter individual fold points
    for pos, vals, color in zip(positions_top, boxes_top, colors_top):
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            continue
        rng = np.random.default_rng(abs(hash(color)) % 9999)
        jitter = rng.uniform(-0.08, 0.08, len(clean))
        ax_top.scatter(np.array([pos] * len(clean)) + jitter, clean,
                       color=color, alpha=0.65, s=40, zorder=5,
                       edgecolors=_WHITE, linewidths=0.6)
        ax_top.scatter([pos], [np.mean(clean)],
                       marker="*", color=_WHITE, s=120, zorder=6,
                       edgecolors=color, linewidths=0.9)

    ax_top.axhline(0.5, color="crimson", linestyle="--",
                   linewidth=1.3, zorder=2)

    ax_top.set_xlim(-0.65, len(methods_shown) - 0.35)
    ax_top.set_xticks(positions_top)
    ax_top.set_xticklabels(
        [METHOD_XLABELS.get(m, m.upper()) for m in methods_shown],
        fontsize=10.5, fontweight="bold",
    )
    ax_top.set_ylabel("Inner CV AUC\n(Stage 1 proxy)", fontsize=9.5, color="#333")
    ax_top.set_ylim(0.0, 1.08)
    ax_top.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_top.tick_params(axis="y", labelsize=8.5)
    ax_top.set_facecolor(_LGRAY)
    ax_top.grid(axis="y", alpha=0.35, linestyle="--", color=_WHITE, linewidth=0.8)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # n= annotation
    ax_top.text(len(methods_shown) - 0.40, 1.05,
                f"n = {n_folds} folds",
                fontsize=8.5, ha="right", va="top", color="#555",
                fontstyle="italic")

    # Chance label
    ax_top.text(len(methods_shown) - 0.40, 0.52, "Chance (0.5)",
                fontsize=8, ha="right", va="bottom", color="crimson",
                fontstyle="italic")

    ax_top.set_title(
        f"Stage 1 — Feature Selection Method Comparison  ·  {TARGET_LABELS.get(target, target)}",
        fontsize=10, fontweight="bold", color="#222", pad=6,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # BOTTOM PANEL — Stage 2 bootstrap AUC horizontal boxplots
    # ═══════════════════════════════════════════════════════════════════════
    boot_data  = []
    auc_values = []
    for clf in CLF_NAMES:
        vals = boot_aucs_for_target.get(clf, [])
        boot_data.append(vals if vals else [float("nan")])
        auc_values.append(float(np.nanmedian(vals)) if vals else float("nan"))

    # X-axis zoom
    all_flat = [v for bv in boot_data for v in bv if not np.isnan(v)]
    x_min = max(0.35, np.nanpercentile(all_flat, 1) - 0.05) if all_flat else 0.40
    x_max = min(1.02, np.nanpercentile(all_flat, 99) + 0.05) if all_flat else 1.02

    y_positions = list(range(n_clfs - 1, -1, -1))   # top = CLF_NAMES[0]

    ax_bot.set_facecolor(_LGRAY)
    for spine in ax_bot.spines.values():
        spine.set_visible(False)
    ax_bot.grid(axis="x", alpha=0.35, linestyle="--", color=_WHITE, linewidth=0.8)

    bp2 = ax_bot.boxplot(
        boot_data,
        positions=y_positions,
        vert=False,
        patch_artist=True,
        widths=0.55,
        manage_ticks=False,
        medianprops=dict(color=_WHITE, linewidth=2.2),
        whiskerprops=dict(linewidth=1.2, color="#666"),
        capprops=dict(linewidth=1.2, color="#666"),
        flierprops=dict(marker="o", markersize=2.5, linestyle="none",
                        alpha=0.30, markerfacecolor="#888"),
    )

    for patch, auc_v in zip(bp2["boxes"], auc_values):
        c = _auc_color(auc_v) if not np.isnan(auc_v) else mcolors.to_rgb(_MGRAY)
        patch.set_facecolor(c)
        patch.set_alpha(0.88)

    label_x = x_max + (x_max - x_min) * 0.01
    for pos, auc_v in zip(y_positions, auc_values):
        label = f"{auc_v:.3f}" if not np.isnan(auc_v) else "n/a"
        ax_bot.text(label_x, pos, label,
                    va="center", ha="left", fontsize=8.5,
                    color=_NAVY, fontweight="bold")

    ax_bot.axvline(0.5, color="crimson", linestyle="--",
                   linewidth=1.3, zorder=2)

    ax_bot.set_yticks(y_positions)
    ax_bot.set_yticklabels(CLF_SHORT, fontsize=9)
    ax_bot.set_xlim(x_min, x_max + (x_max - x_min) * 0.12)
    ax_bot.set_xlabel("Bootstrap AUC (1 000 resamples)", fontsize=9.5, color="#333")
    ax_bot.tick_params(axis="x", labelsize=8.5)
    ax_bot.set_title(
        "Stage 2 — Classifier Test AUC  (Stable Features, Pooled OOF + Bootstrap)",
        fontsize=10, fontweight="bold", color="#222", pad=6)

    # Colorbar
    norm = Normalize(vmin=x_min, vmax=x_max)
    cmap_custom = mcolors.LinearSegmentedColormap.from_list(
        "coral_navy", [_CORAL, _NAVY])
    sm = ScalarMappable(cmap=cmap_custom, norm=norm)
    sm.set_array([])
    bot_pos = gs[1].get_position(fig)
    cax = fig.add_axes([0.90, bot_pos.y0, 0.018, bot_pos.height])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("AUC", fontsize=9, labelpad=4)
    cbar.ax.tick_params(labelsize=8)

    # ── Title and footer ───────────────────────────────────────────────────
    title_label = TARGET_LABELS.get(target, target)
    fig.suptitle(
        f"circRNA Feature Selection & Classification — {title_label}",
        fontsize=14, fontweight="bold", color=_NAVY, y=0.97,
    )
    fig.text(
        0.10, 0.01,
        (f"Feature method: {best_method.upper()}  ·  "
         f"{n_stable_circrna} stable circRNA features  "
         f"(≥ {min_folds}/{n_folds} folds)"),
        fontsize=8.5, color="#555", ha="left", va="bottom", fontstyle="italic",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"[plot] Redesigned figure → {output_path}")


# ============================================================
# PLOT 2: TEST AUC TABLE  (Stage 2, all targets)
# ============================================================

def plot_test_auc_table(stage2_results, stage2_boot,
                        best_methods, stable_counts, output_path):
    """
    Heat-table:  rows = classifiers,  columns = targets.
    Cells are colored from light (0.50) to navy (1.00).
    Shows pooled OOF AUC from Stage 2 stable-feature LOPO.

    stage2_results : {target: {clf_name: pooled_auc}}
    stage2_boot    : {target: {clf_name: list[float]}}
    best_methods   : {target: method_str}
    stable_counts  : {target: int}
    """
    targets = list(stage2_results.keys())
    data    = np.array([
        [stage2_results[t].get(clf, np.nan) for t in targets]
        for clf in CLF_NAMES
    ])

    fig_h = max(5.5, len(CLF_NAMES) * 0.62 + 2.2)
    fig_w = max(6,   len(targets) * 3.0 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#f5f5f5")

    cmap = plt.cm.Blues
    norm = Normalize(vmin=0.50, vmax=1.00)

    for row_i, clf in enumerate(CLF_NAMES):
        for col_j, target in enumerate(targets):
            val = data[row_i, col_j]
            facecolor = cmap(norm(val)) if not np.isnan(val) else "#e8e8e8"
            rect = mpatches.FancyBboxPatch(
                (col_j + 0.04, row_i + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.03",
                linewidth=0,
                facecolor=facecolor,
                transform=ax.transData,
            )
            ax.add_patch(rect)

            if not np.isnan(val):
                text_color = "white" if val > 0.77 else "#1a1a1a"
                ax.text(col_j + 0.50, row_i + 0.55, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=11.5, fontweight="bold", color=text_color)
                boot = stage2_boot.get(target, {}).get(clf, [])
                if boot:
                    ci_lo = np.percentile(boot, 2.5)
                    ci_hi = np.percentile(boot, 97.5)
                    ax.text(col_j + 0.50, row_i + 0.22,
                            f"[{ci_lo:.2f}–{ci_hi:.2f}]",
                            ha="center", va="center",
                            fontsize=7.5,
                            color=text_color if val > 0.77 else "#555",
                            fontstyle="italic")

    ax.set_xlim(0, len(targets))
    ax.set_ylim(0, len(CLF_NAMES))

    ax.set_xticks([j + 0.5 for j in range(len(targets))])
    col_labels = [
        f"{t}\n(best: {best_methods.get(t,'?').upper()}, "
        f"{stable_counts.get(t,0)} stable feats)"
        for t in targets
    ]
    ax.set_xticklabels(col_labels, fontsize=10.5, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks([i + 0.5 for i in range(len(CLF_NAMES))])
    ax.set_yticklabels(CLF_NAMES, fontsize=10.5)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "circRNA Stage 2 — Test AUC  (Stable-Feature Person-level LOPO)\n"
        f"Stability ≥ {STABILITY_THRESHOLD*100:.0f}% of folds  ·  "
        "Pooled OOF AUC + 95% bootstrap CI",
        fontsize=12, fontweight="bold", pad=36,
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("AUC", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Test-AUC table → {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("circRNA POLYSSIFIER LOPO — MULTI-TARGET FEATURE COMPARISON")
    print(f"Targets         : {TARGETS}")
    print(f"Methods         : {METHODS}")
    print(f"  MI/RF top_n   : {FEATURE_COUNTS} (inner CV picks best)")
    print(f"  RFE top_n     : {RFE_TOP_N} (fixed)  step={RFE_STEP}")
    print(f"Stability thresh: >= {STABILITY_THRESHOLD*100:.0f}% of folds")
    print(f"Classifiers     : {CLF_NAMES}")
    print(f"Demographics    : {'yes — Sex, Age appended (excl. target)' if USE_DEMOGRAPHIC_FEATURES else 'no'}")
    print("=" * 72)

    all_fold_records = []
    stage2_results   = {}   # {target → {clf → pooled_auc}}
    stage2_boot      = {}   # {target → {clf → [boot aucs]}}
    best_methods     = {}   # {target → best_method_str}
    stable_counts    = {}   # {target → n_stable_features}
    stage2_info_rows = []

    for target in TARGETS:
        print(f"\n{'#'*72}\n# TARGET: {target}\n{'#'*72}")

        try:
            X, y, groups = load_data(target)
            n_folds   = groups.nunique()
            min_count = int(np.ceil(STABILITY_THRESHOLD * n_folds))

            # ── Stage 1 ───────────────────────────────────────────────────
            fold_records, feature_counters, n_folds = stage1_lopo(
                X, y, groups, target)
            all_fold_records.extend(fold_records)

            # ── Determine best method ─────────────────────────────────────
            df_fold      = pd.DataFrame(fold_records)
            method_mean  = {}
            for method in METHODS:
                vals = df_fold[df_fold["Method"] == method]["FoldAUC"].dropna()
                method_mean[method] = vals.mean() if len(vals) else 0.0
            best_method = max(method_mean, key=method_mean.get)

            print(f"\n  Method comparison (mean fold AUC, all classifiers):")
            for m, v in sorted(method_mean.items(), key=lambda x: -x[1]):
                marker = " ← BEST" if m == best_method else ""
                print(f"    {m.upper():<6} {v:.4f}{marker}")

            # ── Stable features for best method ──────────────────────────
            stable_features = [f for f, c
                                in feature_counters[best_method].most_common()
                                if c >= min_count]
            best_methods[target]  = best_method
            stable_counts[target] = len(stable_features)

            print(f"\n  Stable features ({best_method.upper()}): "
                  f"{len(stable_features)} "
                  f"(>= {min_count}/{n_folds} folds)")

            # Save stable feature CSV
            gene_map  = _CIRC_CACHE.get("gene_map", {})
            feat_rows = [
                {
                    "Target":         target,
                    "Method":         best_method,
                    "Rank":           i + 1,
                    "Feature":        f,
                    "Gene":           gene_map.get(f, "N/A"),
                    "Folds_selected": feature_counters[best_method][f],
                    "N_folds":        n_folds,
                    "Selection_freq": round(
                        feature_counters[best_method][f] / n_folds, 3),
                }
                for i, f in enumerate(stable_features)
            ]
            feat_csv = os.path.join(
                _SCRIPT_DIR,
                f"stable_features_{target.replace('-','').replace(' ','_')}_{ts}.csv",
            )
            pd.DataFrame(feat_rows).to_csv(feat_csv, index=False)
            print(f"  [csv] Stable features → {feat_csv}")

            # ── Stage 2 ───────────────────────────────────────────────────
            pooled_aucs, boot_aucs = stage2_evaluate(
                X, y, groups, stable_features, target, best_method.upper())
            stage2_results[target] = pooled_aucs
            stage2_boot[target]    = boot_aucs

            for clf_name in CLF_NAMES:
                auc  = pooled_aucs.get(clf_name, float("nan"))
                boot = boot_aucs.get(clf_name, [])
                ci_lo = np.percentile(boot, 2.5)  if boot else float("nan")
                ci_hi = np.percentile(boot, 97.5) if boot else float("nan")
                stage2_info_rows.append({
                    "Target":         target,
                    "BestMethod":     best_method,
                    "N_stable_feats": len(stable_features),
                    "N_people":       n_folds,
                    "Classifier":     clf_name,
                    "Test_AUC":       round(auc,   4) if not np.isnan(auc)   else "",
                    "CI_Lo":          round(ci_lo, 4) if not np.isnan(ci_lo) else "",
                    "CI_Hi":          round(ci_hi, 4) if not np.isnan(ci_hi) else "",
                })

            # ── Redesigned per-target figure ──────────────────────────────
            pretty_path = os.path.join(
                _SCRIPT_DIR,
                f"pretty_{target.replace('-','').replace(' ','_')}_{ts}.png",
            )
            plot_per_target_redesigned(
                fold_records         = all_fold_records,
                boot_aucs_for_target = boot_aucs,
                feature_counters     = feature_counters,
                best_method          = best_method,
                target               = target,
                n_folds              = n_folds,
                output_path          = pretty_path,
            )

        except Exception:
            print(f"\n  [ERROR] Target '{target}' failed — skipped.")
            import traceback
            traceback.print_exc()

    # ── Test AUC table (all targets) ─────────────────────────────────────
    if stage2_results:
        table_path = os.path.join(_SCRIPT_DIR, f"test_auc_table_{ts}.png")
        plot_test_auc_table(stage2_results, stage2_boot,
                            best_methods, stable_counts, table_path)

    # ── Save CSV outputs ──────────────────────────────────────────────────
    if all_fold_records:
        fold_csv = os.path.join(_SCRIPT_DIR, f"stage1_results_{ts}.csv")
        pd.DataFrame(all_fold_records).to_csv(fold_csv, index=False)
        print(f"[csv] Stage 1 fold records → {fold_csv}")

    if stage2_info_rows:
        s2_csv = os.path.join(_SCRIPT_DIR, f"stage2_summary_{ts}.csv")
        pd.DataFrame(stage2_info_rows).to_csv(s2_csv, index=False)
        print(f"[csv] Stage 2 summary → {s2_csv}")

    # ── Final console table ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STAGE 2 TEST AUC — SUMMARY  (pooled OOF, stable-feature LOPO)")
    print("=" * 72)
    if stage2_info_rows:
        df_s2 = pd.DataFrame(stage2_info_rows)
        pivot = df_s2.pivot(index="Classifier", columns="Target",
                            values="Test_AUC")
        pivot = pivot.reindex(columns=[t for t in TARGETS if t in pivot.columns])
        print(pivot.to_string())
        print()
        for t in TARGETS:
            if t in best_methods:
                print(f"  {t:<8} best method = {best_methods[t].upper():<6} "
                      f"| {stable_counts.get(t,0)} stable features")

    elapsed = time.time() - t0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\nTotal runtime: {h}h {m}m {s}s")
    print("=" * 72)
