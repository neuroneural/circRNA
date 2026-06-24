"""
============================================================================
FEATURE-SELECTION EVALUATION HARNESS — GROUPED (repeated-measures) version
============================================================================

Same idea as feature_eval_harness.py (CHOOSE -> JUDGE -> CERTIFY), but built
for data where the SAME SUBJECT appears multiple times (e.g. circRNA visits
V1/V2/V3 per person). With repeated measures, an ordinary fold split can put
one visit of a person in training and another visit of the SAME person in the
test fold -> the model "predicts" a near-duplicate it already saw -> leakage
-> inflated AUC. This is the most likely reason raw results "look too good."

THE FIX: group-aware cross-validation. Every fold keeps ALL of a person's
samples on the same side of the split (train OR test, never both). We split by
URSI (the unique person ID). Everything else mirrors the original harness.

Use this version whenever rows are not independent (repeated visits, twins,
multiple sites per subject, etc.). Use the original when each row is a
distinct subject.

HOW TO USE: set TARGET below, then run:  python feature_eval_harness_grouped.py
============================================================================
"""

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
FEATURE_COUNTS = [10, 20, 50, 100]
METHODS = ["mi", "rf"]
MAX_OUTER_FOLDS = 5
MAX_INNER_FOLDS = 3
N_PERMUTATIONS = 25

DATA_DIR = "/Users/matedort/PTSD_Project/data 2"

# Which clinical column to predict. Continuous scores are median-split into
# low(0)/high(1); categorical ones use the explicit mapping below.
TARGET = "GAD-7"

# How to turn each task into a binary 0/1 label.
BINARY_MAP = {"Sex": {"Male": 1, "Female": 0},
              "Response": {"Yes": 1, "No": 0}}
MEDIAN_SPLIT_TARGETS = ["Age", "GAD-7", "IDS-C", "SHAPSC", "TEPS", "YMRS"]


# ============================================================
# PLUG YOUR DATA HERE  (data 2: circRNA counts + clinical metadata)
#
# Returns (X, y, groups, title):
#   X      -> samples x features (circRNA counts, transposed)
#   y      -> binary 0/1 label for TARGET
#   groups -> URSI (person id) per sample, so folds split by PERSON
# ============================================================

META_COLS = ["Chr", "Start", "End", "Gene", "JunctionType",
             "Strand", "Start-End Region"]


def load_data():
    # 1) counts: genes-as-rows -> drop metadata cols -> transpose to samples x features
    counts = pd.read_csv(
        f"{DATA_DIR}/circRNA_counts.csv", index_col="circRNA_id")
    counts = counts.drop(columns=[c for c in META_COLS if c in counts.columns])
    X = counts.T                      # index = map_id, columns = circRNA_id
    X.index.name = "map_id"

    # 2) clinical metadata, indexed by the same map_id
    meta = pd.read_csv(f"{DATA_DIR}/clean_clinical_metadata.csv")
    meta = meta.set_index("map_id")

    # 3) align counts and metadata on the shared samples
    common = [s for s in X.index if s in meta.index]
    X = X.loc[common].astype(float)
    meta = meta.loc[common]

    # 4) build the binary label for TARGET
    if TARGET in BINARY_MAP:
        y = meta[TARGET].map(BINARY_MAP[TARGET])
    elif TARGET in MEDIAN_SPLIT_TARGETS:
        vals = pd.to_numeric(meta[TARGET], errors="coerce")
        y = (vals >= vals.median()).astype(float)
        y[vals.isna()] = np.nan
    else:
        raise ValueError(f"Unknown TARGET '{TARGET}'")

    groups = meta["URSI"]

    # 5) drop samples with no label (e.g. Response is blank on Pre visits)
    keep = y.notna()
    X, y, groups = X[keep], y[keep].astype(int), groups[keep]

    title = (f"{TARGET}  (n={len(y)} samples, "
             f"{groups.nunique()} unique people)")
    return X, y, groups, title


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, y_prob):
    if len(set(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def _groups_per_class(y, groups):
    """Smallest number of distinct people contributing a given class.
    Caps how many group-folds we can stratify into."""
    df = pd.DataFrame({"y": list(y), "g": list(groups)})
    return int(df.groupby("y")["g"].nunique().min())


def make_group_cv(y, groups, max_folds):
    """Group-aware splitter: keeps each person wholly in train OR test.
    Falls back to Leave-One-Group-Out when there are too few people."""
    n_groups = len(set(groups))
    gpc = _groups_per_class(y, groups)
    k = min(max_folds, gpc)
    if k < 2 or n_groups < 3:
        return LeaveOneGroupOut(), "LeaveOneGroupOut"
    return (StratifiedGroupKFold(n_splits=k, shuffle=True,
                                 random_state=RANDOM_STATE),
            f"{k}-group-fold")


# ============================================================
# FEATURE SELECTION METHODS  (same contract: (X, y, top_n) -> feature names)
# ============================================================

def mi_features(X, y, top_n):
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return pd.Series(mi, index=X.columns)\
        .sort_values(ascending=False).head(top_n).index.tolist()


def rfe_features(X, y, top_n):
    lr = LogisticRegression(max_iter=5000, solver="saga",
                            random_state=RANDOM_STATE)
    rfe = RFE(lr, n_features_to_select=top_n, step=0.1)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()


def rf_features(X, y, top_n):
    rf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    return imp.sort_values(ascending=False).head(top_n).index.tolist()


FEATURE_FUNCS = {
    "mi": mi_features,
    "rfe": rfe_features,
    "rf": rf_features,
}


# ============================================================
# INNER SCORER  (CHOOSE) — group-aware
# ============================================================

def inner_score(X_tr, y_tr, g_tr, top_n, method, log=False):
    inner_cv, inner_name = make_group_cv(y_tr, g_tr, MAX_INNER_FOLDS)
    if log:
        print(f"        [inner] {method} top_n={top_n} -> {inner_name} "
              f"on {len(y_tr)} samples / {g_tr.nunique()} people")

    oof_true, oof_prob = [], []
    for tr_idx, val_idx in inner_cv.split(X_tr, y_tr, g_tr):
        X_in_tr, y_in_tr = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_in_val, y_in_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]

        features = FEATURE_FUNCS[method](X_in_tr, y_in_tr, top_n)
        scaler = StandardScaler()
        X_in_tr_s = scaler.fit_transform(X_in_tr[features])
        X_in_val_s = scaler.transform(X_in_val[features])
        svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        svm.fit(X_in_tr_s, y_in_tr)

        oof_true.extend(y_in_val.tolist())
        oof_prob.extend(svm.predict_proba(X_in_val_s)[:, 1].tolist())

    return safe_auc(oof_true, oof_prob)


# ============================================================
# NESTED GROUP CV  (outer = JUDGE, inner = CHOOSE) — group-aware
# ============================================================

def nested_cv(X, y, groups, verbose=True):
    outer_cv, outer_name = make_group_cv(y, groups, MAX_OUTER_FOLDS)
    n_combos = len(METHODS) * len(FEATURE_COUNTS)

    if verbose:
        print(f"[setup] outer: {outer_name} | groups split by PERSON "
              f"(no person spans train/test)")
        print(f"[setup] inner search: {n_combos} combos per fold")

    oof_true, oof_prob = [], []
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y, groups), 1):
        X_tr, y_tr, g_tr = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        if verbose:
            print(f"\n--- Outer fold {fold} ({outer_name}): "
                  f"train={len(y_tr)} ({g_tr.nunique()} people), "
                  f"sealed test={len(y_te)} ({groups.iloc[test_idx].nunique()} people) ---")

        scores = {}
        for method in METHODS:
            for top_n in FEATURE_COUNTS:
                scores[(method, top_n)] = inner_score(
                    X_tr, y_tr, g_tr, top_n, method, log=verbose)

        best_method, best_top_n = max(scores, key=scores.get)
        best_inner_auc = scores[(best_method, best_top_n)]

        if verbose:
            for method in METHODS:
                row = "   ".join(
                    f"{tn}:{scores[(method, tn)]:.3f}" for tn in FEATURE_COUNTS)
                print(f"   {method:<5} {row}")
            print(f"   [CHOSE] {best_method} @ top_n={best_top_n} "
                  f"(inner AUC={best_inner_auc:.4f})")

        features = FEATURE_FUNCS[best_method](X_tr, y_tr, best_top_n)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[features])
        X_te_s = scaler.transform(X_te[features])
        svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        svm.fit(X_tr_s, y_tr)

        oof_true.extend(y_te.tolist())
        oof_prob.extend(svm.predict_proba(X_te_s)[:, 1].tolist())

        fold_records.append({"fold": fold, "chosen_method": best_method,
                             "chosen_top_n": best_top_n,
                             "inner_auc": best_inner_auc})

    overall_auc = safe_auc(oof_true, oof_prob)
    if verbose:
        print(f"\n[result] pooled honest AUC over {len(oof_true)} "
              f"held-out predictions = {overall_auc:.4f}")
    return overall_auc, fold_records


def procedure_pooled_auc(X, y, groups, verbose=False):
    auc, _ = nested_cv(X, y, groups, verbose=verbose)
    return auc


# ============================================================
# PERMUTATION GATE (CERTIFY)
# Labels shuffled at the sample level; group-aware CV still prevents leakage
# in both the real and the null runs, so the comparison stays fair.
# ============================================================

def permutation_gate(X, y, groups, n_permutations=N_PERMUTATIONS):
    real, fold_records = nested_cv(X, y, groups, verbose=True)
    print(f"\n[gate] real procedure pooled AUC = {real:.4f}")
    print(f"[gate] running {n_permutations} label-shuffled permutations...")

    null_scores = []
    for i in range(1, n_permutations + 1):
        rng = np.random.default_rng(RANDOM_STATE + i)
        y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        s = procedure_pooled_auc(X, y_perm, groups, verbose=False)
        null_scores.append(s)
        print(f"   perm {i}/{n_permutations}  null AUC = {s:.4f}")

    arr = np.array(null_scores, dtype=float)
    # Phipson-Smyth: min p with N=30 is 1/31 ~ 0.032
    p_value = (np.sum(arr >= real) + 1) / (n_permutations + 1)
    return real, arr.mean(), p_value, fold_records


# ============================================================
# VERDICT
# ============================================================

def recommend_settings(fold_records):
    combos = [(r["chosen_method"], r["chosen_top_n"]) for r in fold_records]
    (method, top_n), freq = Counter(combos).most_common(1)[0]
    return method, top_n, freq, len(combos)


def print_verdict(title, X, y, groups, real, null_mean, p_value, fold_records):
    bar = "=" * 60
    print(f"\n{bar}\nVERDICT — {title}\n{bar}")

    calibrated = abs(null_mean - 0.5) <= 0.08
    print("\n1. CALIBRATION (is the instrument honest?)")
    print(f"   null AUC = {null_mean:.4f}  [want ~0.50]  ->  "
          f"{'OK, calibrated' if calibrated else 'WARNING: possible leakage'}")

    signal = (p_value < 0.05) and (real > null_mean)
    print("\n2. SIGNAL (real, not luck?) — leakage-free, split by person")
    print(f"   real AUC = {real:.4f} | null = {null_mean:.4f} | "
          f"gap = {real - null_mean:+.4f} | p = {p_value:.4f}")
    print(
        f"   ->  {'SIGNAL DETECTED (p < 0.05)' if signal else 'NO SIGNAL above chance'}")

    method, top_n, freq, n_folds = recommend_settings(fold_records)
    chosen = [f"{r['chosen_method']}@{r['chosen_top_n']}" for r in fold_records]
    print("\n3. RECOMMENDED SETTINGS")
    print(f"   per-fold choices : {chosen}")
    print(f"   most common      : {method} @ top_n={top_n} "
          f"({freq}/{n_folds} folds, "
          f"{'stable' if freq / n_folds >= 0.6 else 'UNSTABLE'})")

    print("\n4. WHAT TO DO NEXT")
    if not calibrated:
        print("   STOP. Null not ~0.50 - still leaking somewhere.")
    elif signal:
        print(f"   PASS. Real signal survives the leakage-free test. Train with "
              f"method='{method}', top_n={top_n}.")
    else:
        print("   NO-GO. Once person-level leakage is removed, the signal is "
              "gone -> the high raw AUC was leakage/overfitting.")
    print(bar)


if __name__ == "__main__":
    X, y, groups, title = load_data()

    print("=" * 60)
    print(f"GROUPED HARNESS — {title}")
    print("=" * 60)
    print(f"X shape: {X.shape}  | y: {y.value_counts().to_dict()}  | "
          f"people: {groups.nunique()}")
    print(f"methods: {METHODS} | counts: {FEATURE_COUNTS} | "
          f"perms: {N_PERMUTATIONS}")

    real, null_mean, p_value, fold_records = permutation_gate(X, y, groups)
    print_verdict(title, X, y, groups, real, null_mean, p_value, fold_records)
