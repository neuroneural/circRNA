"""
============================================================================
FEATURE-SELECTION EVALUATION HARNESS
============================================================================

WHAT THIS IS
    A reusable, data-agnostic tool that, for ANY labeled dataset, answers
    three questions and then tells you what to do next:

      1. CHOOSE  - which feature-selection method + how many features works
                   best?  (inner cross-validation tries every combination)
      2. JUDGE   - how well does that choice REALLY perform on unseen data?
                   (outer cross-validation scores on sealed folds - no leakage)
      3. CERTIFY - is that performance real signal, or luck?
                   (permutation gate: shuffle labels, re-run the WHOLE thing)

    It does NOT produce a deployable model. It tells you whether a trustworthy
    model is possible and what settings to use. You train the real model
    afterward - only if the gate passes.

HOW TO USE IT ON NEW DATA
    Edit ONE function: `load_data()` (look for "PLUG YOUR DATA HERE").
    Return X (samples x features, string column names) and y (0/1 labels).
    Everything else adapts automatically - including the fold strategy, which
    scales itself to the sample size. Then run:  python feature_eval_harness.py

HOW TO CHANGE SETTINGS
    Everything tunable is in the SETTINGS block below (methods, feature counts,
    fold caps, number of permutations). You should not need to touch anything
    else.

HOW TO READ THE RESULT (printed at the end as a VERDICT)
    - Null AUC ~0.50  -> the instrument is calibrated/honest (it scores chance
                         on random labels). If null is far from 0.50, STOP -
                         something is leaking.
    - p < 0.05        -> real signal. Use the recommended method + feature
                         count to train your real model.
    - p >= 0.05       -> no signal above chance at this sample size. Don't
                         report a model; get more data or rethink the target.

CALIBRATION HISTORY
    Sex sandbox (n=27): real 0.50, null 0.50, p=0.55 -> no signal, but proven
    calibrated. Positive controls (planted signal): real ~0.97-0.99, p<0.05
    -> correctly detects real signal. The instrument works both directions.
============================================================================
"""

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


# ============================================================
# SETTINGS  (everything you might want to tune lives here)
# ============================================================

RANDOM_STATE = 42                    # fixed seed -> reproducible runs
FEATURE_COUNTS = [10, 20, 50, 100]   # candidate feature counts the inner loop picks from
METHODS = ["mi", "rf"]               # candidate selection methods (also: "rfe", "shap" - slower)
MAX_OUTER_FOLDS = 5                  # outer CV fold cap; auto-shrinks for small/imbalanced data
MAX_INNER_FOLDS = 3                  # inner CV fold cap; auto-shrinks per training fold
LOO_THRESHOLD = 12                   # below this many samples -> use Leave-One-Out
N_PERMUTATIONS = 30                  # label shuffles for the gate (min p-value = 1/(N+1))


# ============================================================
# PLUG YOUR DATA HERE
#
# Replace the body of load_data() with your own loading code.
# CONTRACT:
#   X -> pandas DataFrame, rows = samples, columns = features
#        (column names should be strings, e.g. gene IDs)
#   y -> pandas Series of binary labels (0 / 1), aligned to X's rows
# Example for a generic CSV dataset:
#   X = pd.read_csv("features.csv", index_col=0)
#   y = pd.read_csv("labels.csv", index_col=0)["label"]
#   return X, y, "MY DATASET NAME"
# ============================================================

def load_data():
    from circRNA_PreProcess_for_Sex_Prediction import load_circrna_data
    data = load_circrna_data(
        base_path="/Users/matedort/PTSD_Project/data", scale=False)
    X = pd.DataFrame(data["X"])
    y = pd.Series(data["y"])
    return X, y, "SEX PREDICTION (circRNA sandbox)"


# ============================================================
# SMALL HELPERS
# ============================================================

def safe_auc(y_true, y_prob):
    """AUC, but return 0.5 (chance) if a fold ended up with only one class,
    which would otherwise crash roc_auc_score."""
    if len(set(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


# ============================================================
# FOLD AUTO-SCALING  (makes it plug-in for any sample size)
#
# Stratified k-fold needs at least k members in the smallest class.
# So we cap the fold count at the smallest class size, and drop to
# Leave-One-Out when the sample is tiny or a class is too small.
# Same code therefore runs on 8 samples or 8000.
# ============================================================

def make_outer_cv(y):
    y = pd.Series(y)
    n = len(y)
    min_class = int(y.value_counts().min())
    if n < LOO_THRESHOLD or min_class < 2:
        return LeaveOneOut(), "LeaveOneOut"
    k = min(MAX_OUTER_FOLDS, min_class)
    return (StratifiedKFold(n_splits=k, shuffle=True,
                            random_state=RANDOM_STATE), f"{k}-fold")


def inner_n_splits(y_tr):
    min_class = int(pd.Series(y_tr).value_counts().min())
    return max(2, min(MAX_INNER_FOLDS, min_class))


# ============================================================
# FEATURE SELECTION METHODS
# Same contract everywhere: take (X, y, top_n) -> list of feature names.
# To add a method (e.g. Lasso), write one function with this signature
# and register it in FEATURE_FUNCS below. Nothing else changes.
# ============================================================

def mi_features(X, y, top_n):
    """FILTER method: score each feature by mutual information with the label,
    keep the top_n. Fast, model-free."""
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return pd.Series(mi, index=X.columns)\
        .sort_values(ascending=False)\
        .head(top_n)\
        .index.tolist()


def rfe_features(X, y, top_n):
    """WRAPPER method: train logistic regression, drop weakest feature,
    repeat until top_n remain. Slower, model-aware."""
    lr = LogisticRegression(max_iter=5000, solver="saga",
                            random_state=RANDOM_STATE)
    rfe = RFE(lr, n_features_to_select=top_n, step=0.1)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()


def rf_features(X, y, top_n):
    """EMBEDDED method: a random forest ranks features by importance;
    keep the top_n. Captures interactions a filter misses."""
    rf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    return imp.sort_values(ascending=False).head(top_n).index.tolist()


def shap_features(X_train, y, top_n):
    """EMBEDDED method: rank features by SHAP importance of an XGBoost model.
    Theoretically strong but slow and needs more data. NOTE: only ever pass
    the TRAINING slice as X_train - never train+test - or you leak."""
    from xgboost import XGBClassifier   # heavy deps, imported only if used
    import shap

    xgb = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_train, y)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_train)
    importance = np.abs(shap_values).mean(axis=0)
    return pd.Series(importance, index=X_train.columns)\
        .sort_values(ascending=False).head(top_n).index.tolist()


FEATURE_FUNCS = {
    "mi": mi_features,
    "rfe": rfe_features,
    "rf": rf_features,
    "shap": shap_features,
}


# ============================================================
# INNER SCORER  (the "CHOOSE" yardstick)
#
# Given a training chunk and one candidate (method, top_n), run a small
# cross-validation INSIDE that chunk and return a pooled AUC. This is how
# the inner loop compares combos. It NEVER sees the outer test fold.
# ============================================================

def inner_score(X_tr, y_tr, top_n, method, log=False):
    k = inner_n_splits(y_tr)
    inner_cv = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    if log:
        print(f"        [inner] {method} top_n={top_n} "
              f"-> {k}-fold CV on {len(y_tr)} train samples")

    oof_true, oof_prob = [], []   # pool held-out predictions across inner folds
    for j, (tr_idx, val_idx) in enumerate(inner_cv.split(X_tr, y_tr), 1):
        X_in_tr, y_in_tr = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_in_val, y_in_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]

        features = FEATURE_FUNCS[method](X_in_tr, y_in_tr, top_n)  # train only

        scaler = StandardScaler()
        X_in_tr_s = scaler.fit_transform(X_in_tr[features])
        X_in_val_s = scaler.transform(X_in_val[features])

        svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        svm.fit(X_in_tr_s, y_in_tr)

        oof_true.extend(y_in_val.tolist())
        oof_prob.extend(svm.predict_proba(X_in_val_s)[:, 1].tolist())
        if log:
            print(f"          inner fold {j}/{k}: {len(features)} features, "
                  f"{len(y_in_val)} held-out")

    auc = safe_auc(oof_true, oof_prob)
    if log:
        print(f"        [inner] {method} top_n={top_n} pooled AUC = {auc:.4f}")
    return auc


# ============================================================
# NESTED CV  (outer = JUDGE, inner = CHOOSE)
# Returns (honest pooled AUC, per-fold choices).
# ============================================================

def nested_cv(X, y, verbose=True):
    outer_cv, outer_name = make_outer_cv(y)
    n_combos = len(METHODS) * len(FEATURE_COUNTS)

    if verbose:
        n_outer = outer_cv.get_n_splits(X, y)
        print(f"[setup] outer strategy: {outer_name} ({n_outer} folds) "
              f"on {len(y)} samples")
        print(f"[setup] inner search: {len(METHODS)} methods x "
              f"{len(FEATURE_COUNTS)} counts = {n_combos} combos per fold")

    oof_true, oof_prob = [], []   # pool held-out predictions across outer folds
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]   # SEALED test fold

        if verbose:
            print(f"\n--- Outer fold {fold} ({outer_name}): "
                  f"train={len(y_tr)}, sealed test={len(y_te)} ---")
            print(f"   [CHOOSE] searching {n_combos} (method, top_n) combos...")

        # ---- CHOOSE: score every combo on the training part only ----
        scores = {}
        for method in METHODS:
            for top_n in FEATURE_COUNTS:
                scores[(method, top_n)] = inner_score(
                    X_tr, y_tr, top_n, method, log=verbose)

        best_method, best_top_n = max(scores, key=scores.get)
        best_inner_auc = scores[(best_method, best_top_n)]

        if verbose:
            for method in METHODS:
                row = "   ".join(
                    f"{tn}:{scores[(method, tn)]:.3f}" for tn in FEATURE_COUNTS)
                print(f"   {method:<5} {row}")
            print(f"   [CHOSE] {best_method} @ top_n={best_top_n} "
                  f"(inner AUC={best_inner_auc:.4f})")

        # ---- JUDGE: re-select on full train part, predict on sealed test ----
        features = FEATURE_FUNCS[best_method](X_tr, y_tr, best_top_n)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[features])
        X_te_s = scaler.transform(X_te[features])
        svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        svm.fit(X_tr_s, y_tr)

        oof_true.extend(y_te.tolist())
        oof_prob.extend(svm.predict_proba(X_te_s)[:, 1].tolist())

        fold_records.append({
            "fold": fold,
            "chosen_method": best_method,
            "chosen_top_n": best_top_n,
            "inner_auc": best_inner_auc,
        })
        if verbose:
            print(f"   [JUDGE] scored sealed fold; "
                  f"{len(oof_true)} held-out predictions pooled so far")

    overall_auc = safe_auc(oof_true, oof_prob)   # ONE honest AUC over the pool
    if verbose:
        print(f"\n[result] pooled honest AUC over all {len(oof_true)} "
              f"held-out predictions = {overall_auc:.4f}")
    return overall_auc, fold_records


def procedure_pooled_auc(X, y, verbose=False):
    """Run the whole nested-CV procedure once, return its POOLED honest AUC
    (concatenated out-of-fold predictions, not the mean of per-fold AUCs)."""
    auc, _ = nested_cv(X, y, verbose=verbose)
    return auc


# ============================================================
# PERMUTATION GATE  (the "CERTIFY" lie-detector)
#
# Shuffle the labels (break the real X<->y link, keep X intact), re-run the
# ENTIRE procedure, build a "luck" distribution, compare the real score to it.
# ============================================================

def permutation_gate(X, y, n_permutations=N_PERMUTATIONS):
    real, fold_records = nested_cv(X, y, verbose=True)
    print(f"\n[gate] real procedure pooled AUC = {real:.4f}")
    print(f"[gate] running {n_permutations} label-shuffled permutations...")

    null_scores = []
    for i in range(1, n_permutations + 1):
        # independent, reproducible shuffle per permutation
        rng = np.random.default_rng(RANDOM_STATE + i)
        y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        s = procedure_pooled_auc(X, y_perm, verbose=False)
        null_scores.append(s)
        print(f"   perm {i}/{n_permutations}  null AUC = {s:.4f}")

    arr = np.array(null_scores, dtype=float)
    # Phipson-Smyth correction: +1 in numerator and denominator avoids p=0 and
    # is conservative for small N. With N=30 the minimum p is 1/31 ~ 0.032.
    p_value = (np.sum(arr >= real) + 1) / (n_permutations + 1)
    return real, arr.mean(), p_value, fold_records


# ============================================================
# VERDICT + RECOMMENDATION  (turns numbers into a decision)
# ============================================================

def recommend_settings(fold_records):
    """Across the outer folds, which (method, top_n) was chosen most often,
    and how stable was that choice?"""
    combos = [(r["chosen_method"], r["chosen_top_n"]) for r in fold_records]
    (method, top_n), freq = Counter(combos).most_common(1)[0]
    return method, top_n, freq, len(combos)


def print_verdict(title, X, y, real, null_mean, p_value, fold_records):
    bar = "=" * 60
    print(f"\n{bar}\nVERDICT — {title}\n{bar}")

    # 1) calibration check
    calibrated = abs(null_mean - 0.5) <= 0.08
    print(f"\n1. CALIBRATION (is the instrument honest?)")
    print(f"   null (shuffled-label) AUC = {null_mean:.4f}  "
          f"[want ~0.50]  ->  {'OK, calibrated' if calibrated else 'WARNING: possible leakage'}")

    # 2) signal check
    signal = (p_value < 0.05) and (real > null_mean)
    print(f"\n2. SIGNAL (is the result real, not luck?)")
    print(f"   real honest AUC = {real:.4f} | null = {null_mean:.4f} | "
          f"gap = {real - null_mean:+.4f} | p = {p_value:.4f}")
    print(f"   ->  {'SIGNAL DETECTED (p < 0.05)' if signal else 'NO SIGNAL above chance (p >= 0.05)'}")

    # 3) recommended settings
    method, top_n, freq, n_folds = recommend_settings(fold_records)
    chosen = [f"{r['chosen_method']}@{r['chosen_top_n']}" for r in fold_records]
    print(f"\n3. RECOMMENDED SETTINGS (for the real model)")
    print(f"   per-fold choices : {chosen}")
    print(f"   most common      : {method} @ top_n={top_n} "
          f"(chosen in {freq}/{n_folds} folds)")
    print(f"   choice stability : {freq}/{n_folds} "
          f"{'(stable)' if freq / n_folds >= 0.6 else '(UNSTABLE - n likely too small)'}")

    # 4) what to do next
    print(f"\n4. WHAT TO DO NEXT")
    if not calibrated:
        print("   STOP. Null is not ~0.50 - the pipeline is leaking. "
              "Fix before trusting any number.")
    elif signal:
        feats = FEATURE_FUNCS[method](X, y, top_n)
        print(f"   PASS. Train the real model with method='{method}', "
              f"top_n={top_n} on all data.")
        print(f"   The {len(feats)} selected features (first 10): {feats[:10]}")
        print(f"   Then deploy/report that model with AUC ~{real:.2f} as the "
              f"honest expectation.")
    else:
        print("   NO-GO. No signal above chance at this sample size. Do NOT "
              "report a model.")
        print("   Options: get more samples, rethink the target, or try more "
              "feature-selection methods (still gated).")
    print(bar)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    X, y, title = load_data()

    print("=" * 60)
    print(f"FEATURE-EVAL HARNESS — {title}")
    print("=" * 60)
    print(f"X shape: {X.shape}  (samples x features)")
    print(f"y distribution: {y.value_counts().to_dict()}")
    print(f"methods: {METHODS} | feature counts: {FEATURE_COUNTS} | "
          f"permutations: {N_PERMUTATIONS}")

    real, null_mean, p_value, fold_records = permutation_gate(X, y)

    print_verdict(title, X, y, real, null_mean, p_value, fold_records)
