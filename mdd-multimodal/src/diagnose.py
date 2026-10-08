"""Confound diagnostics. Run this BEFORE believing any model result.

Derived from our own data rather than taken on trust. Four questions, in order of
how badly a bad answer would invalidate everything downstream.

1. IS DIAGNOSIS CONFOUNDED WITH SITE?
   The headline number is the site-only baseline: cross-validated AUC for
   predicting diagnosis from the one-hot site vector and nothing else. No
   imaging, no features -- just "which scanner was this". If that comes back at
   0.68, then a multimodal model scoring 0.72 has added 0.04 of brain signal to
   0.68 of scanner, and reporting 0.72 as a neuroimaging finding is wrong. Every
   later AUC should be read against this floor, not against 0.5.

2. IS HAMD CONFOUNDED WITH SITE?
   HAMD exists for ~660 subjects. If those came from a handful of sites, then
   "predict HAMD" quietly means "predict subjects from those sites", and all
   three hamd_modes inherit it. The holdout mode is worst affected: it trains on
   subjects WITHOUT HAMD and tests on subjects WITH it, so if HAMD availability
   tracks site, the train/test split is a site split wearing a disguise.

3. HOW MUCH HAMD VARIANCE IS BETWEEN SITES rather than between people?
   The ICC from a random-intercept model. High ICC means severity scores are
   partly a property of the rating site, not the patient.

4. HOW LOUD IS SITE IN THE FEATURES THEMSELVES?
   Per-feature eta-squared by site, summarised per modality. Tells you which
   modality carries the most scanner signature, which is the one to watch.

    python -m src.diagnose --config conf/experiments/hc_vs_mdd_sfnc.yaml
    python -m src.diagnose --config ... --no-features     # design only, instant
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from . import config, data as data_mod


def _rule(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def cramers_v(tab):
    """Effect size for a contingency table: 0 = independent, 1 = deterministic."""
    from scipy.stats import chi2_contingency
    chi2, p, _, _ = chi2_contingency(tab)
    n = tab.values.sum()
    r, k = tab.shape
    phi2 = chi2 / n
    denom = min(r - 1, k - 1)
    return (np.sqrt(phi2 / denom) if denom > 0 else np.nan), p


def site_only_auc(site, y, seed=0, n_folds=10):
    """Cross-validated AUC for predicting y from one-hot site alone."""
    levels = np.unique(site)
    X = np.zeros((len(site), len(levels)), dtype=np.float32)
    for j, s in enumerate(levels):
        X[site == s, j] = 1.0
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    n_folds = min(n_folds, int(np.bincount(y).min()))
    if n_folds < 2:
        return np.nan
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    p = cross_val_predict(
        LogisticRegression(max_iter=2000, C=1.0), X, y, cv=skf, method="predict_proba")
    return float(roc_auc_score(y, p[:, 1]))


def eta_squared(A, groups):
    """Per-column share of variance explained by group. Returns (n_features,)."""
    A = np.asarray(A, dtype=np.float64)
    grand = A.mean(axis=0)
    ss_tot = ((A - grand) ** 2).sum(axis=0)
    ss_bet = np.zeros(A.shape[1])
    for g in np.unique(groups):
        sel = groups == g
        if sel.sum() == 0:
            continue
        ss_bet += sel.sum() * (A[sel].mean(axis=0) - grand) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > 0, ss_bet / ss_tot, np.nan)


def icc_random_intercept(values, groups):
    """ICC from a random-intercept model: between-group variance / total."""
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    v, g = v[ok], np.asarray(groups)[ok]
    if len(v) < 10 or len(np.unique(g)) < 2:
        return np.nan, len(v)
    try:
        import statsmodels.formula.api as smf
        import warnings
        d = pd.DataFrame(dict(y=v, grp=g))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("y ~ 1", d, groups=d["grp"]).fit(reml=True)
        between = float(m.cov_re.iloc[0, 0])
        within = float(m.scale)
        return between / (between + within), len(v)
    except Exception:                                        # noqa: BLE001
        # fall back to the one-way ANOVA estimator
        grand = v.mean()
        ms_b = sum(len(v[g == k]) * (v[g == k].mean() - grand) ** 2
                   for k in np.unique(g)) / max(len(np.unique(g)) - 1, 1)
        ms_w = sum(((v[g == k] - v[g == k].mean()) ** 2).sum()
                   for k in np.unique(g)) / max(len(v) - len(np.unique(g)), 1)
        n_bar = len(v) / len(np.unique(g))
        return max((ms_b - ms_w) / (ms_b + (n_bar - 1) * ms_w + 1e-12), 0.0), len(v)


def run(cfg, with_features=True):
    coh = data_mod.build(cfg)
    site_names = coh.meta["site_names"]
    site = coh.site
    grp = coh.groups
    has_h = coh.meta["has_hamd"]
    h17 = coh.meta["hamd17"]

    # ---------------------------------------------------------------- 1
    _rule("1. DIAGNOSIS vs SITE")
    tab = pd.crosstab(pd.Series(grp, name="group"),
                      pd.Series([site_names[s] for s in site], name="site"))
    print(tab.to_string())

    v, p = cramers_v(tab)
    print(f"\n  Cramer's V = {v:.3f}   (chi-square p = {p:.3g})")
    print("    0.0-0.1 negligible | 0.1-0.3 modest | 0.3-0.5 strong | >0.5 severe")

    pure = [(site_names[s], tab.iloc[:, list(tab.columns).index(site_names[s])])
            for s in np.unique(site) if site_names[s] in tab.columns]
    single = [nm for nm, col in pure if (col > 0).sum() == 1]
    if single:
        print(f"  {len(single)} site(s) contribute ONE group only: {', '.join(single[:10])}")
        print("    For those subjects, site and diagnosis are the same variable.")

    groups_cfg = cfg["data"]["groups"]
    if len(groups_cfg) == 2:
        y = (grp == groups_cfg[0]).astype(int)
        auc0 = site_only_auc(site, y)
        print(f"\n  >> SITE-ONLY BASELINE AUC = {auc0:.4f}")
        print("     Predicting diagnosis from the scanner alone, no imaging at all.")
        if not np.isnan(auc0):
            if auc0 >= 0.65:
                print("     This is high. Read every model AUC against this number, not 0.5;")
                print("     and run leave-one-site-out (train.cv=loso) as the real test.")
            elif auc0 >= 0.55:
                print("     Modest but not nothing. Report it alongside any model AUC.")
            else:
                print("     Low — diagnosis is reasonably balanced across sites.")

    # ---------------------------------------------------------------- 2
    _rule("2. HAMD AVAILABILITY vs SITE")
    n_h = int(has_h.sum())
    print(f"  {n_h} of {len(coh)} subjects have HAMD17+HAMD3")
    if n_h == 0:
        print("  (none in this cohort — skipping)")
    else:
        rows = []
        for s in np.unique(site):
            sel = site == s
            rows.append(dict(site=site_names[s], n=int(sel.sum()),
                             n_hamd=int((sel & has_h).sum()),
                             pct=100.0 * (sel & has_h).sum() / max(sel.sum(), 1)))
        t = pd.DataFrame(rows).sort_values("n_hamd", ascending=False)
        print(t.to_string(index=False, float_format=lambda x: f"{x:5.1f}"))

        contributing = int((t["n_hamd"] > 0).sum())
        top = t.head(3)["n_hamd"].sum()
        print(f"\n  {contributing} of {len(t)} sites contributed any HAMD at all")
        print(f"  top 3 sites hold {top}/{n_h} = {100.0*top/n_h:.0f}% of all HAMD subjects")
        auc_h = site_only_auc(site, has_h.astype(int))
        print(f"\n  >> AUC for predicting HAMD-AVAILABILITY from site alone = {auc_h:.4f}")
        if not np.isnan(auc_h) and auc_h >= 0.75:
            print("     Having a HAMD score is largely a property of the SITE.")
            print("     hamd_mode=holdout is then a site split in disguise: it trains on")
            print("     subjects without HAMD and tests on subjects with it, so the two")
            print("     halves differ by scanner as well as by label. Prefer direct or")
            print("     transfer, and report loso alongside.")

        # -------------------------------------------------------------- 3
        _rule("3. HAMD SEVERITY vs SITE")
        icc, n_used = icc_random_intercept(h17, site)
        print(f"  ICC(HAMD17 | site) = {icc:.3f}   (n = {n_used})")
        print("    the share of severity variance that is between sites, not between people")
        if icc >= 0.10:
            print("    >= 0.10 is substantial for a clinical score. Severity is partly a")
            print("    property of the rating site. Keep site.mode=random_effect for any")
            print("    HAMD target, and treat a raw severity AUC with suspicion.")
        sub = pd.DataFrame(dict(site=[site_names[s] for s in site], h=h17))[has_h]
        if len(sub):
            g = sub.groupby("site")["h"].agg(["count", "mean", "std"]).sort_values(
                "mean", ascending=False)
            print("\n  HAMD17 by site (contributing sites only):")
            print(g.to_string(float_format=lambda x: f"{x:6.2f}"))

    # ---------------------------------------------------------------- 4
    if with_features and coh.X:
        _rule("4. HOW LOUD IS SITE IN THE FEATURES")
        print("  per-feature variance explained by site (eta-squared)\n")
        print(f"  {'modality':<16}{'median':>9}{'90th pct':>10}{'max':>9}"
              f"{'% > 0.10':>10}")
        for m in coh.X:
            real = coh.mask[m]
            if real.sum() < 10:
                continue
            e = eta_squared(coh.X[m][real], site[real])
            e = e[~np.isnan(e)]
            if not len(e):
                continue
            print(f"  {m:<16}{np.median(e):>9.3f}{np.percentile(e, 90):>10.3f}"
                  f"{e.max():>9.3f}{100.0*(e > 0.10).mean():>9.1f}%")
        print("\n  A modality where most features sit above 0.10 is carrying a strong")
        print("  scanner signature. That is what site.mode is for — and what")
        print("  site_check.yaml measures end-to-end.")

    _rule("WHAT TO DO WITH THIS")
    print("""  - Quote the site-only baseline next to every model AUC you report.
  - Run train.cv=loso. Random k-fold puts subjects from the same scanner in
    both train and test, so it cannot detect site leakage; leave-one-site-out
    can. A large gap between the two is the size of the problem.
  - If HAMD availability is site-driven, say so when reporting hamd_mode results
    rather than letting the number stand alone.""")
    return 0


def main():
    cfg, _ = config.load(extra_args=None)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--no-features", action="store_true")
    known, _ = ap.parse_known_args()
    print(config.describe(cfg))
    return run(cfg, with_features=not known.no_features)


if __name__ == "__main__":
    sys.exit(main())
