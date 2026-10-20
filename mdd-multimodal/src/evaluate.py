"""Read the JSON written by train.py and say what it means.

Two things happen here.

1. Fold-level summary. Mean AUC, its spread across folds x repeats, and a
   comparison against any other run files you pass. A difference smaller than
   roughly twice the fold SD is not a result.

2. Linear mixed effects on the fold results. The proposal's stated statistical
   approach (pp. 16, 24, 32) is LME with a random effect. Here the random effect
   is the repeat (folds within a repeat share a seed and a data partition, so
   they are not independent), and the fixed effect is whatever run-level factor
   you are comparing -- site mode, fusion mode, modality set. That is the honest
   test of "does site correction change performance", rather than eyeballing two
   mean AUCs.

    python -m src.evaluate runs/*.json
    python -m src.evaluate runs/*.json --factor site.mode
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def _dig(d, dotted, default=None):
    node = d
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def load_runs(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            res = json.load(f)
        cfg = res.get("config", {})
        for r in res.get("rows", []):
            rows.append(dict(
                run=os.path.basename(p),
                repeat=r.get("repeat", 0),
                fold=r.get("fold", 0),
                auc=r.get("auc"),
                n_train=r.get("n_train"),
                n_test=r.get("n_test"),
                cohort=_dig(cfg, "data.cohort"),
                modalities=",".join(_dig(cfg, "data.modalities", []) or []),
                site_mode=_dig(cfg, "site.mode"),
                fusion=_dig(cfg, "model.fusion.mode"),
                target=_dig(cfg, "label.target"),
                hamd_mode=_dig(cfg, "label.hamd_mode"),
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no fold rows found in those files")
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    return df.dropna(subset=["auc"])


FACTORS = {"site.mode": "site_mode", "model.fusion.mode": "fusion",
           "data.cohort": "cohort", "data.modalities": "modalities",
           "label.hamd_mode": "hamd_mode", "run": "run"}


def summarise(df):
    g = (df.groupby(["run", "target", "cohort", "modalities", "site_mode", "fusion"])
           .agg(folds=("auc", "size"), auc_mean=("auc", "mean"),
                auc_sd=("auc", "std"), n_test=("n_test", "mean"))
           .reset_index().sort_values("auc_mean", ascending=False))
    g["auc_sd"] = g["auc_sd"].fillna(0.0)
    return g


def mixed_model(df, factor_col):
    """AUC ~ factor + (1 | repeat within run). Falls back to OLS if needed."""
    if df[factor_col].nunique() < 2:
        return f"  (only one level of '{factor_col}' — nothing to compare)"
    d = df.copy()
    d["grp"] = d["run"].astype(str) + "_r" + d["repeat"].astype(str)
    try:
        import warnings
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            # "MLE on the boundary" just means the repeat-level variance is ~0,
            # i.e. repeats agree. That is information, not a failure.
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f"auc ~ C({factor_col})", d, groups=d["grp"]).fit(reml=False)
        head = ("  linear mixed effects:  auc ~ C(%s) + (1 | repeat)\n" % factor_col)
        return head + "\n".join("    " + x for x in str(m.summary().tables[1]).splitlines())
    except Exception as e:                                   # noqa: BLE001
        import statsmodels.formula.api as smf
        m = smf.ols(f"auc ~ C({factor_col})", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["grp"]})
        return (f"  (MixedLM did not converge: {e}; "
                "using OLS with cluster-robust SE by repeat)\n"
                + "\n".join("    " + x for x in str(m.summary().tables[1]).splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="result JSONs (globs are fine)")
    ap.add_argument("--factor", default=None,
                    help="run-level factor to test: " + " | ".join(FACTORS))
    ap.add_argument("--target", default=None,
                    help="keep only runs with this label.target (diagnosis|hamd17|hamd3|site)")
    ap.add_argument("--csv", default=None, help="write the fold table here")
    args = ap.parse_args()

    paths = sorted({p for pat in args.runs for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("no files matched")
    df = load_runs(paths)
    if args.target:
        df = df[df["target"] == args.target]
        if df.empty:
            raise SystemExit(f"no runs with target '{args.target}'")

    print("\n" + "=" * 78)
    print(f"{len(paths)} run(s), {len(df)} folds")
    print("=" * 78)
    s = summarise(df)
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(s.to_string(index=False))

    # within-run SD, not the pooled SD: pooling across different targets measures
    # the gap between experiments, not the noise inside one.
    sd = float(df.groupby("run")["auc"].std().median())
    print(f"\n  typical within-run fold SD = {sd:.4f}. "
          f"Differences under ~{2*sd:.4f} should not be called a result.")

    if df["target"].nunique() > 1:
        print("  WARNING: these runs mix targets "
              f"({', '.join(sorted(df['target'].dropna().unique()))}). "
              "An AUC for diagnosis and an AUC for hamd17 are not comparable — "
              "pass --target to compare like with like.")

    if args.factor:
        col = FACTORS.get(args.factor, args.factor)
        if col not in df.columns:
            raise SystemExit(f"unknown factor '{args.factor}' ({' | '.join(FACTORS)})")
        print("\n" + mixed_model(df, col))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n  fold table -> {args.csv}")


if __name__ == "__main__":
    main()
