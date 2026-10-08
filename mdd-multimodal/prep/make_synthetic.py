"""Generate a fake cohort in data/ so the whole pipeline can be run without the server.

Nothing here is real. It exists so that a change to data.py / site.py / train.py can
be checked end-to-end in seconds, including the parts that are easy to get silently
wrong: nested QC tiers, subjects missing sFNC, HAMD on a minority of MDD only, and
a site effect big enough that site.mode actually has something to remove.

    python prep/make_synthetic.py --n 400 --out data
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rois", type=int, default=30)
    ap.add_argument("--n-comp", type=int, default=20)
    ap.add_argument("--n-sites", type=int, default=6)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n = a.n
    ids = np.array([f"sub-{i:05d}" for i in range(n)], dtype=object)

    grp = rng.choice(["MDD", "HC", "Bipolar", "Schizophrenia"], n, p=[.47, .38, .06, .09])
    code = pd.Series(grp).map({"MDD": 1, "HC": 2, "Bipolar": 3, "Schizophrenia": 4})
    site = rng.integers(0, a.n_sites, n)

    # QC tiers, strictly nested, with SCZ never dropped (as in the real data)
    in_clean = (rng.random(n) < 0.72) | (grp == "Schizophrenia")
    in_super = in_clean & (rng.random(n) < 0.96)

    # HAMD, shaped like the real cohort: MDD only, HAMD3 NESTED inside HAMD17,
    # and the HAMD17-without-HAMD3 subjects concentrated in a few sites (in the
    # real data three sites recorded the total but never item 3). Without this
    # structure the target-dependent availability logic in src/data.py is not
    # exercised by any test.
    no_item3_sites = set(range(min(2, a.n_sites)))          # sites that skip item 3
    has_17 = (grp == "MDD") & (rng.random(n) < 0.70)
    has_3 = has_17 & ~np.isin(site, list(no_item3_sites)) & (rng.random(n) < 0.85)
    h17 = np.where(has_17, rng.integers(5, 35, n), np.nan)
    h3 = np.where(has_3, rng.integers(0, 5, n), np.nan)

    df = pd.DataFrame(dict(
        id=ids, file_index=np.arange(n), filename=[f"{s}.nii.gz" for s in ids],
        group_label=grp, group_code=code, site=[f"site{s:02d}" for s in site],
        age=rng.integers(18, 65, n), sex=rng.integers(0, 2, n),
        HAMDTotal17=h17, HAMD3=h3,
        in_clean=in_clean, in_super_clean=in_super))
    df["clean_row"] = np.where(in_clean, np.cumsum(in_clean) - 1, np.nan)
    df["super_clean_row"] = np.where(in_super, np.cumsum(in_super) - 1, np.nan)

    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "mdd_master.csv"), index=False)

    # ---- features: a real group effect, plus a larger site effect on top -------
    def block(d, eff):
        sig = ((grp == "MDD").astype(float) * eff)[:, None] * rng.normal(0, 1, (1, d))
        site_eff = rng.normal(0, 1.5, (a.n_sites, d))[site]
        return (rng.normal(0, 1, (n, d)) + sig + site_eff).astype(np.float32)

    R = a.n_rois
    struct = {m: block(R, e) for m, e in
              (("GM", .45), ("WM", .15), ("CSF", .25), ("FALFF", .35),
               ("FALFF_globalC", .33))}
    np.savez_compressed(os.path.join(a.out, "struct_features.npz"),
                        ids=ids, roi_labels=np.array([f"roi{i}" for i in range(R)], dtype=object),
                        TIV=rng.normal(1.4, .15, (n, 1)).astype(np.float32), **struct)

    # sFNC only for the clean tier — this is the missing-modality case
    k = a.n_comp * (a.n_comp - 1) // 2
    sf = block(k, .40)
    np.savez_compressed(os.path.join(a.out, "sfnc_features.npz"),
                        ids=ids[in_clean], sFNC=sf[in_clean])

    np.savez_compressed(os.path.join(a.out, "dfnc_features.npz"),
                        ids=ids[in_clean], dfnc=block(40, .30)[in_clean],
                        feature_names=np.array([f"f{i}" for i in range(40)], dtype=object))

    print(f"wrote synthetic cohort to {a.out}/")
    print(f"  n={n}  clean={int(in_clean.sum())}  super_clean={int(in_super.sum())}")
    print(f"  HAMD17: {int(has_17.sum())}   HAMD3: {int(has_3.sum())}   "
          f"(HAMD17 without item 3: {int((has_17 & ~has_3).sum())}, all MDD)")
    print(f"  sFNC present for {int(in_clean.sum())}/{n} — the rest are masked")


if __name__ == "__main__":
    main()
