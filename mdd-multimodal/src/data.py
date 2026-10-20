"""Cohort selection, modality assembly, and missing-modality handling.

The three facts this module exists to respect
---------------------------------------------
1. The QC tiers are strictly nested:  ICA-105 (2,426) < ICA-53 (2,426+) < all (3,525).
   Every subject has the structural volumes; sFNC is the scarce one. Requiring
   sFNC discards 999 subjects (490 MDD, 459 HC, 50 BD, 0 SCZ).

2. Because of that, complete-case analysis silently throws away 28% of the sample.
   `allow_missing_modalities` keeps them: each subject carries a per-modality
   availability mask, missing streams are zero-filled, and the model is told which
   are real so it does not learn from the padding.

3. HAMD exists for ~660 subjects only, all MDD. Three ways to use that
   (`label.hamd_mode`): direct / transfer / holdout.

Row alignment
-------------
The GIFT MATs have no subject IDs — `subjects` inside them is a 1..N counter. The
join is on `clean_row` / `super_clean_row` in mdd_master.csv, which are 0-based and
set only on their tier. prep/extract_sfnc.py does that join and writes IDs, so by
the time features reach here everything is keyed by subject ID.
"""

import os
import numpy as np
import pandas as pd

TIER = {"all": None, "sfnc": "in_clean", "super_clean": "in_super_clean"}
HAMD3_SPLITS = {
    "0|1-4":   (lambda h: (h >= 1).astype(int), None),
    "0-1|2-4": (lambda h: (h >= 2).astype(int), None),
    "0-2|3-4": (lambda h: (h >= 3).astype(int), None),
    "0-3|4":   (lambda h: (h >= 4).astype(int), None),
    "0|4":     (lambda h: (h == 4).astype(int), lambda h: (h == 0) | (h == 4)),
}


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def hamd_available(target, h17, h3):
    """Which subjects count as labelled, for THIS target.

    Measured on the real cohort (n = 3,525):

        HAMD17 present            1,394        with sFNC:  984
        HAMD3 (item 3) present      989        with sFNC:  660
        HAMD3 without HAMD17            0   -- HAMD3 is nested inside HAMD17
        HAMD17 without HAMD3        405   -- almost all from 3 sites: IS001
                                             (254), IS008 (47), IS018 (34)

    Requiring BOTH scores regardless of target -- which this function used to do
    -- silently cost 324 subjects on every HAMD17 run, a third of the available
    sample, for a score that run never uses. Availability now follows the target.

    Note for interpretation: HAMD3 is the suicide item, and three whole sites
    recorded HAMD17 but never item 3. So a HAMD3 target does not just have fewer
    subjects, it has fewer SITES -- see src/diagnose.py section 2.
    """
    if target == "hamd17":
        return ~np.isnan(h17)
    if target == "hamd3":
        return ~np.isnan(h3)
    return ~np.isnan(h17) & ~np.isnan(h3)


class Cohort:
    """Assembled features, labels, site and masks for one experiment."""

    def __init__(self, ids, X, mask, y, site, groups, meta):
        self.ids = ids          # (N,)
        self.X = X              # {modality: (N, d)}  zero-filled where absent
        self.mask = mask        # {modality: (N,) bool}  True = real data
        self.y = y              # (N,) int, or float for regression
        self.site = site        # (N,) int codes
        self.groups = groups    # (N,) group_label
        self.meta = meta        # dict of extras (hamd17, hamd3, ...)

    def __len__(self):
        return len(self.ids)

    def subset(self, sel):
        return Cohort(self.ids[sel], {k: v[sel] for k, v in self.X.items()},
                      {k: v[sel] for k, v in self.mask.items()},
                      self.y[sel], self.site[sel], self.groups[sel],
                      {k: (v[sel] if isinstance(v, np.ndarray) else v)
                       for k, v in self.meta.items()})

    def summary(self):
        out = [f"  n = {len(self)}"]
        vc = pd.Series(self.groups).value_counts()
        out.append("  groups: " + ", ".join(f"{k} {v}" for k, v in vc.items()))
        out.append(f"  sites : {len(np.unique(self.site))}")
        for m in self.X:
            n = int(self.mask[m].sum())
            out.append(f"    {m:<16}{self.X[m].shape[1]:>6} dims   present {n}/{len(self)}"
                       + ("" if n == len(self) else "   <- masked"))
        if self.y is not None and self.y.dtype.kind in "iu":
            cnt = np.bincount(self.y)
            out.append(f"  label balance: {dict(enumerate(cnt))}")
        return "\n".join(out)


def load_master(path):
    if not os.path.exists(path):
        raise SystemExit(f"master CSV not found: {path}")
    df = pd.read_csv(path)
    need = ["id", "group_label", "site", "in_clean", "in_super_clean"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"master CSV missing columns: {missing}")
    return df


def _load_npz(path, want, ids_wanted):
    """Return {modality: (N,d)} aligned to ids_wanted, plus availability masks."""
    X, M = {}, {}
    if not want:
        return X, M
    if not os.path.exists(path):
        print(f"    (missing {path} — skipping {want})")
        return X, M
    z = np.load(path, allow_pickle=True)
    if "ids" not in z:
        raise SystemExit(f"{path} has no 'ids' array; cannot align subjects")
    have_ids = np.array([str(x) for x in z["ids"]])
    pos = {s: i for i, s in enumerate(have_ids)}
    idx = np.array([pos.get(s, -1) for s in ids_wanted])
    ok = idx >= 0
    for m in want:
        if m not in z:
            continue
        A = np.asarray(z[m], dtype=np.float32)
        A = A.reshape(A.shape[0], -1)
        full = np.zeros((len(ids_wanted), A.shape[1]), dtype=np.float32)
        full[ok] = A[idx[ok]]
        # a row of all-NaN also counts as absent
        good = ok.copy()
        good[ok] &= ~np.isnan(A[idx[ok]]).all(axis=1)
        full[np.isnan(full)] = 0.0
        X[m], M[m] = full, good
    return X, M


def build(cfg):
    d, l = cfg["data"], cfg["label"]
    df = load_master(cfg["paths"]["master"])

    # ---- 1. QC tier --------------------------------------------------------
    tier_col = TIER.get(d["cohort"])
    if tier_col is None and d["cohort"] != "all":
        raise SystemExit(f"unknown cohort '{d['cohort']}' (all|sfnc|super_clean)")
    sel = (np.ones(len(df), bool) if tier_col is None
           else np.asarray(df[tier_col].map(_truthy).to_numpy(), dtype=bool).copy())
    print(f"  tier '{d['cohort']}': {sel.sum()} / {len(df)}")

    # ---- 2. diagnostic groups ---------------------------------------------
    sel &= df["group_label"].isin(d["groups"]).to_numpy()
    print(f"  after groups {d['groups']}: {sel.sum()}")

    # ---- 2b. site restriction ---------------------------------------------
    # Why this exists: the structural modalities carry a scanner signature so
    # strong (median site eta^2 0.77 GM / 0.91 CSF / 0.87 fALFF, 100% of parcels
    # above 0.10) that a multi-site result cannot easily be told apart from a
    # site-classification result. Restricting to ONE site removes the confound by
    # construction rather than by correction: with a single scanner there is no
    # between-site variance left to leak, so whatever AUC remains is signal.
    #
    # The cost is honest and must be stated with any such result: a single-site
    # model is not shown to generalise to a new scanner. It answers "is there
    # signal here at all", not "does this transfer".
    sites = d.get("sites") or None
    if sites:
        sites = [str(s) for s in (sites if isinstance(sites, (list, tuple))
                                  else [sites])]
        have = set(df["site"].astype(str))
        unknown = [s for s in sites if s not in have]
        if unknown:
            raise SystemExit(
                f"data.sites lists site(s) not in the master CSV: {unknown}\n"
                f"  known sites: {', '.join(sorted(have))}")
        sel &= df["site"].astype(str).isin(sites).to_numpy()
        print(f"  after sites {sites}: {sel.sum()}")
        if sel.sum() == 0:
            raise SystemExit("no subjects left after the site filter")
        kept = df.loc[sel]
        for s in sites:
            n = int((kept["site"].astype(str) == s).sum())
            vc = kept.loc[kept["site"].astype(str) == s, "group_label"].value_counts()
            print(f"      {s}: {n}  (" +
                  ", ".join(f"{k} {v}" for k, v in vc.items()) + ")")
        if len(sites) == 1:
            print("      single site — no site confound; site.mode is inert here")

    # ---- 3. HAMD availability ---------------------------------------------
    h17 = _num(df.get("HAMDTotal17", pd.Series(np.nan, index=df.index))).to_numpy()
    h3 = _num(df.get("HAMD3", pd.Series(np.nan, index=df.index))).to_numpy()
    target = str(l["target"])
    has_hamd = hamd_available(target, h17, h3)

    if target.startswith("hamd"):
        mode = l["hamd_mode"]
        if mode == "direct":
            sel &= has_hamd
        elif mode == "transfer":
            pass          # keep everyone; train.py stages pretrain then fine-tune
        elif mode == "holdout":
            pass          # keep everyone; split is made below
        else:
            raise SystemExit(f"unknown hamd_mode '{mode}' (direct|transfer|holdout)")
        need = {"hamd17": "HAMD17", "hamd3": "HAMD3 (item 3)"}.get(target, "HAMD17+HAMD3")
        print(f"  hamd_mode={mode}, target needs {need}: "
              f"{int((sel & has_hamd).sum())} labelled, "
              f"{int((sel & ~has_hamd).sum())} without")

    sub = df[sel].reset_index(drop=True)
    ids = sub["id"].astype(str).to_numpy()

    # ---- 4. features -------------------------------------------------------
    struct_mods = [m for m in d["modalities"]
                   if m in ("GM", "WM", "CSF", "FALFF", "FALFF_globalC")]
    fnc_mods = [m for m in d["modalities"] if m in ("sFNC", "sFNC105", "spectra")]
    dfnc_mods = [m for m in d["modalities"] if m == "dfnc"]

    X, mask = {}, {}
    for path, want in ((cfg["paths"]["struct"], struct_mods),
                       (cfg["paths"]["sfnc"], fnc_mods),
                       (cfg["paths"]["dfnc"], dfnc_mods)):
        a, b = _load_npz(path, want, ids)
        X.update(a); mask.update(b)

    missing_mods = [m for m in d["modalities"] if m not in X and m not in d.get("covariates", [])]
    if missing_mods:
        raise SystemExit(f"requested modalities not found in any feature file: {missing_mods}")

    # covariates (TIV) ride along inside every stream rather than being one
    cov = None
    covs = [c for c in d.get("covariates", [])]
    if covs:
        cz, _ = _load_npz(cfg["paths"]["struct"], covs, ids)
        if cz:
            cov = np.hstack([cz[c] for c in covs if c in cz])

    # ---- 5. complete-case, if asked ---------------------------------------
    if not d["allow_missing_modalities"] and mask:
        keep = np.ones(len(ids), bool)
        for m in X:
            keep &= mask[m]
        dropped = int((~keep).sum())
        if dropped:
            print(f"  complete-case: dropping {dropped} subjects missing a modality")
        sub = sub[keep].reset_index(drop=True)
        ids = ids[keep]
        X = {k: v[keep] for k, v in X.items()}
        mask = {k: v[keep] for k, v in mask.items()}
        if cov is not None:
            cov = cov[keep]
        h17, h3, has_hamd = None, None, None

    # ---- 6. labels ---------------------------------------------------------
    grp = sub["group_label"].astype(str).to_numpy()
    site_codes, site_uniq = pd.factorize(sub["site"].astype(str))
    site = site_codes.astype(np.int64)

    h17s = _num(sub.get("HAMDTotal17", pd.Series(np.nan, index=sub.index))).to_numpy()
    h3s = _num(sub.get("HAMD3", pd.Series(np.nan, index=sub.index))).to_numpy()
    has_hamd_s = hamd_available(target, h17s, h3s)

    if target == "diagnosis":
        if len(d["groups"]) != 2:
            raise SystemExit("label.target=diagnosis needs exactly 2 entries in data.groups")
        y = (grp == d["groups"][0]).astype(np.int64)
    elif target == "site":
        y = site.copy()
    elif target == "hamd17":
        y = np.where(has_hamd_s, (h17s >= l["hamd17_thresh"]).astype(float), np.nan)
    elif target == "hamd3":
        split = l["hamd3_split"]
        if split not in HAMD3_SPLITS:
            raise SystemExit(f"unknown hamd3_split '{split}' ({list(HAMD3_SPLITS)})")
        pos_fn, filt = HAMD3_SPLITS[split]
        y = np.where(has_hamd_s, pos_fn(np.nan_to_num(h3s, nan=-1)).astype(float), np.nan)
        if filt is not None:
            y = np.where(filt(np.nan_to_num(h3s, nan=-1)), y, np.nan)
    else:
        raise SystemExit(f"unknown label.target '{target}'")

    meta = dict(hamd17=h17s, hamd3=h3s, has_hamd=has_hamd_s,
                site_names=np.array(site_uniq, dtype=object), cov=cov)
    return Cohort(ids, X, mask, y, site, grp, meta)
