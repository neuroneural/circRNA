"""Scanner / site handling — five strategies.

23 sites contributed to DIRECT Phase 2. Sites differ in scanner, sequence and
population, so an untreated model can score well by recognising the scanner. The
proposal's own statistical approach is linear mixed effects with a random effect
(pp. 16, 24, 32), which is where `random_effect` comes from.

Every strategy is fitted on TRAINING data only and applied to test — fitting on
the full sample leaks site structure into the fold and inflates the result.

  ignore        baseline. Useful precisely to measure how much site matters.
  feature       one-hot site appended to the input. Use when site is of interest,
                not when you want a site-invariant model.
  combat        ComBat-style harmonisation: remove per-site location and scale
                from each feature, keeping biological covariates out of it.
  random_effect Proposal-aligned. Per-site random intercepts are estimated on
                train with shrinkage toward the grand mean, then subtracted.
                Shrinkage is what makes it a random rather than fixed effect:
                small sites are pulled toward the global mean instead of having
                their noise removed as if it were signal.
  adversarial   Handled inside the model (a gradient-reversal head predicting
                site); this module only supplies the labels.
"""

import numpy as np

MODES = ("ignore", "feature", "combat", "random_effect", "adversarial")


class SiteHandler:
    def __init__(self, mode, adversarial_weight=0.1):
        if mode not in MODES:
            raise SystemExit(f"unknown site.mode '{mode}' ({'|'.join(MODES)})")
        self.mode = mode
        self.adversarial_weight = adversarial_weight
        self.fitted_ = {}
        self.site_levels_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_by_mod, site, mask_by_mod=None):
        self.site_levels_ = np.unique(site)
        if self.mode in ("ignore", "adversarial", "feature"):
            return self
        for m, A in X_by_mod.items():
            mk = mask_by_mod[m] if mask_by_mod else np.ones(len(A), bool)
            self.fitted_[m] = (self._fit_combat(A, site, mk) if self.mode == "combat"
                               else self._fit_random_effect(A, site, mk))
        return self

    def _fit_combat(self, A, site, mk):
        gmean = A[mk].mean(axis=0) if mk.any() else np.zeros(A.shape[1])
        gstd = A[mk].std(axis=0) if mk.any() else np.ones(A.shape[1])
        gstd[gstd == 0] = 1.0
        per = {}
        for s in self.site_levels_:
            sel = (site == s) & mk
            if sel.sum() < 2:                       # too small to estimate
                per[s] = (np.zeros(A.shape[1]), np.ones(A.shape[1]))
                continue
            mu = A[sel].mean(axis=0) - gmean
            sd = A[sel].std(axis=0) / gstd
            sd[sd == 0] = 1.0
            per[s] = (mu, sd)
        return dict(gmean=gmean, gstd=gstd, per=per)

    def _fit_random_effect(self, A, site, mk):
        """Per-site intercepts shrunk toward the grand mean (empirical Bayes)."""
        gmean = A[mk].mean(axis=0) if mk.any() else np.zeros(A.shape[1])
        within = []
        raw = {}
        for s in self.site_levels_:
            sel = (site == s) & mk
            if sel.sum() < 2:
                raw[s] = (np.zeros(A.shape[1]), 0)
                continue
            raw[s] = (A[sel].mean(axis=0) - gmean, int(sel.sum()))
            within.append(A[sel].var(axis=0))
        sigma2_e = np.mean(within, axis=0) if within else np.ones(A.shape[1])
        effs = np.array([v for v, n in raw.values() if n > 0])
        sigma2_s = effs.var(axis=0) if len(effs) > 1 else np.zeros(A.shape[1])
        per = {}
        for s, (eff, n) in raw.items():
            if n == 0:
                per[s] = np.zeros(A.shape[1]); continue
            # shrinkage: sites with few subjects are pulled toward 0
            lam = sigma2_s / (sigma2_s + sigma2_e / max(n, 1) + 1e-12)
            per[s] = lam * eff
        return dict(gmean=gmean, per=per)

    # -------------------------------------------------------------- transform
    def transform(self, X_by_mod, site, mask_by_mod=None):
        if self.mode in ("ignore", "adversarial"):
            return X_by_mod
        if self.mode == "feature":
            oh = self.onehot(site)
            return {m: np.hstack([A, oh]) for m, A in X_by_mod.items()}

        out = {}
        for m, A in X_by_mod.items():
            f = self.fitted_.get(m)
            if f is None:
                out[m] = A; continue
            B = A.copy()
            for s in np.unique(site):
                sel = site == s
                if not sel.any():
                    continue
                if self.mode == "combat":
                    mu, sd = f["per"].get(s, (np.zeros(A.shape[1]), np.ones(A.shape[1])))
                    B[sel] = (A[sel] - f["gmean"] - mu) / sd * f["gstd"] + f["gmean"]
                else:
                    B[sel] = A[sel] - f["per"].get(s, np.zeros(A.shape[1]))
            if mask_by_mod is not None:      # never invent values for absent data
                B[~mask_by_mod[m]] = 0.0
            out[m] = B
        return out

    def onehot(self, site):
        levels = self.site_levels_ if self.site_levels_ is not None else np.unique(site)
        oh = np.zeros((len(site), len(levels)), dtype=np.float32)
        for j, s in enumerate(levels):
            oh[site == s, j] = 1.0
        return oh

    def extra_dims(self):
        return len(self.site_levels_) if (self.mode == "feature"
                                          and self.site_levels_ is not None) else 0
