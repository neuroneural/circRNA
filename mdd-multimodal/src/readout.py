"""Turn a trained transparent layer into sentences.

This module is the reason the rest exists. A transparent layer whose adjacency is
never read is just an oddly-shaped MLP.

    python -m src.readout --config conf/experiments/hc_vs_mdd_sfnc.yaml
    python -m src.readout --config ... --checkpoint runs/model_*.pt --top 30

For panel A it prints, for the strongest edges, which modality drives each end --
so an edge whose endpoints are led by different modalities reads as
"this <modality-1> pattern co-varies with this <modality-2> pattern", which is
the claim the whole architecture was chosen for. For panel B it prints each
modality's own graph separately, since there is no shared node set to compare across.

Stability
---------
A single fold's adjacency is one sample. An edge that appears in 2 of 10 folds is
noise you can tell a story about. This trains one model per fold and reports how
often each edge survives in the top-N, because the honest unit of a finding is
"present in k of n folds", not "large in the model I happened to look at".
"""

import argparse
import sys

import numpy as np
import torch

from . import config, data as data_mod, models, train as train_mod
from .site import SiteHandler
from .transparent import modality_attribution, top_edges


def _fit_one(cfg, coh, tr, dev, multiclass=False, n_classes=1):
    sh = SiteHandler(cfg["site"]["mode"], cfg["site"]["adversarial_weight"])
    Xtr = {m: coh.X[m][tr] for m in coh.X}
    mtr = {m: coh.mask[m][tr] for m in coh.mask}
    sh.fit(Xtr, coh.site[tr], mtr)
    Xtr = sh.transform(Xtr, coh.site[tr], mtr)
    Xtr, _, _ = train_mod._scale(Xtr, Xtr, mtr, mtr)
    dims = {m: Xtr[m].shape[1] for m in Xtr}
    model = models.build(dims, cfg, n_sites=int(coh.site.max()) + 1,
                         n_classes=n_classes).to(dev)
    y = coh.y[tr].astype(np.float32)
    train_mod._fit(model, Xtr, mtr, y, coh.site[tr], cfg,
                   cfg["train"]["epochs"], dev, multiclass=multiclass)
    return model, Xtr, mtr


def main():
    # config.load() rejects unknown flags, so its own parser must know about ours
    cfg, args = config.load(extra_args=["--top", "--folds"])
    known = argparse.Namespace(top=int(args.top or 20), folds=int(args.folds or 5))
    if cfg["model"]["fusion"]["mode"] not in models.TRANSPARENT_MODES:
        raise SystemExit(
            f"fusion.mode is '{cfg['model']['fusion']['mode']}', which has no "
            "transparent layer to read. Use unified (A) or modality_specific (B).")

    coh = data_mod.build(cfg)
    dev = train_mod._device(cfg["train"]["device"])
    print("\n" + config.describe(cfg))
    print(f"\nfitting {known.folds} models to measure edge stability "
          f"(n = {len(coh)}, device {dev})")

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(known.folds, shuffle=True,
                          random_state=cfg["train"]["seed"])
    idx = np.arange(len(coh))

    counts, weights, last = {}, {}, None
    for f, (tr, _) in enumerate(skf.split(idx, coh.y.astype(int)), 1):
        model, Xtr, mtr = _fit_one(cfg, coh, tr, dev)
        last = (model, Xtr, mtr)
        adj = model.adjacency()
        if isinstance(adj, dict):                               # panel B
            for m, A in adj.items():
                for a, b, w in top_edges(A, model.node_names[m], known.top):
                    key = (m, a, b)
                    counts[key] = counts.get(key, 0) + 1
                    weights.setdefault(key, []).append(w)
        else:                                                   # panel A
            for a, b, w in top_edges(adj, model.node_names, known.top):
                key = ("shared", a, b)
                counts[key] = counts.get(key, 0) + 1
                weights.setdefault(key, []).append(w)
        print(f"  fold {f}/{known.folds} done")

    model, Xtr, mtr = last
    print("\n" + model.describe())

    # ---- panel A: who drives each node, and which edges cross modalities ----
    if model.mode == "unified":
        xs, mk = train_mod._tensors(Xtr, mtr, dev)
        attr = modality_attribution(model, xs, mk, model.node_names)
        lead = {name: ld for name, ld, _ in attr}

        print(f"\n{'='*78}\nWHICH MODALITY DRIVES EACH NODE\n{'='*78}")
        by_mod = {}
        for name, ld, share in attr:
            by_mod.setdefault(ld, []).append((name, share[ld]))
        for m, items in sorted(by_mod.items(), key=lambda kv: -len(kv[1])):
            items.sort(key=lambda t: -t[1])
            top = ", ".join(f"{n} ({s:.0%})" for n, s in items[:6])
            print(f"  {m:<14} leads {len(items):>3} of {len(attr)} nodes   {top}")

        print(f"\n{'='*78}\nSTABLE EDGES  (present in the top {known.top} of "
              f"how many folds)\n{'='*78}")
        rows = sorted(counts.items(), key=lambda kv: (-kv[1], -abs(np.mean(weights[kv[0]]))))
        print(f"  {'folds':>6}  {'weight':>9}  {'node A':<12} {'node B':<12} claim")
        for (scope, a, b), c in rows[:known.top]:
            w = float(np.mean(weights[(scope, a, b)]))
            ma, mb = lead.get(a, "?"), lead.get(b, "?")
            claim = (f"{ma} <-> {mb}   CROSS-MODAL" if ma != mb
                     else f"within {ma}")
            print(f"  {c:>3}/{known.folds}  {w:>9.4f}  {a:<12} {b:<12} {claim}")
        cross = sum(1 for (s, a, b), c in rows[:known.top]
                    if lead.get(a) != lead.get(b))
        print(f"\n  {cross} of the top {min(known.top, len(rows))} edges join nodes "
              "led by DIFFERENT modalities.")
        print("  Those are the cross-modal couplings — the sentence panel A exists")
        print("  to produce. Quote only edges stable across most folds.")
        if any(k == "learned" for k in model.proj_kind.values()):
            print("\n  CAVEAT: the modalities listed above with a learned projection")
            print("  have node names that are placeholders for a fitted mapping, not")
            print("  anatomy. Build fixed projections before naming regions in text.")
    else:
        print(f"\n{'='*78}\nSTABLE EDGES PER MODALITY\n{'='*78}")
        for m in model.modalities:
            rows = sorted(((k, c) for k, c in counts.items() if k[0] == m),
                          key=lambda kv: (-kv[1], -abs(np.mean(weights[kv[0]]))))
            print(f"\n  {m}")
            for (_, a, b), c in rows[:10]:
                print(f"    {c:>3}/{known.folds}  "
                      f"{np.mean(weights[(m, a, b)]):>9.4f}  {a} -- {b}")
        print("\n  Panel B has no shared node set, so there is no direct cross-modal")
        print("  edge. The cross-modal comparison B supports is unimodal vs")
        print("  multimodally-informed: train with model.transparent.recon_weight=0")
        print("  and one modality at a time, then diff these graphs against the")
        print("  jointly-trained ones (pre_empt.pdf p.5, panel B).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
