"""Cross-validated training, with the three HAMD label strategies.

    python -m src.train --config conf/experiments/hamd_transfer.yaml
    python -m src.train --config conf/experiments/hc_vs_mdd_sfnc.yaml site.mode=combat

hamd_mode
---------
  direct    Train and test only on the ~660 subjects who have HAMD. Everything
            else is discarded.
  transfer  Pretrain the encoders on DIAGNOSIS using every available subject
            (~2,500), then fine-tune the head on the HAMD subset. The 1,900
            subjects without symptom scores still shape the representation.
  holdout   Train on subjects WITHOUT HAMD (using diagnosis), then predict HAMD
            on those who have it. Nothing about the evaluation set is seen in
            training — it tests whether a diagnosis-trained representation
            carries symptom information at all.

Everything fold-dependent — scaling, site correction, PCA if added — is fitted on
the training split only.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config, data as data_mod, models
from .site import SiteHandler


def _device(pref):
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _scale(Xtr, Xte, masks_tr, masks_te):
    out_tr, out_te, scalers = {}, {}, {}
    for m in Xtr:
        sc = StandardScaler()
        real = masks_tr[m]
        sc.fit(Xtr[m][real] if real.any() else Xtr[m])
        a, b = sc.transform(Xtr[m]), sc.transform(Xte[m])
        a[~masks_tr[m]] = 0.0
        b[~masks_te[m]] = 0.0
        out_tr[m], out_te[m], scalers[m] = a, b, sc
    return out_tr, out_te, scalers


def _tensors(X, masks, dev):
    xs = {m: torch.tensor(v, dtype=torch.float32, device=dev) for m, v in X.items()}
    mk = {m: torch.tensor(masks[m], dtype=torch.bool, device=dev) for m in X}
    return xs, mk


def _split_val(y, ratio, seed):
    """Stratified inner split of the TRAINING data. Returns (fit_idx, val_idx)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or ratio <= 0 or ratio >= 1:
        return np.arange(len(y)), np.empty(0, int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    a, b = next(sss.split(np.zeros(len(y)), y))
    return a, b


def _fit(model, Xtr, mtr, ytr, site_tr, cfg, epochs, dev, tag="",
         multiclass=False, trace=None, eval_fn=None, early_stop=False):
    """Train for a fixed number of epochs.

    `trace`, when a list is passed, collects one row per epoch:
    {epoch, total, task, <each aux term>, val_auc?}. Without it there is no way
    to tell an under-trained model from a converged one -- the fold AUC alone
    cannot distinguish "needs more epochs", "overfit ten epochs ago", and
    "the auxiliary losses are drowning the classification term".

    That last case matters for the transparent panels specifically: they add a
    reconstruction loss and an L1 on the adjacency to the BCE. If those terms
    are much larger than the task term, the optimiser is mostly fitting an
    autoencoder and the classifier is a passenger -- which looks exactly like
    a weak architecture from the outside.
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    xs, mk = _tensors(Xtr, mtr, dev)
    y = torch.tensor(ytr, dtype=torch.long if multiclass else torch.float32, device=dev)
    s = torch.tensor(site_tr, dtype=torch.long, device=dev)
    n = len(ytr)
    bs = cfg["train"]["batch_size"]

    # ---------------------------------------------------- early stopping
    # The validation split comes out of the TRAINING data, never the test fold.
    # Stopping on the test fold would pick the epoch that flatters the number we
    # then report -- the same defect as choosing the best of several classifiers
    # on pooled out-of-fold predictions.
    #
    # Matches the lab's cvbench protocol: 20% validation, patience 30.
    fit_idx = np.arange(n)
    val_idx = np.empty(0, int)
    best = {"auc": -1.0, "epoch": 0, "state": None}
    if early_stop:
        fit_idx, val_idx = _split_val(
            ytr, float(cfg["train"].get("val_ratio", 0.2)),
            int(cfg["train"].get("val_seed", 42)))
        if len(val_idx) == 0:
            early_stop = False
    if early_stop:
        Xv = {m: Xtr[m][val_idx] for m in Xtr}
        mv = {m: mtr[m][val_idx] for m in mtr}
        yv = np.asarray(ytr)[val_idx].astype(int)
        patience = int(cfg["train"].get("patience", 30))
        n = len(fit_idx)
        Xtr = {m: Xtr[m][fit_idx] for m in Xtr}
        mtr = {m: mtr[m][fit_idx] for m in mtr}
        ytr = np.asarray(ytr)[fit_idx]
        site_tr = np.asarray(site_tr)[fit_idx]
        xs, mk = _tensors(Xtr, mtr, dev)
        y = torch.tensor(ytr, dtype=torch.long if multiclass else torch.float32,
                         device=dev)
        s_ = torch.tensor(site_tr, dtype=torch.long, device=dev)
        s = s_
        bad = 0

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        sums, nb = {}, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logit, site_logit, aux = model({m: xs[m][idx] for m in xs},
                                           {m: mk[m][idx] for m in mk})
            task = ce(logit, y[idx]) if multiclass else bce(logit, y[idx])
            loss = task
            if site_logit is not None:
                loss = loss + ce(site_logit, s[idx])
            # transparent modes: reconstruction (panel A's decoder) + an L1 that
            # keeps the adjacency sparse enough to read
            for extra in aux.values():
                loss = loss + extra
            loss.backward()
            opt.step()
            if trace is not None:
                nb += 1
                sums["task"] = sums.get("task", 0.0) + float(task.detach())
                sums["total"] = sums.get("total", 0.0) + float(loss.detach())
                for k, v in aux.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
        if early_stop:
            pv = _predict(model, Xv, mv, dev, multiclass=multiclass)
            va = _auc(yv, pv, multiclass,
                      classes=np.arange(pv.shape[1]) if multiclass else None)
            model.train()
            if not np.isnan(va) and va > best["auc"]:
                best = {"auc": float(va), "epoch": ep + 1,
                        "state": {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            if trace is not None:
                sums["inner_val_auc"] = (va if not np.isnan(va) else 0.5) * max(nb, 1)
            if bad >= patience:
                if trace is not None and nb:
                    row = {"epoch": ep + 1}
                    row.update({k: v / nb for k, v in sums.items()})
                    if eval_fn is not None:
                        row["val_auc"] = eval_fn(model)
                    trace.append(row)
                break

        if trace is not None and nb:
            row = {"epoch": ep + 1}
            row.update({k: v / nb for k, v in sums.items()})
            if eval_fn is not None:
                row["val_auc"] = eval_fn(model)
                model.train()
            trace.append(row)

    if early_stop and best["state"] is not None:
        model.load_state_dict(best["state"])
        if trace is not None:
            print(f"  early stop: best inner-val AUC {best['auc']:.4f} "
                  f"@ epoch {best['epoch']} of {epochs} "
                  f"(patience {patience}); weights restored")
    return model


def print_trace(trace, tag=""):
    """Compact convergence report: is it still improving, and what dominates."""
    if not trace:
        return
    keys = [k for k in trace[0] if k not in ("epoch", "val_auc")]
    keys = [k for k in keys if k != "inner_val_auc"] + (
        ["inner_val_auc"] if "inner_val_auc" in trace[0] else [])
    head = "  ep  " + "".join(f"{k:>12}" for k in keys)
    if "val_auc" in trace[0]:
        head += f"{'val_auc':>10}"
    print(f"\n  ---- loss trace {tag} ----")
    print(head)
    n = len(trace)
    show = sorted({0, 1, 2, n // 4, n // 2, 3 * n // 4, n - 2, n - 1} & set(range(n)))
    for i in show:
        r = trace[i]
        line = f"  {r['epoch']:>3}  " + "".join(f"{r.get(k, 0):>12.4f}" for k in keys)
        if "val_auc" in r:
            line += f"{r['val_auc']:>10.4f}"
        print(line)

    first, last = trace[0], trace[-1]
    tail = trace[max(0, n - 10):]
    drop = first["total"] - last["total"]
    tail_drop = tail[0]["total"] - tail[-1]["total"]
    print(f"  total loss {first['total']:.4f} -> {last['total']:.4f} "
          f"(drop {drop:.4f}; last 10 epochs {tail_drop:+.4f})")
    if tail_drop > 0.01 * max(abs(last["total"]), 1e-9):
        print("  >> STILL FALLING at the last epoch — under-trained. "
              "Raise train.epochs.")
    else:
        print("  >> flat over the last 10 epochs — converged (or stuck).")

    # inner_val_auc rides along in the same dict for convenience but is a METRIC,
    # not a loss term -- counting it as one made the baseline (which has no aux
    # losses at all) print "AUX LOSS EXCEEDS THE TASK LOSS ... lower
    # recon_weight", advice for a model that has no recon_weight.
    aux_keys = [k for k in keys if k not in ("task", "total", "inner_val_auc")]
    if aux_keys:
        a = sum(last.get(k, 0.0) for k in aux_keys)
        t = last.get("task", 0.0)
        print(f"  final task {t:.4f}   aux {a:.4f} "
              + ", ".join(f"{k}={last.get(k, 0):.4f}" for k in aux_keys))
        if a > t:
            print("  >> AUX LOSS EXCEEDS THE TASK LOSS. The model is spending more"
                  " capacity on reconstruction/sparsity than on classification;"
                  " lower model.transparent.recon_weight / sparsity before"
                  " concluding the architecture is weak.")
    if "val_auc" in last:
        best = max(r["val_auc"] for r in trace)
        best_ep = [r["epoch"] for r in trace if r["val_auc"] == best][0]
        print(f"  val AUC best {best:.4f} @ epoch {best_ep} of {n}; "
              f"final {last['val_auc']:.4f}")
        if best_ep < 0.6 * n and last["val_auc"] < best - 0.02:
            print("  >> PEAKED EARLY then declined — overfitting. "
                  "Fewer epochs, or more regularisation.")


@torch.no_grad()
def _predict(model, X, masks, dev, multiclass=False):
    model.eval()
    xs, mk = _tensors(X, masks, dev)
    logit, _, _ = model(xs, mk)
    if multiclass:
        return torch.softmax(logit, dim=1).cpu().numpy()
    return torch.sigmoid(logit).cpu().numpy()


def _auc(y_true, p, multiclass, classes=None):
    """Binary AUC, or one-vs-rest macro AUC when the target is multiclass."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    if not multiclass:
        return roc_auc_score(y_true, p)
    seen = np.unique(y_true)
    if classes is not None and len(seen) < len(classes):
        # a fold can miss a site entirely; score only the classes present
        p = p[:, seen]
        p = p / p.sum(axis=1, keepdims=True).clip(1e-12)
        remap = {c: i for i, c in enumerate(seen)}
        y_true = np.array([remap[v] for v in y_true])
    return roc_auc_score(y_true, p, multi_class="ovr", average="macro")


def run(cfg):
    print("\n" + "=" * 72)
    print(config.describe(cfg))
    print("=" * 72)

    coh = data_mod.build(cfg)
    print("\ncohort:\n" + coh.summary())

    target = str(cfg["label"]["target"])
    hamd_mode = cfg["label"]["hamd_mode"]
    dev = _device(cfg["train"]["device"])
    print(f"\ndevice: {dev}")

    # target=site is the diagnostic run: predict the scanner instead of the
    # diagnosis. That one is multiclass; everything else here is binary.
    multiclass = (target == "site")
    n_classes = int(coh.y.max()) + 1 if multiclass else 1
    if multiclass:
        if cfg["site"]["mode"] != "ignore":
            raise SystemExit("label.target=site with site.mode != ignore removes the "
                             "very thing being predicted — set site.mode=ignore")
        print(f"  predicting site: {n_classes} classes, macro one-vs-rest AUC "
              "(0.5 = site is invisible; near 1.0 = scanner dominates the features)")

    # ---- which subjects are evaluated, and which are available to train on ----
    if target.startswith("hamd"):
        has = coh.meta["has_hamd"] & ~np.isnan(coh.y)
        if hamd_mode == "direct":
            eval_idx = np.where(has)[0]
            train_pool = eval_idx
            pretrain_idx = None
        elif hamd_mode == "transfer":
            eval_idx = np.where(has)[0]
            train_pool = eval_idx
            pretrain_idx = np.arange(len(coh))          # everyone, on diagnosis
        elif hamd_mode == "holdout":
            eval_idx = np.where(has)[0]
            train_pool = np.where(~has)[0]              # disjoint by construction
            pretrain_idx = None
        print(f"  evaluate on {len(eval_idx)} HAMD subjects; "
              f"train pool {len(train_pool)}"
              + (f"; pretrain on {len(pretrain_idx)}" if pretrain_idx is not None else ""))
    else:
        eval_idx = np.arange(len(coh))
        train_pool = eval_idx
        pretrain_idx = None

    if len(eval_idx) < cfg["train"]["n_folds"] * 2:
        raise SystemExit(f"only {len(eval_idx)} evaluable subjects — too few")

    # diagnosis labels for pretraining / holdout training
    y_diag = (coh.groups == cfg["data"]["groups"][0]).astype(np.float32)

    rows = []
    trace_out = None          # set by the one traced fold; stays None if disabled
    t0 = time.time()
    for rep in range(cfg["train"]["n_repeats"]):
        seed = cfg["train"]["seed"] + rep
        torch.manual_seed(seed)
        np.random.seed(seed)

        cv = str(cfg["train"].get("cv", "kfold"))
        if hamd_mode == "holdout" and target.startswith("hamd"):
            # single split: train on non-HAMD, predict all HAMD subjects
            splits = [(train_pool, eval_idx)]
        elif cv == "loso":
            # Leave-one-site-out. Random k-fold puts subjects from the same
            # scanner on both sides of the split, so it cannot see site leakage
            # at all; holding out a whole site can. The gap between loso and
            # kfold is the size of the site problem.
            splits = []
            for s in np.unique(coh.site[eval_idx]):
                te = eval_idx[coh.site[eval_idx] == s]
                tr = eval_idx[coh.site[eval_idx] != s]
                if len(np.unique(coh.y[te].astype(int))) < 2:
                    print(f"    (skipping site {coh.meta['site_names'][s]}: "
                          f"{len(te)} subjects, only one class — AUC undefined)")
                    continue
                if len(te) < cfg["train"].get("loso_min_test", 10):
                    print(f"    (skipping site {coh.meta['site_names'][s]}: "
                          f"only {len(te)} subjects)")
                    continue
                splits.append((tr, te))
            if not splits:
                raise SystemExit("loso: no site has both classes and enough subjects")
            if rep == 0:
                print(f"  leave-one-site-out: {len(splits)} evaluable sites of "
                      f"{len(np.unique(coh.site[eval_idx]))}")
        elif cv == "kfold":
            y_ev = coh.y[eval_idx].astype(int)
            skf = StratifiedKFold(cfg["train"]["n_folds"], shuffle=True, random_state=seed)
            splits = [(eval_idx[tr], eval_idx[te]) for tr, te in skf.split(eval_idx, y_ev)]
        else:
            raise SystemExit(f"unknown train.cv '{cv}' (kfold|loso)")

        for fold, (tr, te) in enumerate(splits, 1):
            # ---- site handling, fitted on train only ----------------------
            sh = SiteHandler(cfg["site"]["mode"], cfg["site"]["adversarial_weight"])
            Xtr_raw = {m: coh.X[m][tr] for m in coh.X}
            Xte_raw = {m: coh.X[m][te] for m in coh.X}
            mtr = {m: coh.mask[m][tr] for m in coh.mask}
            mte = {m: coh.mask[m][te] for m in coh.mask}
            sh.fit(Xtr_raw, coh.site[tr], mtr)
            Xtr = sh.transform(Xtr_raw, coh.site[tr], mtr)
            Xte = sh.transform(Xte_raw, coh.site[te], mte)
            Xtr, Xte, _ = _scale(Xtr, Xte, mtr, mte)

            dims = {m: Xtr[m].shape[1] for m in Xtr}
            n_sites = int(coh.site.max()) + 1
            model = models.build(dims, cfg, n_sites=n_sites,
                                 n_classes=n_classes).to(dev)
            if rep == 0 and fold == 1:
                print(model.describe())
                if hasattr(model, "assert_non_bypassable"):
                    model.assert_non_bypassable()
                    print("  non-bypassable  : verified")

            # ---- stage 1: pretrain encoders on diagnosis (transfer only) ----
            if pretrain_idx is not None and cfg["model"]["smart_init"]:
                pre = np.setdiff1d(pretrain_idx, te)     # never touch the test fold
                Xp_raw = {m: coh.X[m][pre] for m in coh.X}
                mp = {m: coh.mask[m][pre] for m in coh.mask}
                Xp = sh.transform(Xp_raw, coh.site[pre], mp)
                Xp = {m: StandardScaler().fit_transform(Xp[m]) for m in Xp}
                for m in Xp:
                    Xp[m][~mp[m]] = 0.0
                _fit(model, Xp, mp, y_diag[pre], coh.site[pre], cfg,
                     cfg["train"]["pretrain_epochs"], dev, tag="pretrain")

            # ---- stage 2: train on the target ------------------------------
            y_tr = (y_diag[tr] if (hamd_mode == "holdout" and target.startswith("hamd"))
                    else coh.y[tr].astype(np.int64 if multiclass else np.float32))

            # Trace ONE fold, not all thirty: the question "did it converge" is
            # about the optimisation, which is the same shape in every fold, and
            # thirty traces would bury the result. Held-out AUC per epoch is
            # recorded for the SAME fold that is about to be scored -- it is a
            # diagnostic that is printed, never used to choose epochs or a model,
            # which would make the reported AUC optimistic.
            # EVERY fold records its losses; only fold 1 gets the held-out AUC
            # curve and the printed report. Recording all of them is what makes a
            # bad fold diagnosable: a fold that scores 0.36 either drew a hard
            # test set or failed to optimise, and only its final loss tells the
            # two apart. Printing all of them would bury the result, so the other
            # 29 go to the JSON silently.
            full = cfg["train"].get("trace", True)
            do_trace = full and rep == 0 and fold == 1
            trace = [] if full else None
            eval_fn = None
            if do_trace:
                y_te_tr = coh.y[te].astype(int)

                def eval_fn(mdl, _X=Xte, _m=mte, _y=y_te_tr):
                    pp = _predict(mdl, _X, _m, dev, multiclass=multiclass)
                    return float(_auc(_y, pp, multiclass,
                                      classes=np.arange(n_classes)))

            _fit(model, Xtr, mtr, y_tr, coh.site[tr], cfg,
                 cfg["train"]["epochs"], dev, multiclass=multiclass,
                 trace=trace, eval_fn=eval_fn,
                 early_stop=bool(cfg["train"].get("early_stopping", False)))
            if do_trace and trace:
                print_trace(trace, tag=f"(rep 1 fold 1, n_tr={len(tr)})")
                trace_out = trace

            p = _predict(model, Xte, mte, dev, multiclass=multiclass)
            y_te = coh.y[te].astype(int)
            auc = _auc(y_te, p, multiclass, classes=np.arange(n_classes))
            held = (coh.meta["site_names"][coh.site[te][0]]
                    if cv == "loso" and len(te) else None)
            row = dict(repeat=rep, fold=fold, n_train=len(tr), n_test=len(te),
                       auc=round(float(auc), 4), held_out_site=held)
            if trace:
                last = trace[-1]
                row["final_loss"] = {k: round(v, 5) for k, v in last.items()
                                     if k not in ("epoch", "val_auc")}
                # how much the total fell over the last 10 epochs: a fold that is
                # still dropping steeply here did not finish optimising
                tail = trace[max(0, len(trace) - 10):]
                row["tail_drop"] = round(tail[0]["total"] - tail[-1]["total"], 5)
            rows.append(row)
            print(f"    rep {rep+1} fold {fold:2d}  n_tr={len(tr):5d} n_te={len(te):4d}"
                  + (f"  site {held:<8}" if held else "")
                  + f"  AUC {auc:.4f}   ({time.time()-t0:.0f}s)")

    # ---- report -------------------------------------------------------------
    aucs = np.array([r["auc"] for r in rows], dtype=float)
    aucs = aucs[~np.isnan(aucs)]

    # Per-fold convergence: was a bad fold a hard split, or a failed fit?
    fl = [r for r in rows if r.get("final_loss")]
    if len(fl) > 2:
        tot = np.array([r["final_loss"].get("total", np.nan) for r in fl])
        tail = np.array([r.get("tail_drop", np.nan) for r in fl])
        a = np.array([r["auc"] for r in fl], dtype=float)
        ok = ~np.isnan(tot)
        print("\n  ---- per-fold convergence ----")
        print(f"  final total loss  mean {np.nanmean(tot):.4f}  "
              f"sd {np.nanstd(tot, ddof=1):.4f}  "
              f"range {np.nanmin(tot):.4f}-{np.nanmax(tot):.4f}")
        still = int(np.nansum(tail > 0.01 * np.abs(tot)))
        print(f"  folds still falling at the last epoch: {still}/{len(fl)}"
              + ("   >> raise train.epochs" if still > len(fl) // 4 else ""))
        if ok.sum() > 2 and np.nanstd(tot[ok]) > 0:
            z = (tot - np.nanmean(tot)) / np.nanstd(tot, ddof=1)
            # Only WEAK folds are worth explaining, and only a HIGH loss explains
            # them. A loss outlier on a fold that scored well is not a problem,
            # and a two-sigma rule over 30 folds fires ~1.4 times by chance --
            # which would fill this section with false alarms.
            weak = [i for i in range(len(fl))
                    if not np.isnan(a[i]) and a[i] < 0.5]
            if weak:
                print("  folds that scored below chance:")
                for i in weak:
                    r = fl[i]
                    verdict = ("FIT FAILED — final loss is far above the others"
                               if z[i] > 3 else
                               "hard test split — its fit looks normal")
                    print(f"    rep {r['repeat']+1} fold {r['fold']:2d}  "
                          f"AUC {r['auc']:.4f}  final loss "
                          f"{r['final_loss'].get('total', float('nan')):.4f} "
                          f"(z={z[i]:+.1f})   {verdict}")
            blown = [i for i in range(len(fl)) if z[i] > 3]
            if blown and not weak:
                print(f"  {len(blown)} fold(s) ended at a much higher loss than the "
                      "rest — the fit did not take there:")
                for i in blown:
                    r = fl[i]
                    print(f"    rep {r['repeat']+1} fold {r['fold']:2d}  "
                          f"AUC {r['auc']:.4f}  final loss "
                          f"{r['final_loss'].get('total', float('nan')):.4f} "
                          f"(z={z[i]:+.1f})")
            if not weak and not blown:
                print("  every fold converged to a similar loss — the AUC spread is")
                print("  the splits, not the optimisation.")
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(str(cfg.get("_config_file", "run"))))[0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    res = dict(config=dict(cfg), rows=rows,
               auc_mean=float(aucs.mean()) if len(aucs) else None,
               auc_sd=float(aucs.std()) if len(aucs) else None,
               n_eval=int(len(eval_idx)),
               loss_trace=trace_out)      # per-epoch losses for rep 1 fold 1
    path = os.path.join(out_dir, f"{name}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print("\n" + "#" * 72)
    if len(aucs):
        print(f"#  AUC {aucs.mean():.4f}  (sd {aucs.std():.4f} over "
              f"{len(aucs)} folds x repeats)")
        print(f"#  Treat differences under ~2x the sd as noise.")
    print(f"#  -> {path}")
    print("#" * 72)
    return res


def main():
    cfg, _ = config.load()
    run(cfg)


if __name__ == "__main__":
    main()
