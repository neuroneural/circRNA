"""End-to-end checks. Run from the repo root:  python tests/test_pipeline.py

Five things are checked, and they are the five that fail silently rather than
loudly if they are wrong:

  1. The GIFT MAT join lands each row on the right subject. A wrong join produces
     a perfectly plausible matrix with every label attached to the wrong brain.
  2. A MAT whose row count matches no QC tier is REFUSED, not guessed at.
  3. Missing modalities are masked, not read as zeros that mean something.
  4. Site correction is fitted on train only -- refitting on the test fold would
     quietly inflate every AUC in the repo.
  5. Leave-one-site-out actually detects site confounding. Built on features with
     ZERO within-site diagnosis signal, where only the MDD RATE differs by site:
     k-fold scores high by recognising the scanner, LOSO must fall to chance. If
     this test ever stops failing under k-fold, the fixture broke, not the model.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "data")
    subprocess.run([sys.executable, "prep/make_synthetic.py", "--n", "300",
                    "--out", data], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    master = os.path.join(data, "mdd_master.csv")
    df = pd.read_csv(master)

    # ---- 1 & 2: the MAT join --------------------------------------------
    import h5py
    clean = df[df.in_clean == True].sort_values("clean_row")            # noqa: E712
    N, C = len(clean), 8
    M = np.zeros((N, C, C))
    for k in range(N):
        M[k] = k                       # fingerprint each row with its clean_row
    good = os.path.join(tmp, "good.mat")
    with h5py.File(good, "w") as f:
        f.create_dataset("fnc_corrs_all", data=M.T)     # MATLAB column order
        f.create_dataset("subjects", data=np.arange(1, N + 1, dtype=float))

    out = os.path.join(tmp, "sf.npz")
    r = subprocess.run([sys.executable, "prep/extract_sfnc.py", "--mat", good,
                        "--master", master, "--out", out],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        check("MAT join runs", False, r.stderr.strip().splitlines()[-1][:90])
    else:
        z = np.load(out, allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        expect = clean.set_index("id")["clean_row"].to_dict()
        wrong = [s for i, s in enumerate(ids) if abs(z["sFNC"][i, 0] - expect[s]) > 1e-6]
        check("MAT rows land on the right subjects", not wrong,
              f"{len(ids)} subjects" if not wrong else f"{len(wrong)} mislabelled")

    bad = os.path.join(tmp, "bad.mat")
    with h5py.File(bad, "w") as f:
        f.create_dataset("fnc_corrs_all", data=np.zeros((C, C, N - 7)))
    r = subprocess.run([sys.executable, "prep/extract_sfnc.py", "--mat", bad,
                        "--master", master, "--out", os.path.join(tmp, "x.npz")],
                       cwd=ROOT, capture_output=True, text=True)
    check("a MAT matching no QC tier is refused",
          r.returncode != 0 and "matches no QC tier" in (r.stdout + r.stderr))

    # ---- 3: missing modalities are masked --------------------------------
    from src import config, data as data_mod
    cfg, _ = config.load([
        "--config", os.path.join(ROOT, "conf", "experiments", "hc_vs_mdd_sfnc.yaml"),
        "data.cohort=all",
        f"paths.master={master}",
        f"paths.struct={os.path.join(data, 'struct_features.npz')}",
        f"paths.sfnc={os.path.join(data, 'sfnc_features.npz')}",
        f"paths.dfnc={os.path.join(data, 'dfnc_features.npz')}",
    ])
    coh = data_mod.build(cfg)
    absent = ~coh.mask["sFNC"]
    check("subjects without sFNC are kept", absent.sum() > 0,
          f"{int(absent.sum())} masked of {len(coh)}")
    check("absent sFNC rows are zero-filled, not garbage",
          bool(np.all(coh.X["sFNC"][absent] == 0)))
    check("present sFNC rows are not zero",
          bool(np.any(coh.X["sFNC"][coh.mask["sFNC"]] != 0)))

    # ---- 4: site correction sees training data only ----------------------
    from src.site import SiteHandler
    tr = np.arange(0, len(coh), 2)
    te = np.arange(1, len(coh), 2)
    sh = SiteHandler("random_effect")
    Xtr = {m: coh.X[m][tr] for m in coh.X}
    mtr = {m: coh.mask[m][tr] for m in coh.mask}
    sh.fit(Xtr, coh.site[tr], mtr)
    before = {m: {s: v.copy() for s, v in sh.fitted_[m]["per"].items()} for m in sh.fitted_}
    sh.transform({m: coh.X[m][te] for m in coh.X}, coh.site[te],
                 {m: coh.mask[m][te] for m in coh.mask})
    unchanged = all(np.allclose(before[m][s], sh.fitted_[m]["per"][s])
                    for m in before for s in before[m])
    check("transform() does not refit on the test fold", unchanged)

    # a site the training fold never saw must not crash transform()
    sh2 = SiteHandler("combat")
    only = coh.site[tr] != coh.site[tr].max()
    sh2.fit({m: v[only] for m, v in Xtr.items()}, coh.site[tr][only],
            {m: v[only] for m, v in mtr.items()})
    try:
        sh2.transform({m: coh.X[m][te] for m in coh.X}, coh.site[te],
                      {m: coh.mask[m][te] for m in coh.mask})
        check("an unseen site at test time is handled", True)
    except Exception as e:                                   # noqa: BLE001
        check("an unseen site at test time is handled", False, str(e)[:80])

    # ---- 5: LOSO detects a confound that k-fold cannot --------------------
    # Build features with ZERO within-site diagnosis signal; only the MDD RATE
    # differs by site. A model can then score well under k-fold purely by
    # recognising the scanner. LOSO must expose that.
    rng = np.random.default_rng(3)
    n_sites, per = 6, 50
    fracs = [0.95, 0.9, 0.8, 0.2, 0.1, 0.05]
    rows = []
    for si in range(n_sites):
        for k in range(per):
            gg = 1 if rng.random() < fracs[si] else 2
            rows.append(dict(id=f"IS{si+1:03d}-{gg}-{k:04d}", file_index=len(rows),
                             filename="x.nii", group_label={1: "MDD", 2: "HC"}[gg],
                             group_code=gg, site=f"IS{si+1:03d}", age=40, sex=0,
                             HAMDTotal17=np.nan, HAMD3=np.nan,
                             in_clean=True, in_super_clean=True))
    d2 = pd.DataFrame(rows)
    d2["clean_row"] = range(len(d2)); d2["super_clean_row"] = range(len(d2))
    cdir = os.path.join(tmp, "conf_fx"); os.makedirs(cdir, exist_ok=True)
    d2.to_csv(os.path.join(cdir, "m.csv"), index=False)
    sidx = pd.factorize(d2.site)[0]
    blk = lambda d: (rng.standard_normal((len(d2), d))
                     + rng.normal(0, 1.5, (n_sites, d))[sidx]).astype(np.float32)
    np.savez_compressed(os.path.join(cdir, "s.npz"),
                        ids=d2.id.to_numpy(dtype=object),
                        roi_labels=np.array([f"r{i}" for i in range(20)], dtype=object),
                        TIV=np.ones((len(d2), 1), np.float32),
                        GM=blk(20), WM=blk(20), CSF=blk(20), FALFF=blk(20))
    np.savez_compressed(os.path.join(cdir, "f.npz"),
                        ids=d2.id.to_numpy(dtype=object), sFNC=blk(60))

    from src import train as train_mod
    from src.diagnose import site_only_auc
    base = site_only_auc(sidx, (d2.group_label == "MDD").astype(int).to_numpy())
    check("site-only baseline detects the planted confound", base > 0.75,
          f"AUC {base:.3f} from site alone")

    def _auc_for(cv):
        c, _ = config.load([
            "--config", os.path.join(ROOT, "conf", "experiments", "hc_vs_mdd_sfnc.yaml"),
            f"paths.master={os.path.join(cdir, 'm.csv')}",
            f"paths.struct={os.path.join(cdir, 's.npz')}",
            f"paths.sfnc={os.path.join(cdir, 'f.npz')}",
            f"paths.dfnc={os.path.join(cdir, 'f.npz')}",
            f"paths.out_dir={os.path.join(tmp, 'runs')}",
            f"train.cv={cv}", "site.mode=ignore", "train.n_folds=5",
            "train.n_repeats=1", "train.epochs=25", "train.device=cpu",
        ])
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            return train_mod.run(c)["auc_mean"]

    a_kf, a_loso = _auc_for("kfold"), _auc_for("loso")
    check("k-fold is fooled by the confound", a_kf > 0.70, f"AUC {a_kf:.3f}")
    check("LOSO exposes it", a_loso < a_kf - 0.15,
          f"kfold {a_kf:.3f} vs loso {a_loso:.3f}")

    # ---- 6: HAMD availability follows the TARGET, not "both scores" -------
    # HAMD3 is nested inside HAMD17 in the real data (989 vs 1394 cohort-wide;
    # 660 vs 984 at cohort=sfnc). Requiring both regardless of target silently
    # cost 324 subjects on every HAMD17 run.
    from src.data import hamd_available
    m3 = pd.read_csv(master)
    a17 = pd.to_numeric(m3.HAMDTotal17, errors="coerce").to_numpy()
    a3 = pd.to_numeric(m3.HAMD3, errors="coerce").to_numpy()
    n17 = int(hamd_available("hamd17", a17, a3).sum())
    n3 = int(hamd_available("hamd3", a17, a3).sum())
    check("HAMD3 is nested inside HAMD17", int((~np.isnan(a3) & np.isnan(a17)).sum()) == 0)
    check("hamd17 target sees more subjects than hamd3", n17 > n3,
          f"hamd17 {n17} vs hamd3 {n3}")

    # ---- 7: the transparent layer is real, and does not invent findings ----
    import io, contextlib
    from src import models as models_mod

    for mode in ("unified", "modality_specific"):
        c, _ = config.load([
            "--config", os.path.join(ROOT, "conf", "experiments", "hc_vs_mdd_sfnc.yaml"),
            f"paths.master={master}",
            f"paths.struct={os.path.join(data, 'struct_features.npz')}",
            f"paths.sfnc={os.path.join(data, 'sfnc_features.npz')}",
            f"paths.dfnc={os.path.join(data, 'dfnc_features.npz')}",
            f"model.fusion.mode={mode}", "train.device=cpu",
        ])
        cc = data_mod.build(c)
        dd = {m: cc.X[m].shape[1] for m in cc.X}
        with contextlib.redirect_stdout(io.StringIO()):
            mdl = models_mod.build(dd, c, n_sites=int(cc.site.max()) + 1)
        # p.4: the layer must be a non-bypassable bottleneck
        try:
            mdl.assert_non_bypassable()
            check(f"{mode}: transparent layer is non-bypassable", True)
        except AssertionError as e:
            check(f"{mode}: transparent layer is non-bypassable", False, str(e)[:70])

        # the adjacency must exist, be symmetric, and be the readable object
        adj = mdl.adjacency()
        As = list(adj.values()) if isinstance(adj, dict) else [adj]
        sym = all(bool(torch.allclose(A, A.t(), atol=1e-5)) for A in As)
        check(f"{mode}: adjacency is symmetric", sym)

        # panel A must keep modality as a separate channel block, or no edge can
        # ever be attributed across modalities
        if mode == "unified":
            xs = {m: torch.tensor(cc.X[m][:8], dtype=torch.float32) for m in cc.X}
            mk = {m: torch.tensor(cc.mask[m][:8]) for m in cc.mask}
            H = mdl.node_features(xs, mk)
            expect = (8, mdl.k, mdl.channels * len(mdl.modalities))
            check("unified: node features are modality-blocked",
                  tuple(H.shape) == expect, f"{tuple(H.shape)} == {expect}")
            # zeroing one modality must change only its own channel block
            xs2 = dict(xs); xs2[mdl.modalities[0]] = torch.zeros_like(xs[mdl.modalities[0]])
            H2 = mdl.node_features(xs2, mk)
            ch = mdl.channels
            other_unchanged = bool(torch.allclose(H[:, :, ch:], H2[:, :, ch:], atol=1e-6))
            check("unified: one modality cannot leak into another's channels",
                  other_unchanged)

    # ---- 8: Neuromark label files parse, and disagreements are refused -----
    from src.transparent import load_node_names
    nmdir = os.path.join(tmp, "nm"); os.makedirs(nmdir, exist_ok=True)
    dom = os.path.join(nmdir, "Neuromark_fMRI_1.0.txt")
    with open(dom, "w") as f:
        f.write("SC, 1, 2, 3, 4, 5\nAU, 6, 7\n"
                "SM, 8, 9, 10, 11, 12, 13, 14, 15, 16\n"
                "VI, 17, 18, 19, 20, 21, 22, 23, 24, 25\n"
                "CC, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42\n"
                "DM, 43, 44, 45, 46, 47, 48, 49\nCB, 50, 51, 52, 53\n")
    with open(os.path.join(nmdir, "Neuromark_fMRI_1.0_indexall.txt"), "w") as f:
        f.write("SC, 69 53 98 99 45\nAU, 21 56\nSM, 3 9 2 11 27 54 66 80 72\n"
                "VI, 16 5 62 15 12 93 20 8 77\n"
                "CC, 68 33 43 70 61 55 63 79 84 96 88 48 81 37 67 38 83\n"
                "DM, 32 40 23 71 17 51 94\nCB, 13 18 4 7\n")
    with contextlib.redirect_stdout(io.StringIO()):
        nm_names, nm_ok = load_node_names(dom, 53)
    check("Neuromark domain file parses to 53 named nodes",
          nm_ok and len(nm_names) == 53 and nm_names[5] == "AU_06",
          f"{nm_names[0]}, {nm_names[5]}, {nm_names[-1]}")

    # a truncated file must NOT silently produce the wrong number of labels
    trunc = os.path.join(nmdir, "trunc.txt")
    with open(trunc, "w") as f:
        f.write(open(dom).read().replace(", 53\n", "\n"))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            load_node_names(trunc, 53)
        check("a truncated label file is refused", False, "it was accepted")
    except SystemExit:
        check("a truncated label file is refused", True)

    # and the two Neuromark files must agree with each other
    bad = os.path.join(nmdir, "bad"); os.makedirs(bad, exist_ok=True)
    import shutil
    shutil.copy(dom, os.path.join(bad, "Neuromark_fMRI_1.0.txt"))
    with open(os.path.join(bad, "Neuromark_fMRI_1.0_indexall.txt"), "w") as f:
        f.write("SC, 69 53 98 99 45\nAU, 21 56\nSM, 3 9 2 11 27 54 66 80 72\n"
                "VI, 16 5 62 15 12 93 20 8 77\n"
                "CC, 68 33 43 70 61 55 63 79 84 96 88 48 81 37 67 38 83\n"
                "DM, 32 40 23 71 17 51 94\nCB, 13 18 4\n")       # one short
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            load_node_names(os.path.join(bad, "Neuromark_fMRI_1.0.txt"), 53)
        check("disagreeing Neuromark files are refused", False, "accepted")
    except SystemExit:
        check("disagreeing Neuromark files are refused", True)

    # ---- 9: FNC shape handling. The real GIFT file is (N, 1, 53, 53) -------
    # A singleton session/window axis used to fall through to a flat reshape and
    # yield 2809 columns instead of 1378 -- silently, in extract_sfnc.py.
    sys.path.insert(0, os.path.join(ROOT, "prep"))
    from extract_sfnc import vectorise as _vec
    for shp in [(40, 1, 53, 53), (40, 53, 53), (40, 1378), (40, 2809),
                (40, 1, 1, 53, 53)]:
        got = _vec(np.zeros(shp, np.float32)).shape
        check(f"FNC {shp} -> 1378 edges", got == (40, 1378), f"got {got}")
    # and the values must be the upper triangle, not a reshuffle
    M = np.arange(53 * 53, dtype=np.float32).reshape(1, 1, 53, 53)
    M = (M + M.transpose(0, 1, 3, 2)) / 2          # symmetric
    iu = np.triu_indices(53, k=1)
    check("FNC values are the true upper triangle",
          bool(np.allclose(_vec(M)[0], M[0, 0][iu])))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
