"""Dynamic connectivity -> data/dfnc_features.npz.

The proposal names dynamic connectivity explicitly (p. 4), and it is the one
modality that is not already sitting on the server as a finished file. Two ways
in, in order of preference:

  --mat  a GIFT dFNC postprocess MAT, if the consortium ran one. Joined to the
         QC tier exactly as extract_sfnc.py does it.

  --tc   ICA timecourses (subjects x time x components). This script then does
         the sliding-window computation itself: window the timecourse, correlate
         the components within each window, cluster the windows into states, and
         summarise each subject by how they move between states.

Per-subject features from the timecourse route
----------------------------------------------
  state occupancy     fraction of windows spent in each state (k numbers)
  mean dwell time     average consecutive windows per visit (k numbers)
  n transitions       how often the subject switches state (1 number)
  variability         SD across windows of each connectivity edge

Occupancy and dwell are the summaries the dFNC literature reports; edge
variability is what carries the fine-grained signal. Together they are a fixed
vector per subject, so dfnc drops into the fusion model as just another stream.

A warning about the clustering: k-means over windows is fitted across ALL
subjects at once, which means it has seen the test fold. Occupancy features are
therefore mildly optimistic. The clean version refits the clustering inside each
training fold; that is a bigger change than this file, and the honest thing is
to say so rather than bury it. Treat dFNC results as exploratory until then.

    python prep/extract_dfnc.py --tc .../ica_timecourses.npz --master data/mdd_master.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from extract_sfnc import load_mat, pick, vectorise, resolve_tier   # noqa: E402

DFNC_KEYS = ("dfnc_corrs", "FNCdyn", "dfnc", "dfnc_corrs_all")


def sliding_windows(tc, win, step, taper=True):
    """(T, C) -> (n_win, C*(C-1)/2) vectorised correlation per window."""
    T, C = tc.shape
    iu = np.triu_indices(C, k=1)
    w = np.hanning(win) if taper else np.ones(win)
    out = []
    for s in range(0, T - win + 1, step):
        seg = tc[s:s + win] * w[:, None]
        seg = seg - seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0)
        sd[sd == 0] = 1.0
        R = (seg.T @ seg) / (len(seg) * np.outer(sd, sd))
        out.append(R[iu])
    return np.asarray(out, dtype=np.float32)


def state_features(labels, k):
    occ = np.bincount(labels, minlength=k).astype(np.float32) / max(len(labels), 1)
    dwell = np.zeros(k, np.float32)
    counts = np.zeros(k, np.float32)
    run, cur = 1, labels[0]
    for x in labels[1:]:
        if x == cur:
            run += 1
        else:
            dwell[cur] += run; counts[cur] += 1
            cur, run = x, 1
    dwell[cur] += run; counts[cur] += 1
    dwell = dwell / np.maximum(counts, 1)
    ntrans = float((np.diff(labels) != 0).sum())
    return occ, dwell, ntrans


def from_timecourses(path, k, win, step, seed):
    from sklearn.cluster import KMeans
    z = np.load(path, allow_pickle=True)
    if "ids" not in z:
        raise SystemExit(f"{path} has no 'ids' array; cannot align subjects")
    ids = np.array([str(x) for x in z["ids"]])
    tcs = z["tc"] if "tc" in z else z[[k_ for k_ in z.files if k_ != "ids"][0]]
    tcs = np.asarray(tcs, dtype=np.float32)          # (N, T, C)
    print(f"  timecourses {tcs.shape}; window {win}, step {step}, k {k}")

    per_subj, bounds = [], [0]
    for i in range(len(ids)):
        W = sliding_windows(tcs[i], win, step)
        per_subj.append(W)
        bounds.append(bounds[-1] + len(W))
    allW = np.concatenate(per_subj, axis=0)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(allW)
    lab = km.labels_

    feats = []
    for i in range(len(ids)):
        a, b = bounds[i], bounds[i + 1]
        occ, dwell, ntrans = state_features(lab[a:b], k)
        var = per_subj[i].std(axis=0)
        feats.append(np.concatenate([occ, dwell, [ntrans], var]))
    X = np.asarray(feats, dtype=np.float32)
    names = ([f"occ_{i}" for i in range(k)] + [f"dwell_{i}" for i in range(k)]
             + ["n_transitions"] + [f"var_{i}" for i in range(per_subj[0].shape[1])])
    return ids, X, np.array(names, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mat", help="GIFT dFNC postprocess MAT")
    g.add_argument("--tc", help="NPZ with ids + tc (N, T, C) ICA timecourses")
    ap.add_argument("--master", default="data/mdd_master.csv")
    ap.add_argument("--out", default="data/dfnc_features.npz")
    ap.add_argument("--k", type=int, default=5, help="number of connectivity states")
    ap.add_argument("--window", type=int, default=22, help="window length in TRs")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mat:
        d = load_mat(args.mat)
        key, A = pick(d, DFNC_KEYS)
        if A is None:
            raise SystemExit(f"no dFNC variable found (looked for {DFNC_KEYS})")
        X = vectorise(A)
        print(f"  using '{key}' -> {X.shape}")
        ids = resolve_tier(X.shape[0], pd.read_csv(args.master))
        names = np.array([f"f{i}" for i in range(X.shape[1])], dtype=object)
    else:
        ids, X, names = from_timecourses(args.tc, args.k, args.window, args.step, args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids, dtype=object),
                        dfnc=X.astype(np.float32), feature_names=names)
    print(f"\nwrote {args.out}  ({len(ids)} subjects, {X.shape[1]} features)")
    print("  NOTE: state clustering was fitted across all subjects — "
          "occupancy/dwell features are mildly optimistic. Exploratory only.")


if __name__ == "__main__":
    main()
