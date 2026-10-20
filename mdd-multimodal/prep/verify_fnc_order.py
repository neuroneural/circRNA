"""Are the sFNC columns in Neuromark template order? Test it, don't assume it.

Why this matters
----------------
The node labels (SC_01 ... CB_53) are attached to sFNC columns 1..53 on the
assumption that GIFT wrote them in template order. The edge count confirms there
are 53 components; it says nothing about their ORDER. If the order differs, every
label is wrong, the readout names the wrong networks, and nothing downstream
reveals it -- the model trains fine and the figure looks plausible.

The test
--------
Neuromark's domains are not arbitrary groupings: components within a domain are
more strongly connected to each other than to distant domains, which is why an
sFNC matrix in template order shows blocks along the diagonal at the domain
boundaries (after 5, 7, 16, 25, 42, 49, 53 for NM1.0).

So: compute the mean |FNC| within domains versus between domains. Then compare
that contrast against the distribution you get from randomly reassigning
components to domains. If the columns really are in template order, the observed
within-domain coherence sits far out in the tail of that null. If the order is
wrong, the domain labels are effectively a random partition and the observed
value lands in the middle of the null -- which is exactly the signal we want.

This is evidence, not proof: a partially-correct order (say, domains right but
components shuffled within a domain) would still pass. It rules out the failure
that matters most, which is a wholesale mismatch.

    python prep/verify_fnc_order.py \
        --sfnc data/sfnc_features.npz \
        --domains /data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.txt
"""

import argparse
import os
import sys

import numpy as np


# --- MAT reading, inlined so this script is self-contained -------------------
# It deliberately does NOT import prep/extract_sfnc.py: this file gets copied
# around on its own, and an ImportError for a sibling module is a confusing way
# to fail when all you wanted was to read one matrix.
FNC_KEYS = ("fnc_corrs_all", "fnc_corrs", "FNCM", "sFNC", "fnc",
            "fnc_corrs_all_harmonized")
SKIP_KEYS = ("spectra", "spectrum", "psd", "freq", "fft", "power")


def _read_mat(path):
    """{name: array} from a MAT of either version. v7.3 is HDF5; v7 is not."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic != b"\x89HDF\r\n\x1a\n":
        try:
            from scipy.io import loadmat
            m = loadmat(path, squeeze_me=True, struct_as_record=False)
            return {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")}
        except (NotImplementedError, ValueError):
            pass                     # actually v7.3 despite the header; fall through
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                # h5py returns MATLAB arrays transposed
                out[name] = np.asarray(obj).T if obj.ndim > 1 else np.asarray(obj)
        f.visititems(visit)
    return out


def _pick_fnc(d):
    """The FNC variable, by name then by shape. Spectra are excluded by name."""
    for k in FNC_KEYS:
        for have in d:
            if have.lower().endswith(k.lower()) or have.lower() == k.lower():
                return have, d[have]
    # nothing matched by name: take the largest array that could be an FNC and
    # is not obviously spectra
    best = None
    for k, v in d.items():
        if any(s in k.lower() for s in SKIP_KEYS):
            continue
        a = np.asarray(v)
        if a.ndim in (2, 3) and a.size > 1000:
            if best is None or a.size > np.asarray(d[best]).size:
                best = k
    return (best, d[best]) if best else (None, None)


def _to_edges(A):
    """(N, ...) FNC in whatever shape GIFT wrote -> (N, K*(K-1)/2) upper triangle.

    Real GIFT output has been seen as (2526, 1, 53, 53): a singleton session or
    window axis between subjects and the matrix. Squeezing it is safe; squeezing
    the SUBJECT axis is not, so axis 0 is always preserved.

    Also accepts an already-vectorised (N, 1378) and a full square (N, 53, 53),
    taking the upper triangle of the latter. A flat (N, 2809) -- a square matrix
    someone already flattened -- is recognised and re-triangulated rather than
    rejected.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim > 2:                       # drop singleton axes, never axis 0
        keep = [0] + [i for i in range(1, A.ndim) if A.shape[i] > 1]
        A = A.transpose(keep + [i for i in range(1, A.ndim) if i not in keep])
        A = A.reshape([A.shape[0]] + [s for s in A.shape[1:] if s > 1])
    if A.ndim == 3 and A.shape[1] == A.shape[2]:
        iu = np.triu_indices(A.shape[1], k=1)
        return A[:, iu[0], iu[1]]
    X = A.reshape(A.shape[0], -1)
    n = X.shape[1]
    root = int(round(np.sqrt(n)))
    if root * root == n and root > 2:     # a flattened square matrix
        iu = np.triu_indices(root, k=1)
        return X.reshape(-1, root, root)[:, iu[0], iu[1]]
    return X


def load_domains(path):
    """-> (labels list length K, domain name per component)."""
    dom = {}
    with open(path) as f:
        for line in f:
            parts = [t.strip() for t in line.replace("\t", ",").split(",") if t.strip()]
            if len(parts) < 2:
                continue
            for tok in parts[1:]:
                try:
                    dom[int(tok)] = parts[0]
                except ValueError:
                    pass
    if not dom:
        raise SystemExit(f"no domains parsed from {path}")
    lo, hi = min(dom), max(dom)
    missing = [i for i in range(lo, hi + 1) if i not in dom]
    if missing:
        raise SystemExit(f"{path} is missing components {missing} — truncated?")
    return [dom[i] for i in range(lo, hi + 1)]


def load_fnc(path, key=None):
    """Mean FNC matrix across subjects, (K, K), from our npz or a GIFT MAT."""
    if not os.path.exists(path):
        raise SystemExit(
            f"not found: {path}\n"
            "  data/sfnc_features.npz is written by prep/extract_sfnc.py, which\n"
            "  needs the GIFT postprocess MAT. You do not have to build it first --\n"
            "  this script reads a GIFT MAT directly. Point it at one:\n"
            "      sbatch scripts/run_verify_fnc.sh /path/to/postprocess_results.mat\n"
            "  Look in the ICA output directory, e.g.\n"
            "      ls /data/users3/bbaker/projects/MDD_preproc/results/*.mat\n"
            "      ls /data/users3/bbaker/projects/MDD_preproc/results/ica/\n"
            "  Prefer a RAW postprocess MAT over a harmonized one: harmonisation\n"
            "  can flatten the domain block structure this test measures.")
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        key = key or next((k for k in ("sFNC", "sFNC105") if k in z), None)
        if key is None:
            raise SystemExit(f"{path} has no sFNC array. Keys present: "
                             f"{list(z.files)}. Pass --key to name one.")
        X = _to_edges(z[key])
    else:
        d = _read_mat(path)
        name, A = _pick_fnc(d)
        if A is None:
            raise SystemExit(
                f"no FNC variable in {path}.\n"
                f"  variables present: "
                + ", ".join(f"{k}{tuple(np.shape(v))}" for k, v in d.items())
                + "\n  Pass --key to name the right one.")
        print(f"  using variable '{name}' {tuple(np.shape(A))}")
        X = _to_edges(A)

    n_edge = X.shape[1]
    k = int((1 + np.sqrt(1 + 8 * n_edge)) / 2)
    if k * (k - 1) // 2 != n_edge:
        raise SystemExit(
            f"{n_edge} values per subject is not an upper triangle of any K.\n"
            "  Expected 1378 for 53 components. Check the variable named above —\n"
            "  pass --key if a different one holds the FNC.")
    M = np.zeros((k, k))
    iu = np.triu_indices(k, k=1)
    mean_edge = np.nanmean(X, axis=0)
    M[iu[0], iu[1]] = mean_edge
    M = M + M.T
    return M, k, X.shape[0]


def contrast(M, labels):
    """mean |FNC| within domains minus between domains."""
    lab = np.asarray(labels)
    same = lab[:, None] == lab[None, :]
    off = ~np.eye(len(lab), dtype=bool)
    within = np.abs(M[same & off])
    between = np.abs(M[~same & off])
    return float(within.mean() - between.mean()), float(within.mean()), float(between.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfnc", default="data/sfnc_features.npz",
                    help="npz from prep/extract_sfnc.py, or the GIFT MAT directly")
    ap.add_argument("--key", default=None)
    ap.add_argument("--domains", required=True,
                    help="Neuromark_fMRI_1.0.txt")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = load_domains(args.domains)
    M, k, n_sub = load_fnc(args.sfnc, args.key)
    print(f"\nFNC: {k} components, averaged over {n_sub} subjects")
    print(f"domains: {len(labels)} components from {os.path.basename(args.domains)}")
    if len(labels) != k:
        raise SystemExit(f"domain file describes {len(labels)} components but the "
                         f"FNC has {k}. These are not the same template.")

    names, first = [], {}
    for i, d in enumerate(labels):
        first.setdefault(d, i)
        names.append(d)
    order = sorted(first, key=lambda d: first[d])
    bounds = [first[d] for d in order] + [k]
    print("  domain blocks: " + ", ".join(
        f"{d} {bounds[j]+1}-{bounds[j+1]}" for j, d in enumerate(order)))

    # ---- the domain-by-domain mean, which is the block structure in numbers --
    print(f"\n{'':6}" + "".join(f"{d:>8}" for d in order))
    for a in order:
        ia = [i for i in range(k) if labels[i] == a]
        row = []
        for b in order:
            ib = [i for i in range(k) if labels[i] == b]
            blk = np.abs(M[np.ix_(ia, ib)])
            if a == b and len(ia) > 1:
                m = blk[~np.eye(len(ia), dtype=bool)].mean()
            elif a == b:
                m = np.nan
            else:
                m = blk.mean()
            row.append(m)
        print(f"{a:>6}" + "".join(f"{v:>8.3f}" if not np.isnan(v) else f"{'--':>8}"
                                  for v in row))
    print("  (mean |FNC|; the diagonal should be the largest entry in its row)")

    # ---- permutation test ---------------------------------------------------
    obs, w, b = contrast(M, labels)
    rng = np.random.default_rng(args.seed)
    null = np.empty(args.n_perm)
    lab = np.asarray(labels)
    for t in range(args.n_perm):
        null[t] = contrast(M, rng.permutation(lab))[0]
    p = float((np.sum(null >= obs) + 1) / (args.n_perm + 1))
    z = float((obs - null.mean()) / (null.std() + 1e-12))

    print(f"\n{'='*74}\nWITHIN-DOMAIN COHERENCE\n{'='*74}")
    print(f"  mean |FNC| within domains   {w:.4f}")
    print(f"  mean |FNC| between domains  {b:.4f}")
    print(f"  contrast                    {obs:+.4f}")
    print(f"  null (random domain labels) {null.mean():+.4f} +/- {null.std():.4f}")
    print(f"  z = {z:.1f},  p = {p:.2g}  ({args.n_perm} permutations)")

    print(f"\n{'='*74}\nVERDICT\n{'='*74}")
    if p < 0.01 and z > 3:
        print("  CONSISTENT with template order.")
        print("  The domain labels explain the FNC block structure far better than")
        print("  a random partition does, which is what you would see only if the")
        print("  columns are ordered as the template says. Labelling nodes SC_01..")
        print("  CB_53 is safe.")
        print("\n  Caveat: this cannot detect a shuffle WITHIN a domain — two visual")
        print("  components swapped would still pass. It rules out the wholesale")
        print("  mismatch, which is the error that would invalidate every label.")
        rc = 0
    elif z > 1.5:
        print("  WEAKLY consistent. The block structure is there but not sharp.")
        print("  Check a per-domain heatmap by eye before trusting node names, and")
        print("  consider whether this FNC was harmonised or residualised in a way")
        print("  that flattens the domain structure.")
        rc = 0
    else:
        print("  NOT CONSISTENT with template order. !!")
        print("  The domain labels explain the FNC no better than a random")
        print("  partition. Either the columns are in a different order, or this")
        print("  FNC is not from this template. DO NOT attach node names until")
        print("  this is resolved — ask whoever ran the ICA what order GIFT wrote.")
        rc = 1
    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
