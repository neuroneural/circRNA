"""GIFT postprocess MAT -> data/sfnc_features.npz, keyed by subject ID.

Group ICA through GIFT is the proposal's own feature-generation step (p. 16).
What comes out of the postprocess stage is a static FNC matrix per subject, plus
component spectra.

The join, which is the whole point of this file
-----------------------------------------------
The MAT has no subject IDs. Its `subjects` variable is a 1..N counter and carries
no identity. The rows are in QC-tier order, so row k of an N=2,526 MAT is the
subject whose `clean_row` == k in mdd_master.csv, and row k of an N=2,426 MAT is
the subject whose `super_clean_row` == k. Joining on any index that counts all
3,525 files instead produces a plausible-looking matrix with every label attached
to the wrong brain. This script picks the tier by matching N, refuses if N matches
neither, and writes real IDs so nothing downstream repeats the mistake.

MAT v7.3 is HDF5: scipy.io.loadmat cannot read it, h5py can, and h5py returns
arrays transposed relative to MATLAB. Both readers are tried.

    python prep/extract_sfnc.py --mat .../postprocess_results.mat \
        --master data/mdd_master.csv --out data/sfnc_features.npz
"""

import argparse
import os

import numpy as np
import pandas as pd

FNC_KEYS = ("fnc_corrs_all", "fnc_corrs", "FNCM", "sFNC", "fnc")
SPEC_KEYS = ("spectra_tc_all", "spectra", "spectra_tc")


def is_hdf5(path):
    with open(path, "rb") as f:
        head = f.read(512)
    return head[:8] == b"\x89HDF\r\n\x1a\n" or b"\x89HDF\r\n\x1a\n" in head[:256]


def load_mat(path):
    """Return {name: ndarray} for the variables we care about, either MAT format."""
    if not is_hdf5(path):
        try:
            from scipy.io import loadmat
            m = loadmat(path, squeeze_me=True, struct_as_record=False)
            return {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")}
        except (NotImplementedError, ValueError):
            pass                               # v7.3 -> HDF5, fall through
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            try:
                a = np.asarray(f[k])
            except Exception:                  # noqa: BLE001
                continue
            out[k] = a.T if a.ndim > 1 else a   # h5py gives MATLAB arrays transposed
    return out


def pick(d, keys):
    for k in keys:
        if k in d:
            return k, np.asarray(d[k])
    lower = {k.lower(): k for k in d}
    for k in keys:
        if k.lower() in lower:
            kk = lower[k.lower()]
            return kk, np.asarray(d[kk])
    return None, None


def vectorise(A):
    """FNC in whatever shape GIFT wrote -> (N, C*(C-1)/2) upper triangle.

    Real GIFT output has been seen as (2526, 1, 53, 53) -- a singleton session or
    window axis sits between the subject axis and the matrix. The previous version
    of this function did not squeeze it, fell through to a flat reshape, and
    produced 2809 columns (the full 53x53, upper and lower triangle plus a zero
    diagonal) instead of 1378. That does not raise: it writes a feature file that
    is twice as wide as it should be, with every edge duplicated. Downstream
    everything still runs.
    """
    A = np.asarray(A, dtype=np.float32)
    if A.ndim > 2:                       # drop singleton axes, never axis 0
        A = A.reshape([A.shape[0]] + [s for s in A.shape[1:] if s > 1])
    if A.ndim == 3 and A.shape[1] == A.shape[2]:
        iu = np.triu_indices(A.shape[1], k=1)
        return A[:, iu[0], iu[1]]
    X = A.reshape(A.shape[0], -1)
    n = X.shape[1]
    root = int(round(np.sqrt(n)))
    if root * root == n and root > 2:     # a square matrix someone flattened
        iu = np.triu_indices(root, k=1)
        return X.reshape(-1, root, root)[:, iu[0], iu[1]]
    return X


def resolve_tier(n_rows, df):
    """Match the MAT's row count to a QC tier and return that tier's IDs in order."""
    for col_flag, col_row, name in (("in_clean", "clean_row", "clean / ICA-53"),
                                    ("in_super_clean", "super_clean_row", "super_clean / ICA-105")):
        if col_flag not in df or col_row not in df:
            continue
        sub = df[df[col_flag].astype(str).str.lower().isin(("true", "1", "1.0", "yes"))]
        if len(sub) != n_rows:
            continue
        sub = sub.dropna(subset=[col_row]).copy()
        sub[col_row] = sub[col_row].astype(int)
        if sorted(sub[col_row]) != list(range(n_rows)):
            raise SystemExit(f"{name}: {col_row} is not a clean 0..{n_rows-1} sequence — "
                             "rebuild the index before trusting this join")
        sub = sub.sort_values(col_row)
        print(f"  MAT has {n_rows} rows -> tier '{name}', joining on {col_row}")
        return sub["id"].astype(str).to_numpy()
    raise SystemExit(
        f"MAT has {n_rows} rows, which matches no QC tier in the master CSV. "
        "Refusing to guess — a wrong join silently mislabels every subject.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--master", default="data/mdd_master.csv")
    ap.add_argument("--out", default="data/sfnc_features.npz")
    ap.add_argument("--name", default="sFNC", help="modality name to write (sFNC | sFNC105)")
    ap.add_argument("--with-spectra", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    d = load_mat(args.mat)
    print("variables in MAT: " + ", ".join(f"{k}{tuple(np.shape(v))}" for k, v in d.items()))

    key, fnc = pick(d, FNC_KEYS)
    if fnc is None:
        raise SystemExit(f"no FNC variable found (looked for {FNC_KEYS})")
    X = vectorise(fnc)
    print(f"  using '{key}' -> {X.shape}")

    ids = resolve_tier(X.shape[0], df)
    payload = {args.name: X.astype(np.float32)}

    if args.with_spectra:
        skey, sp = pick(d, SPEC_KEYS)
        if sp is not None:
            S = vectorise(sp)
            if S.shape[0] == X.shape[0]:
                payload["spectra"] = S.astype(np.float32)
                print(f"  using '{skey}' -> {S.shape}")
            else:
                print(f"  (spectra has {S.shape[0]} rows, not {X.shape[0]} — skipped)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids, dtype=object), **payload)
    print(f"\nwrote {args.out}  ({len(ids)} subjects, "
          + ", ".join(f"{k} {v.shape[1]}d" for k, v in payload.items()) + ")")


if __name__ == "__main__":
    main()
