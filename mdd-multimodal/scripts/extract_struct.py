"""Parcellate the structural and voxelwise-fALFF volumes -> data/struct_features.npz.

Modalities written
------------------
  GM, WM, CSF        tissue probability maps (probseg), parcel means
  FALFF              voxelwise fALFF from the consortium release, parcel means
  FALFF_globalC      the globally-corrected variant, where present
  TIV                total intracranial volume, one number per subject (a covariate,
                     not a modality -- listed under data.covariates in the config)

Atlas
-----
Schaefer-2018 by default. Note what that costs: Schaefer is cortex-only, so
hippocampus, amygdala and the rest of the subcortex are absent. For MDD that is a
real gap. `--subcortical harvard_oxford` appends subcortical parcels from the
Harvard-Oxford subcortical atlas so those structures are represented.

nilearn's atlas.labels includes a 'Background' entry that NiftiLabelsMasker does
not return a column for. Dropping it is not cosmetic -- keeping it shifts every
ROI name by one.

    python prep/extract_struct.py --root /data/users2/ppopov1/datasets/MDD_DIRECT \
        --master data/mdd_master.csv --out data/struct_features.npz --n-rois 100
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# Modality -> (substrings that must ALL appear, substrings that must NOT).
# Matched against the filename, not the directory, because the extraction is flat:
# Pavel's dir holds files like
#   IS001-1-0001_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz
#   IS001-1-0001_fALFF.nii.gz
#   IS001-1-0001_fALFF_globalC.nii.gz
# all directly under $ROOT. The previous version assumed $ROOT/Data_BIDS/GM/ and
# so matched nothing -- and then wrote an all-NaN feature file without complaint,
# which is the failure mode this rewrite exists to remove.
# NB: matched against the LOWERCASED filename, so every pattern here must be
# lowercase. Writing "label-GM_probseg" silently matches nothing.
MOD_PATTERNS = {
    "GM":            (["label-gm_probseg"],  []),
    "WM":            (["label-wm_probseg"],  []),
    "CSF":           (["label-csf_probseg"], []),
    "FALFF":         (["falff"],             ["globalc", "probseg"]),
    "FALFF_globalC": (["falff_globalc"],     ["probseg"]),
}
assert all(t == t.lower() for pats in MOD_PATTERNS.values() for grp in pats
           for t in grp), "MOD_PATTERNS must be lowercase — filenames are lowercased"
SUBJ_RE = re.compile(r"([A-Za-z]*\d{2,}-\d-\d{3,})")


def build_masker(n_rois, subcortical, resolution=2):
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.image import new_img_like
    import nibabel as nib

    sch = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=resolution)
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
    labels = [l for l in labels if l.strip().lower() != "background"]   # off-by-one guard
    img = nib.load(sch.maps) if isinstance(sch.maps, str) else sch.maps

    if subcortical == "harvard_oxford":
        from nilearn.image import resample_to_img
        ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        ho_img = nib.load(ho.maps) if isinstance(ho.maps, str) else ho.maps
        ho_img = resample_to_img(ho_img, img, interpolation="nearest")
        a = np.asanyarray(img.dataobj).astype(np.int16)
        b = np.asanyarray(ho_img.dataobj).astype(np.int16)
        ho_labels = [str(l) for l in ho.labels]
        keep = [i for i, l in enumerate(ho_labels)
                if l.strip().lower() not in ("background",)
                and "cerebral cortex" not in l.lower()
                and "cerebral white matter" not in l.lower()]
        offset = int(a.max())
        for j, i in enumerate(keep, start=1):
            a[(b == i) & (a == 0)] = offset + j
            labels.append("SubCtx_" + ho_labels[i].replace(" ", ""))
        img = new_img_like(img, a)

    masker = NiftiLabelsMasker(labels_img=img, standardize=False,
                               resampling_target="labels", verbose=0)
    return masker, labels


def index_files(root, mods):
    """Walk ROOT once and index {modality: {subject_id: path}}.

    Walking once matters: the previous version called os.listdir per subject per
    modality, which is 3,525 x 5 = 17,625 directory listings. This is one walk.
    """
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")
    idx = {m: {} for m in mods}
    n_seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            if not f.endswith((".nii", ".nii.gz")):
                continue
            n_seen += 1
            low = f.lower()
            sm = SUBJ_RE.search(f)
            if not sm:
                continue
            sid = sm.group(1)
            for m in mods:
                must, mustnt = MOD_PATTERNS[m]
                if all(t in low for t in must) and not any(t in low for t in mustnt):
                    idx[m].setdefault(sid, os.path.join(dirpath, f))
                    break
    print(f"  scanned {n_seen} NIfTI files under {root}")
    for m in mods:
        print(f"    {m:<16} {len(idx[m]):>6} subjects matched")

    empty = [m for m in mods if not idx[m]]
    if empty:
        # Fail here, not silently three hours later with an all-NaN file.
        sample = []
        for dirpath, _, filenames in os.walk(root):
            sample += [f for f in filenames if f.endswith((".nii", ".nii.gz"))][:5]
            if sample:
                break
        raise SystemExit(
            f"\nNo files matched for: {', '.join(empty)}.\n"
            f"  Looked for these substrings in filenames under {root}:\n"
            + "".join(f"    {m:<16} needs {MOD_PATTERNS[m][0]}"
                      f"{' and not ' + str(MOD_PATTERNS[m][1]) if MOD_PATTERNS[m][1] else ''}\n"
                      for m in empty)
            + (f"  Example filenames actually present:\n"
               + "".join(f"    {f}\n" for f in sample[:5]) if sample
               else "  No .nii/.nii.gz files found there at all.\n")
            + "  Fix --root, or adjust MOD_PATTERNS at the top of this file.")
    return idx


def tiv(paths):
    """GM+WM+CSF probability mass x voxel volume, in litres. NaN if unreadable.

    This function must never raise. It used to: a subject whose GM file was
    indexed but could not be opened at read time killed a 1h34m job outright,
    3,000 subjects in, discarding everything. The per-modality reads above were
    already wrapped; this one was not, so it was the single unguarded read in the
    loop -- and on a shared filesystem an unreadable file is a WHEN, not an IF.

    A missing TIV is a NaN for one covariate on one subject. That is not worth a
    lost job.
    """
    import nibabel as nib
    total = 0.0
    for p in paths:
        if p is None:
            return np.nan
        try:
            img = nib.load(p)
            vox = float(np.abs(np.linalg.det(img.affine[:3, :3])))
            total += float(np.nansum(np.asanyarray(img.dataobj))) * vox
        except Exception:                                        # noqa: BLE001
            return np.nan
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/users2/ppopov1/datasets/MDD_DIRECT")
    ap.add_argument("--master", default="data/mdd_master.csv")
    ap.add_argument("--out", default="data/struct_features.npz")
    ap.add_argument("--modalities", default="GM,WM,CSF,FALFF")
    ap.add_argument("--n-rois", type=int, default=100)
    ap.add_argument("--subcortical", default="none", choices=["none", "harvard_oxford"])
    ap.add_argument("--limit", type=int, default=0, help="first N subjects, for a smoke test")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.partial.npz if it exists")
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    if args.limit:
        df = df.head(args.limit)
    ids = df["id"].astype(str).tolist()
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]

    # index BEFORE building the atlas: if nothing matches, fail in seconds
    # rather than after the atlas download and 3,525 subjects of no-ops
    need = sorted(set(mods) | {"GM", "WM", "CSF"})     # TIV needs all three
    index = index_files(args.root, [m for m in need if m in MOD_PATTERNS])

    overlap = sum(1 for s in ids if any(s in index[m] for m in mods))
    print(f"  {overlap}/{len(ids)} master-CSV subjects have at least one file")
    if overlap == 0:
        raise SystemExit(
            "No subject in the master CSV matched any file on disk.\n"
            f"  master CSV ids look like: {', '.join(map(str, ids[:3]))}\n"
            f"  file ids look like:       "
            f"{', '.join(list(index[mods[0]])[:3]) if index[mods[0]] else '(none)'}\n"
            "  These must be the same form for the join to work.")

    masker, labels = build_masker(args.n_rois, args.subcortical)
    print(f"atlas: {len(labels)} parcels ({args.n_rois} cortical"
          + (f" + {len(labels)-args.n_rois} subcortical)" if args.subcortical != "none" else ")"))

    out = {m: np.full((len(ids), len(labels)), np.nan, np.float32) for m in mods}
    tivs = np.full(len(ids), np.nan, np.float32)

    # ---------------------------------------------------------------- resume
    # A crash at subject 2,900 of 3,525 used to discard 1h34m of work. The run
    # now checkpoints, so a restart re-reads only what is still missing.
    ckpt = args.out + ".partial.npz"
    done = np.zeros(len(ids), bool)
    if args.resume and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        if list(z["ids"]) == ids and z[mods[0]].shape[1] == len(labels):
            for m in mods:
                if m in z:
                    out[m] = z[m]
            tivs = z["TIV"].ravel()
            done = z["done"]
            print(f"  resumed from {ckpt}: {int(done.sum())}/{len(ids)} already done")
        else:
            print(f"  ignoring {ckpt}: it was built for a different "
                  "subject list or atlas")

    def save_ckpt():
        np.savez(ckpt, ids=np.array(ids, dtype=object), done=done,
                 TIV=tivs.reshape(-1, 1), **out)

    # Fit the masker ONCE. `fit_transform` refits on every call, and across
    # 17,625 calls that is what drove RSS to 37.9 GB of a 48 GB limit -- on
    # course to be killed around subject 3,400. The labels image never changes,
    # so there is nothing to refit.
    masker.fit()

    import gc
    n_fail = 0
    for i, sid in enumerate(ids):
        if done[i]:
            continue
        for m in mods:
            p = index[m].get(sid)
            if p is None:
                continue
            try:
                out[m][i] = masker.transform(p).ravel()
            except Exception as e:                            # noqa: BLE001
                n_fail += 1
                print(f"  {sid} {m}: {e}", flush=True)
        tivs[i] = tiv([index[t].get(sid) for t in ("GM", "WM", "CSF")])
        done[i] = True
        if (i + 1) % 250 == 0:
            gc.collect()
            save_ckpt()
            print(f"  {i+1}/{len(ids)}   (checkpointed)", flush=True)

    if n_fail:
        print(f"\n  {n_fail} file(s) could not be read; those entries are NaN")

    # refuse to write a file of NaNs -- the old behaviour, and invisible downstream
    got = {m: int((~np.isnan(out[m]).all(axis=1)).sum()) for m in mods}
    if all(v == 0 for v in got.values()):
        raise SystemExit(
            "Every modality is empty for every subject — not writing an all-NaN "
            "feature file.\n  The files were indexed but none could be read or "
            "parcellated; check the messages above.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids, dtype=object),
                        roi_labels=np.array(labels, dtype=object),
                        TIV=tivs.reshape(-1, 1), **out)
    print(f"\nwrote {args.out}")
    for m in mods:
        print(f"  {m:<16} present for {int((~np.isnan(out[m]).all(axis=1)).sum())}/{len(ids)}")
    print(f"  TIV              present for {int((~np.isnan(tivs)).sum())}/{len(ids)}")


if __name__ == "__main__":
    main()
