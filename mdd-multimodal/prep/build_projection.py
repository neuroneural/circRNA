"""Fixed parcel -> Neuromark-component projection, for panel A's shared node set.

The problem this solves
-----------------------
Panel A needs every modality on ONE node set. sFNC already is: it is indexed by
component pairs, so component i's FNC row is node i's features, free. The
structural modalities are not -- GM/WM/CSF/fALFF are parcel means, and a parcel
is not a component.

Without this script, `src/models.py` falls back to a LEARNED linear projection.
That still shares the node set, but what a node MEANS for GM is fitted by the
optimiser, so calling node 31 "CC_31" for the GM channel is not an anatomical
claim. With this script the mapping is fixed by the template, and it is.

How
---
A Neuromark component is a whole-brain spatial map. A parcel is a region. The
overlap is: how much of component c's weight falls inside parcel p.

    P[p, c] = mean( |component_c(v)| for voxels v in parcel p )

then each column is normalised so a node is a weighted average over parcels
rather than a sum that scales with parcel size. Absolute value is used because
ICA component signs are arbitrary -- a component's negative lobe is as much
"that network" as its positive one, and we are asking where the network IS, not
which direction it loads.

The atlas must be the SAME one extract_struct.py used, in the same order, or the
rows of P address the wrong parcels. Pass the same --n-rois and --subcortical.

    python prep/build_projection.py \\
        --template /data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.nii \\
        --n-rois 200 --subcortical harvard_oxford \\
        --modalities GM,WM,CSF,FALFF,FALFF_globalC \\
        --out data/projection.npz

Then point the config at it:

    model:
      transparent:
        projection: data/projection.npz

Caveat worth stating in any writeup: this makes the node's GM value a
component-weighted average of parcel means, which is two lossy steps from the
voxels (voxels -> parcels -> components). A voxelwise projection would be
cleaner; this is the version that works with the features we already have.
"""

import argparse
import os
import sys

import numpy as np


def load_atlas(n_rois, subcortical, resolution=2):
    """Same atlas construction as prep/extract_struct.py — keep these in step."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_struct import build_masker
    masker, labels = build_masker(n_rois, subcortical, resolution)
    return masker, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True,
                    help="Neuromark_fMRI_1.0.nii (4D: x, y, z, n_components)")
    ap.add_argument("--n-rois", type=int, default=200,
                    help="must match what extract_struct.py used")
    ap.add_argument("--subcortical", default="harvard_oxford",
                    choices=["none", "harvard_oxford"],
                    help="must match what extract_struct.py used")
    ap.add_argument("--modalities", default="GM,WM,CSF,FALFF,FALFF_globalC",
                    help="which modalities get this projection (they share the atlas)")
    ap.add_argument("--out", default="data/projection.npz")
    args = ap.parse_args()

    import nibabel as nib
    from nilearn.image import resample_to_img

    if not os.path.exists(args.template):
        raise SystemExit(f"template not found: {args.template}")

    tpl = nib.load(args.template)
    if tpl.ndim != 4:
        raise SystemExit(
            f"{args.template} is {tpl.ndim}D with shape {tpl.shape}. Expected 4D "
            "(x, y, z, n_components). If this is a 3D label image rather than a "
            "stack of component maps, it is the wrong file for this purpose.")
    k = tpl.shape[3]
    print(f"template: {os.path.basename(args.template)}  {tpl.shape}  -> {k} components")

    masker, labels = load_atlas(args.n_rois, args.subcortical)
    n_parcels = len(labels)
    print(f"atlas   : {n_parcels} parcels "
          f"(n_rois={args.n_rois}, subcortical={args.subcortical})")

    # Parcellating |component| with the same masker used for the subject data
    # guarantees the rows of P line up with the feature columns.
    P = np.zeros((n_parcels, k), dtype=np.float32)
    for c in range(k):
        comp = tpl.slicer[..., c]
        absmap = nib.Nifti1Image(np.abs(np.asanyarray(comp.dataobj)),
                                 comp.affine, comp.header)
        try:
            P[:, c] = masker.fit_transform(absmap).ravel()
        except Exception as e:                                    # noqa: BLE001
            raise SystemExit(f"component {c+1}: {e}")
        if (c + 1) % 10 == 0:
            print(f"  {c+1}/{k} components")

    empty = int((P.sum(axis=0) == 0).sum())
    if empty:
        print(f"  !! {empty} components have zero overlap with every parcel. "
              "Check that the template and atlas are in the same space.")
    dead = int((P.sum(axis=1) == 0).sum())
    if dead:
        print(f"  note: {dead} parcels overlap no component — their GM value will "
              "not reach any node. Expected for cerebellar/brainstem parcels if "
              "the template's mask excludes them.")

    # column-normalise: each node is a weighted average over parcels
    P = P / P.sum(axis=0, keepdims=True).clip(1e-8)

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    payload = {m: P for m in mods}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, roi_labels=np.array(labels, dtype=object),
                        n_components=k, **payload)

    print(f"\nwrote {args.out}")
    print(f"  ({n_parcels} parcels x {k} components), applied to: {', '.join(mods)}")
    print("\n  strongest parcel per component (a sanity check — each should look")
    print("  anatomically plausible for its domain):")
    for c in range(min(k, 8)):
        j = int(np.argmax(P[:, c]))
        print(f"    component {c+1:>3}  <- {labels[j]}  ({P[j, c]:.3f})")
    print("\n  Set model.transparent.projection to this file. The model will then")
    print("  report these modalities as 'fixed' rather than 'learned', and node")
    print("  names become anatomical claims rather than placeholders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
