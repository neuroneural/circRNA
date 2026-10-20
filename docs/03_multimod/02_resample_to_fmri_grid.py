#!/usr/bin/env python3
"""
Resample DIRECT's fALFF maps and tissue probability maps onto the grid of the
preprocessed fMRI, so every modality shares one voxel lattice.

Target grid is taken from each subject's own `Np` BOLD file -- SPM's prefix
convention, read right to left: p = resaved, N = normalised, Sm = smoothed.
In practice that is 53x63x52 @ 3 mm, affine diag(-3,3,3) + (78,-112,-70), LAS.

    source                       grid                  ratio
    fALFF, fALFF_globalC         97x115x97 @ 2 mm      1.5x down
    GM / WM / CSF probseg        193x229x193 @ 1 mm    3x   down

NO WARPING IS INVOLVED. Everything is already in MNI152NLin2009cAsym and every
affine is diagonal, so composing the target affine with the inverse source
affine gives another diagonal map: a per-axis scale and offset. The probseg
files are RAS and the target is LAS; that sign flip falls out of the affine
composition automatically, which is exactly why this must go through the
affines and never through array reshaping.

Because all subjects share identical geometry, the index map is computed once
per (source grid, target grid) pair and cached -- two maps for the whole run.

Per-modality handling
---------------------
probseg  Plain trilinear resample. Linear interpolation is convex, so values
         stay in [0,1] without clipping (asserted). No masking, no smoothing.

fALFF    DIRECT ships these pre-masked to a fixed group mask (~238,955 voxels)
         with hard zeros outside. Interpolating across that cliff averages real
         values against structural zeros and leaves a rim of artificially low
         fALFF one voxel deep around the brain -- systematic, and exactly the
         kind of artifact a classifier will happily treat as signal. So the
         mask is resampled through the identical kernel and the data divided by
         it (normalized convolution: the resampled mask is the sum of
         interpolation weights landing on valid voxels, so the quotient is the
         valid-weighted mean), then the thresholded mask is re-applied.

NO SMOOTHING is applied to anything. This script is a purely geometric
operation and is meant to stay reversible. Smoothing is a modelling choice that
must be made jointly across modalities -- do it downstream, consistently.

--prefilter FWHM applies a Gaussian anti-alias filter before sampling. It is
OFF by default. Measured on this data, omitting it shifts values by ~11% RMS at
the 3x ratio, but every subject shares identical source and target geometry, so
the distortion is deterministic and identical for all subjects rather than
subject-varying noise -- absorbable by a model. Turn it on if you ever pool a
cohort with different geometry, or compare voxelwise maps to atlas coordinates.

Requires nibabel and scipy.

--chunk K selects rows [K*chunk_size, (K+1)*chunk_size) so the cohort can be
split across a SLURM array. Each chunk writes its own log
(resample_log_chunk000.csv) -- parallel jobs sharing one log would clobber it.
A chunk index past the end of the CSV exits 0 with a message, so an oversized
array range is harmless.

Usage:
    python3 02_resample_to_fmri_grid.py
    python3 02_resample_to_fmri_grid.py --limit 5 --overwrite
    python3 02_resample_to_fmri_grid.py --chunk 3 --chunk-size 100
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "data", "mdd_direct_volume_paths.csv")
DEFAULT_OUT = "/data/users2/ppopov1/datasets/MDD_DIRECT_proc"
LOG_NAME = "resample_log.csv"
CHUNK_LOG = "resample_log_chunk{:03d}.csv"

TARGET_COL = "fmri_np_path"
MASK_THRESH = 0.5          # keep an fALFF voxel if >50% of its weight was valid
SFORM_CODE = 4             # NIFTI_XFORM_MNI_152; Np itself carries a stale 2

#   csv column          output tag        kind
MODALITIES = [
    ("gm_probseg_path",   "GM_probseg",     "probseg"),
    ("wm_probseg_path",   "WM_probseg",     "probseg"),
    ("csf_probseg_path",  "CSF_probseg",    "probseg"),
    ("falff_c_path",      "fALFF",          "falff"),
    ("falff_globalc_path", "fALFF_globalC", "falff"),
]

LOG_FIELDS = ["subject_id", "modality", "kind", "status", "out_path",
              "src_mass", "out_mass", "pct_change",
              "n_nonzero_out", "vmin", "vmax", "src_path"]


def build_index_map(src_affine, tgt_affine, tgt_shape):
    """Target voxel indices -> source voxel indices, via world coordinates."""
    ijk = np.indices(tgt_shape, dtype=np.float64).reshape(3, -1)
    world = tgt_affine[:3, :3] @ ijk + tgt_affine[:3, 3:4]
    return np.linalg.inv(src_affine[:3, :3]) @ (world - src_affine[:3, 3:4])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help=f"default: {DEFAULT_CSV}")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    ap.add_argument("--prefilter", type=float, default=0.0, metavar="FWHM",
                    help="Gaussian anti-alias FWHM in mm before sampling (0 = off)")
    ap.add_argument("--chunk", type=int, metavar="K",
                    help="process rows [K*chunk_size, (K+1)*chunk_size) "
                         "(for SLURM arrays); writes its own per-chunk log")
    ap.add_argument("--chunk-size", type=int, default=100,
                    help="subjects per chunk (default 100)")
    ap.add_argument("--limit", type=int,
                    help="process only the first N of the selected subjects")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import nibabel as nib
        from scipy.ndimage import map_coordinates, gaussian_filter
    except ImportError as exc:
        sys.exit(f"needs nibabel and scipy: {exc}")

    all_rows = list(csv.DictReader(open(args.csv)))
    log_name = LOG_NAME
    if args.chunk is not None:
        if args.chunk < 0:
            sys.exit("--chunk must be >= 0")
        n_chunks = -(-len(all_rows) // args.chunk_size)      # ceil
        start = args.chunk * args.chunk_size
        if start >= len(all_rows):
            print(f"chunk {args.chunk} starts at row {start} but the CSV has only "
                  f"{len(all_rows)} rows ({n_chunks} chunks of {args.chunk_size}, "
                  f"i.e. 0..{n_chunks - 1}). Nothing to do.")
            sys.exit(0)
        rows = all_rows[start:start + args.chunk_size]
        log_name = CHUNK_LOG.format(args.chunk)
        print(f"chunk    : {args.chunk} of 0..{n_chunks - 1}  "
              f"(rows {start}..{start + len(rows) - 1} of {len(all_rows)})")
    else:
        rows = all_rows
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} subjects from {args.csv}")
    print(f"output   : {args.out}")
    print(f"prefilter: {'off' if args.prefilter <= 0 else f'{args.prefilter} mm FWHM'}")
    print(f"mode     : {'DRY RUN' if args.dry_run else 'WRITE'}\n")

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    maps = {}          # (src grid, tgt grid) -> index map, computed twice total
    log = []
    n_written = n_skipped = n_failed = 0

    for i, row in enumerate(rows, 1):
        sid = row["subject_id"]
        tgt_path = row[TARGET_COL]
        if not tgt_path or not os.path.exists(tgt_path):
            print(f"  {sid}: no target ({TARGET_COL}), skipping subject", file=sys.stderr)
            continue

        # nibabel is lazy: this reads the header, not the 80 MB of data
        tgt_img = nib.load(tgt_path)
        tgt_affine = tgt_img.affine
        tgt_shape = tuple(int(x) for x in tgt_img.shape[:3])
        tgt_voxvol = abs(np.linalg.det(tgt_affine[:3, :3]))

        for col, tag, kind in MODALITIES:
            src_path = row.get(col, "")
            rec = {f: "" for f in LOG_FIELDS}
            rec.update(subject_id=sid, modality=tag, kind=kind, src_path=src_path)
            dest = os.path.join(args.out, f"{sid}_{tag}_3mm.nii.gz")
            rec["out_path"] = dest

            if not src_path or not os.path.exists(src_path):
                rec["status"] = "missing_source"; log.append(rec); n_failed += 1
                continue
            if os.path.exists(dest) and os.path.getsize(dest) > 0 and not args.overwrite:
                rec["status"] = "skipped_exists"; log.append(rec); n_skipped += 1
                continue
            if args.dry_run:
                rec["status"] = "would_write"; log.append(rec)
                continue

            try:
                src_img = nib.load(src_path)
                src = np.asanyarray(src_img.dataobj, dtype=np.float64)
                src_affine = src_img.affine
                src_voxvol = abs(np.linalg.det(src_affine[:3, :3]))
                rec["src_mass"] = f"{src.sum() * src_voxvol:.1f}"

                key = (src_affine.tobytes(), src.shape, tgt_affine.tobytes(), tgt_shape)
                if key not in maps:
                    maps[key] = build_index_map(src_affine, tgt_affine, tgt_shape)
                    print(f"  built index map for {src.shape} -> {tgt_shape}",
                          file=sys.stderr)
                coords = maps[key]

                def sample(vol):
                    if args.prefilter > 0:
                        vox = np.abs(np.diag(src_affine[:3, :3]))
                        vol = gaussian_filter(vol, (args.prefilter / 2.3548) / vox,
                                              mode="constant")
                    return map_coordinates(vol, coords, order=1,
                                           mode="constant", cval=0).reshape(tgt_shape)

                if kind == "probseg":
                    out = sample(src)
                    # linear interpolation is convex, so no clipping is needed
                    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6, \
                        f"{sid} {tag}: probseg left [0,1]: [{out.min()}, {out.max()}]"
                    out = np.clip(out, 0.0, 1.0)
                else:
                    mask = (src != 0).astype(np.float64)
                    data_r = sample(src)
                    mask_r = sample(mask)
                    keep = mask_r > MASK_THRESH
                    out = np.zeros(tgt_shape, dtype=np.float64)
                    out[keep] = data_r[keep] / mask_r[keep]   # normalized convolution

                out_mass = out.sum() * tgt_voxvol
                rec["out_mass"] = f"{out_mass:.1f}"
                src_mass = src.sum() * src_voxvol
                if src_mass != 0:
                    rec["pct_change"] = f"{100 * (out_mass - src_mass) / src_mass:+.2f}"
                rec["n_nonzero_out"] = str(int((out != 0).sum()))
                rec["vmin"] = f"{out.min():.4f}"
                rec["vmax"] = f"{out.max():.4f}"

                img = nib.Nifti1Image(out.astype(np.float32), tgt_affine)
                img.header.set_xyzt_units("mm")
                # Np carries a stale sform_code of 2; these really are MNI
                img.set_sform(tgt_affine, code=SFORM_CODE)
                img.set_qform(tgt_affine, code=SFORM_CODE)
                # nibabel infers the format from the extension, so the temp
                # name has to keep a real one -- dest + ".part" fails outright
                tmp = dest[:-len(".nii.gz")] + ".part.nii.gz"
                nib.save(img, tmp)
                os.replace(tmp, dest)

                rec["status"] = "written"; n_written += 1
            except Exception as exc:
                rec["status"] = f"error: {exc}"; n_failed += 1
                print(f"  FAILED {sid} {tag}: {exc}", file=sys.stderr)

            log.append(rec)

        if i % 50 == 0:
            print(f"  ... {i}/{len(rows)} subjects "
                  f"(written {n_written}, skipped {n_skipped}, failed {n_failed})",
                  flush=True)

    if not args.dry_run:
        log_path = os.path.join(args.out, log_name)
        with open(log_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            w.writeheader()
            w.writerows(log)
        print(f"\nlog written to {log_path}")

    print(f"\nwritten {n_written}, skipped {n_skipped}, failed {n_failed}")

    # Volume conservation is a bug detector, not a quality metric: a wrong
    # affine, an uninverted matrix or a mirrored axis shows up here as tens of
    # percent. A drift of well under 1% is expected and harmless.
    pcts = [float(r["pct_change"]) for r in log
            if r["kind"] == "probseg" and r["pct_change"]]
    if pcts:
        pcts = np.array(pcts)
        print(f"probseg volume change: mean {pcts.mean():+.2f}%  "
              f"range [{pcts.min():+.2f}, {pcts.max():+.2f}]%")
        if np.abs(pcts).max() > 5:
            print("WARNING: >5% volume change -- check the affines", file=sys.stderr)

    nz = [int(r["n_nonzero_out"]) for r in log
          if r["kind"] == "falff" and r["n_nonzero_out"]]
    if nz:
        print(f"fALFF nonzero voxels: {min(nz)}..{max(nz)} "
              f"({'constant' if min(nz) == max(nz) else 'VARIES - group mask expected constant'})")

    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
