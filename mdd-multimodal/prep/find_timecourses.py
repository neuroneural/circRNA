"""Find out whether ICA component TIME COURSES exist on the server, or only sFNC.

Why this needs a script
-----------------------
GIFT writes a lot of files and the names do not say what is inside them. The
postprocess MAT we already use holds sFNC (subjects x 1378) -- correlations
between component time courses, with time already collapsed away. The dynamic
fMRI encoder (pre_empt.pdf p.6: biLSTM per component -> self-attention -> a graph
per timepoint) needs the time courses themselves, subjects x T x 53. You cannot
recover one from the other: sFNC is a summary, and the summarising is lossy.

So this script looks for arrays whose SHAPE says "time course" -- one axis equal
to the component count (53 or 105) and another in a plausible TR range -- rather
than trusting filenames.

What it checks, in order
------------------------
1. Filename patterns GIFT uses for time courses and back-reconstruction.
2. File size. A per-subject time course is tiny (150 x 53 x 4 bytes = ~32 KB).
   A per-subject spatial map is enormous (53 x ~900k voxels x 4 = ~190 MB).
   Size alone separates them, which is useful when there are thousands of files.
3. The actual arrays inside a sample of files (MAT v7 via scipy, v7.3 via h5py,
   NIfTI via nibabel), reporting every variable whose shape looks like a time
   course and every variable that looks like sFNC instead.

    python prep/find_timecourses.py --root /data/users2/ppopov1/datasets/MDD_DIRECT
    python prep/find_timecourses.py --root ... --inspect 5 --components 53,105

Step 1 and 2 are pure filesystem and run anywhere. Step 3 imports scipy/h5py/
nibabel, so run the whole thing inside a SLURM job (scripts/run_find_tc.sh) --
conda is disabled on the login node.
"""

import argparse
import os
import sys

# GIFT's own naming. Time courses first, then things that are NOT time courses
# but are easy to mistake for them.
TC_PATTERNS = [
    "timecourses", "time_courses", "_tc_", "tc.mat",
    "_ica_br", "_ica_c1-1", "_ica_c",     # back-reconstruction: holds compset.tc
    "_sub", "subject_loadings",
]
NOT_TC_PATTERNS = ["component_ica", "_agg_", "mask", "mean_", "std_", "tmap"]
DFNC_PATTERNS = ["dfnc", "dynamic"]

TC_KEYS = ("tc", "timecourses", "time_courses", "TC", "compset", "tc_all",
           "icatb_tc", "sub_tc", "timecourse")
FNC_KEYS = ("fnc_corrs_all", "fnc_corrs", "FNCM", "sFNC")
# Power spectra are the trap: spectra_tc_all is (N, 53, 129) -- a component axis
# and a second axis that lands squarely in any plausible TR range. It is
# frequency, not time, and it cannot drive a dynamic encoder. The name is the
# only thing that distinguishes it, so the name is what we check.
SPECTRA_KEYS = ("spectra", "spectrum", "psd", "freq", "fft", "power")


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024.0


def looks_like_timecourse(shape, comps, tr_range):
    """One axis == a component count, another axis in a plausible TR range.

    An (N, C, C) FNC stack also has a C axis and an N that can fall in the TR
    range, so a repeated component axis disqualifies it -- that is a square
    connectivity matrix, not time.
    """
    s = [int(x) for x in shape if x and x > 1]
    if len(s) < 2:
        return False
    for c in comps:
        if s.count(c) >= 2:                 # C x C -> connectivity, not time
            return False
    for c in comps:
        if c in s:
            rest = list(s)
            rest.remove(c)
            if any(tr_range[0] <= r <= tr_range[1] for r in rest):
                return True
    return False


def looks_like_fnc(shape, comps):
    """(N, C*(C-1)/2) or (N, C, C) -- time already collapsed."""
    s = [int(x) for x in shape if x and x > 1]
    for c in comps:
        k = c * (c - 1) // 2
        if k in s or (len(s) >= 3 and s.count(c) >= 2):
            return True
    return False


# ----------------------------------------------------------------- scanning
def scan(root, max_files):
    hits = {"tc": [], "dfnc": [], "other_mat": [], "nii": []}
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            n += 1
            if n > max_files:
                print(f"  (stopped after {max_files} files; narrow --root or raise --max-files)")
                return hits, n
            low = f.lower()
            if not low.endswith((".mat", ".nii", ".nii.gz")):
                continue
            p = os.path.join(dirpath, f)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            rec = (p, sz)
            if any(k in low for k in DFNC_PATTERNS):
                hits["dfnc"].append(rec)
            elif (any(k in low for k in TC_PATTERNS)
                  and not any(k in low for k in NOT_TC_PATTERNS)):
                hits["tc"].append(rec)
            elif low.endswith(".mat"):
                hits["other_mat"].append(rec)
            else:
                hits["nii"].append(rec)
    return hits, n


def summarise_group(name, recs, limit=8):
    if not recs:
        print(f"  {name:<26} none")
        return
    total = sum(s for _, s in recs)
    sizes = sorted(s for _, s in recs)
    med = sizes[len(sizes) // 2]
    print(f"  {name:<26} {len(recs):>6} files, {human(total):>9} total, "
          f"median {human(med)}")
    by_dir = {}
    for p, s in recs:
        by_dir.setdefault(os.path.dirname(p), []).append(s)
    for d, ss in sorted(by_dir.items(), key=lambda kv: -len(kv[1]))[:limit]:
        print(f"      {len(ss):>6} x median {human(sorted(ss)[len(ss)//2]):>9}  {d}")


# --------------------------------------------------------------- inspection
def read_any(path):
    """{name: shape} for a MAT (either version) or a NIfTI."""
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return {"<nifti>": nib.load(path).shape}
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head[:8] != b"\x89HDF\r\n\x1a\n":
        try:
            from scipy.io import loadmat
            m = loadmat(path, squeeze_me=True, struct_as_record=False)
            out = {}
            for k, v in m.items():
                if k.startswith("__"):
                    continue
                if hasattr(v, "_fieldnames"):        # a MATLAB struct, e.g. compset
                    for fn in v._fieldnames:
                        sub = getattr(v, fn)
                        out[f"{k}.{fn}"] = getattr(sub, "shape", ())
                else:
                    out[k] = getattr(v, "shape", ())
            return out
        except (NotImplementedError, ValueError):
            pass
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = tuple(reversed(obj.shape))   # h5py transposes MATLAB
        f.visititems(visit)
    return out


def inspect(paths, comps, tr_range):
    found_tc, found_fnc, found_spec = [], [], []
    for p in paths:
        print(f"\n  --- {p}  ({human(os.path.getsize(p))})")
        try:
            shapes = read_any(p)
        except Exception as e:                                   # noqa: BLE001
            print(f"      could not read: {e}")
            continue
        if not shapes:
            print("      (no arrays)")
        for k, sh in sorted(shapes.items()):
            tag = ""
            if any(t in k.lower() for t in SPECTRA_KEYS):
                tag = "   <-- SPECTRA (frequency axis, not time)"
                found_spec.append((p, k, sh))
            elif looks_like_fnc(sh, comps):         # test FNC first: a C x C
                tag = "   <-- sFNC (time already collapsed)"   # stack can mimic
                found_fnc.append((p, k, sh))                   # a time course
            elif looks_like_timecourse(sh, comps, tr_range):
                tag = "   <== TIME COURSE"
                found_tc.append((p, k, sh))
            elif any(t in k.lower() for t in TC_KEYS):
                tag = "   (name suggests tc, shape does not)"
            print(f"      {k:<34} {str(tuple(sh)):<24}{tag}")
    return found_tc, found_fnc, found_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+",
                    default=["/data/qneuromark/Data/Depression/MDD_DIRECT"],
                    help="one or more directories to walk")
    ap.add_argument("--components", default="53,105",
                    help="component counts that mark a time-course axis")
    ap.add_argument("--tr-range", default="80,1200",
                    help="plausible number of timepoints, min,max")
    ap.add_argument("--inspect", type=int, default=3,
                    help="how many candidate files to open and report shapes for")
    ap.add_argument("--max-files", type=int, default=400000)
    args = ap.parse_args()

    roots = [r for r in (args.root or []) if r]
    comps = [int(x) for x in args.components.split(",")]
    lo, hi = (int(x) for x in args.tr_range.split(","))

    all_hits = {"tc": [], "dfnc": [], "other_mat": [], "nii": []}
    total_files = 0
    for root in roots:
        if not os.path.isdir(root):
            print(f"!! not a directory: {root}")
            continue
        print(f"\nwalking {root}")
        h, n = scan(root, args.max_files)
        total_files += n
        for k in all_hits:
            all_hits[k].extend(h[k])

    print(f"\n{'='*74}\nfile counts ({total_files} entries walked)\n{'='*74}")
    summarise_group("time-course candidates", all_hits["tc"])
    summarise_group("dFNC candidates", all_hits["dfnc"])
    summarise_group("other .mat", all_hits["other_mat"])
    summarise_group("other .nii", all_hits["nii"])

    print(f"\n{'='*74}\nopening a sample to check ACTUAL array shapes\n{'='*74}")
    sample = ([p for p, _ in sorted(all_hits["tc"], key=lambda r: r[1])[:args.inspect]]
              + [p for p, _ in sorted(all_hits["dfnc"], key=lambda r: r[1])[:args.inspect]]
              + [p for p, _ in sorted(all_hits["other_mat"],
                                      key=lambda r: -r[1])[:args.inspect]])
    tc, fnc, spec = inspect(sample, comps, (lo, hi)) if sample else ([], [], [])

    print(f"\n{'='*74}\nVERDICT\n{'='*74}")
    if tc:
        print("  TIME COURSES FOUND — the dynamic fMRI encoder is buildable.")
        for p, k, sh in tc[:10]:
            print(f"    {tuple(sh)}  {k}  in  {p}")
        print("\n  Next: confirm coverage — is this per-subject (how many files?)"
              "\n  or one array with a subjects axis, and does the count match a QC tier"
              "\n  (3525 / 2526 / 2426)?")
    elif fnc or spec:
        if spec and not fnc:
            print("  ONLY SPECTRA FOUND — frequency, not time. Do not mistake")
            print("  spectra_tc_all for time courses; the '_tc_' in its name refers")
            print("  to the time courses it was COMPUTED FROM, which are not here.")
        print("  ONLY sFNC/SPECTRA FOUND. Time was collapsed at the postprocess step and")
        print("  cannot be recovered from these files. The dynamic encoder would need")
        print("  GIFT re-run to write time courses, or the back-reconstruction files")
        print("  (*_ica_br*.mat) if they were kept. Static view works today.")
    else:
        print("  NOTHING CONCLUSIVE. Either the search root is wrong or the files are")
        print("  named unusually. Re-run with --root pointed at the GIFT output")
        print("  directory, and raise --inspect.")
    print()
    return 0 if (tc or fnc) else 1


if __name__ == "__main__":
    sys.exit(main())
