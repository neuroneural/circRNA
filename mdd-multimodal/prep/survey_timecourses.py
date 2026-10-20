"""Survey the Neuromark time courses: which template, how many subjects, how long.

find_timecourses.py answers "do they exist". This answers "can we actually train
on them", which is three separate questions:

  1. WHICH TEMPLATE. NM1.0 (53 components, named domains), NM2.x (~105,
     multi-scale), NM-WM (white-matter networks). Detected from the component
     axis, not the path, because directory names are not a contract.

  2. COVERAGE. How many subjects, and do they line up with a QC tier
     (3,525 / 2,526 / 2,426) and with the subjects we have labels for. A
     template that covers only part of the cohort changes which experiments
     are possible.

  3. LENGTH, BY SITE. This is the one that can quietly ruin a dynamic model.
     DIRECT Phase 2 spans 23 sites with different TRs and run lengths. sFNC
     hides that -- correlation collapses time. A biLSTM does not: if site A is
     always 150 TRs and site B always 240, sequence length alone identifies the
     scanner, and no amount of feature harmonisation removes it because it is
     not in the features, it is in the shape. This script prints T by site and
     says outright whether length is separable by site.

Shapes are read from headers only (scipy.io.whosmat, h5py .shape, nibabel
.shape), so nothing large is loaded and this stays fast over thousands of files.

    python prep/survey_timecourses.py --root /data/qneuromark \
        --master data/mdd_master.csv --out data/tc_manifest.csv
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# component count -> template name. Detected, then labelled; a count we do not
# recognise is reported as-is rather than forced into one of these.
KNOWN_TEMPLATES = {53: "NM1.0", 105: "NM2.x", 106: "NM2.x"}

SUBJ_RE = [
    # DIRECT / REST-meta-MDD form, e.g. IS023-2-0031 -> site IS023, group 2.
    # This must come first: the generic patterns below do not match it, and a
    # failed parse shows up as "0 subjects matched", not as an error.
    re.compile(r"([A-Za-z]*\d{2,}-\d-\d{3,})"),
    re.compile(r"(sub-[A-Za-z0-9]+)"),
    re.compile(r"(S\d{4,})"),
    re.compile(r"_sub_?(\d{3,})"),
    re.compile(r"([A-Za-z]{2,}\d{4,})"),
]
SKIP_DIR = {".git", ".datalad", "tmp", "logs"}


def subject_from(path):
    base = os.path.basename(path)
    for rx in SUBJ_RE:
        m = rx.search(base)
        if m:
            return m.group(1)
    parent = os.path.basename(os.path.dirname(path))
    for rx in SUBJ_RE:
        m = rx.search(parent)
        if m:
            return m.group(1)
    return None


def shapes_of(path):
    """[(name, shape)] read from headers only — nothing large is loaded."""
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return [("<nifti>", nib.load(path).shape)]
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic != b"\x89HDF\r\n\x1a\n":
        from scipy.io import whosmat
        try:
            return [(n, s) for n, s, _ in whosmat(path)]
        except (NotImplementedError, ValueError):
            pass
    import h5py
    out = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out.append((name, tuple(reversed(obj.shape))))
        f.visititems(visit)
    return out


def classify(shape, comps_hint, tr_range):
    """Return (n_components, n_timepoints) if this looks like a time course."""
    s = [int(x) for x in shape if x and x > 1]
    if len(s) != 2:
        # (subjects, T, C) style is handled by the caller
        return None
    for c in (comps_hint or []):
        if c in s:
            other = s[1] if s[0] == c else s[0]
            if tr_range[0] <= other <= tr_range[1]:
                return c, other
    # no hint matched: guess the smaller axis is components if it is plausible
    lo, hi = min(s), max(s)
    if 20 <= lo <= 400 and tr_range[0] <= hi <= tr_range[1]:
        return lo, hi
    return None


def walk(roots, max_files, exts=(".mat", ".nii", ".nii.gz"), every=2000):
    """Yield candidate files, printing progress so a long job is not silent.

    Walking a large archive can take a long time and there is no way to know in
    advance how long -- so the job says where it is, how fast it is going, and
    what it has found. Flushed, so `tail -f` shows it live.
    """
    t0 = time.time()
    total = 0
    for root in roots:
        if not os.path.isdir(root):
            print(f"!! not a directory: {root}", file=sys.stderr, flush=True)
            continue
        print(f"\n  walking {root}", flush=True)
        n = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIR and not d.startswith(".")]
            for f in filenames:
                if f.lower().endswith(exts):
                    n += 1
                    total += 1
                    if total % every == 0:
                        el = time.time() - t0
                        print(f"    {total:>8} files  {el/60:6.1f} min  "
                              f"{total/max(el, 1e-9):6.0f} files/s   "
                              f"{dirpath[-60:]}", flush=True)
                    if n > max_files:
                        print(f"  (hit --max-files in {root}; narrow the root "
                              f"or raise --max-files)", flush=True)
                        break
                    yield os.path.join(dirpath, f)
            else:
                continue
            break
        print(f"    {n} candidate files under {root} "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", default=[
        # the MDD subtree, not all of /data/qneuromark (the whole archive)
        "/data/qneuromark/Data/Depression/MDD_DIRECT",
        "/data/users3/bbaker/projects/MDD_preproc/results/ica"])
    ap.add_argument("--trinfo", default=None,
                    help="TRInfo.tsv from Data_BIDS — turns run length into seconds "
                         "and shows whether TR itself marks the site")
    ap.add_argument("--master", default="data/mdd_master.csv")
    ap.add_argument("--out", default="data/tc_manifest.csv")
    ap.add_argument("--components", default="53,105,106",
                    help="expected component counts; others are still reported")
    ap.add_argument("--tr-range", default="50,3000")
    ap.add_argument("--max-files", type=int, default=500000)
    ap.add_argument("--progress-every", type=int, default=2000,
                    help="print a progress line every N files scanned")
    ap.add_argument("--sample-per-dir", type=int, default=0,
                    help="read shapes for at most N files per directory (0 = all)")
    args = ap.parse_args()

    comps_hint = [int(x) for x in args.components.split(",")]
    lo, hi = (int(x) for x in args.tr_range.split(","))

    rows, per_dir, skipped = [], {}, 0
    print(f"scanning {len(args.root)} root(s). Progress every "
          f"{args.progress_every} files; nothing is loaded but headers.",
          flush=True)
    for p in walk(args.root, args.max_files, every=args.progress_every):
        d = os.path.dirname(p)
        per_dir[d] = per_dir.get(d, 0) + 1
        if args.sample_per_dir and per_dir[d] > args.sample_per_dir:
            continue
        try:
            shp = shapes_of(p)
        except Exception:                                        # noqa: BLE001
            skipped += 1
            continue
        for name, s in shp:
            hit = classify(s, comps_hint, (lo, hi))
            if hit is None:
                continue
            c, t = hit
            rows.append(dict(path=p, dir=d, var=name, subject=subject_from(p),
                             n_components=c, n_timepoints=t,
                             template=KNOWN_TEMPLATES.get(c, f"unknown-{c}")))
            break

    if not rows:
        print("\nNo time-course-shaped arrays found under: " + ", ".join(args.root))
        print("Widen --tr-range, or check the root. (Files unreadable: %d)" % skipped)
        return 1

    df = pd.DataFrame(rows)
    print(f"\n{'='*78}\n{len(df)} time-course files found"
          + (f"  ({skipped} unreadable)" if skipped else "") + f"\n{'='*78}")

    # ---- 1. template ------------------------------------------------------
    print("\nTEMPLATES")
    for (tpl, c), g in df.groupby(["template", "n_components"]):
        t = g["n_timepoints"]
        print(f"  {tpl:<12} {c:>4} components  {len(g):>6} files   "
              f"T min {t.min()} / median {int(t.median())} / max {t.max()}"
              + ("   <- variable length" if t.min() != t.max() else "   (fixed length)"))
        for d, gg in list(g.groupby("dir"))[:4]:
            print(f"        {len(gg):>6} in {d}")

    # ---- 2. coverage ------------------------------------------------------
    ids = df["subject"].dropna().unique()
    print(f"\nCOVERAGE\n  distinct subject IDs parsed from filenames: {len(ids)}"
          + ("   <- parsing failed on some files" if df["subject"].isna().any() else ""))
    for n, label in ((3525, "all"), (2526, "clean / NM1.0 tier"), (2426, "super_clean tier")):
        if abs(len(ids) - n) <= 2:
            print(f"  matches the {label} tier ({n})")

    master = None
    if os.path.exists(args.master):
        master = pd.read_csv(args.master)
        master["id"] = master["id"].astype(str)
        have = set(ids)
        master["has_tc"] = master["id"].isin(have)
        n_hit = int(master["has_tc"].sum())
        print(f"  joined to {args.master}: {n_hit}/{len(master)} subjects matched")
        if n_hit == 0:
            print("  !! zero matched — the subject IDs in these filenames do not look")
            print("     like the IDs in the master CSV. Fix the join before trusting")
            print("     anything downstream; a silent mismatch mislabels every subject.")
        else:
            if "group_label" in master:
                print("  by group: " + ", ".join(
                    f"{k} {int(v)}" for k, v in
                    master[master.has_tc]["group_label"].value_counts().items()))
            for col in ("in_clean", "in_super_clean"):
                if col in master:
                    flag = master[col].astype(str).str.lower().isin(("true", "1", "1.0"))
                    print(f"  {col}: {int((master.has_tc & flag).sum())} of "
                          f"{int(flag.sum())} have time courses")
    else:
        print(f"  (no master CSV at {args.master} — skipping the label join)")

    # ---- 3. length by site, the confound check ---------------------------
    if master is not None and "site" in master and master["has_tc"].any():
        m = master.merge(df.groupby("subject")["n_timepoints"].median().reset_index(),
                         left_on="id", right_on="subject", how="inner")

        # TRInfo.tsv, if given, turns TRs into seconds and exposes TR-by-site
        if args.trinfo and os.path.exists(args.trinfo):
            tr = pd.read_csv(args.trinfo, sep="\t")
            idc = next((c for c in tr.columns
                        if c.lower() in ("id", "subid", "subject", "participant_id")),
                       tr.columns[0])
            trc = next((c for c in tr.columns if "tr" == c.lower()
                        or "repetition" in c.lower()), None)
            if trc:
                tr[idc] = tr[idc].astype(str)
                m = m.merge(tr[[idc, trc]].rename(columns={idc: "id", trc: "TR"}),
                            on="id", how="left")
                m["seconds"] = m["n_timepoints"] * pd.to_numeric(m["TR"], errors="coerce")
                print("\nTR BY SITE  (from %s)" % args.trinfo)
                print(m.groupby("site")[["TR", "n_timepoints", "seconds"]]
                        .agg(["nunique", "median"]).to_string())
                if m.groupby("site")["TR"].nunique().max() == 1 and m["TR"].nunique() > 1:
                    print("  !! TR is constant within site and differs between sites —")
                    print("     TR is a site label too, not just run length.")
            else:
                print(f"\n  (no TR column found in {args.trinfo}; "
                      f"columns are {list(tr.columns)[:8]})")
        if len(m):
            print("\nLENGTH BY SITE  (the dynamic-encoder confound)")
            g = m.groupby("site")["n_timepoints"].agg(["count", "min", "median", "max"])
            print(g.to_string())
            per_site = m.groupby("site")["n_timepoints"].nunique()
            constant_within = bool((per_site == 1).all())
            n_lengths = m["n_timepoints"].nunique()
            print()
            if n_lengths == 1:
                print("  All subjects share one length. No length confound; a fixed-size")
                print("  encoder works and nothing special is needed.")
            elif constant_within and n_lengths > 1:
                print("  !! Length is CONSTANT WITHIN each site and DIFFERS BETWEEN sites.")
                print("     Sequence length identifies the scanner exactly. A dynamic")
                print("     encoder can score well by reading length alone, and feature-")
                print("     level site correction cannot remove it — it is not in the")
                print("     features. Crop every subject to a common length before")
                print("     training, and run site_check.yaml on the cropped data.")
            else:
                print("  Length varies both within and between sites. Cropping to a common")
                print("  length is still the safe default; check the table above for how")
                print("  much data that costs.")
            common = int(m["n_timepoints"].min())
            print(f"  shortest run = {common} TRs — cropping everyone to that keeps "
                  f"{100.0*common/float(m['n_timepoints'].median()):.0f}% of the median run")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nmanifest -> {args.out}  ({len(df)} rows)")
    print("Columns: path, dir, var, subject, n_components, n_timepoints, template")
    return 0


if __name__ == "__main__":
    sys.exit(main())
