"""MDD vs HC — per-fold AUC, lab benchmark vs this project's multimodal models.

Two cohorts are plotted side by side and they are NOT the same experiment:
  cvbench   1,944 subjects, ICA-53 timecourses (LR uses the 1,378 sFNC edges),
            10 folds, 20% inner validation, patience 30.
  ours      2,055 subjects (sFNC QC tier, MDD+HC), 3 repeats x 10 folds.
The figure says so, because a reader comparing a box to a box would otherwise
assume one protocol.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BENCH = {
 "meanMLP":[0.7713,0.7179,0.7448,0.7003,0.7238,0.7871,0.7193,0.7055,0.8413,0.6986],
 "BolT":[0.7144,0.7218,0.7364,0.6996,0.6809,0.7625,0.7043,0.6754,0.8242,0.7176],
 "Glacier":[0.7323,0.7203,0.7194,0.7012,0.68,0.7848,0.6125,0.7057,0.7879,0.7598],
 "LR":[0.6946,0.683,0.6886,0.7023,0.682,0.7426,0.6775,0.7003,0.7917,0.7266],
 "BrainNetCNN":[0.7346,0.6832,0.7167,0.6879,0.6395,0.7281,0.665,0.6901,0.7776,0.7184]}

OURS = {
 "Baseline\n4 modalities":[0.7043,0.7365,0.8280,0.7181,0.7095,0.7236,0.7310,0.6952,0.7342,0.7522,
   0.6759,0.7193,0.7210,0.7046,0.7474,0.7488,0.7196,0.8198,0.7708,0.7390,
   0.7322,0.7740,0.7608,0.7494,0.7401,0.6806,0.7369,0.6547,0.7706,0.7491],
 "Baseline\n+ early stop":[0.6911,0.7332,0.7973,0.7327,0.7117,0.7274,0.7127,0.7067,0.7436,0.7514,
   0.6830,0.7082,0.6947,0.6836,0.7634,0.7574,0.7017,0.8003,0.7464,0.7235,
   0.7482,0.7445,0.7985,0.7759,0.7648,0.6747,0.7283,0.6660,0.7709,0.7189],
 "Panel A\n4 modalities":[0.6480,0.6705,0.7060,0.5995,0.5903,0.6473,0.6934,0.6443,0.6464,0.6967,
   0.5403,0.6260,0.5970,0.6189,0.6575,0.6496,0.6618,0.6770,0.6426,0.6585,
   0.7400,0.6878,0.6394,0.6598,0.6550,0.6122,0.6201,0.5632,0.6731,0.6483],
 "Panel A\nsFNC only":[0.6321,0.6818,0.6190,0.6919,0.6509,0.6498,0.5794,0.6547,0.6408,0.6042,
   0.6199,0.6230,0.6266,0.5727,0.6271,0.6307,0.5975,0.6976,0.6089,0.6723,
   0.6855,0.7185,0.5743,0.6629,0.6475,0.6182,0.5795,0.5823,0.7127,0.6839],
 "Panel A\n+ early stop":[0.6297,0.5959,0.6646,0.6436,0.5595,0.6424,0.6334,0.5854,0.6329,0.6431,
   0.5686,0.6445,0.6424,0.6077,0.5788,0.6287,0.6193,0.6859,0.6288,0.6775,
   0.6637,0.6082,0.6338,0.6046,0.6186,0.6323,0.5753,0.5591,0.6963,0.6863]}

SITE_ONLY = 0.5860
C_BENCH, C_OURS = "#3A6EA5", "#C4622D"
INK, MUTED = "#1c1c1c", "#6b6b6b"

items = ([(k, np.array(v), "bench") for k, v in BENCH.items()]
         + [(k, np.array(v), "ours") for k, v in OURS.items()])
items.sort(key=lambda t: -t[1].mean())                 # rank by mean, not by group

fig, ax = plt.subplots(figsize=(12.5, 7.2))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

data = [t[1] for t in items]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                medianprops=dict(color="white", linewidth=2.0),
                whiskerprops=dict(color="#555", linewidth=1.2),
                capprops=dict(color="#555", linewidth=1.2),
                boxprops=dict(linewidth=0.8, edgecolor="#333"),
                flierprops=dict(marker="o", markerfacecolor="none",
                                markeredgecolor="#888", markersize=4.5))
for patch, (_, _, grp) in zip(bp["boxes"], items):
    patch.set_facecolor(C_BENCH if grp == "bench" else C_OURS)
    patch.set_alpha(0.80)

# mean marker + per-fold points: the distribution, not just the summary
rng = np.random.default_rng(0)
for i, (_, v, grp) in enumerate(items, start=1):
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v,
               s=13, color="#2b2b2b", alpha=0.30, zorder=4, linewidths=0)
    ax.scatter([i], [v.mean()], marker="D", s=34, color="white",
               edgecolors=INK, linewidths=1.0, zorder=6)

labels = [f"{k}\n{v.mean():.3f} ± {v.std(ddof=1):.3f}" for k, v, _ in items]
ax.set_xticks(range(1, len(items) + 1))
ax.set_xticklabels(labels, fontsize=8.5, color=INK)

ax.axhline(0.50, color="#999", ls="--", lw=1.2, zorder=2)
ax.axhline(SITE_ONLY, color="#B3261E", ls=":", lw=1.6, zorder=2)
ax.text(len(items) + 0.55, 0.50, "chance\n0.500", va="center", fontsize=8.5, color=MUTED)
ax.text(len(items) + 0.55, SITE_ONLY, "site only\n0.586", va="center",
        fontsize=8.5, color="#B3261E")

ax.set_ylim(0.487, 0.872)   # chance line needs air below it, or it merges with the axis
ax.set_ylabel("test AUC", fontsize=11.5, color=INK)
ax.set_title("MDD vs HC — per-fold test AUC", fontsize=13.5, color=INK, pad=14)
ax.grid(axis="y", color="#e2e2e2", lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#cccccc")
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_xlim(0.4, len(items) + 1.6)

ax.legend(handles=[
    Patch(facecolor=C_BENCH, alpha=0.8, edgecolor="#333",
          label="cvbench — unimodal fMRI, n=1,944, 10 folds"),
    Patch(facecolor=C_OURS, alpha=0.8, edgecolor="#333",
          label="this project — multimodal, n=2,055, 3×10 folds")],
    loc="upper right", frameon=False, fontsize=9.5)

fig.text(0.012, 0.015,
         "Diamond = mean; box = quartiles; dots = individual folds.  The two sets "
         "are different cohorts and protocols — compare with care.",
         fontsize=8.2, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig("mdd_vs_hc_comparison.png", dpi=170, facecolor="white")
print("wrote mdd_vs_hc_comparison.png")
for k, v, g in items:
    print(f"  {k.replace(chr(10),' '):<26} {v.mean():.4f} ± {v.std(ddof=1):.4f}  n={len(v)}  [{g}]")
