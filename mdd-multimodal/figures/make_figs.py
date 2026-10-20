"""Two figures.

fig1  multimodal only: four architectures on the same four modalities.
fig2  the same four, multimodal vs sFNC-only, to show what the three
      structural modalities are worth to each one.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fold_data import MULTI, SFNC, SITE_ONLY

C_BASE, C_PANEL = "#3A6EA5", "#C4622D"
C_MULTI, C_SFNC = "#3A6EA5", "#C4622D"
INK, MUTED, GRID = "#1c1c1c", "#6b6b6b", "#e2e2e2"


def dress(ax, lo, hi, title):
    ax.axhline(SITE_ONLY, color="#B3261E", ls=":", lw=1.6, zorder=2)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("test AUC", fontsize=11.5, color=INK)
    ax.set_title(title, fontsize=13.5, color=INK, pad=14)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=MUTED, labelsize=9.5)


def box(ax, pos, vals, color, width=0.62):
    bp = ax.boxplot([vals], positions=[pos], widths=width, patch_artist=True,
                    showfliers=True,
                    medianprops=dict(color="white", linewidth=2.0),
                    whiskerprops=dict(color="#555", linewidth=1.2),
                    capprops=dict(color="#555", linewidth=1.2),
                    boxprops=dict(linewidth=0.8, edgecolor="#333"),
                    flierprops=dict(marker="o", markerfacecolor="none",
                                    markeredgecolor="#888", markersize=4.5))
    bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.80)
    rng = np.random.default_rng(int(pos * 97))
    ax.scatter(np.full(len(vals), pos) + rng.uniform(-0.12, 0.12, len(vals)), vals,
               s=12, color="#2b2b2b", alpha=0.28, zorder=4, linewidths=0)
    ax.scatter([pos], [np.mean(vals)], marker="D", s=34, color="white",
               edgecolors=INK, linewidths=1.0, zorder=6)


# ---------------------------------------------------------------- figure 1
order = sorted(MULTI, key=lambda k: -np.mean(MULTI[k]))
fig, ax = plt.subplots(figsize=(9.2, 6.4))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
for i, k in enumerate(order, start=1):
    box(ax, i, np.array(MULTI[k]), C_PANEL if "Panel A" in k else C_BASE)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels([f"{k}\n{np.mean(MULTI[k]):.3f} ± {np.std(MULTI[k], ddof=1):.3f}"
                    for k in order], fontsize=9, color=INK)
ax.set_xlim(0.4, len(order) + 1.15)
dress(ax, 0.52, 0.845,
      "MDD vs HC — four architectures, all four modalities\n"
      "n = 2,055 · GM + CSF + fALFF + sFNC · 3×10 folds")
ax.text(len(order) + 0.5, SITE_ONLY + 0.004, "site only\n0.586", va="bottom",
        fontsize=8.5, color="#B3261E", linespacing=1.25)
ax.legend(handles=[Patch(facecolor=C_BASE, alpha=.8, edgecolor="#333",
                         label="baseline — no transparent layer"),
                   Patch(facecolor=C_PANEL, alpha=.8, edgecolor="#333",
                         label="Panel A — shared 53-node transparent graph")],
          loc="lower left", frameon=False, fontsize=9.5)
fig.text(0.012, 0.015, "Diamond = mean; box = quartiles; dots = individual folds.",
         fontsize=8.2, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig("fig1_multimodal.png", dpi=170, facecolor="white")

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(11.2, 6.6))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
centres, labels = [], []
for i, k in enumerate(order):
    c = i * 1.0 + 1
    box(ax, c - 0.19, np.array(MULTI[k]), C_MULTI, width=0.33)
    box(ax, c + 0.19, np.array(SFNC[k]), C_SFNC, width=0.33)
    d = np.mean(SFNC[k]) - np.mean(MULTI[k])
    centres.append(c)
    labels.append(f"{k}\n{np.mean(MULTI[k]):.3f}  vs  {np.mean(SFNC[k]):.3f}"
                  f"\n({d:+.3f})")
ax.set_xticks(centres); ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.set_xlim(0.45, len(order) + 0.95)
dress(ax, 0.36, 0.845,
      "What the three structural modalities are worth\n"
      "n = 2,055 · all four modalities vs sFNC alone · 3×10 folds")
ax.text(len(order) + 0.5, SITE_ONLY + 0.006, "site only\n0.586", va="bottom",
        fontsize=8.5, color="#B3261E", linespacing=1.25)
ax.legend(handles=[Patch(facecolor=C_MULTI, alpha=.8, edgecolor="#333",
                         label="GM + CSF + fALFF + sFNC"),
                   Patch(facecolor=C_SFNC, alpha=.8, edgecolor="#333",
                         label="sFNC only")],
          loc="lower left", frameon=False, fontsize=9.5)
fig.text(0.012, 0.030,
         "Diamond = mean; box = quartiles; dots = individual folds. Same protocol "
         "throughout: 60 epochs, no early stopping.",
         fontsize=8.2, color=MUTED)
fig.text(0.012, 0.008,
         "Panel A's sFNC-only fit had NOT converged — held-out AUC was still "
         "rising in all 30 folds; 1000 epochs with early stopping reaches 0.638.",
         fontsize=8.2, color=MUTED)
fig.tight_layout(rect=[0, 0.062, 1, 1])
fig.savefig("fig2_multimodal_vs_sfnc.png", dpi=170, facecolor="white")
print("wrote fig1_multimodal.png and fig2_multimodal_vs_sfnc.png")
