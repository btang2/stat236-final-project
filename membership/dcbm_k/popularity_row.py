"""Full-width redraw of report Figure 6 (membership vs number of active owners) for the top of a two-column page.

One panel per profile, both dates overlaid (2006 darker, 2008 lighter), Spearman correlation with log owners
annotated. Writes out/popularity_row.{pdf,png}.
Run: python -m membership.dcbm_k.popularity_row
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.stats import spearmanr

from membership.dcbm_k.k_test import OUT
from membership.fit import COLORS, K, NAMES, fit

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "pdf.fonttype": 42, "axes.linewidth": 0.5,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                     "ytick.major.size": 2, "legend.frameon": False})


def lighten(c, f=0.55):
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f)


def main():
    res = fit()
    fig, axes = plt.subplots(1, K, figsize=(7.0, 1.95), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.9, bottom=0.2, wspace=0.07)
    out = {}
    for k, ax in enumerate(axes):
        rhos = []
        for y, shade, z in (("2008", lighten(COLORS[k]), 1), ("2006", COLORS[k], 2)):
            r = res[y]
            n = r["x"].n_active_owners.values
            p = r["Pi"][:, k]
            ax.scatter(n, p, s=3.5, color=shade, lw=0, alpha=0.8, zorder=z, label=y)
            rhos.append((y, spearmanr(n, p)[0]))
        rhos = dict(rhos)
        out[NAMES[k]] = rhos
        ax.text(0.97 if rhos["2006"] < 0.5 else 0.03, 0.96,
                f"$\\rho$ = {rhos['2006']:+.2f} / {rhos['2008']:+.2f}", transform=ax.transAxes,
                ha="right" if rhos["2006"] < 0.5 else "left", va="top", fontsize=6.3, color=INK)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(NAMES[k], loc="left", fontsize=7.5, color=COLORS[k] if k != 3 else "#a87300",
                     fontweight="bold", pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("membership $\\hat\\pi_{ik}$")
    fig.supxlabel("number of active owners (log scale)", fontsize=7, color=INK, y=0.02)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, color=INK), plt.Line2D([], [], ls="", marker="o", ms=3,
                                                                           color=lighten(INK, 0.6))]
    fig.legend(h, ["2006 (darker)", "2008 (lighter)"], loc="lower left", ncol=2, fontsize=6.3,
               bbox_to_anchor=(0.06, -0.01), handletextpad=0.2, columnspacing=1)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"popularity_row.{ext}", dpi=300)
    plt.close(fig)
    print({k: {y: round(v, 2) for y, v in d.items()} for k, d in out.items()})


if __name__ == "__main__":
    main()
