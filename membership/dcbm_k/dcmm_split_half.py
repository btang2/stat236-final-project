"""Single-column redraw of the report's Figure 3 (DCMM split-half reliability by K), styled like dcbm_figures.py.

Reads results/membership/tables/s1_split_half.csv (from s1_choose_K.py: 100 manager halvings per date).
Writes results/dcbm_k/dcmm_split_half.{pdf,png}.

Run: python -m membership.dcbm_k.dcmm_split_half
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from membership.core import OUT as MOUT
from membership.dcbm_k.dcbm_figures import GRID, INK, MUTED, W, save, style
from membership.fit import COLORS

C_MM = COLORS[0]
YEARS = ["2006", "2008"]


def main():
    s = pd.read_csv(MOUT / "tables" / "s1_split_half.csv", dtype={"year": str})
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.95), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.99, top=0.88, bottom=0.2, wspace=0.08)
    for ax, y in zip(axes, YEARS):
        g = s[s.year == y]
        ax.axvspan(3.6, 4.4, color="#eaf1fb", lw=0, zorder=0)
        for K, gk in g.groupby("K"):
            ax.plot([K, K], [gk.r_q10.min(), gk.split_half_r_median.max()], color=GRID, lw=3.2,
                    solid_capstyle="butt", zorder=1)
            ok = gk.split_half_r_median >= 0.7
            ax.scatter([K] * ok.sum(), gk.split_half_r_median[ok], s=14, color=C_MM, edgecolor="white", lw=0.5,
                       zorder=3)
            ax.scatter([K] * (~ok).sum(), gk.split_half_r_median[~ok], s=14, facecolor="white", edgecolor=C_MM,
                       lw=0.9, zorder=3)
        ax.axhline(0.7, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
        ax.set_title(y, fontsize=7, color=INK, loc="left")
        ax.set_xticks(range(2, 7))
        ax.set_xlim(1.5, 6.5)
        ax.set_xlabel("K")
        ax.set_ylim(0.15, 1.02)
        style(ax)
    axes[0].set_ylabel("split-half correlation of $\\hat\\pi_k$\n(median over 100 splits)")
    axes[0].text(1.6, 0.71, "0.7", color=MUTED, fontsize=6, ha="left", va="bottom")
    axes[1].text(4, 0.18, "K = 4", color=C_MM, fontsize=6, ha="center", va="bottom")
    save(fig, "dcmm_split_half")


if __name__ == "__main__":
    main()
