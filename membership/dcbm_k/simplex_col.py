"""Single-column redraw of the report's SCORE simplex figure (report_figures.g3_simplex), dates stacked vertically,
vertices labelled Cluster 1-4 (same order as fig_matrix_row: 1 Blue chip, 2 Income, 3 Growth, 4 Small value).

Writes out/simplex_col.{pdf,png}. Run: python -m membership.dcbm_k.simplex_col
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe

from membership.dcbm_k.k_test import OUT
from membership.report_figures import GRID, INK, LABELS, MUTED, best_plane
from membership.fit import COLORS, NAMES, data

W = 3.35
EDGE = "#c0392b"
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "pdf.fonttype": 42})
LABEL_COL = [COLORS[0], COLORS[1], COLORS[2], "#a87300"]


def panel(ax, y, tag):
    d = data(y)
    R, V, P, x = d["R"], d["V"], d["Pi"], d["x"]
    c = V.mean(0)
    Wp = best_plane(V - c)
    r2, v2 = (R - c) @ Wp.T, (V - c) @ Wp.T
    a = np.arctan2(v2[0, 1], v2[0, 0]) - np.pi / 2          # cluster 1 at the top
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    r2, v2 = r2 @ rot, v2 @ rot
    if v2[1, 0] > v2[3, 0]:                                  # cluster 2 on the left
        r2[:, 0] *= -1
        v2[:, 0] *= -1
    dom, strength, keep = P.argmax(1), P.max(1), ~d["pruned"]
    for k in range(4):
        m = keep & (dom == k)
        ax.scatter(r2[m, 0], r2[m, 1], s=3 + 7 * strength[m], color=COLORS[k], alpha=0.35 + 0.55 * strength[m],
                   lw=0, zorder=2)
    ax.scatter(r2[~keep, 0], r2[~keep, 1], s=5, facecolor="none", edgecolor=MUTED, lw=0.4, zorder=2)
    for i in range(4):
        for j in range(i + 1, 4):
            ax.plot(*v2[[i, j]].T, color=EDGE, lw=0.7, zorder=3)
    span = np.ptp(v2, 0).max()
    for k in range(4):
        ax.scatter(*v2[k], marker="*", s=70, color=COLORS[k], edgecolor=INK, lw=0.5, zorder=5)
        out = v2[k] - v2.mean(0)
        if abs(out[0]) > abs(out[1]):          # side vertex: label above the star, aligned inward
            pos, ha, va = v2[k] + np.array([-0.02 if out[0] > 0 else 0.02, 0.06]) * span, \
                ("right" if out[0] > 0 else "left"), "bottom"
        else:                                  # top/bottom vertex: label further out
            pos, ha, va = v2[k] + out / np.linalg.norm(out) * 0.13 * span, "center", "center"
        ax.text(*pos, NAMES[k], ha=ha, va=va, fontsize=6.5, fontweight="bold", color=LABEL_COL[k],
                zorder=7, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    tick = x.ticker.values
    for t in LABELS[y]:
        i = np.where(tick == t)[0]
        if not len(i):
            continue
        i = i[0]
        ax.scatter(*r2[i], s=8, facecolor="none", edgecolor=INK, lw=0.5, zorder=4)
        ax.annotate(t, r2[i], xytext=(2, 1.5), textcoords="offset points", fontsize=4.8, color=INK, zorder=6)
    lo = np.minimum(np.quantile(r2, 0.01, 0), v2.min(0))
    hi = np.maximum(np.quantile(r2, 0.99, 0), v2.max(0))
    pad = 0.13 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.5)
    ax.set_title(f"{tag} {'December 2006' if y == '2006' else 'June 2008'} ({len(R)} stocks)", loc="left",
                 fontsize=7, color=INK, pad=3)


def main():
    fig, axes = plt.subplots(2, 1, figsize=(W, 6.1))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.01, hspace=0.09)
    for ax, y, tag in zip(axes, ("2006", "2008"), ("(a)", "(b)")):
        panel(ax, y, tag)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"simplex_col.{ext}", dpi=300)
    plt.close(fig)
    print("saved", OUT / "simplex_col.png")


if __name__ == "__main__":
    main()
