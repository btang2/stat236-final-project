"""Figure 2 redrawn as one row for the top of a two-column paper (LaTeX figure*, ~7 in wide).

(a) cosine co-ownership matrix A in data order
(b) A with stocks sorted by dominant profile, then by membership within profile (strip = membership vector)
(c) degree divided out as in SCORE: rank-4 signal part Omega_hat_ij / (xi_1(i) xi_1(j)), same order

Same data, fit and ordering as report_figures.g1_matrix, panels (a), (b), (d).
Run: python -m membership.dcbm_k.fig_matrix_row   (writes results/dcbm_k/fig_matrix_row.{pdf,png})
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from membership.core import cosine
from membership.dcbm_k.k_test import OUT
from membership.report_figures import CMAP, GRID, INK, order
from membership.fit import COLORS, NAMES, data

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "pdf.fonttype": 42})
SHORT = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]


def main(year="2006", strips=True):
    d = data(year)
    P, A = d["Pi"], cosine(d["B"])
    lam, xi = np.linalg.eigh(A)
    top = np.argsort(-np.abs(lam))[:4]
    lam, xi = lam[top], xi[:, top]
    xi[:, 0] *= np.sign(xi[:, 0].sum())
    Om = (xi * lam) @ xi.T
    Om_free = Om / np.outer(xi[:, 0], xi[:, 0])
    o, dom = order(P)
    n = len(P)
    cuts = np.cumsum(np.bincount(dom, minlength=4))[:-1]

    panels = [(A, np.arange(n), "(a) $A$, data order", False, "cosine similarity"),
              (A, o, "(b) $A$, sorted by cluster", True, "cosine similarity"),
              (Om_free, o, r"(c) $\hat\Omega_{ij}/\hat\xi_1(i)\hat\xi_1(j)$: degree removed", True,
               "relative affinity")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.17, wspace=0.04)
    for ax, (M, idx, title, strip, cblab) in zip(axes, panels):
        S = M[np.ix_(idx, idx)].copy()
        np.fill_diagonal(S, np.nan)
        lo, hi = np.nanquantile(S, [0.01, 0.99])
        im = ax.imshow(S, cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest", aspect="equal")
        ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=11 if strips else 4)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
            s.set_linewidth(0.5)
        if strip:
            for c in cuts:
                ax.axhline(c - 0.5, color=INK, lw=0.4)
                ax.axvline(c - 0.5, color=INK, lw=0.4)
            if strips:
                sax = ax.inset_axes([0, 1.008, 1, 0.045])
                bottom = np.zeros(n)
                for k in range(4):
                    sax.bar(np.arange(n), P[idx, k], bottom=bottom, width=1.0, color=COLORS[k], lw=0)
                    bottom += P[idx, k]
                sax.set_xlim(-0.5, n - 0.5)
                sax.set_ylim(0, 1)
                sax.axis("off")
            starts, ends = np.r_[0, cuts], np.r_[cuts, n]
            for k in range(4):
                ax.text((starts[k] + ends[k]) / 2 - 0.5, n + 4, SHORT[k], ha="center", va="top", fontsize=6,
                        color=COLORS[k] if k != 3 else "#a87300")
        cax = ax.inset_axes([0.2, -0.13, 0.6, 0.03])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.3)
        cb.ax.tick_params(labelsize=5.5, width=0.3, length=1.5, pad=1)
        cb.locator = matplotlib.ticker.MaxNLocator(3)
        cb.update_ticks()
        cb.set_label(cblab, fontsize=6, labelpad=1)
        if cblab == "relative affinity":
            cb.set_ticks([])
            cb.set_label("relative affinity (low → high)", fontsize=6, labelpad=1)
    name = "fig_matrix_row" if strips else "fig_matrix_row_nostrip"
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf", "n =", n, "profile sizes", np.bincount(dom), NAMES)


if __name__ == "__main__":
    main()
    main(strips=False)
