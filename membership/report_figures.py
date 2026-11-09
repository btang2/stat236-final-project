"""Figures added to REPORT.md to show the communities themselves, each modelled on a lecture figure.

g1  Sorted co-ownership matrix (Lec 8 slide 29: A = Omega + W with low-rank Omega; Lec 9 slides 6 and 15: permuting
    nodes reveals blocks, and under DCBM each block is scaled by theta). Panels: A in data order, A sorted by
    dominant profile then membership, the rank-4 signal part Omega_hat, and Omega_hat with degree divided out.
g2  Community roster (Lec 10 slide 28: named political books with their estimated memberships). The ten purest
    stocks of each profile at each date, with their full membership vector, sector, size and dividend yield.
g3  The SCORE simplex in two dimensions (Lec 10 slide 6: rows of R_hat for the trade network inside a triangle;
    slide 30: college football in dimension 3 with the tetrahedron drawn). K = 4 so R_hat is 3-dimensional; we
    project onto the plane that best spreads the four estimated vertices.
Table: average membership within each GICS sector (Lec 10 slide 30: within-conference average pi_hat).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from membership.core import OUT, cosine
from membership.fit import COLORS, NAMES, data

F = OUT / "figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SEC = {"10": "Energy", "15": "Materials", "20": "Industrials", "25": "Cons. disc.", "30": "Cons. staples",
       "35": "Health care", "40": "Financials", "45": "Info tech", "50": "Telecom", "55": "Utilities"}
CMAP = LinearSegmentedColormap.from_list("ink", ["#ffffff", "#c9d6ea", "#5b7fb5", "#1d2b4a"])


def order(P):
    dom = P.argmax(1)
    return np.lexsort((-P[np.arange(len(P)), dom], dom)), dom


def g1_matrix(year="2006"):
    d = data(year)
    P, A = d["Pi"], cosine(d["B"])
    lam, xi = np.linalg.eigh(A)
    top = np.argsort(-np.abs(lam))[:4]
    lam, xi = lam[top], xi[:, top]
    xi[:, 0] *= np.sign(xi[:, 0].sum())
    Om = (xi * lam) @ xi.T
    theta = xi[:, 0]
    Om_free = Om / np.outer(theta, theta)
    o, dom = order(P)
    cuts = np.cumsum(np.bincount(dom, minlength=4))[:-1]

    fig = plt.figure(figsize=(12, 11.4))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.18)
    panels = [(A, np.arange(len(P)), "(a) $A$ in data order (by CRSP permno)", False),
              (A, o, "(b) $A$ with stocks sorted by dominant profile, then membership", True),
              (Om, o, r"(c) rank-4 signal part $\hat\Omega=\sum_{k\leq4}\hat\lambda_k\hat\xi_k\hat\xi_k^\top$, same order",
               True),
              (Om_free, o, r"(d) $\hat\Omega_{ij}/(\hat\xi_1(i)\hat\xi_1(j))$: degree divided out, same order", True)]
    for j, (M, idx, title, strip) in enumerate(panels):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        S = M[np.ix_(idx, idx)].copy()
        np.fill_diagonal(S, np.nan)
        lo, hi = np.nanquantile(S, [0.01, 0.99])
        im = ax.imshow(S, cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest", aspect="equal")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=22 if strip else 6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        if strip:
            for c in cuts:
                ax.axhline(c - 0.5, color=INK, lw=0.6)
                ax.axvline(c - 0.5, color=INK, lw=0.6)
            # membership strip along the top: stacked pi of each stock in the sorted order
            n = len(idx)
            sax = ax.inset_axes([0, 1.005, 1, 0.045])
            bottom = np.zeros(n)
            for k in range(4):
                sax.bar(np.arange(n), P[idx, k], bottom=bottom, width=1.0, color=COLORS[k], lw=0)
                bottom += P[idx, k]
            sax.set_xlim(-0.5, n - 0.5)
            sax.set_ylim(0, 1)
            sax.axis("off")
            starts = np.r_[0, cuts]
            ends = np.r_[cuts, n]
            for k in range(4):
                ax.text((starts[k] + ends[k]) / 2, n + n * 0.035, f"{NAMES[k]}\n(n={ends[k] - starts[k]})",
                        ha="center", va="top", fontsize=8, color=COLORS[k] if k != 3 else "#a87300")
        cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        if j == 3:
            cb.set_label("relative affinity (scale arbitrary)", fontsize=7.5)
    fig.suptitle(f"Co-ownership matrix of {len(P)} S&P 500 stocks, {d['t'][:7]}: permuting reveals the blocks; "
                 "removing degree equalizes them", x=0.06, ha="left", y=0.97, fontsize=11, color=INK)
    fig.savefig(F / "g1_sorted_matrix.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    # block means of the degree-free signal matrix: an estimate of P up to scale
    Pblk = np.array([[np.nanmean(np.where(np.eye(len(P), dtype=bool), np.nan, Om_free)[np.ix_(dom == a, dom == b)])
                      for b in range(4)] for a in range(4)])
    Pblk = Pblk / np.sqrt(np.outer(np.diag(Pblk), np.diag(Pblk)))
    pd.DataFrame(Pblk, index=NAMES, columns=NAMES).round(3).to_csv(OUT / "tables" / f"g1_block_affinity_{year}.csv")
    return Pblk


def title_name(s, n=22):
    drop = {"Inc", "Corp", "Co", "Com", "Cos", "New", "Ltd", "The"}
    s = " ".join(w for w in str(s).title().replace(".", "").split() if w not in drop)
    return s if len(s) <= n else s[:n - 1].rstrip() + "."


def g2_roster(n=10):
    ds = {y: data(y) for y in ("2006", "2008")}
    fig, axes = plt.subplots(4, 2, figsize=(14, 15.5), gridspec_kw={"hspace": 0.42, "wspace": 1.3})
    for k in range(4):
        for j, y in enumerate(("2006", "2008")):
            d = ds[y]
            x, P = d["x"], d["Pi"]
            idx = np.argsort(-P[:, k])[:n]
            ax = axes[k, j]
            yy = np.arange(n)[::-1]
            left = np.zeros(n)
            for q in range(4):
                ax.barh(yy, P[idx, q], left=left, color=COLORS[q], height=0.72, lw=0)
                left += P[idx, q]
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_yticks(yy, [f"{x.ticker.iloc[i]}  {title_name(x.name.iloc[i])}" for i in idx], fontsize=8.3)
            ax.tick_params(axis="y", length=0)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=7)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            for i, yv in zip(idx, yy):
                dy = x.div_yield.iloc[i]
                txt = f"{SEC.get(str(x.gsector.iloc[i]), '?')} · ${x.cap_bn.iloc[i]:,.0f}bn · " \
                      f"{'' if np.isnan(dy) else f'{dy * 100:.1f}%'} · {x.ret12.iloc[i] * 100:+.0f}%"
                ax.text(1.02, yv, txt, va="center", fontsize=7.3, color=MUTED, transform=ax.get_yaxis_transform())
            med = lambda v: np.nanmedian(v[idx])
            ax.set_title(f"{NAMES[k]}, {'Dec 2006' if y == '2006' else 'Jun 2008'}   "
                         f"(median ${med(x.cap_bn.values):.0f}bn, yield {med(x.div_yield.values) * 100:.1f}%, "
                         f"{med(d['B'].sum(1)):.0f} owners)",
                         loc="left", fontsize=9.3, color=COLORS[k] if k != 3 else "#a87300", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, NAMES, ncol=4, loc="upper left", bbox_to_anchor=(0.06, 0.935), frameon=False, fontsize=9)
    fig.suptitle("The ten purest stocks of each community. Bars: full membership vector. "
                 "Right: sector · market cap · dividend yield · trailing 12-month return",
                 x=0.06, ha="left", y=0.955, fontsize=10.5, color=INK)
    fig.savefig(F / "g2_roster.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


LABELS = {"2006": ["GE", "ED", "BSC", "SWK", "PHM", "LEH", "GOOG", "C", "AIG"],
          "2008": ["JNJ", "WPO", "SWN", "PHM", "LEH", "MOT", "GM", "X", "HPQ", "GOOG", "C", "AIG", "NCC"]}


def best_plane(Vc):
    """Plane (2 x 3 orthonormal rows) maximizing the projected area of the vertex quadrilateral, requiring all four
    vertices on its convex hull, so every vertex is visible and the two hidden edges are the diagonals."""
    from scipy.spatial import ConvexHull
    best, W = -1, None
    for th in np.linspace(0, np.pi, 90):
        for ph in np.linspace(0, 2 * np.pi, 180, endpoint=False):
            nrm = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            e1 = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(nrm, e1)
            Q = np.vstack([e1, e2])
            h = ConvexHull(Vc @ Q.T)
            if len(h.vertices) == 4 and h.volume > best:
                best, W = h.volume, Q
    return W


def g3_simplex():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    for ax, y in zip(axes, ("2006", "2008")):
        d = data(y)
        R, V, P, x = d["R"], d["V"], d["Pi"], d["x"]
        c = V.mean(0)
        W = best_plane(V - c)
        r2, v2 = (R - c) @ W.T, (V - c) @ W.T
        # rotate in the plane so Blue chip is at the top, then mirror so Income is on the left
        a = np.arctan2(v2[0, 1], v2[0, 0]) - np.pi / 2
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        r2, v2 = r2 @ rot, v2 @ rot
        if v2[1, 0] > v2[3, 0]:
            r2[:, 0] *= -1
            v2[:, 0] *= -1
        dom = P.argmax(1)
        strength = P.max(1)
        keep = ~d["pruned"]
        for k in range(4):
            m = keep & (dom == k)
            ax.scatter(r2[m, 0], r2[m, 1], s=10 + 22 * strength[m], color=COLORS[k], alpha=0.35 + 0.55 * strength[m],
                       lw=0, zorder=2)
        ax.scatter(r2[~keep, 0], r2[~keep, 1], s=14, facecolor="none", edgecolor=MUTED, lw=0.6, zorder=2)
        for a in range(4):
            for b in range(a + 1, 4):
                ax.plot(*v2[[a, b]].T, color="#c0392b", lw=1.1, zorder=3)
        for k in range(4):
            ax.scatter(*v2[k], marker="*", s=230, color=COLORS[k], edgecolor=INK, lw=0.8, zorder=5)
            off = (v2[k] - v2.mean(0))
            off = off / np.linalg.norm(off) * 0.11 * np.ptp(v2, 0).max()
            ax.text(*(v2[k] + off), NAMES[k], ha="center", va="center", fontsize=10, fontweight="bold",
                    color=COLORS[k] if k != 3 else "#a87300", zorder=6)
        tick = x.ticker.values
        for t in LABELS[y]:
            i = np.where(tick == t)[0]
            if not len(i):
                continue
            i = i[0]
            ax.scatter(*r2[i], s=26, facecolor="none", edgecolor=INK, lw=0.9, zorder=4)
            ax.annotate(t, r2[i], xytext=(4, 3), textcoords="offset points", fontsize=7.6, color=INK, zorder=6)
        lo = np.minimum(np.quantile(r2, 0.01, 0), v2.min(0))
        hi = np.maximum(np.quantile(r2, 0.99, 0), v2.max(0))
        pad = 0.12 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(f"{'Dec 2006' if y == '2006' else 'Jun 2008'}: rows of $\\hat R$ ({len(R)} stocks)",
                     loc="left", fontsize=10, color=INK)
    fig.suptitle("Rows of the SCORE matrix projected onto the plane where all four estimated vertices are visible "
                 "(red: edges of the estimated simplex, a tetrahedron; the two inner edges cross in projection)", x=0.02, ha="left", y=1.0, fontsize=10.5, color=INK)
    fig.text(0.02, -0.02, "Dot colour: dominant profile; size and opacity: largest membership. Open grey circles: "
             "pruned before vertex hunting. Outlined: named stocks discussed in the text. "
             "Each date is a separate fit, so the two panels are oriented differently.", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(F / "g3_simplex_2d.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def sector_table():
    rows = []
    for y in ("2006", "2008"):
        d = data(y)
        x, P = d["x"], d["Pi"]
        s = x.gsector.astype(str).map(SEC).values
        for sec in SEC.values():
            m = s == sec
            rows.append(dict(year=y, sector=sec, n=int(m.sum()),
                             **{NAMES[k]: P[m, k].mean() for k in range(4)},
                             **{f"sd_{NAMES[k]}": P[m, k].std() for k in range(4)}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "tables" / "g_sector_average_pi.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(g1_matrix())
    g2_roster()
    g3_simplex()
    print(sector_table().round(2).to_string(index=False))
