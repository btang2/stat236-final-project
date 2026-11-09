"""DCBM-only single-column figures (3.35 in wide) and LaTeX tables.

Message: under DCBM, K = 2 is the only K whose hard clusters replicate across halves of the managers, but the K = 2
split is popularity, not a community. A degree-matched network with no communities passes the same test, and one cut
on the number of owners reproduces most of the observed K = 2 labels. Beyond K = 2 the data hold structure the null
lacks, but hard labels cannot keep it: the stocks that flip are the mixed ones.

Figures (out/dcbm_*.pdf, .png):
  dcbm_stability      least-stable cluster's split-half Jaccard by K, observed vs no-community null
  dcbm_flip_purity    label-flip rate at K = 4 by the stock's largest DCMM membership
  dcbm_k2_popularity  K = 2 SCORE coordinate vs number of owners, observed and one null draw, with the best owner cut
Tables: out/dcbm_tables.tex (booktabs).

Run: python -m membership.dcbm_k.dcbm_figures  (dcbm_k2_popularity always; the rest after k_test.py and null_k2.py)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from membership.core import cosine, spectrum
from membership.dcbm_k.k_test import OUT, score_labels, summarize_t1
from membership.fit import COLORS
from membership.s1_choose_K import get_data
from src.curveball import curveball

W = 3.35
INK, MUTED, GRID, NULLC = "#1f1f1e", "#6b6a64", "#e4e3de", "#9a998f"
C_BM = COLORS[1]
YEARS = ["2006", "2008"]
KS = range(2, 7)
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "pdf.fonttype": 42, "axes.linewidth": 0.5,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                     "ytick.major.size": 2, "legend.frameon": False})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def best_cut(z, owners):
    """Single owner-count threshold that best reproduces a 2-cluster label: (accuracy, threshold)."""
    o = np.argsort(owners)
    zs, os_ = z[o], owners[o]
    best = (0, None)
    for lab in (0, 1):                      # label `lab` above the cut
        below = np.r_[0, np.cumsum(zs != lab)]
        above = np.r_[np.cumsum((zs == lab)[::-1])[::-1], 0]
        acc = (below + above) / len(z)
        j = int(np.argmax(acc))
        if acc[j] > best[0]:
            thr = os_[j - 1] if j > 0 else os_[0]
            best = (acc[j], (thr + (os_[j] if j < len(z) else thr)) / 2)
    return best


# --- figures -------------------------------------------------------------------------------------
def fig_stability(s, nd):
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.85), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.99, top=0.80, bottom=0.2, wspace=0.08)
    ks = list(KS)
    for ax, y in zip(axes, YEARS):
        g = s[s.year == y]
        nul = nd[(nd.year == y) & (nd.kind == "null")]
        q = np.array([nul[f"min_jaccard_K{K}"].quantile([0.1, 0.5, 0.9]).values for K in ks])
        ax.fill_between(ks, q[:, 0], q[:, 2], color=NULLC, alpha=0.25, lw=0)
        ax.plot(ks, q[:, 1], color=NULLC, lw=1.2, ls=(0, (3, 2)), label="null: no communities")
        ax.fill_between(g.K, g.dcbm_min_jaccard_q10, g.dcbm_min_jaccard_q90, color=C_BM, alpha=0.15, lw=0)
        ax.plot(g.K, g.dcbm_min_jaccard, color=C_BM, lw=1.5, label="observed")
        ok = g.dcbm_min_jaccard >= 0.7
        ax.scatter(g.K[ok], g.dcbm_min_jaccard[ok], s=16, color=C_BM, zorder=3, edgecolor="white", lw=0.6)
        ax.scatter(g.K[~ok], g.dcbm_min_jaccard[~ok], s=16, facecolor="white", edgecolor=C_BM, lw=1, zorder=3)
        ax.axhline(0.7, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
        ax.set_title(y, fontsize=7, color=INK, loc="left")
        ax.set_xticks(ks)
        ax.set_xlabel("K")
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("least-stable cluster,\nsplit-half Jaccard")
    axes[1].text(6.1, 0.72, "0.7", color=MUTED, fontsize=6, ha="right", va="bottom")
    axes[0].annotate("both pass\nat K = 2", xy=(2.08, 0.86), xytext=(2.9, 0.93), fontsize=6, color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5), va="center")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.55, 1.0),
               handlelength=1.8)
    save(fig, "dcbm_stability")


def fig_flip(f):
    bins = ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]
    labels = ["< 0.5", "0.5–0.7", "0.7–0.9", "≥ 0.9"]
    fig, ax = plt.subplots(figsize=(W, 1.75))
    fig.subplots_adjust(left=0.165, right=0.99, top=0.95, bottom=0.27)
    x = np.arange(4)
    for j, (y, c) in enumerate(zip(YEARS, [C_BM, "#f3b597"])):
        g = f[f.year == y].set_index("purity_bin").reindex(bins)
        ax.bar(x + (j - 0.5) * 0.36, g.flip_rate * 100, width=0.34, color=c, label=y, lw=0)
        for xi, v in zip(x, g.flip_rate * 100):
            ax.text(xi + (j - 0.5) * 0.36, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6, color=INK)
    share = f.groupby("purity_bin").share_of_stocks.mean().reindex(bins)
    ax.set_xticks(x, [f"{l}\n{s * 100:.0f}% of stocks" for l, s in zip(labels, share)], fontsize=6.5)
    ax.set_xlabel("stock's largest DCMM membership (full sample, K = 4)", labelpad=2)
    ax.set_ylabel("DCBM label flips\nbetween halves (%)", labelpad=2)
    ax.set_ylim(0, 65)
    ax.legend(loc="upper right", fontsize=6.5, ncol=2, handlelength=1, columnspacing=1)
    style(ax)
    save(fig, "dcbm_flip_purity")


def fig_k2(year="2006"):
    d = get_data()[year]
    B = d["B"]
    rng = np.random.default_rng([236, 9])
    Bn = curveball(B.astype(bool), 50, rng).astype(float)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.84, bottom=0.21, wspace=0.08)
    for ax, (M, title) in zip(axes, [(B, "(a) observed"), (Bn, "(b) null: no communities")]):
        A = cosine(M)
        lam, xi = spectrum(A)
        r = np.clip(xi[:, 1] / xi[:, 0], -np.log(len(A)), np.log(len(A)))
        z = score_labels(A, 2)
        owners = M.sum(1)
        if np.corrcoef(r, np.log(owners))[0, 1] > 0:   # orient: more owners -> lower coordinate, as observed
            r = -r
        acc, thr = best_cut(z, owners)
        hi = np.argmax([owners[z == c].mean() for c in (0, 1)])
        for c, col, lab in [(hi, INK, "cluster 1"), (1 - hi, "#bdbcb4", "cluster 2")]:
            m = z == c
            ax.scatter(owners[m], r[m], s=4, color=col, lw=0, alpha=0.85, label=lab)
        ax.axvline(thr, color=C_BM, lw=0.9, ls=(0, (3, 1.5)))
        ax.text(thr * 1.07, 0.97, f"owner cut:\n{acc * 100:.0f}% of\nlabels", transform=ax.get_xaxis_transform(),
                fontsize=5.8, color=INK, va="top")
        ax.set_xscale("log")
        ax.set_xlabel("active owners (log scale)")
        ax.set_title(title, fontsize=6.8, loc="left", color=INK)
        style(ax)
        out[title] = dict(acc=acc, thr=thr, n=np.bincount(z))
    axes[0].set_ylabel("$K = 2$ SCORE coordinate\n$\\hat\\xi_2/\\hat\\xi_1$", labelpad=1)
    axes[0].legend(loc="lower left", fontsize=6, handletextpad=0.1, markerscale=2, borderaxespad=0.2)
    save(fig, "dcbm_k2_popularity")
    return out


# --- tables --------------------------------------------------------------------------------------
def tables(s, nd, f):
    def b(v):
        return rf"\textbf{{{v:.2f}}}" if v >= 0.7 else f"{v:.2f}"

    def iq(x):
        q = x.quantile([0.1, 0.5, 0.9]).values
        return f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]"

    L = [r"% Requires \usepackage{booktabs}", ""]
    # Table 1: stability by K, observed vs null
    L += [r"\begin{table}[t]", r"\centering", r"\small",
          r"\caption{DCBM (SCORE + $k$-means) split-half stability by $K$: Jaccard overlap of the least-stable "
          r"matched cluster across disjoint halves of the managers, median over splits (observed: 100 splits). "
          r"Null: degree-preserving "
          r"random networks with no communities (Curveball; 30 draws, 20 splits each). Bold: $\geq 0.7$.}",
          r"\label{tab:dcbm-stability}",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r" & \multicolumn{2}{c}{2006} & \multicolumn{2}{c}{2008} \\",
          r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
          r"$K$ & Observed & Null & Observed & Null \\", r"\midrule"]
    for K in KS:
        cells = []
        for y in YEARS:
            o = s[(s.year == y) & (s.K == K)].dcbm_min_jaccard.iloc[0]
            n = nd[(nd.year == y) & (nd.kind == "null")][f"min_jaccard_K{K}"].median()
            cells += [b(o), b(n)]
        L.append(f"{K} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 2: K = 2 diagnostics
    L += [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
          r"\caption{The DCBM $K=2$ split is popularity. Popularity accuracy: share of stocks whose $K=2$ label "
          r"is reproduced by one cut on the number of owners. Null: median over 30 degree-preserving networks with no "
          r"communities (every 10th--90th percentile range is within $\pm0.02$ of the median).}",
          r"\label{tab:dcbm-k2}",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r" & \multicolumn{2}{c}{2006} & \multicolumn{2}{c}{2008} \\",
          r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
          r" & Observed & Null & Observed & Null \\", r"\midrule"]
    for col, lab in [("min_jaccard_K2", r"Split-half Jaccard, $K=2$"),
                     ("popularity_accuracy", "Popularity accuracy"),
                     ("abs_spearman_owners", r"$|\rho|$, $\hat\xi_2/\hat\xi_1$ vs.\ log owners"),
                     ("lam2_over_lam1", r"$\lambda_2/\lambda_1$")]:
        cells = []
        for y in YEARS:
            o = nd[(nd.year == y) & (nd.kind == "observed")][col].iloc[0]
            cells += [f"{o:.2f}", f"{nd[(nd.year == y) & (nd.kind == 'null')][col].median():.2f}"]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 3: flips by purity
    bins = ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]
    lab = ["$<0.5$", "$0.5$--$0.7$", "$0.7$--$0.9$", r"$\geq 0.9$"]
    L += [r"\begin{table}[t]", r"\centering", r"\small",
          r"\caption{DCBM label flips between manager halves at $K=4$, by each stock's largest DCMM membership "
          r"$\max_k\hat\pi_{ik}$ (full-sample fit). Flips occur almost only in mixed stocks.}",
          r"\label{tab:dcbm-flip}",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r" & \multicolumn{2}{c}{Share of stocks} & \multicolumn{2}{c}{Flip rate} \\",
          r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
          r"$\max_k\hat\pi_{ik}$ & 2006 & 2008 & 2006 & 2008 \\", r"\midrule"]
    fi = f.set_index(["year", "purity_bin"])
    for bb, ll in zip(bins, lab):
        v = [fi.loc[(y, bb)] for y in YEARS]
        L.append(f"{ll} & {v[0].share_of_stocks * 100:.0f}\\% & {v[1].share_of_stocks * 100:.0f}\\% & "
                 f"{v[0].flip_rate * 100:.0f}\\% & {v[1].flip_rate * 100:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    caps = {
        "dcbm_stability": r"DCBM split-half stability of the least-stable cluster by $K$ (median and 10th--90th "
                          r"percentile). A degree-preserving network with no communities passes at $K=2$ as "
                          r"easily as the data, so passing at $K=2$ is not evidence of communities. For $K\geq3$ "
                          r"the data keep far more structure than the null, but no hard partition replicates.",
        "dcbm_flip_purity": r"DCBM label flips between halves at $K=4$, by the stock's largest DCMM membership. "
                            r"Nearly pure stocks keep their label; mixed stocks flip up to half the time.",
        "dcbm_k2_popularity": r"The DCBM $K=2$ split, December 2006. (a) Observed. (b) A degree-preserving network "
                              r"with no communities. In both, the split separates widely held from less widely "
                              r"held stocks, and one cut on the number of owners reproduces nearly all labels.",
    }
    L += [r"% Figures: one column each", ""]
    for name, cap in caps.items():
        L += [r"\begin{figure}[t]", r"\centering", rf"\includegraphics[width=\columnwidth]{{{name}.png}}",
              rf"\caption{{{cap}}}", rf"\label{{fig:{name.replace('_', '-')}}}", r"\end{figure}", ""]
    (OUT / "dcbm_tables.tex").write_text("\n".join(L))


def main():
    print("K=2 panels:", fig_k2())          # paper figure (dcbm_k2_popularity); needs only the data
    need = [OUT / "t1_splits.pkl", OUT / "null_k2_draws.csv", OUT / "t3_flip_by_purity.csv"]
    if not all(p.exists() for p in need):
        print("skipping supplementary DCBM figures/tables: run k_test.py and null_k2.py first")
        return
    s = summarize_t1(pd.read_pickle(OUT / "t1_splits.pkl"))
    s["year"] = s.year.astype(str)
    nd = pd.read_csv(OUT / "null_k2_draws.csv", dtype={"year": str}, keep_default_na=False)  # "null" is a label
    f = pd.read_csv(OUT / "t3_flip_by_purity.csv", dtype={"year": str})
    fig_stability(s, nd)
    fig_flip(f)
    tables(s, nd, f)
    print("wrote", OUT / "dcbm_tables.tex")


if __name__ == "__main__":
    main()
