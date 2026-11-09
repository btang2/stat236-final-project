"""Eigengap view of K for the cosine co-ownership matrix A (observed data only, no null).

Single-column figure (3.35 in): (a) scree plot of lambda_k, log scale; (b) eigengap ratio lambda_k / lambda_{k+1}
for k >= 2. The largest gap after the trivial first one is between lambda_2 and lambda_3 at both dates, so the
eigengap rule picks K = 2. Also writes out/eigengap_table.tex and out/eigengap.csv.

Run: python -m membership.dcbm_k.eigengap
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from membership.core import cosine, spectrum
from membership.dcbm_k.k_test import OUT
from membership.fit import COLORS
from membership.s1_choose_K import get_data

W = 3.35
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
HL = COLORS[1]
YEARS = ["2006", "2008"]
YC = {"2006": INK, "2008": "#a9a8a0"}
NK = 12
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "pdf.fonttype": 42, "axes.linewidth": 0.5,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                     "ytick.major.size": 2, "legend.frameon": False})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def main():
    lam = {y: spectrum(cosine(d["B"]))[0][:NK] for y, d in get_data().items()}
    rows = [dict(year=y, k=k + 1, lam=l[k], ratio_next=l[k] / l[k + 1] if k + 1 < NK else np.nan,
                 scoreplus=1 - l[k + 1] / l[k] if k + 1 < NK else np.nan) for y, l in lam.items() for k in range(NK)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eigengap.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.9))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.83, bottom=0.21, wspace=0.42)
    k = np.arange(1, NK + 1)
    for y in YEARS:
        a1.plot(k, lam[y], color=YC[y], lw=1.2, marker="o", ms=3, mec="white", mew=0.4, label=y)
    a1.set_yscale("log")
    a1.set_xticks([1, 2, 3, 6, 9, 12])
    a1.set_xlabel("k")
    a1.set_ylabel("$\\lambda_k$ of $A$ (log scale)")
    a1.set_title("(a) scree", fontsize=6.8, loc="left", color=INK)
    l6 = lam["2006"]
    a1.annotate("", xy=(2.9, l6[2]), xytext=(2.1, l6[1]),
                arrowprops=dict(arrowstyle="-|>", color=HL, lw=0.8, mutation_scale=6))
    a1.text(3.3, np.sqrt(l6[1] * l6[2]) * 1.4, f"$\\lambda_2/\\lambda_3$ = {l6[1] / l6[2]:.1f}", fontsize=6,
            color=INK)
    a1.legend(loc="upper right", fontsize=6.3, handlelength=1.4)
    style(a1)

    kk = np.arange(2, NK)
    wbar = 0.38
    for j, y in enumerate(YEARS):
        r = lam[y][1:NK - 1] / lam[y][2:NK]
        cols = [HL if kv == 2 else YC[y] for kv in kk]
        a2.bar(kk + (j - 0.5) * wbar, r, width=wbar - 0.04, color=cols, lw=0,
               alpha=1 if y == "2006" else 0.55)
    v6, v8 = (lam[y][1] / lam[y][2] for y in YEARS)
    a2.text(2.25, max(v6, v8) + 0.08, f"{v6:.1f} / {v8:.1f}", ha="left", va="bottom", fontsize=5.8, color=INK)
    a2.axhline(1, color=MUTED, lw=0.5)
    a2.set_xticks([2, 4, 6, 8, 10])
    a2.set_xlabel("k")
    a2.set_ylabel("$\\lambda_k / \\lambda_{k+1}$")
    a2.set_ylim(0, 4.3)
    a2.set_title("(b) eigengap ratio, k ≥ 2", fontsize=6.8, loc="left", color=INK)
    style(a2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"dcbm_eigengap.{ext}", dpi=300)
    plt.close(fig)

    # table: lambda_1..6 and gap ratios
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Leading eigenvalues of the cosine co-ownership matrix $A$ and eigengap ratios "
         r"$\lambda_k/\lambda_{k+1}$. After the trivial first gap, the largest is between $\lambda_2$ and "
         r"$\lambda_3$ at both dates, so the eigengap rule selects $K=2$.}",
         r"\label{tab:eigengap}",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"$k$ & 1 & 2 & 3 & 4 & 5 & 6 \\", r"\midrule"]
    for y in YEARS:
        l = lam[y]
        L.append(rf"$\lambda_k$, {y} & " + " & ".join(f"{v:.1f}" for v in l[:6]) + r" \\")
    L.append(r"\addlinespace")
    for y in YEARS:
        l = lam[y]
        cells = [f"{l[k] / l[k + 1]:.2f}" for k in range(6)]
        cells[1] = rf"\textbf{{{cells[1]}}}"
        L.append(rf"$\lambda_k/\lambda_{{k+1}}$, {y} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\columnwidth]{dcbm_eigengap.png}",
          r"\caption{Spectrum of the cosine co-ownership matrix $A$. (a) Eigenvalues on a log scale: one large "
          r"drop after $\lambda_2$, then a slow decay. (b) Eigengap ratios $\lambda_k/\lambda_{k+1}$ for $k\geq2$ "
          r"(dark: 2006, light: 2008). The gap after $k=2$ is about 3.6 at both dates; every later gap is below "
          r"1.7. The eigengap rule therefore selects $K=2$.}",
          r"\label{fig:dcbm-eigengap}", r"\end{figure}", ""]
    (OUT / "eigengap_table.tex").write_text("\n".join(L))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
