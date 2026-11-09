"""Is the DCBM K = 2 split a community? Run the same DCBM pipeline on networks with no communities.

Null networks: Curveball draws (src/curveball.py) keep every stock's number of owners and every manager's number
of holdings fixed and randomize who holds what. They have degree heterogeneity and nothing else.

For the observed network and each null draw:
  - eigenvalue ratio lambda_2 / lambda_1 of the cosine matrix A
  - |Spearman| of the K = 2 SCORE coordinate xi_2/xi_1 with log(number of owners)
  - popularity accuracy: share of stocks whose K = 2 DCBM label is reproduced by a single cut on the number of
    owners (best cut). 1.0 = the split is exactly "more owners vs fewer owners"
  - split-half stability of DCBM at K = 2..6 (least-stable cluster's Jaccard, as in k_test.py)
If the null passes the K = 2 test as easily as the data, passing it says nothing about communities.

Run: python -m membership.dcbm_k.null_k2  (writes out/null_k2_draws.csv, out/null_k2_summary.csv)
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from membership.core import cosine, spectrum
from membership.dcbm_k.k_test import OUT, match_labels, score_labels
from membership.s1_choose_K import get_data
from src.curveball import curveball

N_NULL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_SPLIT = 20
KS = range(2, 7)


def popularity_accuracy(z, owners):
    """Best accuracy of predicting a 2-cluster label from one threshold on owners."""
    o = np.argsort(owners)
    zs = z[o]
    n = len(z)
    # label 1 above the cut, or label 0 above the cut
    c1 = np.r_[0, np.cumsum(zs == 0)] + np.r_[np.cumsum((zs == 1)[::-1])[::-1], 0]
    c0 = np.r_[0, np.cumsum(zs == 1)] + np.r_[np.cumsum((zs == 0)[::-1])[::-1], 0]
    return max(c1.max(), c0.max()) / n


def describe(B, rng, kind, year, draw):
    A = cosine(B)
    lam, xi = spectrum(A)
    owners = B.sum(1)
    r = xi[:, 1] / xi[:, 0]
    z = score_labels(A, 2)
    row = dict(kind=kind, year=year, draw=draw, lam2_over_lam1=lam[1] / lam[0], lam3_over_lam2=lam[2] / lam[1],
               abs_spearman_owners=abs(spearmanr(r, np.log(owners))[0]),
               popularity_accuracy=popularity_accuracy(z, owners))
    jac = {K: [] for K in KS}
    for _ in range(N_SPLIT):
        h = rng.permutation(B.shape[1]) < B.shape[1] // 2
        ok = (B[:, h].sum(1) > 0) & (B[:, ~h].sum(1) > 0)
        Aa, Ab = cosine(B[ok][:, h]), cosine(B[ok][:, ~h])
        for K in KS:
            jac[K].append(match_labels(score_labels(Aa, K), score_labels(Ab, K), K)[1].min())
    for K in KS:
        row[f"min_jaccard_K{K}"] = float(np.median(jac[K]))
    return row


def main():
    rng = np.random.default_rng([236, 2])
    rows = []
    for y, d in get_data().items():
        B = d["B"]
        rows.append(describe(B, rng, "observed", y, -1))
        print(y, "observed", {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        Bb = B.astype(bool)
        for s in range(N_NULL):
            Bn = curveball(Bb, 50, rng).astype(float)
            rows.append(describe(Bn, rng, "null", y, s))
            if s % 5 == 0:
                print(y, "null", s, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)},
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "null_k2_draws.csv", index=False)
    cols = [c for c in df.columns if c not in ("kind", "year", "draw")]
    summ = df.groupby(["year", "kind"])[cols].median().reset_index()
    summ.to_csv(OUT / "null_k2_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
