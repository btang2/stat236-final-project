"""Step 1: which ownership profiles (vertices) are real? Choose K by replication, not by eigengap.

For K = 2..6 and each date: split the active managers at random into two halves, fit Mixed-SCORE on each
half separately, align both to the full-sample fit, and record per-vertex correlation of pi_k across stocks
between the two halves (split-half reliability; 100 splits). A vertex is called reproducible if its median
split-half correlation >= 0.7. Halves have half the data, so this is conservative for the full sample.
Also: eigenvalues, eigenvector localization (IPR), and vertex match across dates.
"""
import pickle

import numpy as np
import pandas as pd

from membership.core import DATES, OUT, align, cfg, col_corr, cosine, load, mixed_score, spectrum

N_SPLIT = 100
KS = range(2, 7)


MIN_OWNERS = 50


def _drop_artifacts(d):
    """Drop stocks with < MIN_OWNERS active owners. Only 2008 has any: ABI (1 owner), SSP, ACAS; every other
    stock has >= 80. These are 13F identifier breaks around restructurings (Applera tracking stock,
    Scripps spin-off), not real ownership patterns."""
    keep = d["B"].sum(1) >= MIN_OWNERS
    return dict(d, B=d["B"][keep], W=d["W"][keep], rows=d["rows"][keep], x=d["x"][keep],
                dropped=list(d["x"].ticker[~keep]))


def get_data():
    p = OUT / "data_cache.pkl"
    if p.exists():
        d = pickle.load(open(p, "rb"))
    else:
        c = cfg()
        d = {y: load(c, y) for y in DATES}
        pickle.dump(d, open(p, "wb"))
    return {y: _drop_artifacts(v) for y, v in d.items()}


def main():
    data = get_data()
    rng = np.random.default_rng(236)
    rows, spec = [], []
    for y, d in data.items():
        B = d["B"]
        A = cosine(B)
        lam, xi = spectrum(A)
        for k in range(8):
            spec.append(dict(year=y, k=k + 1, lam=lam[k], ipr=(xi[:, k] ** 4).sum(),
                             top_loading=d["x"].ticker.iloc[int(np.argmax(np.abs(xi[:, k])))]))
        for K in KS:
            full = mixed_score(A, K)["Pi"]
            cors = []
            for s in range(N_SPLIT):
                h = rng.permutation(B.shape[1]) < B.shape[1] // 2
                ok = (B[:, h].sum(1) > 0) & (B[:, ~h].sum(1) > 0)
                Pa = mixed_score(cosine(B[ok][:, h]), K, n_init=5)["Pi"]
                Pb = mixed_score(cosine(B[ok][:, ~h]), K, n_init=5)["Pi"]
                Pa, _ = align(full[ok], Pa)
                Pb, _ = align(full[ok], Pb)
                cors.append(col_corr(Pa, Pb))
            cors = np.array(cors)
            for k in range(K):
                rows.append(dict(year=y, K=K, vertex=k, split_half_r_median=np.median(cors[:, k]),
                                 r_q10=np.quantile(cors[:, k], 0.1), share_argmax=(full.argmax(1) == k).mean(),
                                 purest=", ".join(d["x"].ticker.iloc[np.argsort(-full[:, k])[:6]])))
            print(y, K, np.round(np.median(cors, 0), 2), flush=True)
    rel = pd.DataFrame(rows)
    rel.to_csv(OUT / "tables" / "s1_split_half.csv", index=False)
    pd.DataFrame(spec).to_csv(OUT / "tables" / "s1_spectrum.csv", index=False)
    print(rel.round(2).to_string())
    print(pd.DataFrame(spec).round(3).to_string())


if __name__ == "__main__":
    main()
