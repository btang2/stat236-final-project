"""Step 3: is a profile more than popularity (number of owners)?

Blue-chip membership tracks the number of active owners (Spearman ~0.85) and Small/neglected tracks it
negatively (~ -0.7). Two tests separate "who owns it" from "how many own it":

(a) Degree null. 100 Curveball draws (uniform over 0/1 matrices with every stock's and every manager's
    degree fixed; 50 palindromic cycles, sampler in src/curveball.py). Fit K = 4 on each draw and
    record (i) the share of membership variance explained by log degree (cubic in log n_i) and
    (ii) split-half reliability of each null profile (10 splits per draw). If observed profiles are only
    degree, the null should reproduce both.
(b) Beyond-degree replication on the observed data. For 100 random halvings of the managers, fit each
    half, residualize each half's pi_k on a cubic in the stock's full-sample log degree, and correlate the
    residuals across halves. A positive, stable correlation = replicable ownership information that
    popularity does not explain.
"""
import numpy as np
import pandas as pd

from membership.core import OUT, align, col_corr, cosine, mixed_score
from membership.fit import K, NAMES, fit
from src.curveball import curveball

N_NULL, N_SPLIT_NULL, N_SPLIT = 100, 10, 100


def deg_basis(n):
    z = np.log(n)
    z = (z - z.mean()) / z.std()
    return np.column_stack([np.ones_like(z), z, z ** 2, z ** 3])


def resid(y, X):
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def r2_degree(Pi, X):
    return np.array([1 - resid(Pi[:, k], X).var() / Pi[:, k].var() for k in range(Pi.shape[1])])


def split_half(B, ref, rng, n, X=None):
    out = []
    for _ in range(n):
        h = rng.permutation(B.shape[1]) < B.shape[1] // 2
        ok = (B[:, h].sum(1) > 0) & (B[:, ~h].sum(1) > 0)
        Pa, _ = align(ref[ok], mixed_score(cosine(B[ok][:, h]), K, n_init=5)["Pi"])
        Pb, _ = align(ref[ok], mixed_score(cosine(B[ok][:, ~h]), K, n_init=5)["Pi"])
        if X is not None:
            Xo = X[ok]
            Pa = np.column_stack([resid(Pa[:, k], Xo) for k in range(K)])
            Pb = np.column_stack([resid(Pb[:, k], Xo) for k in range(K)])
        out.append(col_corr(Pa, Pb))
    return np.array(out)


def main():
    res = fit()
    rng = np.random.default_rng([236, 3])
    rows = []
    for y, r in res.items():
        B, Pi = r["B"], r["Pi"]
        X = deg_basis(B.sum(1))
        obs_r2 = r2_degree(Pi, X)
        obs_rel = np.median(split_half(B, Pi, rng, N_SPLIT), 0)
        beyond = split_half(B, Pi, rng, N_SPLIT, X)
        null_r2, null_rel = [], []
        Bb = B.astype(bool)
        for s in range(N_NULL):
            Bn = curveball(Bb, 50, rng).astype(float)
            Pn = mixed_score(cosine(Bn), K, n_init=5)["Pi"]
            # order null profiles by their degree correlation, like observed (BC most +, Small most -)
            null_r2.append(np.sort(r2_degree(Pn, X))[::-1])
            null_rel.append(np.sort(np.median(split_half(Bn, Pn, rng, N_SPLIT_NULL), 0))[::-1])
            if s % 10 == 0:
                print(y, s, np.round(null_r2[-1], 2), np.round(null_rel[-1], 2), flush=True)
        null_r2, null_rel = np.array(null_r2), np.array(null_rel)
        for k in range(K):
            rows.append(dict(year=y, profile=NAMES[k], R2_degree_obs=obs_r2[k],
                             split_half_obs=obs_rel[k],
                             beyond_degree_r_median=np.median(beyond[:, k]),
                             beyond_degree_r_q05=np.quantile(beyond[:, k], 0.05),
                             beyond_degree_share_pos=(beyond[:, k] > 0).mean()))
        rows.append(dict(year=y, profile="NULL (sorted best..worst)",
                         R2_degree_obs=np.round(null_r2.mean(0), 2).tolist(),
                         split_half_obs=np.round(null_rel.mean(0), 2).tolist(),
                         beyond_degree_r_median=np.round(np.quantile(null_rel, 0.95, axis=0), 2).tolist()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "tables" / "s3_degree.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    main()
