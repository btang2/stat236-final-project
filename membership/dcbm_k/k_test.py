"""DCBM vs DCMM: does K = 4 hold under a hard-label block model, and which model fits better?

Standalone check, separate from the report pipeline. Same data and cosine matrix A as membership/core.py.

Design (mirrors s1_choose_K.py so the two models are compared on identical splits):
  For each date, K = 2..6 and N_SPLIT random halves of the managers, build A on each half and fit
    DCBM  SCORE: k-means (K clusters) on the SCORE ratios R = xi_2..K / xi_1  -> hard labels z_i
    DCMM  Mixed-SCORE (core.mixed_score)                                   -> memberships pi_i
  T1  K under DCBM. Match the two halves' clusters (Hungarian on overlap) and record, per cluster, the Jaccard
      overlap of matched clusters, plus ARI overall. Mirror of the DCMM rule: K is supported if every cluster's
      median Jaccard >= 0.7 (DCMM: every profile's median split-half r >= 0.7).
  T2  Same splits, DCMM: per-profile split-half r (the report's statistic) and ARI of argmax(pi), so hard labels
      from both models are compared on the same scale.
  T3  Which model? Held-out fit. Fit Omega = Theta M P M' Theta on half a (M = one-hot z for DCBM, Pi for DCMM;
      theta and P by alternating least squares, same number of theta/P parameters in both) and score R^2 on the
      off-diagonal of half b's A (and b -> a). The two halves share no managers, so this rewards memberships that
      describe the stocks, not the particular funds.
      Also: where do DCBM labels flip between halves? Flip rate by the stock's full-sample DCMM purity max_k pi_ik.
      If flips concentrate in mixed stocks, hard labels are arbitrary exactly where DCMM says stocks are mixed.

Run: python -m membership.dcbm_k.k_test   (writes results/dcbm_k/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from membership.core import align, col_corr, cosine, mixed_score, spectrum
from membership.s1_choose_K import get_data

OUT = Path(__file__).resolve().parents[2] / "results" / "dcbm_k"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
KS = range(2, 7)
PURITY_BINS = [0, 0.5, 0.7, 0.9, 1.01]


# --- DCBM estimator ------------------------------------------------------------------------------
def score_labels(A, K, seed=236, n_init=10):
    """SCORE (Jin 2015): k-means on the ratio rows R, clipped at +-log n as in core.mixed_score."""
    lam, xi = spectrum(A)
    n = A.shape[0]
    R = np.clip(xi[:, 1:K] / xi[:, [0]], -np.log(n), np.log(n))
    return KMeans(K, n_init=n_init, random_state=seed).fit_predict(R)


def match_labels(za, zb, K):
    """Permute zb's labels to best match za; return (zb relabelled, per-cluster Jaccard of matched pairs)."""
    C = np.zeros((K, K))
    np.add.at(C, (za, zb), 1)
    r, c = linear_sum_assignment(-C)
    perm = np.empty(K, int)
    perm[c] = r
    zb2 = perm[zb]
    jac = np.array([((za == k) & (zb2 == k)).sum() / max(((za == k) | (zb2 == k)).sum(), 1) for k in range(K)])
    return zb2, jac


# --- block-model fit for held-out R^2 ------------------------------------------------------------
def fit_omega(A, M, iters=15):
    """Least-squares fit of A_ij ~ theta_i theta_j m_i' P m_j over i != j (P symmetric, theta > 0)."""
    n, K = M.shape
    iu = np.triu_indices(n, 1)
    a = A[iu]
    theta = np.sqrt(np.maximum(A.sum(1), 1e-9))
    theta /= theta.mean()
    pairs = [(p, q) for p in range(K) for q in range(p, K)]
    for _ in range(iters):
        U = theta[:, None] * M
        X = np.column_stack([U[iu[0], p] * U[iu[1], q] + (U[iu[0], q] * U[iu[1], p] if p != q else 0)
                             for p, q in pairs])
        coef = np.linalg.lstsq(X, a, rcond=None)[0]
        P = np.zeros((K, K))
        for (p, q), v in zip(pairs, coef):
            P[p, q] = P[q, p] = v
        C = (M @ P @ U.T)                     # C_ij = m_i' P (theta_j m_j)
        np.fill_diagonal(C, 0)
        theta = np.maximum((A * C).sum(1) / np.maximum((C ** 2).sum(1), 1e-12), 1e-6)
    U = theta[:, None] * M
    Om = U @ P @ U.T
    np.fill_diagonal(Om, 0)
    return Om


def r2_off(A, Om):
    iu = np.triu_indices(A.shape[0], 1)
    a, o = A[iu], Om[iu]
    return 1 - ((a - o) ** 2).sum() / ((a - a.mean()) ** 2).sum()


def rank_k(A, K):
    lam, xi = spectrum(A)
    Om = (xi[:, :K] * lam[:K]) @ xi[:, :K].T
    np.fill_diagonal(Om, 0)
    return Om


def summarize_t1(t1):
    """Stability by K. DCBM cluster numbers are arbitrary in each split, so per-cluster medians across splits are not
    defined. Instead, sort each split's values and take medians by rank; rank K (the least stable cluster/profile)
    is the pass statistic for both models: median over splits of the minimum, >= 0.7."""
    rows = []
    for (y, K), g in t1.groupby(["year", "K"]):
        J = np.sort(np.vstack(g.jac_dcbm), 1)[:, ::-1]
        R = np.sort(np.vstack(g.r_dcmm), 1)[:, ::-1]
        jmin, rmin = np.median(J[:, -1]), np.median(R[:, -1])
        rows.append(dict(year=y, K=K,
                         dcbm_jaccard_by_rank=" / ".join(f"{v:.2f}" for v in np.median(J, 0)),
                         dcbm_min_jaccard=jmin, dcbm_min_jaccard_q10=np.quantile(J[:, -1], 0.1),
                         dcbm_min_jaccard_q90=np.quantile(J[:, -1], 0.9), dcbm_ari=g.ari_dcbm.median(),
                         dcmm_r_by_rank=" / ".join(f"{v:.2f}" for v in np.median(R, 0)),
                         dcmm_min_r=rmin, dcmm_min_r_q10=np.quantile(R[:, -1], 0.1),
                         dcmm_min_r_q90=np.quantile(R[:, -1], 0.9), dcmm_argmax_ari=g.ari_dcmm.median(),
                         dcbm_pass=bool(jmin >= 0.7), dcmm_pass=bool(rmin >= 0.7)))
    s12 = pd.DataFrame(rows)
    s12.to_csv(OUT / "t1_t2_summary.csv", index=False)
    return s12


# --- main ----------------------------------------------------------------------------------------
def main():
    data = get_data()
    rng = np.random.default_rng(236)
    t1, t3, flips = [], [], []
    for y, d in data.items():
        B = d["B"]
        A = cosine(B)
        full_pi = {K: mixed_score(A, K)["Pi"] for K in KS}
        purity4 = full_pi[4].max(1)
        for s in range(N_SPLIT):
            h = rng.permutation(B.shape[1]) < B.shape[1] // 2
            ok = (B[:, h].sum(1) > 0) & (B[:, ~h].sum(1) > 0)
            Aa, Ab = cosine(B[ok][:, h]), cosine(B[ok][:, ~h])
            for K in KS:
                # DCBM
                za, zb = score_labels(Aa, K), score_labels(Ab, K)
                zb2, jac = match_labels(za, zb, K)
                # DCMM (same statistic as the report) + hard labels
                Pa = mixed_score(Aa, K, n_init=5)["Pi"]
                Pb = mixed_score(Ab, K, n_init=5)["Pi"]
                Pa, _ = align(full_pi[K][ok], Pa)
                Pb, _ = align(full_pi[K][ok], Pb)
                r = col_corr(Pa, Pb)
                _, jac_mm = match_labels(Pa.argmax(1), Pb.argmax(1), K)
                t1.append(dict(year=y, split=s, K=K,
                               ari_dcbm=adjusted_rand_score(za, zb), ari_dcmm=adjusted_rand_score(Pa.argmax(1),
                                                                                                  Pb.argmax(1)),
                               min_jac_dcbm=jac.min(), min_jac_dcmm_argmax=jac_mm.min(), min_r_dcmm=r.min(),
                               jac_dcbm=jac.tolist(), r_dcmm=r.tolist()))
                # held-out fit, both directions
                Za, Zb = np.eye(K)[za], np.eye(K)[zb]
                for src, Asrc, Atgt, Z, P in [("a", Aa, Ab, Za, Pa), ("b", Ab, Aa, Zb, Pb)]:
                    t3.append(dict(year=y, split=s, K=K, dir=src,
                                   r2_dcbm=r2_off(Atgt, fit_omega(Asrc, Z)),
                                   r2_dcmm=r2_off(Atgt, fit_omega(Asrc, P)),
                                   r2_rankK=r2_off(Atgt, rank_k(Asrc, K)),
                                   r2_theta_only=r2_off(Atgt, fit_omega(Asrc, np.ones((Asrc.shape[0], 1))))))
                if K == 4:
                    pur = purity4[ok]
                    flips.append(pd.DataFrame(dict(year=y, split=s, purity=pur, flip=(za != zb2))))
            if s % 10 == 0:
                print(y, "split", s, flush=True)

    t1, t3, flips = pd.DataFrame(t1), pd.DataFrame(t3), pd.concat(flips)
    t1.to_pickle(OUT / "t1_splits.pkl")
    t3.to_csv(OUT / "t3_heldout.csv", index=False)

    s12 = summarize_t1(t1)

    s3 = t3.groupby(["year", "K"])[["r2_theta_only", "r2_dcbm", "r2_dcmm", "r2_rankK"]].median().reset_index()
    win = t3.assign(dcmm_better=t3.r2_dcmm > t3.r2_dcbm).groupby(["year", "K"]).dcmm_better.mean()
    s3 = s3.merge(win.rename("share_splits_dcmm_better").reset_index())
    s3.to_csv(OUT / "t3_summary.csv", index=False)

    flips["purity_bin"] = pd.cut(flips.purity, PURITY_BINS, right=False,
                                 labels=["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"])
    sf = flips.groupby(["year", "purity_bin"], observed=True).agg(
        share_of_stocks=("flip", "size"), flip_rate=("flip", "mean")).reset_index()
    sf["share_of_stocks"] = sf.share_of_stocks / sf.groupby("year").share_of_stocks.transform("sum")
    sf.to_csv(OUT / "t3_flip_by_purity.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nT1/T2: split-half stability by K\n", s12.round(3).to_string(index=False))
    print("\nT3: held-out R^2 on the other half's A (median over splits)\n", s3.round(4).to_string(index=False))
    print("\nT3: DCBM label flips at K = 4 by full-sample DCMM purity\n", sf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
