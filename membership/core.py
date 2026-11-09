"""Core methods: cosine co-ownership matrix, Mixed-SCORE, Hungarian alignment, manager tilts.

Network. B = binary incidence, stocks x active managers (passive managers removed):
B_ik = 1 if manager k reports holding stock i at the 13F date t. The stock-by-stock matrix
    A_ij = cos(B_i, B_j) = (# owners i and j share) / sqrt(n_i n_j),   A_ii = 0
is a degree-normalized co-ownership (Gram / "covariance") matrix. Binary, not weighted: weighted
holdings localize on blockholder stocks and give unstable memberships.

Model. Degree-corrected mixed membership (DCMM): E[A] = Theta Pi P Pi' Theta. Each stock i has a membership
vector pi_i on the K-simplex (how much of its ownership pattern comes from each of K pure "ownership
profiles"); theta_i absorbs popularity, so pi is about *which* managers own a stock, not *how many*.

Estimator. Mixed-SCORE (Jin, Ke & Luo 2023):
  1. top-K eigenpairs of A by |lambda|; SCORE ratios r_i = (xi_2(i)..xi_K(i)) / xi_1(i), clipped at +-log n.
  2. Pruning (step recommended in the paper): drop the PRUNE share of points with the largest distance to
     their 10th nearest neighbour before vertex hunting, so a single odd stock cannot become a vertex.
  3. Vertex hunting: k-means (L = 10K centres) on the kept points, then the successive projection algorithm
     on the centres picks the K centres spanning the simplex.
  4. Barycentric coordinates of every r_i (pruned points included); pi_i(k) prop. to w_i(k) / b_1(k),
     b_1(k) = [lambda_1 + v_k' diag(lambda_2..K) v_k]^(-1/2); negatives -> 0; renormalize.

Owners (second benchmark: core.manager_tilt_popularity draws stocks in proportion to their popularity).
Manager m's exposure to profile k is the mean membership of the stocks it holds,
e_mk = (1/n_m) sum_{i held} pi_ik. Its tilt is a z-score against a random portfolio of the same size drawn
without replacement from the universe: z = (e_mk - mean_i pi_ik) / sqrt(var_i(pi_ik)/n_m * (N-n_m)/(N-1)).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, raw  # noqa: E402
from src.ext_common import characteristics, load_net  # noqa: E402

# numpy + macOS Accelerate raises spurious FP warnings in matmul on valid inputs
warnings.filterwarnings("ignore", r".*encountered in (matmul|dot)", RuntimeWarning)

OUT = ROOT / "results" / "membership"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
DATES = {"2006": "2006-12-31", "2008": "2008-06-30"}
PRUNE = 0.05
TYPECODE = {1: "bank", 2: "insurance", 3: "investment co.", 4: "investment adviser", 5: "other"}


# --- data ---------------------------------------------------------------------------------------
def load(cfg, year):
    """Binary active-manager network, manager table and stock characteristics at date t."""
    t = DATES[year]
    W, rows, cols = load_net(cfg, t, "without")
    Wall, _, cols_all = load_net(cfg, t, "all")
    B = (W.toarray() > 0).astype(np.float64)
    x = characteristics(cfg, t, rows, Wall, cols_all)
    x["n_active_owners"] = B.sum(1)
    names = raw(cfg, "stocknames")
    names = names[names.namedt <= t].sort_values("namedt").groupby("permno").comnam.last()
    x["name"] = names.reindex(rows).str.title().values
    return dict(t=t, B=B, W=W.toarray(), rows=rows, cols=cols, x=x)


def cosine(B):
    G = B @ B.T
    d = np.sqrt(np.diag(G))
    with np.errstate(invalid="ignore", divide="ignore"):
        A = G / np.outer(d, d)
    A = np.nan_to_num(A)
    np.fill_diagonal(A, 0)
    return A


def spectrum(A):
    lam, xi = np.linalg.eigh(A)
    o = np.argsort(-np.abs(lam))
    lam, xi = lam[o], xi[:, o]
    if xi[:, 0].sum() < 0:
        xi[:, 0] = -xi[:, 0]
    return lam, xi


# --- Mixed-SCORE --------------------------------------------------------------------------------
def spa(X, K):
    """Successive projection algorithm on rows of X (n x K, rows = [1, r]); returns K row indices."""
    R = X.copy()
    idx = []
    for _ in range(K):
        j = int(np.argmax((R ** 2).sum(1)))
        idx.append(j)
        u = R[j] / np.linalg.norm(R[j])
        R = R - np.outer(R @ u, u)
    return idx


def mixed_score(A, K, seed=236, prune=PRUNE, n_init=20):
    lam, xi = spectrum(A)
    lam, xi = lam[:K], xi[:, :K]
    n = A.shape[0]
    R = np.clip(xi[:, 1:] / xi[:, [0]], -np.log(n), np.log(n))
    if prune > 0:
        d = cKDTree(R).query(R, k=11)[0][:, -1]
        keep = d <= np.quantile(d, 1 - prune)
    else:
        keep = np.ones(n, bool)
    L = min(10 * K, keep.sum() - 1)
    with np.errstate(all="ignore"):
        C = KMeans(L, n_init=n_init, random_state=seed).fit(R[keep]).cluster_centers_
    V = C[spa(np.column_stack([np.ones(L), C]), K)]
    V = V[np.lexsort(V.T[::-1])] if K > 1 else V
    M = np.vstack([V.T, np.ones(K)])
    Wt = np.linalg.solve(M, np.vstack([R.T, np.ones(n)])).T
    b1 = 1 / np.sqrt(np.abs(lam[0] + np.einsum("kj,j,kj->k", V, lam[1:], V)))
    Pi = np.maximum(Wt / b1[None, :], 0)
    Pi = Pi / Pi.sum(1, keepdims=True)
    return dict(Pi=Pi, V=V, R=R, lam=lam, pruned=~keep)


def align(P, Q):
    """Column permutation of Q matching P (Hungarian on -correlation). Returns (Q aligned, perm)."""
    K = P.shape[1]
    C = np.nan_to_num(np.corrcoef(P.T, Q.T)[:K, K:])
    _, j = linear_sum_assignment(-C)
    return Q[:, j], j


def col_corr(P, Q):
    return np.array([np.corrcoef(P[:, k], Q[:, k])[0, 1] for k in range(P.shape[1])])


# --- owners -------------------------------------------------------------------------------------
def manager_exposure(B, Pi):
    """e (managers x K) = mean pi over holdings; z = tilt vs. same-size random portfolio."""
    n_m = B.sum(0)
    N = B.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (B.T @ Pi) / n_m[:, None]
        mu, var = Pi.mean(0), Pi.var(0)
        se = np.sqrt(var[None, :] / n_m[:, None] * (N - n_m[:, None]) / (N - 1))
        z = (e - mu[None, :]) / se
    return e, z


def popularity_null(B, Pi, rng, n_draw=5000):
    """Null moments of e_k(n) for a portfolio of n stocks drawn one at a time, without replacement, with
    probability proportional to stock popularity (number of active owners) among those not yet drawn.
    Efraimidis-Spirakis keys u^(1/p): sorting once gives a successive sample whose every prefix of length n
    is a valid size-n draw, so one ordering serves all portfolio sizes. Returns mean, sd: (N+1) x K."""
    p = B.sum(1) / B.sum()
    N, K = Pi.shape
    s1, s2 = np.zeros((N + 1, K)), np.zeros((N + 1, K))
    n = np.arange(1, N + 1)[:, None]
    for _ in range(n_draw):
        o = np.argsort(-np.log(rng.uniform(size=N)) / p)
        e = np.cumsum(Pi[o], 0) / n
        s1[1:] += e
        s2[1:] += e ** 2
    mu = s1 / n_draw
    sd = np.sqrt(np.maximum(s2 / n_draw - mu ** 2, 0))
    return mu, sd, p


def manager_tilt_popularity(B, Pi, null):
    """Tilt vs. a same-size portfolio from the popularity-weighted benchmark: 'more profile-k than a typical
    institution would hold', which nets out the market-wide preference for widely held large caps."""
    mu, sd, _ = null
    n_m = B.sum(0).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (B.T @ Pi) / n_m[:, None]
        z = (e - mu[n_m]) / sd[n_m]
    return z


# --- multiple testing ---------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def cfg():
    return load_config()
