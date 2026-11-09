"""Simulation check: Mixed-SCORE (with pruning) recovers planted DCMM memberships on a bipartite
stock x manager incidence like ours (n = 470 stocks, 2,500 managers)."""
import numpy as np

from membership.core import align, cosine, mixed_score


def simulate(n=470, m=2500, K=3, seed=0, outliers=5):
    rng = np.random.default_rng(seed)
    Pi = rng.dirichlet(np.full(K, 0.4), n)
    pure = rng.choice(n, 30 * K, replace=False)
    Pi[pure] = np.eye(K)[np.arange(len(pure)) % K]
    theta = rng.uniform(0.3, 1.0, n)
    Z = rng.choice(K, m, p=np.full(K, 1 / K))                 # manager styles
    act = rng.uniform(0.02, 0.4, m)                           # manager breadth
    P = np.full((K, K), 0.35) + 0.65 * np.eye(K)
    prob = np.clip(theta[:, None] * (Pi @ P[:, Z]) * act[None, :], 0, 1)
    B = (rng.uniform(size=prob.shape) < prob).astype(float)
    # a few odd stocks held only by a handful of managers (the failure the pruning guards against)
    B[:outliers] = 0
    for i in range(outliers):
        B[i, rng.choice(m, 8, replace=False)] = 1
    return B, Pi


def test_recovery():
    for prune in (0.05, 0.0):
        errs = []
        for s in range(5):
            B, Pi = simulate(seed=s)
            est = mixed_score(cosine(B), 3, prune=prune)["Pi"]
            Q, _ = align(Pi[5:], est[5:])
            errs.append(np.abs(Q - Pi[5:]).sum(1).mean() / 2)
        print(f"prune={prune}: mean TV error {np.mean(errs):.3f} (per seed {np.round(errs, 3)})")
        if prune:
            assert np.mean(errs) < 0.15


if __name__ == "__main__":
    test_recovery()
