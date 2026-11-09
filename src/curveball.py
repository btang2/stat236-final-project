"""Curveball sampler: random 0/1 matrices with every row sum and column sum fixed.

Used for the degree-preserving null networks: each stock keeps its number of owners and each manager keeps its
number of holdings, but who holds what is randomized, so the null has degree heterogeneity and no communities.
"""
import numpy as np


def _sweep(B, rng):
    """One sweep of Curveball trades over a random pairing of rows (in place). Trades on a pairing
    act on disjoint rows, so they commute; each trade is reversible w.r.t. the uniform law."""
    p = rng.permutation(B.shape[0])
    for a, b in zip(p[0::2], p[1::2]):
        d = B[a] ^ B[b]
        idx = np.flatnonzero(d)
        na = int(np.count_nonzero(B[a, idx]))
        if na == 0 or na == idx.size:
            continue
        rng.shuffle(idx)
        B[a, idx] = False
        B[b, idx] = False
        B[a, idx[:na]] = True
        B[b, idx[na:]] = True


def curveball(B, n_cycles, rng):
    """n_cycles palindromic cycles (row sweep, column sweep, row sweep) on a boolean matrix. Returns a new matrix."""
    B = B.copy()
    for _ in range(n_cycles):
        _sweep(B, rng)
        BT = np.ascontiguousarray(B.T)
        _sweep(BT, rng)
        B = np.ascontiguousarray(BT.T)
        _sweep(B, rng)
    return B
