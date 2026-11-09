"""Manager bootstrap at each date: how much would memberships move with no real change?

Resample the active managers with replacement (columns of B), rebuild the cosine matrix, refit K = 4
Mixed-SCORE and align each refit to that date's full fit (which fit.py already aligned to the 2006 names).
Two independent refits at the SAME date differ only by estimation noise. That is the noise floor for the
2006 -> 2008 comparison, which also compares two independently estimated fits.

Output: out/boot.pkl = {"2006": Pi_b (N_BOOT x n x K), "2008": ...}. ~3 min.
"""
import pickle
from pathlib import Path

import numpy as np

from membership.core import align, cosine, mixed_score
from membership.fit import K, fit

OUT = Path(__file__).resolve().parents[2] / "results" / "dynamics"
N_BOOT = 200


def boot():
    p = OUT / "boot.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    OUT.mkdir(parents=True, exist_ok=True)
    res = fit()
    rng = np.random.default_rng(236)
    out = {}
    for y, r in res.items():
        B, full = r["B"], r["Pi"]
        draws = []
        for b in range(N_BOOT):
            Bb = B[:, rng.integers(0, B.shape[1], B.shape[1])]
            Pb = mixed_score(cosine(Bb), K, n_init=5)["Pi"]
            Pb, _ = align(full, Pb)
            draws.append(Pb)
            if b % 50 == 0:
                print(y, b, flush=True)
        out[y] = np.array(draws)
    pickle.dump(out, open(p, "wb"))
    return out


if __name__ == "__main__":
    o = boot()
    for y, P in o.items():
        print(y, P.shape, "mean per-stock bootstrap sd:", np.round(P.std(0).mean(0), 3))
