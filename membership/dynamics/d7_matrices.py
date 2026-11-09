"""Profile-by-profile (4x4) matrices comparing the two snapshots, on the 430 stocks in the index at both dates.

m1_identity.csv        corr(pi_k 2006, pi_l 2008) across stocks: row = 2006 profile, column = 2008 profile.
                       Diagonal = how much each profile labels the same stocks at both dates.
m2_transition.csv      dominant profile 2006 (row) -> 2008 (column): counts.
m3_transition_share.csv  the same as row shares (each row sums to 1).
m4_soft_transition.csv mean 2008 membership of stocks grouped by 2006 dominant profile (rows sum to 1):
                       a transition matrix that uses the whole membership vector, not just the argmax.
m5_corr_2006.csv       corr(pi_k, pi_l) across stocks within 2006.
m6_corr_2008.csv       the same within 2008.

Movement (cross-date minus same-date), from the manager bootstrap draws (d1):
m7_same_2006.csv       corr(pi_k refit a, pi_l refit b), two refits at 2006, median over 100 disjoint pairs.
m8_same_2008.csv       the same at 2008.
m9_same_pooled.csv     median over both dates' pairs: the matrix a profile set with NO change would produce.
m10_cross_boot.csv     corr(pi_k refit 2006, pi_l refit 2008), median over 200 paired draws.
m11_movement.csv       m10 - m9. Diagonal < 0 = identity lost beyond noise; off-diagonal > 0 = flow k -> l.
m12_movement_ci.csv    2.5% / 97.5% quantiles of (cross draw - m9) per cell.
m13_flows.csv          m11 with only the positive off-diagonal cells kept (others 0).
"""
import numpy as np
import pandas as pd

from membership.dynamics.common import K, NAMES, TAB, noise_pairs, setup


def cross_corr(P, Q):
    return np.array([[np.corrcoef(P[:, k], Q[:, l])[0, 1] for l in range(K)] for k in range(K)])


def main():
    s = setup()
    P06, P08 = s["P06"], s["P08"]
    d06, d08 = P06.argmax(1), P08.argmax(1)
    frame = lambda M, r="2006", c="2008": pd.DataFrame(M, pd.Index(NAMES, name=r), pd.Index(NAMES, name=c))

    counts = np.array([[((d06 == k) & (d08 == l)).sum() for l in range(K)] for k in range(K)])
    out = {
        "m1_identity": frame(cross_corr(P06, P08)),
        "m2_transition": frame(counts),
        "m3_transition_share": frame(counts / counts.sum(1, keepdims=True)),
        "m4_soft_transition": frame(np.array([P08[d06 == k].mean(0) for k in range(K)])),
        "m5_corr_2006": frame(np.corrcoef(P06.T), "2006", "2006"),
        "m6_corr_2008": frame(np.corrcoef(P08.T), "2008", "2008"),
    }
    B06, B08 = s["B06"], s["B08"]
    sa = np.array([cross_corr(p, q) for p, q in zip(*noise_pairs(B06))])
    sb = np.array([cross_corr(p, q) for p, q in zip(*noise_pairs(B08))])
    same = np.median(np.concatenate([sa, sb]), 0)
    cross = np.array([cross_corr(p, q) for p, q in zip(B06, B08)])
    move = np.median(cross, 0) - same
    lo, hi = np.quantile(cross, 0.025, 0) - same, np.quantile(cross, 0.975, 0) - same
    ci = np.array([[f"{lo[k, l]:+.2f} to {hi[k, l]:+.2f}" for l in range(K)] for k in range(K)])
    flows = np.where((move > 0) & ~np.eye(K, dtype=bool), move, 0.0)
    out.update({
        "m7_same_2006": frame(np.median(sa, 0), "2006 refit a", "2006 refit b"),
        "m8_same_2008": frame(np.median(sb, 0), "2008 refit a", "2008 refit b"),
        "m9_same_pooled": frame(same, "refit a", "refit b"),
        "m10_cross_boot": frame(np.median(cross, 0)),
        "m11_movement": frame(move),
        "m12_movement_ci": frame(ci),
        "m13_flows": frame(flows),
    })
    for name, df in out.items():
        df.to_csv(TAB / f"{name}.csv")
        print(f"\n{name}\n{(df if df.dtypes.iloc[0] == object else df.round(2)).to_string()}")


if __name__ == "__main__":
    main()
