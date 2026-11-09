"""Step 2: what kind of stock sits at each ownership profile?

(a) Joint regression. For each profile k: pi_k ~ standardized characteristics (+/- GICS sector FE), HC3 SEs.
    Partial coefficients answer "holding the others fixed, which trait moves a stock toward profile k?".
    R^2 decomposition: characteristics only, sector only, both -> how much of a profile is style vs sector.
(b) Sector enrichment. Mean pi_k within each GICS sector minus the overall mean; p-value from 10,000
    permutations of pi rows across stocks; Benjamini-Hochberg q across all sector x profile cells.
(c) Popularity check: Spearman of pi_k with the number of active owners (DCMM should remove degree).
(d) Return validation (holdings-independent). Daily-return correlations over the 12 months before t:
    mean correlation of pairs sharing a dominant profile minus pairs that do not, within the same sector and
    across sectors. Null: dominant labels permuted within sector (2,000 draws), keeping sector composition.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from membership.core import OUT, bh, cfg, raw
from membership.fit import K, NAMES, fit

CHARS = {"logME": "size (log mkt cap)", "logBM": "book-to-market", "ret12": "12m return", "vol12": "volatility",
         "beta12": "market beta", "div_yield": "dividend yield", "years_in_index": "years in S&P 500"}
N_PERM = 10000


def joint_regression(r):
    x, Pi = r["x"], r["Pi"]
    Z = x[list(CHARS)].apply(lambda s: (s - s.mean()) / s.std())
    ok = Z.notna().all(1).values & x.gsector.notna().values
    Z = Z[ok]
    sec = pd.get_dummies(x.gsector[ok], prefix="s", drop_first=True).astype(float)
    rows, r2 = [], []
    for k in range(K):
        y = Pi[ok, k]
        f_c = sm.OLS(y, sm.add_constant(Z)).fit(cov_type="HC3")
        f_s = sm.OLS(y, sm.add_constant(sec)).fit()
        f_b = sm.OLS(y, sm.add_constant(pd.concat([Z, sec], axis=1))).fit(cov_type="HC3")
        r2.append(dict(profile=NAMES[k], R2_chars=f_c.rsquared, R2_sector=f_s.rsquared, R2_both=f_b.rsquared,
                       n=int(ok.sum())))
        for c in CHARS:
            rows.append(dict(profile=NAMES[k], char=c, coef=f_c.params[c], se=f_c.bse[c], p=f_c.pvalues[c],
                             coef_secFE=f_b.params[c], p_secFE=f_b.pvalues[c],
                             spearman=spearmanr(y, Z[c])[0]))
    t = pd.DataFrame(rows)
    t["q"] = bh(t.p)
    t["q_secFE"] = bh(t.p_secFE)
    return t, pd.DataFrame(r2)


SECTORS = {"10": "Energy", "15": "Materials", "20": "Industrials", "25": "Cons. discretionary",
           "30": "Cons. staples", "35": "Health care", "40": "Financials", "45": "Info tech", "50": "Telecom",
           "55": "Utilities"}


def sector_enrichment(r, rng):
    x, Pi = r["x"], r["Pi"]
    ok = x.gsector.notna().values
    g, P = x.gsector[ok].values, Pi[ok]
    secs = sorted(set(g))
    obs = np.array([P[g == s].mean(0) for s in secs]) - P.mean(0)
    null = np.empty((N_PERM,) + obs.shape)
    for b in range(N_PERM):
        Q = P[rng.permutation(len(P))]
        null[b] = np.array([Q[g == s].mean(0) for s in secs]) - P.mean(0)
    p = (1 + (np.abs(null) >= np.abs(obs)[None]).sum(0)) / (1 + N_PERM)
    rows = [dict(sector=SECTORS.get(s, s), n=int((g == s).sum()), profile=NAMES[k], excess=obs[i, k], p=p[i, k])
            for i, s in enumerate(secs) for k in range(K)]
    t = pd.DataFrame(rows)
    t["q"] = bh(t.p)
    return t


def return_validation(c, r, rng, n_perm=2000):
    t = pd.Timestamp(r["t"])
    d = raw(c, "dsf")
    d = d[d.permno.isin(r["rows"]) & (d.date > t - pd.DateOffset(months=12)) & (d.date <= t)]
    R = d.pivot(index="date", columns="permno", values="ret").reindex(columns=r["rows"])
    C = R.corr(min_periods=150).values
    lab = r["Pi"].argmax(1)
    sec = r["x"].gsector.fillna("NA").values
    iu = np.triu_indices(len(lab), 1)
    cc, same_sec = C[iu], (sec[:, None] == sec[None, :])[iu]
    okp = ~np.isnan(cc)

    def stat(l):
        same = (l[:, None] == l[None, :])[iu]
        out = {}
        for nm, m in (("within_sector", same_sec), ("across_sector", ~same_sec)):
            mm = m & okp
            out[nm] = cc[mm & same].mean() - cc[mm & ~same].mean()
        # per profile: pairs both dominated by k vs. pairs with one in k and the other not (across sector)
        for k in range(K):
            both = (l[:, None] == k)[iu[0], 0] & (l[iu[1]] == k)
            one = ((l[iu[0]] == k) ^ (l[iu[1]] == k))
            mm = ~same_sec & okp
            out[f"across_{NAMES[k]}"] = cc[mm & both].mean() - cc[mm & one].mean()
        return out

    obs = stat(lab)
    null = []
    for _ in range(n_perm):
        l2 = lab.copy()
        for s in np.unique(sec):
            m = sec == s
            l2[m] = rng.permutation(l2[m])
        null.append(stat(l2))
    null = pd.DataFrame(null)
    return pd.DataFrame([dict(stat=k, observed=v, null_mean=null[k].mean(), null_sd=null[k].std(),
                              p=(1 + (np.abs(null[k] - null[k].mean()) >= abs(v - null[k].mean())).sum())
                              / (1 + n_perm)) for k, v in obs.items()])


def main():
    c = cfg()
    res = fit()
    rng = np.random.default_rng([236, 2])
    for y, r in res.items():
        reg, r2 = joint_regression(r)
        enr = sector_enrichment(r, rng)
        deg = pd.DataFrame([dict(profile=NAMES[k], spearman_vs_n_owners=spearmanr(r["Pi"][:, k],
                                                                                   r["x"].n_active_owners)[0])
                            for k in range(K)])
        rv = return_validation(c, r, rng)
        for nm, t in (("regression", reg), ("r2", r2), ("sector", enr), ("degree", deg), ("returns", rv)):
            t.to_csv(OUT / "tables" / f"s2_{nm}_{y}.csv", index=False)
        print(f"\n===== {y}")
        print(r2.round(3).to_string(index=False))
        print(reg.pivot(index="char", columns="profile", values="coef").round(3)[NAMES].to_string())
        print("q<0.05 (no FE):\n", reg.pivot(index="char", columns="profile", values="q").round(3)[NAMES].to_string())
        print("with sector FE coef:\n", reg.pivot(index="char", columns="profile", values="coef_secFE").round(3)[NAMES].to_string())
        print(enr[enr.q < 0.05].round(3).to_string(index=False))
        print(deg.round(3).to_string(index=False))
        print(rv.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
