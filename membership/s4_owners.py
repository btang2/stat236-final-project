"""Step 4: who owns each profile? Manager-side memberships.

Exposure e_mk = mean pi_k over the stocks manager m holds. Main tilt z_mk: vs. a same-size portfolio drawn
with probability proportional to stock popularity (core.manager_tilt_popularity), i.e. relative to a
typical institution; this nets out the near-universal preference for widely held large caps, which made
91% of managers 'Blue chip' under a uniform benchmark (zunif_*, kept for reference). Managers with >= 20 holdings only (thin portfolios: noisy).

(a) Specialization: share of managers with a significant positive tilt (two-sided normal p, BH q < 0.05
    within each profile). Calibration: the same statistic on random portfolios of the same sizes must give
    about 0 significant managers; we draw one random portfolio per manager and report the count.
(b) Persistence: Spearman of z_mk between 2006 and 2008 for managers present at both dates. Mandates are
    sticky, so real profiles should show persistent manager tilts.
(c) Identity: largest managers (by active dollars in the universe) with q < 0.05 positive tilt per profile.
(d) Type and size: 13F type code and log dollar holdings by each manager's strongest significant tilt
    (chi-square / Kruskal-Wallis). Type codes after 1998 are known to be unreliable (many advisers coded 5).
(e) Who holds the dollars: for stocks dominated by profile k, share of active institutional dollars held by
    managers significantly tilted to k, vs. those managers' share of all active dollars (lift).
(f) Exclusivity: correlation of tilts across profiles (are Income owners anti-Growth?).
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, norm, spearmanr

from membership.core import OUT, TYPECODE, bh, manager_exposure, manager_tilt_popularity, popularity_null
from membership.fit import K, NAMES, fit

MIN_HELD = 20


def _pop_sample(B, sizes, rng):
    """Fresh random portfolios from the popularity-weighted successive-sampling benchmark (calibration)."""
    p = B.sum(1) / B.sum()
    Br = np.zeros((B.shape[0], len(sizes)))
    for j, n in enumerate(sizes):
        Br[np.argsort(-np.log(rng.uniform(size=len(p))) / p)[:n], j] = 1
    return Br


def manager_table(r, rng):
    B, W, Pi, cols, x = r["B"], r["W"], r["Pi"], r["cols"], r["x"]
    e, z_unif = manager_exposure(B, Pi)
    null = popularity_null(B, Pi, rng)
    z = manager_tilt_popularity(B, Pi, null)   # main tilt: vs. a typical institution
    dollars = W * np.exp(x.logME.values)[:, None]
    m = pd.DataFrame({"mgrno": cols.mgrno.values, "mgrname": cols.mgrname.values,
                      "type": cols.typecode.map(TYPECODE).fillna("missing").values,
                      "n_held": B.sum(0), "usd_m": dollars.sum(0) / 1e6})
    for k in range(K):
        m[f"e_{NAMES[k]}"], m[f"z_{NAMES[k]}"], m[f"zunif_{NAMES[k]}"] = e[:, k], z[:, k], z_unif[:, k]
    m = m[m.n_held >= MIN_HELD].reset_index(drop=True)
    for k in range(K):
        p = 2 * norm.sf(np.abs(m[f"z_{NAMES[k]}"]))
        m[f"q_{NAMES[k]}"] = bh(p)
    sig = np.column_stack([(m[f"q_{n}"] < 0.05) & (m[f"z_{n}"] > 0) for n in NAMES])
    zz = m[[f"z_{n}" for n in NAMES]].values
    m["home"] = np.where(sig.any(1), np.array(NAMES)[np.argmax(np.where(sig, zz, -np.inf), 1)], "none")
    # calibration: one random same-size portfolio per manager from the benchmark itself
    Br = _pop_sample(B, m.n_held.astype(int).values, rng)
    zr = manager_tilt_popularity(Br, Pi, null)
    calib = {NAMES[k]: int(((bh(2 * norm.sf(np.abs(zr[:, k]))) < 0.05) & (zr[:, k] > 0)).sum()) for k in range(K)}
    return m, calib, dollars


def main():
    res = fit()
    rng = np.random.default_rng([236, 4])
    mt, out = {}, []
    for y, r in res.items():
        m, calib, dollars = manager_table(r, rng)
        mt[y] = m
        m.to_csv(OUT / "tables" / f"s4_managers_{y}.csv", index=False)
        lab = r["Pi"].argmax(1)
        mask_m = r["B"].sum(0) >= MIN_HELD
        Dm = dollars[:, mask_m]
        for k, nm in enumerate(NAMES):
            s = (m[f"q_{nm}"] < 0.05) & (m[f"z_{nm}"] > 0)
            home_usd = Dm[lab == k][:, s.values].sum() / Dm[lab == k].sum()
            all_usd = Dm[:, s.values].sum() / Dm.sum()
            big = m[s].sort_values("usd_m", ascending=False).head(12)
            out.append(dict(year=y, profile=nm, n_managers=len(m), n_sig_pos=int(s.sum()),
                            share_sig_pos=s.mean(), calib_random_sig=calib[nm],
                            home_dollar_share=home_usd, overall_dollar_share=all_usd, lift=home_usd / all_usd,
                            biggest_tilted="; ".join(f"{a.title()} (z={b:.1f})" for a, b in
                                                     zip(big.mgrname, big[f"z_{nm}"]))))
        # type & size by home profile
        ct = pd.crosstab(m.home, m.type)
        chi = chi2_contingency(ct)
        ct.to_csv(OUT / "tables" / f"s4_type_by_home_{y}.csv")
        groups = [g.usd_m.apply(np.log).values for _, g in m.groupby("home")]
        kw = kruskal(*groups)
        med = m.groupby("home").agg(n=("mgrno", "size"), median_usd_m=("usd_m", "median"),
                                    median_n_held=("n_held", "median"))
        med.to_csv(OUT / "tables" / f"s4_size_by_home_{y}.csv")
        zc = m[[f"z_{n}" for n in NAMES]].corr(method="spearman")
        zc.to_csv(OUT / "tables" / f"s4_tilt_corr_{y}.csv")
        print(f"\n===== {y}: {len(m)} managers with >= {MIN_HELD} holdings")
        print(ct.to_string(), f"\nchi2 p = {chi[1]:.2g}, Cramer's V = "
              f"{np.sqrt(chi[0] / (ct.values.sum() * (min(ct.shape) - 1))):.3f}")
        print(med.round(1).to_string(), f"\nKruskal-Wallis log $ p = {kw.pvalue:.2g}")
        print(zc.round(2).to_string())
    o = pd.DataFrame(out)
    # persistence
    a, b = mt["2006"].set_index("mgrno"), mt["2008"].set_index("mgrno")
    c = a.index.intersection(b.index)
    pers = pd.DataFrame([dict(profile=n, n_common=len(c), spearman=spearmanr(a.loc[c, f"z_{n}"], b.loc[c, f"z_{n}"])[0],
                              home_same=(a.loc[c, "home"] == b.loc[c, "home"])[a.loc[c, "home"] == n].mean())
                         for n in NAMES])
    pers.to_csv(OUT / "tables" / "s4_persistence.csv", index=False)
    o.to_csv(OUT / "tables" / "s4_summary.csv", index=False)
    pd.set_option("display.max_colwidth", 400)
    print(o.drop(columns="biggest_tilted").round(3).to_string(index=False))
    print(pers.round(3).to_string(index=False))
    for _, rr in o.iterrows():
        print(rr.year, rr.profile, ":", rr.biggest_tilted)


if __name__ == "__main__":
    main()
