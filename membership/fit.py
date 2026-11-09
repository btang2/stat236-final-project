"""Fit K = 4 Mixed-SCORE at both dates, align 2008 profiles to 2006 (common stocks), add stock features.

K = 4 is the largest K at which every profile replicates across independent halves of the managers at
both dates (s1_choose_K.py; split-half r >= 0.72). Profiles are numbered by the 2006 fit and named after
the evidence in s2/s3 (see README); names are labels for reading, not inputs to any test.
"""
import pickle

import numpy as np
import pandas as pd

from membership.core import OUT, align, cfg, cosine, mixed_score, raw
from membership.s1_choose_K import get_data

K = 4
NAMES = ["Blue chip", "Income", "Growth", "Small value"]   # named after s2/s4 evidence; order = 2006 vertex order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4 (dataviz validator: CVD pass)


def extra_features(c, t, rows):
    """Trailing 12-month dividend yield (sum of ret - retx), market beta and 12-month return, all <= t."""
    t = pd.Timestamp(t)
    d = raw(c, "dsf")
    d = d[d.permno.isin(rows) & (d.date > t - pd.DateOffset(months=12)) & (d.date <= t)]
    m = raw(c, "mkt").set_index("date").vwretd
    d = d.assign(div=d.ret - d.retx, mkt=m.reindex(d.date).values)
    g = d.groupby("permno")
    out = pd.DataFrame(index=pd.Index(rows, name="permno"))
    out["div_yield"] = g["div"].sum().reindex(rows)
    out["beta12"] = g.apply(lambda z: np.cov(z.ret, z.mkt)[0, 1] / z.mkt.var() if len(z) > 100 else np.nan
                            ).reindex(rows)
    out["ret12"] = g.ret.apply(lambda r: np.prod(1 + r.dropna()) - 1).reindex(rows)
    return out


def fit():
    p = OUT / "fit_cache.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    data, c = get_data(), cfg()
    res = {}
    for y, d in data.items():
        print(d["t"], "dropped (13F identifier artifacts, < 50 active owners):", d["dropped"])
        r = mixed_score(cosine(d["B"]), K)
        x = d["x"].join(extra_features(c, d["t"], d["rows"]))
        res[y] = dict(d, Pi=r["Pi"], R=r["R"], V=r["V"], pruned=r["pruned"], x=x)
    # align 2008 to 2006 on common stocks
    a, b = res["2006"], res["2008"]
    common = np.intersect1d(a["rows"], b["rows"])
    ia = pd.Index(a["rows"]).get_indexer(common)
    ib = pd.Index(b["rows"]).get_indexer(common)
    _, perm = align(a["Pi"][ia], b["Pi"][ib])
    b["Pi"], b["V"] = b["Pi"][:, perm], b["V"][perm]
    for y, r in res.items():
        pd.DataFrame(r["Pi"], index=pd.Index(r["rows"], name="permno"), columns=NAMES).assign(
            ticker=r["x"].ticker.values, name=r["x"].name.values, gsector=r["x"].gsector.values,
            pruned=r["pruned"]).to_csv(OUT / "tables" / f"memberships_{y}.csv")
    pickle.dump(res, open(p, "wb"))
    return res


def data(year):
    """The fit at one date, with display units added: B/M (cached book equity is in $m, market cap in $) and
    market cap in $bn."""
    r = fit()[year]
    x = r["x"].copy()
    x["BM"] = np.exp(x.logBM) * 1e6
    x["cap_bn"] = np.exp(x.logME) / 1e9
    return dict(r, x=x, n_held=r["B"].sum(0))


if __name__ == "__main__":
    res = fit()
    for y, r in res.items():
        Pi = r["Pi"]
        print(y, "dominant shares", np.round(np.bincount(Pi.argmax(1), minlength=K) / len(Pi), 2),
              "pure>=0.8", np.round((Pi >= 0.8).sum(0), 0), "max<0.5", round((Pi.max(1) < 0.5).mean(), 2))
        for k in range(K):
            print(f"  {NAMES[k]:11s}", ", ".join(r["x"].ticker.iloc[np.argsort(-Pi[:, k])[:12]]))
