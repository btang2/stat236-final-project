"""Load the cached ownership matrix and stock characteristics built by prepare_data.py (stages 1, 2, 4).

All inputs are dated <= the 13F report date t.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import out_path, raw, tag
from .stage2_holdings import variant


def load_net(cfg, t, which="without"):
    """W (stocks x managers, w = shares held / shares outstanding) restricted to manager set `which`:
    all | without (passive managers removed; used throughout the paper) | passive."""
    tg, v = tag(t), variant(cfg)
    rows = pd.read_parquet(out_path(cfg, "cache", f"W_rows_{tg}_{v}.parquet")).permno.values
    cols = pd.read_parquet(out_path(cfg, "cache", f"W_cols_{tg}_{v}.parquet"))
    W = sp.load_npz(out_path(cfg, "cache", f"W_{tg}_{v}.npz")).tocsc()
    keep = {"all": np.ones(len(cols), bool), "without": ~cols.passive.values,
            "passive": cols.passive.values}[which]
    W = W[:, keep].tocsr()
    if which == "without":   # must match the stage 2 cache exactly
        Wc = sp.load_npz(out_path(cfg, "cache", f"W_nopassive_{tg}_{v}.npz")).tocsr()
        assert W.shape == Wc.shape and abs(W - Wc).max() < 1e-12
    return W, rows, cols[keep].reset_index(drop=True)


def characteristics(cfg, t, rows, W_all=None, cols_all=None):
    """Stock characteristics at t: log ME, log BM, inst_own, passive ownership, 12-1 month momentum,
    12-month daily volatility, years in the S&P 500 (current spell), GICS sector, ticker."""
    t = pd.Timestamp(t)
    tg, v = tag(t), variant(cfg)
    c = pd.read_parquet(out_path(cfg, "cache", f"controls_{tg}_{v}.parquet")).set_index("permno")
    u = pd.read_parquet(out_path(cfg, "cache", f"universe_{tg}.parquet")).set_index("permno")
    x = pd.DataFrame(index=pd.Index(rows, name="permno"))
    x["ticker"] = u.ticker.reindex(rows)
    x["gsector"] = c.gsector.reindex(rows)
    x["gind"] = c.gind.reindex(rows)
    x["logME"] = np.log(c.ME.reindex(rows))
    x["logBM"] = np.log(c.BM.reindex(rows))
    x["inst_own"] = c.inst_own.reindex(rows)

    d = raw(cfg, "dsf")[["permno", "date", "ret"]]
    d = d[d.permno.isin(rows) & (d.date <= t)]
    y0, m1 = t - pd.DateOffset(months=12), t - pd.DateOffset(months=1)
    mom = d[(d.date > y0) & (d.date <= m1)].groupby("permno").ret.apply(lambda r: np.prod(1 + r.dropna()) - 1)
    vol = d[d.date > y0].groupby("permno").ret.std() * np.sqrt(252)
    x["mom12_1"] = mom.reindex(rows)
    x["vol12"] = vol.reindex(rows)

    s = raw(cfg, "sp500")
    s = s[(s.start <= t) & (s.ending.fillna(pd.Timestamp(cfg["open_date"])) >= t)]
    start = s.groupby("permno").start.max()
    x["years_in_index"] = ((t - start.reindex(rows)).dt.days / 365.25).values

    if W_all is not None:
        P = np.asarray(W_all[:, cols_all.passive.values].sum(axis=1)).ravel()
        T = np.asarray(W_all.sum(axis=1)).ravel()
        x["passive_own"] = P
        x["passive_share"] = np.where(T > 0, P / T, np.nan)
    return x
