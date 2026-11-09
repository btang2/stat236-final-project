"""Shared setup for the 2006 -> 2008 dynamics analysis: aligned fits, common stocks, bootstrap draws."""
from pathlib import Path

import numpy as np
import pandas as pd

from membership.dynamics.d1_bootstrap import boot
from membership.fit import K, NAMES, COLORS, fit  # noqa: F401

OUT = Path(__file__).resolve().parents[2] / "results" / "dynamics"
TAB = OUT / "tables"
FIG = OUT / "figures"
SECTOR = {10: "Energy", 15: "Materials", 20: "Industrials", 25: "Cons. Disc.", 30: "Cons. Staples",
          35: "Health Care", 40: "Financials", 45: "Info Tech", 50: "Telecom", 55: "Utilities"}


def tv(P, Q):
    return np.abs(P - Q).sum(-1) / 2


def setup():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res, bt = fit(), boot()
    a, b = res["2006"], res["2008"]
    common = np.intersect1d(a["rows"], b["rows"])
    ia = pd.Index(a["rows"]).get_indexer(common)
    ib = pd.Index(b["rows"]).get_indexer(common)
    x = a["x"].iloc[ia].copy()
    x["sector"] = x.gsector.astype(float).map(SECTOR)
    x["ret_0707_0806"] = b["x"].ret12.iloc[ib].values          # 12m return to Jun 2008
    x["ret_2006"] = a["x"].ret12.iloc[ia].values               # 12m return to Dec 2006
    x["owners_2006"] = a["B"][ia].sum(1)
    x["owners_2008"] = b["B"][ib].sum(1)
    return dict(res=res, a=a, b=b, common=common, ia=ia, ib=ib, x=x,
                P06=a["Pi"][ia], P08=b["Pi"][ib], B06=bt["2006"][:, ia], B08=bt["2008"][:, ib])


def noise_pairs(Pb):
    """Disjoint pairs of bootstrap refits at one date: (draw 0 vs 1), (2 vs 3), ..."""
    n = Pb.shape[0] // 2 * 2
    return Pb[0:n:2], Pb[1:n:2]
