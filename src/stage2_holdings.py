"""Stage 2: 13F holdings -> ownership matrix W (stocks x managers) as of formation date t."""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import check, last_trading_day, out_path, raw, report, tag


def variant(cfg):
    h = cfg["holdings"]
    return f"{h['late_vintage']}_{h['over_one']}"


def _cfacshr_on(cfg, permnos, date):
    """cfacshr per permno on the last trading day <= date."""
    d = last_trading_day(cfg, date)
    dsf = raw(cfg, "dsf")
    return dsf[(dsf.date == d) & dsf.permno.isin(permnos)].set_index("permno").cfacshr, d


def run(t, cfg):
    t = pd.Timestamp(t)
    key, hc = str(t.date()), cfg["holdings"]
    exceptions = {}

    uni = pd.read_parquet(out_path(cfg, "cache", f"universe_{tag(t)}.parquet"))
    cus2perm = uni.set_index("ncusip").permno

    # 1-2. Holdings on rdate == t; guard the first-vintage rule.
    s34 = raw(cfg, "s34")
    s34 = s34[s34.rdate == t]
    first = s34.groupby(["mgrno", "rdate"]).fdate.transform("min")
    n_bad = int((s34.fdate != first).sum())
    check(cfg, t, "fdate == min fdate within (mgrno, rdate)", n_bad == 0, f"{n_bad} rows dropped")
    s34 = s34[s34.fdate == first]

    h = s34[s34.cusip.isin(cus2perm.index)].reset_index(drop=True)
    h["permno"] = h.cusip.map(cus2perm).astype(int)

    # 3. Later-vintage rows.
    late = h.fdate > h.rdate
    late_mgrs = sorted(h.loc[late, "mgrno"].unique())
    report(cfg, t, "later-vintage rows on universe stocks",
           f"{int(late.sum())} rows from {len(late_mgrs)} managers "
           f"({', '.join(f'{m} {n}' for m, n in h.loc[late, ['mgrno', 'mgrname']].drop_duplicates().values)})")
    if hc["late_vintage"] == "rescale" and late.any():
        cf_t, _ = _cfacshr_on(cfg, h.permno.unique(), t)
        factor = pd.Series(1.0, index=h.index)
        for fd, idx in h[late].groupby("fdate").groups.items():
            cf_f, d_f = _cfacshr_on(cfg, h.loc[idx, "permno"].unique(), fd)
            factor[idx] = h.loc[idx, "permno"].map(cf_f).values / h.loc[idx, "permno"].map(cf_t).values
            exceptions[f"dsf.cfacshr@{d_f.date()}"] = {
                "reason": "split factor only, used to restate later-vintage shares to the rdate "
                          "basis (spec stage 2.3 DECISION main); no information after t enters W"}
        changed = h.loc[late & (factor != 1)]
        h["shares"] = h.shares * factor
        report(cfg, t, "stocks changed by later-vintage rescaling",
               f"{changed.permno.nunique()} stocks ({', '.join(map(str, sorted(changed.permno.unique())))}), "
               f"{len(changed)} rows")
    elif hc["late_vintage"] == "drop_managers":
        h = h[~h.mgrno.isin(late_mgrs)]
        report(cfg, t, "robustness: dropped later-vintage managers", str(late_mgrs))

    # 4. Denominator: shares outstanding on the last trading day <= t.
    d_so = last_trading_day(cfg, t)
    dsf = raw(cfg, "dsf")
    so = dsf[(dsf.date == d_so) & dsf.permno.isin(uni.permno)].set_index("permno").shrout * 1000
    check(cfg, t, "every universe stock has a denominator", so.reindex(uni.permno).notna().all(),
          f"denominator date {d_so.date()}")

    # 5. Ownership shares.
    h["w"] = h.shares / h.permno.map(so)
    check(cfg, t, "max individual w_ik <= 1", h.w.max() <= 1, f"max = {h.w.max():.4f}")
    tot = h.groupby("permno").w.sum()
    over = tot[tot > 1].sort_values(ascending=False)
    names = uni.set_index("permno").ticker
    report(cfg, t, f"stocks with sum_k w_ik > 1 ({len(over)})",
           ", ".join(f"{names[p]}={v:.3f}" for p, v in over.items()))
    if hc["over_one"] == "capped":
        h["w"] = h.w / h.permno.map(tot.clip(lower=1))

    # 6. Zero-holding stocks.
    zero = sorted(set(uni.permno) - set(h.permno))
    report(cfg, t, f"universe stocks with zero holdings, dropped ({len(zero)})",
           ", ".join(f"{p} {names[p]}" for p in zero) or "none")

    thin = tot[tot < 0.2].sort_values()
    report(cfg, t, f"thin 13F coverage, sum_k w_ik < 0.2 (kept per spec) ({len(thin)})",
           ", ".join(f"{p} {names[p]}={v:.3f}" for p, v in thin.items()) or "none")

    rows = np.sort(h.permno.unique())
    cols = np.sort(h.mgrno.unique())
    ri, ci = np.searchsorted(rows, h.permno), np.searchsorted(cols, h.mgrno)
    W = sp.csr_matrix((h.w.values, (ri, ci)), shape=(len(rows), len(cols)))
    density = W.nnz / np.prod(W.shape)
    if hc["late_vintage"] == "rescale":  # spec shapes are for the main variant
        exp = tuple(hc["expect_W_shape"][key])
        check(cfg, t, "W shape", W.shape == exp, f"{W.shape} vs expected {exp}")
    lo, hi = hc["expect_density"]
    check(cfg, t, "W density ~18-19%", lo <= density <= hi, f"{density:.4f}")

    # 7. Passive managers: hold > 90% of that date's network stocks.
    held = np.asarray((W > 0).sum(axis=0)).ravel()
    passive = held > hc["passive_share"] * W.shape[0]
    mgr = (h.drop_duplicates("mgrno").set_index("mgrno").loc[cols, ["mgrname", "typecode"]]
             .reset_index().assign(n_held=held, passive=passive))
    pl = mgr[mgr.passive]
    report(cfg, t, f"passive managers ({len(pl)}, >{hc['passive_share']:.0%} of {W.shape[0]} stocks)",
           "; ".join(f"{m.mgrno} {m.mgrname} ({m.n_held})" for m in pl.itertuples()))
    for mno, label in [(7900, "Barclays Bank PLC"), (90457, "Vanguard")]:
        check(cfg, t, f"passive list includes {label} ({mno})", mno in set(pl.mgrno))
    ss = pl[pl.mgrname.str.contains(r"^STATE STR(?:EET)?\b", regex=True)]  # filed as "STATE STR CORPORATION"
    check(cfg, t, "passive list includes State Street", len(ss) > 0,
          "; ".join(f"{m.mgrno} {m.mgrname}" for m in ss.itertuples()))

    W_np = W[:, ~passive]
    empty = rows[np.asarray(W_np.sum(axis=1)).ravel() == 0]
    report(cfg, t, "stocks with no holdings once passive managers are removed",
           ", ".join(map(str, empty)) or "none")

    # 8. Save.
    v, tg = variant(cfg), tag(t)
    stocks = pd.DataFrame({"permno": rows, "so": so.loc[rows].values, "so_date": d_so,
                           "inst_own": np.asarray(W.sum(axis=1)).ravel()})
    sp.save_npz(out_path(cfg, "cache", f"W_{tg}_{v}.npz"), W)
    sp.save_npz(out_path(cfg, "cache", f"W_nopassive_{tg}_{v}.npz"), W_np)
    stocks.to_parquet(out_path(cfg, "cache", f"W_rows_{tg}_{v}.parquet"), index=False)
    mgr.to_parquet(out_path(cfg, "cache", f"W_cols_{tg}_{v}.parquet"), index=False)
    pd.DataFrame({"permno": zero}).to_parquet(out_path(cfg, "cache", f"zero_holdings_{tg}.parquet"))

    used = {"s34.rdate": h.rdate.max(), "dsf.date(denominator)": d_so}
    out_path(cfg, "cache", f"dates_stage2_{tg}_{v}.json").write_text(json.dumps(
        {"used": {k: str(x.date()) for k, x in used.items()}, "exceptions": exceptions}, indent=1))

    return dict(W=W, W_nopassive=W_np, permno=rows, mgrno=cols, managers=mgr, stocks=stocks,
                zero_holdings=zero, passive=pl, over_one=over, late_rows=int(late.sum()),
                late_mgrs=late_mgrs, density=density)
