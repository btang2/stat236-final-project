"""Stage 4: stock-level controls as of formation date t (gvkey, GICS, ME, B/M, institutional ownership)."""
import json

import numpy as np
import pandas as pd

from .common import active, check, last_trading_day, out_path, raw, report, tag
from .stage2_holdings import variant


def run(t, cfg):
    t = pd.Timestamp(t)
    tg = tag(t)
    uni = pd.read_parquet(out_path(cfg, "cache", f"universe_{tg}.parquet"))
    names = uni.set_index("permno").ticker

    # 1. CRSP-Compustat link valid on t.
    lk = raw(cfg, "link")
    lk = lk[lk.linktype.isin(["LU", "LC"]) & lk.linkprim.isin(["P", "C"])]
    lk = active(lk, "linkdt", "linkenddt", t, cfg)
    lk = lk[lk.permno.isin(uni.permno)]
    n_multi = int(lk.groupby("permno").gvkey.nunique().gt(1).sum())
    check(cfg, t, "one gvkey per permno", n_multi == 0, f"{n_multi} permnos with >1 gvkey")
    lk = lk.drop_duplicates("permno")[["permno", "gvkey"]]
    unlinked = sorted(set(uni.permno) - set(lk.permno))
    report(cfg, t, f"unlinked stocks ({len(unlinked)})",
           ", ".join(f"{p} {names[p]}" for p in unlinked) or "none")

    # 2. GICS valid on t.
    gi = active(raw(cfg, "gics"), "indfrom", "indthru", t, cfg)
    check(cfg, t, "one GICS row per gvkey", not gi.gvkey.duplicated().any())
    df = uni[["permno"]].merge(lk, on="permno", how="left").merge(
        gi[["gvkey", "gsector", "gind"]], on="gvkey", how="left")
    coded = int(df.gind.notna().sum())
    check(cfg, t, "at least 470 stocks GICS-coded", coded >= 470, f"{coded} coded")
    unc = df[df.gind.isna()]
    report(cfg, t, f"stocks without GICS on t ({len(unc)})",
           ", ".join(f"{r.permno} {names[r.permno]} (gvkey {r.gvkey})" for r in unc.itertuples()) or "none")
    bac = df.loc[df.gvkey == "007647", "gind"]
    report(cfg, t, "Bank of America (gvkey 007647) gind", ", ".join(bac) or "missing")
    check(cfg, t, "Bank of America gind == 402010", list(bac) == ["402010"])

    # 3. Book equity from the latest fiscal year ending <= t - 6 months.
    cutoff = t - pd.DateOffset(months=6)
    fa = raw(cfg, "funda")
    fa = fa[fa.datadate <= cutoff].sort_values("datadate").drop_duplicates("gvkey", keep="last")
    pref = fa.pstkrv.fillna(fa.pstkl).fillna(fa.pstk).fillna(0)
    fa["BE"] = fa.seq + fa.txditc.fillna(0) - pref
    df = df.merge(fa[["gvkey", "datadate", "BE"]], on="gvkey", how="left")

    # 4. ME on the last trading day <= t.
    d_me = last_trading_day(cfg, t)
    dsf = raw(cfg, "dsf")
    px = dsf[dsf.date == d_me].set_index("permno")
    df["ME"] = df.permno.map(px.prc.abs() * px.shrout * 1000)
    check(cfg, t, "every stock has ME", df.ME.notna().all(), f"ME date {d_me.date()}")

    nonpos = df[df.BE <= 0]
    missing = df[df.BE.isna()]
    report(cfg, t, f"BE <= 0, dropped from B/M ({len(nonpos)})",
           ", ".join(f"{p} {names[p]}" for p in nonpos.permno) or "none")
    report(cfg, t, f"BE missing ({len(missing)})",
           ", ".join(f"{p} {names[p]}" for p in missing.permno) or "none")
    df["BM"] = np.where(df.BE > 0, df.BE / df.ME, np.nan)

    # 5. Institutional ownership from W (NaN for stocks dropped in stage 2).
    rows = pd.read_parquet(out_path(cfg, "cache", f"W_rows_{tg}_{variant(cfg)}.parquet"))
    df["inst_own"] = df.permno.map(rows.set_index("permno").inst_own)

    out = df[["permno", "gvkey", "gsector", "gind", "ME", "BM", "inst_own"]]
    out.to_parquet(out_path(cfg, "cache", f"controls_{tg}_{variant(cfg)}.parquet"), index=False)

    used = {"gics.indfrom": gi.indfrom.max(), "funda.datadate": fa.datadate.max(), "dsf.date(ME)": d_me}
    used["link.linkdt"] = active(raw(cfg, "link"), "linkdt", "linkenddt", t, cfg).linkdt.max()
    out_path(cfg, "cache", f"dates_stage4_{tg}.json").write_text(
        json.dumps({"used": {k: str(v.date()) for k, v in used.items()}}, indent=1))
    return out
