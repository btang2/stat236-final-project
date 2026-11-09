"""Stage 1: S&P 500 universe and identifiers as of formation date t."""
import json

import pandas as pd

from .common import active, check, out_path, raw, report, tag


def run(t, cfg):
    t = pd.Timestamp(t)
    key = str(t.date())
    u = cfg["universe"]

    # 1. Members on t.
    sp = active(raw(cfg, "sp500"), "start", "ending", t, cfg)
    mem = sp[["permno"]].drop_duplicates()
    check(cfg, t, "S&P 500 member count", len(mem) == u["expect_members"][key],
          f"{len(mem)} vs expected {u['expect_members'][key]}")

    # 2. Names valid on t, at most one per permno.
    sn = active(raw(cfg, "stocknames"), "namedt", "nameenddt", t, cfg)
    df = mem.merge(sn, on="permno", how="left")
    n_dup = int(df.permno.duplicated().sum())
    check(cfg, t, "at most one stocknames row per permno", n_dup == 0, f"{n_dup} duplicates")
    no_name = df[df.shrcd.isna()]
    if len(no_name):
        report(cfg, t, "members with no stocknames row on t",
               ", ".join(map(str, no_name.permno)))

    # 3. Common shares only.
    keep = df.shrcd.isin(u["shrcd_keep"])
    excl = df[~keep]
    report(cfg, t, f"excluded by shrcd ({len(excl)})",
           "; ".join(f"{r.ticker} {r.comnam} (permno {r.permno}, shrcd {r.shrcd:g})"
                     if pd.notna(r.shrcd) else f"permno {r.permno} (no name row)"
                     for r in excl.sort_values("shrcd").itertuples()))
    df = df[keep]
    check(cfg, t, "count after shrcd filter", len(df) == u["expect_after_shrcd"][key],
          f"{len(df)} vs expected {u['expect_after_shrcd'][key]}")

    # 4. Output.
    out = df[["permno", "ncusip", "ticker", "comnam"]].sort_values("permno").reset_index(drop=True)
    out.to_parquet(out_path(cfg, "cache", f"universe_{tag(t)}.parquet"), index=False)

    # Golden-rule record: interval starts used are all <= t by construction.
    used = {"sp500.start": sp.start.max(), "stocknames.namedt": sn.namedt.max()}
    out_path(cfg, "cache", f"dates_stage1_{tag(t)}.json").write_text(
        json.dumps({"used": {k: str(v.date()) for k, v in used.items()}}, indent=1))
    return out
