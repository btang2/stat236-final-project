"""Shared helpers: config, raw-data loading, interval filtering, cache and log I/O."""
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Identifier columns that must stay strings so leading zeros survive.
STR_COLS = {"cusip": str, "ncusip": str, "gvkey": str, "gind": str, "gsector": str}

RAW = {
    "sp500": ("sp500_permno.csv", ["start", "ending"]),
    "stocknames": ("stocknames.csv", ["namedt", "nameenddt"]),
    "s34": ("s34_dec2006_jun2008.csv", ["rdate", "fdate"]),
    "dsf": ("dsf_jan2006_dec2008.csv", ["date"]),
    "mkt": ("market_returns_jan2006_dec2008.csv", ["date"]),
    "link": ("crsp_compustat_link.csv", ["linkdt", "linkenddt"]),
    "gics": ("gics_classification.csv", ["indfrom", "indthru"]),
    "funda": ("bm_control.csv", ["datadate"]),
}


def load_config(path=ROOT / "config.yaml", **overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    return cfg


def out_path(cfg, *parts):
    p = Path(cfg["out_dir"])
    p = p if p.is_absolute() else ROOT / p
    p = p.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=None)
def _read_raw(data_dir, name):
    fname, dates = RAW[name]
    df = pd.read_csv(Path(data_dir) / fname, index_col=0, dtype=STR_COLS, parse_dates=dates)
    for c in ("permno", "mgrno"):
        if c in df:
            df[c] = df[c].astype("Int64").astype(int) if df[c].notna().all() else df[c].astype("Int64")
    return df


def raw(cfg, name):
    """Read-only raw table (cached). Callers get a copy so the cache is never mutated."""
    d = Path(cfg["data_dir"])
    d = d if d.is_absolute() else ROOT / d
    return _read_raw(str(d), name).copy()


def active(df, start, end, t, cfg):
    """Interval rule: valid_from <= t <= valid_to.fillna(open_date)."""
    t = pd.Timestamp(t)
    return df[(df[start] <= t) & (df[end].fillna(pd.Timestamp(cfg["open_date"])) >= t)]


def last_trading_day(cfg, t):
    """Last dsf trading day <= t. Never filter on date == t."""
    d = raw(cfg, "dsf").date
    return d[d <= pd.Timestamp(t)].max()


def tag(t):
    return pd.Timestamp(t).strftime("%Y%m%d")


def log(cfg, text):
    with open(out_path(cfg, cfg.get("log_file", "log.md")), "a") as f:
        f.write(text.rstrip() + "\n")


def report(cfg, t, item, body):
    log(cfg, f"- **REPORT** [{pd.Timestamp(t).date()}] {item}: {body}")


def check(cfg, t, name, cond, detail=""):
    """Assertion that is also written to the log."""
    status = "PASS" if cond else "FAIL"
    log(cfg, f"- **ASSERT {status}** [{pd.Timestamp(t).date()}] {name}" + (f" ({detail})" if detail else ""))
    assert cond, f"{name}: {detail}"


def start_log(cfg, title):
    log(cfg, f"\n## {title} — run {datetime.now():%Y-%m-%d %H:%M:%S}\n")
