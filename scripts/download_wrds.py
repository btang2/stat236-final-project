"""Download the raw WRDS tables used in the paper into data/.

Requires a WRDS account with access to CRSP, Compustat and Thomson Reuters 13F (s34). The `wrds` package asks
for your WRDS username and password on first use and can save them to ~/.pgpass.

Run from the repository root: python scripts/download_wrds.py
"""
from pathlib import Path

import wrds

DATA = Path(__file__).resolve().parents[1] / "data"

UNIVERSE = """
WITH universe AS (
    SELECT DISTINCT permno
    FROM crsp.dsp500list
    WHERE (start <= '2006-12-31' AND ending >= '2006-12-31')
       OR (start <= '2008-06-30' AND ending >= '2008-06-30')
)
"""

GVKEYS = """
, gvkeys AS (
    SELECT DISTINCT l.gvkey
    FROM crsp.ccmxpf_lnkhist AS l
    JOIN universe AS u ON l.lpermno = u.permno
    WHERE l.linktype IN ('LU', 'LC') AND l.linkprim IN ('P', 'C')
)
"""

QUERIES = {
    # 13F holdings, first filing vintage of each manager-report date
    "s34_dec2006_jun2008.csv": ("""
WITH first_vintage AS (
    SELECT mgrno, rdate, MIN(fdate) AS fdate
    FROM tr_13f.s34
    WHERE rdate IN ('2006-12-31', '2008-06-30')
    GROUP BY mgrno, rdate
)
SELECT s.rdate, s.fdate, s.mgrno, s.mgrname, s.typecode,
       s.cusip, s.shares, s.prc, s.shrout1, s.shrout2
FROM tr_13f.s34 AS s
JOIN first_vintage AS fv
  ON  s.mgrno = fv.mgrno AND s.rdate = fv.rdate AND s.fdate = fv.fdate
WHERE s.rdate IN ('2006-12-31', '2008-06-30')
""", ["rdate", "fdate"]),
    # CRSP daily stock file: returns, prices, shares outstanding, split factors
    "dsf_jan2006_dec2008.csv": (UNIVERSE + """
SELECT d.permno, d.date, d.ret, d.retx, d.prc, d.shrout, d.cfacshr
FROM crsp.dsf AS d
JOIN universe AS u ON d.permno = u.permno
WHERE d.date BETWEEN '2006-01-01' AND '2008-12-31'
""", ["date"]),
    # CRSP value-weighted market return (for market beta)
    "market_returns_jan2006_dec2008.csv": ("""
SELECT date, vwretd
FROM crsp.dsi
WHERE date BETWEEN '2006-01-01' AND '2008-12-31'
""", ["date"]),
    # CRSP-Compustat link
    "crsp_compustat_link.csv": (UNIVERSE + """
SELECT l.gvkey, l.lpermno AS permno, l.linktype, l.linkprim, l.linkdt, l.linkenddt
FROM crsp.ccmxpf_lnkhist AS l
JOIN universe AS u ON l.lpermno = u.permno
WHERE l.linktype IN ('LU', 'LC') AND l.linkprim IN ('P', 'C')
""", ["linkdt", "linkenddt"]),
    # Compustat historical GICS
    "gics_classification.csv": (UNIVERSE + GVKEYS + """
SELECT g.gvkey, g.gsector, g.ggroup, g.gind, g.gsubind, g.indfrom, g.indthru
FROM comp.co_hgic AS g
JOIN gvkeys AS k ON g.gvkey = k.gvkey
""", ["indfrom", "indthru"]),
    # Compustat annual book equity components (for book-to-market)
    "bm_control.csv": (UNIVERSE + GVKEYS + """
SELECT f.gvkey, f.datadate, f.fyear, f.seq, f.ceq, f.txditc, f.pstkrv, f.pstkl, f.pstk
FROM comp.funda AS f
JOIN gvkeys AS k ON f.gvkey = k.gvkey
WHERE f.indfmt = 'INDL' AND f.datafmt = 'STD' AND f.popsrc = 'D' AND f.consol = 'C'
  AND f.datadate BETWEEN '2004-01-01' AND '2007-12-31'
""", ["datadate"]),
}


def main():
    DATA.mkdir(exist_ok=True)
    conn = wrds.Connection()
    # S&P 500 membership history and CRSP names (CUSIP -> PERMNO), full tables
    conn.get_table(library="crsp", table="dsp500list").to_csv(DATA / "sp500_permno.csv")
    conn.get_table(library="crsp", table="stocknames").to_csv(DATA / "stocknames.csv")
    for fname, (sql, dates) in QUERIES.items():
        print("downloading", fname, flush=True)
        conn.raw_sql(sql, date_cols=dates).to_csv(DATA / fname)
    conn.close()
    print("done:", sorted(p.name for p in DATA.glob("*.csv")))


if __name__ == "__main__":
    main()
