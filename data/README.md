# Data

The raw data come from Wharton Research Data Services (WRDS) and are **not included** in this repository: the
WRDS license does not allow redistribution. With a WRDS account, download them with

```bash
python scripts/download_wrds.py
```

which writes these files here:

| file | WRDS source | used for |
|---|---|---|
| `s34_dec2006_jun2008.csv` | Thomson Reuters 13F (`tr_13f.s34`), report dates 2006-12-31 and 2008-06-30, first filing vintage | manager holdings (about 150 MB) |
| `sp500_permno.csv` | CRSP `dsp500list` | S&P 500 membership on each date |
| `stocknames.csv` | CRSP `stocknames` | CUSIP → PERMNO, tickers, names, share codes |
| `dsf_jan2006_dec2008.csv` | CRSP daily stock file | shares outstanding, prices, split factors, returns, dividend yield |
| `market_returns_jan2006_dec2008.csv` | CRSP `dsi` (`vwretd`) | market beta |
| `crsp_compustat_link.csv` | CRSP/Compustat merged link history | PERMNO → GVKEY |
| `gics_classification.csv` | Compustat `co_hgic` | GICS sector |
| `bm_control.csv` | Compustat `funda` | book equity for book-to-market |

`prepare_data.py` checks the counts it expects (for example 476 / 473 common-share S&P 500 stocks and the passive
manager list) and stops if a download differs.
