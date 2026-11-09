# Ownership communities in 13F filings: DCMM and Mixed-SCORE on S&P 500 co-ownership networks

Code for the STAT 236 project report on community detection in institutional ownership networks. From 13F
holdings at two dates (December 2006, June 2008), we build a cosine co-ownership network of S&P 500 stocks over
*active* managers, show that the eigengap choice K = 2 under a DCBM is a popularity split, and fit a
degree-corrected mixed membership model (DCMM) with Mixed-SCORE. Split-half reliability across disjoint halves
of the managers selects K = 4 profiles (Blue chip, Income, Growth, Small value), whose movement between the two
dates we then measure against a bootstrap noise floor.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_wrds.py      # needs a WRDS account; see data/README.md
./run_all.sh                         # about 1 hour; writes everything to results/
```

Every figure and table of the paper then sits in `results/paper/`: figures under the file names used in the LaTeX
source, one CSV per table, and `results/paper/RESULTS.md` with all tables in one place.

The raw data are not included because the WRDS license does not allow redistribution.

## Where each result comes from

| paper (LaTeX label) | script | output |
|---|---|---|
| `tab:network_summary` | `prepare_data.py`, `membership/fit.py` | `tables/network_summary.csv` |
| `fig:dcbm-eigengap` | `membership/dcbm_k/eigengap.py` | `figures/dcbm_eigengap.png` |
| `fig:dcbm_k2_popularity` | `membership/dcbm_k/dcbm_figures.py` | `figures/dcbm_k2_popularity.png` |
| `fig:dcmm-split-half` | `membership/s1_choose_K.py` → `membership/dcbm_k/dcmm_split_half.py` | `figures/dcmm_split_half.png` |
| `fig:matrix` | `membership/dcbm_k/fig_matrix_row.py` | `figures/fig_matrix.png` |
| `tab:naming` | `membership/s2_stocks.py`, `membership/s4_owners.py` | `tables/naming.csv` |
| `fig:simplex` | `membership/dcbm_k/simplex_col.py` | `figures/dcmm_simplex.png` |
| `fig:popularity` | `membership/dcbm_k/popularity_row.py` | `figures/popularity_row.png` |
| `tab:degree` | `membership/s3_degree.py` | `tables/degree.csv` |
| `tab:sector-pi` | `membership/report_figures.py` | `tables/sector_membership.csv` |
| `tab:dynamics`, `tab:appendix-clusters` | `membership/dynamics/d1_bootstrap.py`, `d7_matrices.py` | `tables/dynamics_*.csv` |
| `tab:tilted-managers` | `membership/s4_owners.py` | `tables/tilted_managers.csv` |
| `tab:pure-stocks`, `tab:pure-stocks-2008` | `membership/fit.py` | `tables/pure_stocks.csv` |
| `tab:flow-stocks` | `membership/dynamics/d2_overall.py` | `tables/flow_summary.csv`, `tables/flow_stocks.csv` |

Output paths are relative to `results/paper/`. `generate-figures.py` assembles them from the per-step outputs
in `results/membership/`, `results/dcbm_k/` and `results/dynamics/`.

Two appendix tables show a hand-picked subset of rows: the named managers (`tab:tilted-managers`) and the example
stocks per flow (`tab:flow-stocks`). The CSVs hold the full ranked lists they were chosen from. Cluster names were assigned by the
authors from the trait regressions, purest stocks and owners; the code numbers profiles by the 2006 vertex order
and only attaches the names as labels.

## Method in brief

1. **Network** (`prepare_data.py`, `src/`). S&P 500 common stocks on each date; 13F holdings matched by CUSIP.
   Passive managers (holding more than 90% of the universe, e.g. Vanguard, Barclays, State Street) are removed.
   `B` is the binary stock × active-manager incidence matrix, and `A_ij = (BBᵀ)_ij / √(n_i n_j)` with `A_ii = 0`.
2. **Mixed-SCORE** (`membership/core.py`). Top-K eigenvectors of `A`, SCORE ratios `ξ_k / ξ_1`, pruning of the 5%
   most isolated rows, vertex hunting (k-means with 10K centres, then the successive projection algorithm), and
   memberships from barycentric coordinates with the `b_1` correction. 2008 profiles are matched to 2006 by the
   Hungarian algorithm.
3. **Choosing K** (`membership/s1_choose_K.py`). For K = 2…6, 100 random splits of the managers into disjoint
   halves; K = 4 is the largest K at which every profile's median split-half correlation is at least 0.7 at both
   dates.
4. **Popularity** (`membership/s3_degree.py`). R² of each profile on a cubic in standardized log owners;
   degree-preserving Curveball null networks (`src/curveball.py`).
5. **Dynamics** (`membership/dynamics/`). 200 manager-bootstrap refits per date give a same-date baseline `S`; the
   cross-date correlation `C` minus `S` is the movement between profiles.

All random steps use fixed seeds (236), so reruns give the same numbers.

## Repository layout

```
prepare_data.py          stages 1, 2, 4: universe, ownership matrix, stock controls
run_all.sh               full pipeline
generate-figures.py      collects paper tables and figures into results/paper/
config.yaml              dates, filters and the counts prepare_data.py checks
scripts/download_wrds.py WRDS queries for the raw data
src/                     data preparation and shared helpers
membership/              cosine matrix, Mixed-SCORE, K selection, fit, traits, popularity, owners
membership/dcbm_k/       eigengap, DCBM vs DCMM checks, paper figures
membership/dynamics/     bootstrap noise floor and 2006 → 2008 movement
tests/test_core.py       Mixed-SCORE recovers planted DCMM memberships in simulation
```

## Supplementary checks (not needed for the paper)

```bash
python -m membership.dcbm_k.k_test 100       # DCBM vs DCMM split-half stability and held-out fit (~30 min)
python -m membership.dcbm_k.null_k2 30       # DCBM pipeline on 30 no-community null networks per date
python -m membership.dcbm_k.dcbm_figures     # then also writes the DCBM stability/flip figures and tables
pytest tests                                 # Mixed-SCORE simulation check (no data needed)
```

## Notes

- Fits are cached (`results/membership/data_cache.pkl`, `fit_cache.pkl`, `results/dynamics/boot.pkl`). Delete
  `results/` after changing the data or settings.
- `prepare_data.py` logs every filter, exclusion and sanity check to `results/pipeline/log.md`.
- Runtime on a laptop: data preparation and fit under a minute, `s1_choose_K` about 5 minutes, `s3_degree` about
  40 minutes, bootstrap about 3 minutes.

## License

Code: MIT (see `LICENSE`). The WRDS data are not covered and must be obtained under your own WRDS license.
