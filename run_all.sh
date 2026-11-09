#!/usr/bin/env bash
# Reproduce every figure and table in the paper from the raw WRDS files in data/.
# Total runtime about 1 hour on a laptop; s3_degree (degree-preserving null, 100 draws per date) is most of it.
# Intermediate fits are cached in results/; delete results/ to start from scratch.
set -euo pipefail
cd "$(dirname "$0")"

step() { echo; echo "=== $*"; }

step "0. Build ownership matrices (stages 1, 2, 4)";        python prepare_data.py
step "1. Choose K by split-half reliability (~5 min)";     python -m membership.s1_choose_K
step "2. Final K = 4 Mixed-SCORE fit at both dates";       python -m membership.fit
step "3. Stock traits and sector enrichment";             python -m membership.s2_stocks
step "4. Popularity: R^2 on owners, degree null (~40 min)"; python -m membership.s3_degree
step "5. Manager tilts";                                  python -m membership.s4_owners
step "6. Sector averages and extra figures";              python -m membership.report_figures
step "7. Paper figures";
python -m membership.dcbm_k.eigengap
python -m membership.dcbm_k.dcbm_figures
python -m membership.dcbm_k.dcmm_split_half
python -m membership.dcbm_k.fig_matrix_row
python -m membership.dcbm_k.simplex_col
python -m membership.dcbm_k.popularity_row
step "8. Dynamics 2006 -> 2008 (bootstrap ~3 min)";
python -m membership.dynamics.d1_bootstrap
python -m membership.dynamics.d2_overall
python -m membership.dynamics.d7_matrices
step "9. Collect paper tables and figures into results/paper/"; python generate-figures.py
