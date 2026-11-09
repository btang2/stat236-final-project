"""Collect every table and figure in the paper into results/paper/.

Run after the analysis steps in run_all.sh. Writes
  results/paper/figures/   the six figures, under the file names used in the LaTeX source
  results/paper/tables/    one CSV per table
  results/paper/RESULTS.md all tables in one readable file, labelled by their LaTeX \\label

Some paper tables show a hand-picked subset of rows (illustrative managers and flow stocks). For those the CSV
holds the full ranked list the subset was chosen from.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from membership.fit import NAMES, data, fit
from membership.report_figures import sector_table

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
MEM, DCB, DYN = RES / "membership" / "tables", RES / "dcbm_k", RES / "dynamics" / "tables"
PAPER = RES / "paper"
YEARS = ["2006", "2008"]

FIGURES = {  # paper file name -> produced by
    "dcbm_eigengap.png": DCB / "dcbm_eigengap.png",            # membership/dcbm_k/eigengap.py
    "dcbm_k2_popularity.png": DCB / "dcbm_k2_popularity.png",  # membership/dcbm_k/dcbm_figures.py
    "dcmm_split_half.png": DCB / "dcmm_split_half.png",        # membership/dcbm_k/dcmm_split_half.py
    "fig_matrix.png": DCB / "fig_matrix_row.png",              # membership/dcbm_k/fig_matrix_row.py
    "dcmm_simplex.png": DCB / "simplex_col.png",               # membership/dcbm_k/simplex_col.py
    "popularity_row.png": DCB / "popularity_row.png",          # membership/dcbm_k/popularity_row.py
}


def network_summary(res):
    rows = []
    for y in YEARS:
        B = res[y]["B"]
        sd, md = B.sum(1), B.sum(0)
        rows.append({"date": res[y]["t"], "stocks_n": B.shape[0], "active_managers_m": B.shape[1],
                     "density_B": B.mean(),
                     "stock_degree_min/med/max": f"{sd.min():.0f} / {np.median(sd):.0f} / {sd.max():.0f}",
                     "manager_degree_min/med/max": f"{md.min():.0f} / {np.median(md):.0f} / {md.max():.0f}"})
    return pd.DataFrame(rows).set_index("date").T


def eigengap():
    e = pd.read_csv(DCB / "eigengap.csv", dtype={"year": str})
    return e[e.k <= 6].pivot(index="k", columns="year", values=["lam", "ratio_next"])


def split_half():
    s = pd.read_csv(MEM / "s1_split_half.csv", dtype={"year": str})
    t = s.pivot_table(index=["K", "vertex"], columns="year", values="split_half_r_median")
    passes = s.groupby(["year", "K"]).split_half_r_median.min().unstack(0) >= 0.7
    return t, passes


def naming(res):
    """Table: purest stocks, trait coefficients (joint regression, per SD) and most-tilted owners."""
    s4 = pd.read_csv(MEM / "s4_summary.csv", dtype={"year": str}).set_index(["year", "profile"])
    rows = []
    for k, nm in enumerate(NAMES):
        r = {"profile": nm}
        for y in YEARS:
            P, x = res[y]["Pi"], res[y]["x"]
            r[f"purest_{y}"] = ", ".join(x.ticker.iloc[np.argsort(-P[:, k])[:4]])
            reg = pd.read_csv(MEM / f"s2_regression_{y}.csv").set_index(["profile", "char"])
            for c, lab in (("logME", "size"), ("div_yield", "div_yield"), ("logBM", "book_to_market")):
                r[f"{lab}_coef_{y}"] = reg.loc[(nm, c), "coef"]
                r[f"{lab}_q_{y}"] = reg.loc[(nm, c), "q"]
            r[f"top_tilted_owners_{y}"] = s4.loc[(y, nm), "biggest_tilted"]
        rows.append(r)
    return pd.DataFrame(rows)


def popularity(res):
    """Spearman of membership with number of active owners (text of Section III.I)."""
    return pd.DataFrame({y: [spearmanr(res[y]["x"].n_active_owners, res[y]["Pi"][:, k])[0] for k in range(4)]
                         for y in YEARS}, index=NAMES)


def degree():
    d = pd.read_csv(MEM / "s3_degree.csv", dtype={"year": str})
    d = d[d.profile.isin(NAMES)]
    out = d.pivot(index="profile", columns="year", values=["R2_degree_obs", "split_half_obs"]).reindex(NAMES)
    out.columns = [f"{'R2_on_owners' if a == 'R2_degree_obs' else 'split_half_r'}_{y}" for a, y in out.columns]
    return out.astype(float)


def sectors():
    t = sector_table()
    n = t.pivot(index="sector", columns="year", values="n")
    pi = t.pivot(index="sector", columns="year", values=NAMES) * 100
    out = pd.concat([n.add_prefix("n_")] + [pi[nm].add_prefix(f"{nm}_") for nm in NAMES], axis=1)
    order = list(dict.fromkeys(t.sector))
    out = out.reindex(order)
    allrow = {f"n_{y}": n[y].sum() for y in YEARS}
    for nm in NAMES:      # weighted by sector size = mean over stocks with a GICS sector
        for y in YEARS:
            allrow[f"{nm}_{y}"] = (pi[nm][y] * n[y]).sum() / n[y].sum()
    out.loc["All stocks"] = allrow
    return out


def dynamics():
    rd = lambda f: pd.read_csv(DYN / f"{f}.csv", index_col=0)
    return {"identity": rd("m1_identity"), "same_date_baseline": rd("m9_same_pooled"),
            "cross_date": rd("m10_cross_boot"), "movement": rd("m11_movement"), "flows": rd("m13_flows")}


def tilted_managers(n=8):
    """Managers (>= 20 holdings) whose largest tilt z is toward the profile and positive, largest by dollars."""
    rows = []
    for y in YEARS:
        m = pd.read_csv(MEM / f"s4_managers_{y}.csv")
        z = m[[f"z_{nm}" for nm in NAMES]].values
        for k, nm in enumerate(NAMES):
            s = m[(z.argmax(1) == k) & (z[:, k] > 0)].nlargest(n, "usd_m")
            rows += [dict(year=y, profile=nm, manager=r.mgrname, stocks=int(r.n_held), value_bn=r.usd_m / 1e3,
                          mean_pi_k=r[f"e_{nm}"], tilt_z=r[f"z_{nm}"]) for _, r in s.iterrows()]
    return pd.DataFrame(rows)


def pure_stocks(n=4):
    rows = []
    for y in YEARS:
        d = data(y)
        x, P = d["x"], d["Pi"]
        for k, nm in enumerate(NAMES):
            for i in np.argsort(-P[:, k])[:n]:
                rows.append(dict(year=y, profile=nm, ticker=x.ticker.iloc[i], name=x.name.iloc[i], pi_k=P[i, k],
                                 cap_bn=x.cap_bn.iloc[i], div_yield=x.div_yield.iloc[i], book_to_market=x.BM.iloc[i],
                                 owners=int(x.n_active_owners.iloc[i])))
    return pd.DataFrame(rows)


def flow_stocks(n=6):
    """For each positive flow k -> l: stocks whose dominant profile moved k -> l (count, median Jul 2007-Jun 2008
    return) and the stocks contributing most, ranked by min(loss of pi_k, gain of pi_l)."""
    o = pd.read_csv(DYN / "o3_stock_movement.csv", index_col=0)
    fl = pd.read_csv(DYN / "m13_flows.csv", index_col=0)
    flows = sorted(((fl.loc[k, l], k, l) for k in NAMES for l in NAMES if fl.loc[k, l] > 0), reverse=True)
    summ, top = [], []
    for v, k, l in flows:
        g = o[(o.dom_2006 == k) & (o.dom_2008 == l)]
        summ.append(dict(origin=k, destination=l, flow=v, n_dominant_movers=len(g),
                         median_return=g.ret_0707_0806.median()))
        s = o[(o[f"d_{k}"] < 0) & (o[f"d_{l}"] > 0)].copy()
        s["score"] = (-s[f"d_{k}"]).clip(upper=s[f"d_{l}"])
        for _, r in s.nlargest(n, "score").iterrows():
            top.append(dict(flow=f"{k} -> {l}", ticker=r.ticker, name=r["name"],
                            origin_pi=f"{r[f'pi06_{k}']:.2f} -> {r[f'pi08_{k}']:.2f}",
                            destination_pi=f"{r[f'pi06_{l}']:.2f} -> {r[f'pi08_{l}']:.2f}",
                            return_0707_0806=r.ret_0707_0806,
                            owners=f"{r.owners_2006:.0f} -> {r.owners_2008:.0f}", score=r.score))
    return pd.DataFrame(summ), pd.DataFrame(top)


def main():
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    for name, src in FIGURES.items():
        shutil.copy(src, PAPER / "figures" / name)

    res = fit()
    sh, sh_pass = split_half()
    fsum, ftop = flow_stocks()
    dyn = dynamics()
    tables = [
        ("tab:network_summary", "Network summary", "network_summary", network_summary(res), 3),
        ("fig:dcbm-eigengap", "Eigenvalues of A and eigengap ratios", "eigengap", eigengap(), 2),
        ("fig:dcmm-split-half", "Split-half r of each profile by K (median over 100 splits)", "split_half", sh, 2),
        ("fig:dcmm-split-half", "Every profile >= 0.7 (K = 4 is the largest passing at both dates)",
         "split_half_pass", sh_pass, 2),
        ("tab:naming", "Purest stocks, trait coefficients (per SD) and most-tilted owners", "naming", naming(res), 2),
        ("fig:popularity", "Spearman of membership with number of active owners", "popularity_spearman",
         popularity(res), 2),
        ("tab:degree", "R^2 on owners and split-half r", "degree", degree(), 2),
        ("tab:sector-pi", "Average membership within GICS sector (%)", "sector_membership", sectors(), 0),
        ("tab:dynamics", "Identity correlation, 2006 (rows) vs 2008 (columns)", "dynamics_identity",
         dyn["identity"], 2),
        ("tab:dynamics", "Movement M = C - S", "dynamics_movement", dyn["movement"], 2),
        ("tab:dynamics", "Flows (positive off-diagonal movement)", "dynamics_flows", dyn["flows"], 2),
        ("tab:appendix-clusters", "Same-date baseline S", "dynamics_same_date_baseline",
         dyn["same_date_baseline"], 2),
        ("tab:appendix-clusters", "Cross-date correlation C", "dynamics_cross_date", dyn["cross_date"], 2),
        ("tab:tilted-managers", "Largest managers tilted toward each community (paper shows a selection)",
         "tilted_managers", tilted_managers(), 2),
        ("tab:pure-stocks, tab:pure-stocks-2008", "Purest stocks of each community", "pure_stocks",
         pure_stocks(), 2),
        ("tab:flow-stocks", "Flows: dominant-profile movers and median return", "flow_summary", fsum, 2),
        ("tab:flow-stocks", "Main stocks behind each flow (paper shows a selection)", "flow_stocks", ftop, 2),
    ]
    md = ["# Paper results", "", "Generated by `generate-figures.py`. CSVs in `tables/`, figures in `figures/`.", ""]
    for label, title, fname, df, dig in tables:
        df.to_csv(PAPER / "tables" / f"{fname}.csv")
        md += [f"## {title}", f"`{label}` · `tables/{fname}.csv`", "", "```",
               df.round(dig).to_string(), "```", ""]
    (PAPER / "RESULTS.md").write_text("\n".join(md))
    print(f"wrote {len(FIGURES)} figures and {len(tables)} tables to {PAPER}")


if __name__ == "__main__":
    main()
