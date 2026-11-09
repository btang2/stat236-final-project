"""Overall movement 2006 -> 2008, measured against a noise floor.

Every cross-date number is paired with the same number computed between two bootstrap refits at ONE date
(d1_bootstrap). Those two refits differ only by estimation noise, so the gap between the cross-date value and
the same-date value is the movement the second snapshot actually shows.

Tables (out/tables/):
  o1_profile_summary.csv  per profile: mean membership 2006 vs 2008 (common stocks) with bootstrap CI;
                          share dominant; stay rate observed vs noise-only stay rate.
  o2_transition.csv       dominant-profile transition counts: observed, expected from noise alone, excess.
  o3_stock_movement.csv   per stock: TV observed, noise median / 95th pct, p-value, per-profile change + CI.
  o4_by_purity.csv        movement by 2006 purity band (pure >= 0.8, 0.5-0.8, mixed < 0.5).
  o5_universe.csv         change in each profile's total weight split into movement of continuing stocks vs
                          index exits and entries.
"""
import numpy as np
import pandas as pd

from membership.core import bh
from membership.dynamics.common import K, NAMES, TAB, noise_pairs, setup, tv


def stay_rate(P, Q):
    """Per profile: among stocks dominant in k under P, share still dominant in k under Q."""
    ap, aq = P.argmax(-1), Q.argmax(-1)
    return np.array([(aq[ap == k] == k).mean() if (ap == k).any() else np.nan for k in range(K)])


def transition(P, Q):
    return pd.crosstab(pd.Categorical(np.array(NAMES)[P.argmax(1)], NAMES),
                       pd.Categorical(np.array(NAMES)[Q.argmax(1)], NAMES), dropna=False).values


def main():
    s = setup()
    P06, P08, B06, B08, x = s["P06"], s["P08"], s["B06"], s["B08"], s["x"]
    n = len(x)
    # same-date noise pairs (both dates pooled)
    na06, nb06 = noise_pairs(B06)
    na08, nb08 = noise_pairs(B08)
    tv_noise = np.concatenate([tv(na06, nb06), tv(na08, nb08)])          # pairs x stocks
    tv_obs = tv(P06, P08)

    # --- o1: profile summary ------------------------------------------------------------------
    shift_b = B08.mean(1) - B06.mean(1)                                    # paired bootstrap draws
    stay_obs = stay_rate(P06, P08)
    stay_noise = np.nanmean([stay_rate(p, q) for p, q in zip(np.concatenate([na06, na08]),
                                                             np.concatenate([nb06, nb08]))], 0)
    stay_boot = np.array([stay_rate(p, q) for p, q in zip(B06, B08)])
    o1 = pd.DataFrame({
        "profile": NAMES,
        "mean_pi_2006": P06.mean(0), "mean_pi_2008": P08.mean(0),
        "shift": P08.mean(0) - P06.mean(0),
        "shift_lo": np.quantile(shift_b, 0.025, 0), "shift_hi": np.quantile(shift_b, 0.975, 0),
        "n_dominant_2006": np.bincount(P06.argmax(1), minlength=K),
        "n_dominant_2008": np.bincount(P08.argmax(1), minlength=K),
        "n_pure_2006": (P06 >= 0.8).sum(0), "n_pure_2008": (P08 >= 0.8).sum(0),
        "stay_observed": stay_obs, "stay_noise_only": stay_noise,
        "stay_boot_lo": np.nanquantile(stay_boot, 0.025, 0), "stay_boot_hi": np.nanquantile(stay_boot, 0.975, 0),
    })
    o1["excess_leaving"] = o1.stay_noise_only - o1.stay_observed
    o1.to_csv(TAB / "o1_profile_summary.csv", index=False)

    # --- o2: transitions vs noise ------------------------------------------------------------
    obs = transition(P06, P08)
    noise_all = np.array([transition(p, q) for p, q in zip(np.concatenate([na06, na08]),
                                                           np.concatenate([nb06, nb08]))])
    noise = noise_all.mean(0)
    rows = []
    for i, f in enumerate(NAMES):
        for j, t in enumerate(NAMES):
            ge = (noise_all[:, i, j] >= obs[i, j]).sum() if i != j else (noise_all[:, i, j] <= obs[i, j]).sum()
            rows.append(dict(from_2006=f, to_2008=t, observed=obs[i, j], noise_expected=noise[i, j],
                             noise_q95=np.quantile(noise_all[:, i, j], 0.95), excess=obs[i, j] - noise[i, j],
                             p_vs_noise=(1 + ge) / (1 + len(noise_all))))
    pd.DataFrame(rows).to_csv(TAB / "o2_transition.csv", index=False)

    # --- o3: per-stock movement ---------------------------------------------------------------
    p_tv = (1 + (tv_noise >= tv_obs[None, :]).sum(0)) / (1 + tv_noise.shape[0])
    d_b = B08 - B06
    o3 = x[["ticker", "name", "sector"]].copy()
    o3["tv"], o3["tv_noise_med"] = tv_obs, np.median(tv_noise, 0)
    o3["tv_noise_q95"], o3["p_tv"] = np.quantile(tv_noise, 0.95, 0), p_tv
    o3["q_tv"] = bh(p_tv)
    o3["moved"] = o3.q_tv < 0.10
    for k, nm in enumerate(NAMES):
        o3[f"pi06_{nm}"], o3[f"pi08_{nm}"] = P06[:, k], P08[:, k]
        o3[f"d_{nm}"] = P08[:, k] - P06[:, k]
        o3[f"d_lo_{nm}"] = np.quantile(d_b[:, :, k], 0.025, 0)
        o3[f"d_hi_{nm}"] = np.quantile(d_b[:, :, k], 0.975, 0)
    o3["dom_2006"], o3["dom_2008"] = np.array(NAMES)[P06.argmax(1)], np.array(NAMES)[P08.argmax(1)]
    o3["max_pi_2006"] = P06.max(1)
    o3["ret_0707_0806"] = x.ret_0707_0806.values
    o3["owners_2006"], o3["owners_2008"] = x.owners_2006.values, x.owners_2008.values
    o3.to_csv(TAB / "o3_stock_movement.csv")

    # --- o4: by purity band -------------------------------------------------------------------
    band = pd.cut(P06.max(1), [0, 0.5, 0.8, 1.01], labels=["mixed <0.5", "leaning 0.5-0.8", "pure >=0.8"],
                  right=False)
    nb_all, nn_all = np.concatenate([na06, na08]), np.concatenate([nb06, nb08])
    agree_noise = (nb_all.argmax(-1) == nn_all.argmax(-1)).mean(0)        # per stock
    o4 = pd.DataFrame({"band": band, "tv": tv_obs, "tv_noise": tv_noise.mean(0), "moved": o3.moved.values,
                       "stay": P06.argmax(1) == P08.argmax(1), "stay_noise": agree_noise}).groupby(
        "band", observed=True).agg(n=("tv", "size"), tv=("tv", "mean"), tv_noise=("tv_noise", "mean"),
                                   share_moved=("moved", "mean"), stay=("stay", "mean"),
                                   stay_noise=("stay_noise", "mean"))
    o4["tv_excess"] = o4.tv - o4.tv_noise
    o4.to_csv(TAB / "o4_by_purity.csv")

    # --- o5: universe turnover ----------------------------------------------------------------
    a, b = s["a"], s["b"]
    out_ = ~np.isin(a["rows"], s["common"])
    in_ = ~np.isin(b["rows"], s["common"])
    tot06, tot08 = a["Pi"].sum(0), b["Pi"].sum(0)
    o5 = pd.DataFrame({"profile": NAMES, "weight_2006": tot06, "weight_2008": tot08,
                       "from_movement": P08.sum(0) - P06.sum(0),
                       "from_exits": -a["Pi"][out_].sum(0), "from_entries": b["Pi"][in_].sum(0),
                       "n_exit_dominant": np.bincount(a["Pi"][out_].argmax(1), minlength=K),
                       "n_entry_dominant": np.bincount(b["Pi"][in_].argmax(1), minlength=K)})
    o5.to_csv(TAB / "o5_universe.csv", index=False)
    ex = a["x"][out_].assign(dom=np.array(NAMES)[a["Pi"][out_].argmax(1)], pimax=a["Pi"][out_].max(1))
    en = b["x"][in_].assign(dom=np.array(NAMES)[b["Pi"][in_].argmax(1)], pimax=b["Pi"][in_].max(1))
    ex[["ticker", "name", "dom", "pimax"]].to_csv(TAB / "o5_exits.csv")
    en[["ticker", "name", "dom", "pimax"]].to_csv(TAB / "o5_entries.csv")

    # --- print --------------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"common stocks {n}; exits {out_.sum()}, entries {in_.sum()}")
    print(f"mean TV observed {tv_obs.mean():.3f} vs same-date noise {tv_noise.mean():.3f}; "
          f"stocks moved beyond noise (BH q<0.10): {o3.moved.sum()} ({o3.moved.mean():.0%})")
    print(f"dominant kept: observed {(P06.argmax(1) == P08.argmax(1)).mean():.2f} vs noise {agree_noise.mean():.2f}")
    print(o1.round(3).to_string())
    print(pd.DataFrame(obs, NAMES, NAMES))
    print(pd.DataFrame(noise, NAMES, NAMES).round(1))
    print(pd.read_csv(TAB / "o2_transition.csv").query("p_vs_noise < 0.05").round(3).to_string())
    print("stocks with p_tv < 0.05:", (o3.p_tv < 0.05).sum(), "expected by chance", round(0.05 * n, 1))
    print(o4.round(3))
    print(o5.round(1).to_string())


if __name__ == "__main__":
    main()
