"""Stage 10 / P0-3: seed-ensemble uncertainty for severity intervals (heteroscedastic split conformal). SYNTHETIC. The 5 saved checkpoints' predictions are the ensemble; nothing is retrained.
Families with 5 seeds: GRU, KAN, KAN-reg, M1, M1-static (GBT has one run: excluded). Tasks: 8 severity tasks (nowcast + 7/14/21 d forecasts).
Ensemble mean mu = mean over the 5 seeds, disagreement sigma = std over seeds (ddof=1). Methods (fixed in advance):
  single : Stage 6 method on ONE seed (|y - yhat|, constant width), averaged over the 5 seeds (reproduced from predictions; checked against Stage 6 CSV in tests)
  ens_const : constant width around mu (|y - mu|)
  ens_norm  : normalised score |y - mu| / (sigma + eps), interval mu +- q (sigma + eps); eps = EPS_MULT x median(sigma over calibration-split eligible origins). Primary EPS_MULT = 1;
              0.25 and 4 are reported as a labelled sensitivity (not used to choose anything).
Calibration protocol identical to Stage 6: 200 calibration participants, one origin each (uniform 27..98; the SAME draws as Stage 6: primary seed 20261114 + 1000 repeats seed 7),
quantile ceil((n+1)(1-alpha))-th smallest, levels 90% / 95%. Test used only for reporting.
Reported: coverage (primary / mean of 1000 draws), mean width (unclipped and clipped to 0..27), Winkler interval score (lower better; combines width and misses),
coverage by sigma tertile (cut points from calibration), Spearman(sigma, |residual|), RMSE of ensemble mean vs mean single-seed RMSE, coverage by sex/age group.
Usage: python extension_stage10/p03_ensemble.py"""
import json
import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import common as C  # noqa: E402
import conformal as cf  # noqa: E402  (Stage 6, read-only import for constants and the draw rule)

OUT = "results/stage10_ensemble"
FAMS = ["gru", "kan", "kan_reg", "m1", "m1_static"]
ALPHAS = {"90%": 0.10, "95%": 0.05}
EPS_MULTS = {"primary": 1.0, "sens_0.25": 0.25, "sens_4": 4.0}
N_ALL_DRAWS = 200            # draws used for all-origin and subgroup evaluation
LOCAL = "tertile"


# ------------------------------------------------------------------ core (unit-tested)
def ensemble(P):
    """P (n_seeds, ...) -> (mean, std ddof=1)."""
    return P.mean(0), P.std(0, ddof=1)


def quantile_rows(S, alpha):
    """S (D, n) nonconformity scores -> (D,) finite-sample split-conformal quantile ceil((n+1)(1-alpha))-th smallest (+inf if absent)."""
    n = S.shape[1]
    k = math.ceil((n + 1) * (1 - alpha))
    return np.sort(S, axis=1)[:, k - 1] if k <= n else np.full(S.shape[0], np.inf)


def winkler(y, lo, hi, alpha):
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def one_method(mu_c, sd_c, y_c, mu_t, sd_t, y_t, alpha, scale):
    """scale: function sd -> positive scale (ones for constant width). mu/sd/y shapes (D, n). Returns q (D,), lo, hi (D, n_t)."""
    q = quantile_rows(np.abs(y_c - mu_c) / scale(sd_c), alpha)
    w = q[:, None] * scale(sd_t)
    return q, mu_t - w, mu_t + w


# ------------------------------------------------------------------ pipeline
def main():
    os.chdir(C.ROOT)
    os.makedirs(OUT, exist_ok=True)
    D = {s: C.load_split(s) for s in ("cal", "test")}
    n = {s: len(D[s]["pid"]) for s in D}
    gm = C.group_masks(D["test"])
    rng = np.random.default_rng(cf.ORIGIN_SEED)
    t_cal, t_test = cf.draw_origins(n["cal"], rng), cf.draw_origins(n["test"], rng)
    rrng = np.random.default_rng(cf.REPEAT_SEED)
    tr_cal, tr_test = rrng.integers(cf.ORIG_LO, cf.ORIG_HI + 1, (cf.N_REPEATS, n["cal"])), rrng.integers(cf.ORIG_LO, cf.ORIG_HI + 1, (cf.N_REPEATS, n["test"]))
    Tc, Tt = np.vstack([t_cal[None], tr_cal]), np.vstack([t_test[None], tr_test])          # row 0 = primary draw, rows 1.. = repeats
    ac, at = np.arange(n["cal"]), np.arange(n["test"])
    rows, grows, trows = [], [], []
    for fam in FAMS:
        for task in cf.SEV_TASKS:
            Pc = np.stack([C.load_pred(fam, task, s, "cal")[0] for s in range(5)])      # (5, N, 120)
            Pt = np.stack([C.load_pred(fam, task, s, "test")[0] for s in range(5)])
            Yc, Yt = D["cal"][f"y_{task}"].astype(float), D["test"][f"y_{task}"].astype(float)
            mu_c, sd_c = ensemble(Pc)
            mu_t, sd_t = ensemble(Pt)
            pic, tic = C.eligible(Yc)
            pit, tit = C.eligible(Yt)
            eps_base = float(np.median(sd_c[pic, tic]))                                   # calibration-only epsilon base
            cuts = np.quantile(sd_c[pic, tic], [1 / 3, 2 / 3])                           # calibration-only tertile cut points
            # origins
            yc, yt = Yc[ac[None, :], Tc], Yt[at[None, :], Tt]
            assert np.isfinite(yc).all() and np.isfinite(yt).all()
            mc, sc_ = mu_c[ac[None, :], Tc], sd_c[ac[None, :], Tc]
            mt, st_ = mu_t[at[None, :], Tt], sd_t[at[None, :], Tt]
            # all-origin test rows
            ya, ma, sa = Yt[pit, tit], mu_t[pit, tit], sd_t[pit, tit]
            rho = float(spearmanr(sa, np.abs(ya - ma)).statistic)
            bins = np.digitize(sa, cuts)
            rmse_ens = float(np.sqrt(np.mean((ya - ma) ** 2)))
            rmse_single = float(np.mean([np.sqrt(np.mean((ya - Pt[s][pit, tit]) ** 2)) for s in range(5)]))
            trows.append(dict(family=fam, task=task, spearman_sigma_abs_resid=rho, rmse_ensemble_mean=rmse_ens, rmse_single_seed_mean=rmse_single, eps_base=eps_base, sigma_median_test=float(np.median(sa))))
            for lvl, alpha in ALPHAS.items():
                meths = [("single", None, None)] + [("ens_const", None, None)] + [("ens_norm", k, m) for k, m in EPS_MULTS.items()]
                for meth, tag, mult in meths:
                    if meth == "single":
                        sub_runs = [(Pc[s], Pt[s]) for s in range(5)]
                    else:
                        sub_runs = [(None, None)]
                    acc = []
                    for pcs, pts in sub_runs:
                        if meth == "single":
                            mcs, mts = pcs[ac[None, :], Tc], pts[at[None, :], Tt]
                            scale, sdc, sdt, sda = (lambda s_: np.ones_like(s_)), np.zeros_like(mcs), np.zeros_like(mts), np.zeros_like(ma)
                            m_c, m_t, m_a = mcs, mts, pts[pit, tit]
                        else:
                            m_c, m_t, m_a = mc, mt, ma
                            if meth == "ens_const":
                                scale, sdc, sdt, sda = (lambda s_: np.ones_like(s_)), sc_, st_, sa
                            else:
                                eps = mult * eps_base
                                scale, sdc, sdt, sda = (lambda s_, e=eps: s_ + e), sc_, st_, sa
                        q, lo, hi = one_method(m_c, sdc, yc, m_t, sdt, yt, alpha, scale)
                        cov = (yt >= lo) & (yt <= hi)
                        wid, widc = hi - lo, np.clip(hi, 0, 27) - np.clip(lo, 0, 27)
                        wk = winkler(yt, lo, hi, alpha)
                        # all-origin (first N_ALL_DRAWS draws): q from each draw applied to every test origin
                        Sa = np.abs(ya - m_a)[None, :] / scale(sda)[None, :]
                        cov_a = Sa <= q[:N_ALL_DRAWS, None]
                        wid_a = 2 * q[:N_ALL_DRAWS, None] * scale(sda)[None, :]
                        d = dict(cov_primary=float(cov[0].mean()), cov_rep=float(cov[1:].mean()), cov_rep_p2_5=float(np.percentile(cov[1:].mean(1), 2.5)), cov_rep_p97_5=float(np.percentile(cov[1:].mean(1), 97.5)),
                                 width_primary=float(wid[0].mean()), width_rep=float(wid[1:].mean()), width_clipped_rep=float(widc[1:].mean()), winkler_primary=float(wk[0].mean()), winkler_rep=float(wk[1:].mean()),
                                 all_cov=float(cov_a.mean()), all_width=float(wid_a.mean()),
                                 **{f"all_cov_sigma_tertile{b + 1}": float(cov_a[:, bins == b].mean()) for b in range(3)},
                                 **{f"all_width_sigma_tertile{b + 1}": float(wid_a[:, bins == b].mean()) for b in range(3)})
                        acc.append(d)
                        if meth != "single" and tag in ("primary", None):
                            for g, m in gm.items():
                                r = m[pit]
                                grows.append(dict(family=fam, task=task, level=lvl, method=meth if tag is None else "ens_norm", group=g, n_participants=int(m.sum()),
                                                  coverage=float(cov_a[:, r].mean()), width=float(wid_a[:, r].mean()),
                                                  flag=("UNRELIABLE (n<%d)" % C.MIN_N) if (int(m.sum()) < C.MIN_N and g != "all") else "ok"))
                    r = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
                    rows.append(dict(family=fam, task=task, level=lvl, nominal=1 - alpha, method=meth, eps_setting=tag or "", **r))
        print("done", fam, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ensemble_conformal_by_task.csv"), index=False)
    key = ["family", "level", "method", "eps_setting"]
    num = [c for c in df.columns if c not in ("task", "nominal") + tuple(key)]
    df.groupby(key)[num].mean().reset_index().to_csv(os.path.join(OUT, "ensemble_conformal_summary.csv"), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(OUT, "ensemble_diagnostics_by_task.csv"), index=False)
    gdf = pd.DataFrame(grows)
    gdf.to_csv(os.path.join(OUT, "ensemble_subgroup_by_task.csv"), index=False)
    gdf.groupby(["family", "level", "method", "group"]).agg(n_participants=("n_participants", "first"), flag=("flag", "first"), coverage=("coverage", "mean"), width=("width", "mean")).reset_index().to_csv(os.path.join(OUT, "ensemble_subgroup_summary.csv"), index=False)
    json.dump({"SYNTHETIC": True, "families": FAMS, "tasks": cf.SEV_TASKS, "levels": list(ALPHAS), "eps_rule": "EPS_MULT x median(sigma over calibration eligible origins); primary 1.0; sensitivity 0.25, 4",
               "tertile_cuts": "calibration sigma terciles", "draws": "identical to Stage 6 (origin seed %d, repeat seed %d, %d repeats)" % (cf.ORIGIN_SEED, cf.REPEAT_SEED, cf.N_REPEATS),
               "all_origin_draws": N_ALL_DRAWS, "ensemble": "5 saved seeds; sigma = std ddof=1", "gbt": "excluded (single deterministic run)"}, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    plots(df, pd.DataFrame(trows))
    print("done ->", OUT)


def plots(df, tr):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    s = df[(df.level == "90%") & (df.eps_setting.isin(["", "primary"]))].groupby(["family", "method"]).mean(numeric_only=True).reset_index()
    cols = {"single": "#eb6834", "ens_const": "#2a78d6", "ens_norm": "#1baf7a"}
    for meth, c in cols.items():
        x = s[s.method == meth]
        axs[0].scatter(x.width_rep, x.cov_rep, color=c, label=meth)
        for _, r in x.iterrows():
            axs[0].annotate(r.family, (r.width_rep, r.cov_rep), fontsize=6)
    axs[0].axhline(0.9, color="k", lw=0.8)
    axs[0].set_xlabel("mean width (severity points)")
    axs[0].set_ylabel("coverage (mean of 1000 draws)")
    axs[0].legend(fontsize=8)
    axs[0].set_title("90%: coverage vs width", fontsize=9)
    for meth, c in cols.items():
        x = df[(df.level == "90%") & (df.method == meth) & (df.eps_setting.isin(["", "primary"]))].mean(numeric_only=True)
        axs[1].plot([1, 2, 3], [x[f"all_cov_sigma_tertile{b}"] for b in (1, 2, 3)], "o-", color=c, label=meth)
    axs[1].axhline(0.9, color="k", lw=0.8)
    axs[1].set_xticks([1, 2, 3])
    axs[1].set_xticklabels(["low sigma", "mid", "high sigma"])
    axs[1].set_ylabel("coverage (all test origins)")
    axs[1].set_title("90%: conditional coverage by ensemble-disagreement tertile", fontsize=9)
    fig.suptitle("Seed-ensemble conformal intervals, severity (SYNTHETIC; mean over families and tasks)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_ensemble_conformal.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
