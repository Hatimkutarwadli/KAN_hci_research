"""Stage 6: split conformal prediction intervals for severity + subgroup reliability (SYNTHETIC data).

Reuses saved predictions (nothing is retrained). Procedure, fixed in advance:
  * Nonconformity score: absolute residual s_i = |y_i - yhat_i| (symmetric intervals yhat +- q).
  * Calibration set = the existing CALIBRATION participants (200); test set = TEST participants (200, final evaluation only).
  * Participant-level calibration: each participant contributes exactly ONE (origin, target) pair, so scores are not
    correlated daily observations. The origin is drawn uniformly from days 27..98 (28-day lookback satisfied and
    t+21 <= 119, so every horizon is eligible at every origin) with a fixed seed (ORIGIN_SEED), identical for every model/task.
  * q = the ceil((n+1)(1-alpha))-th smallest calibration score (finite-sample split-conformal quantile).
  * Primary result = that single pre-specified origin draw. Robustness = N_REPEATS further random origin draws (same rule).
Nominal levels 90% and 95%. No formal coverage guarantee over time is claimed (see INTERPRETATION.md).
Usage: python src/conformal.py results/stage6_conformal
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import data as dd

ORIGIN_SEED = 20261114
ORIG_LO, ORIG_HI = 27, 98
N_REPEATS = 1000
REPEAT_SEED = 7
ALPHAS = {"90%": 0.10, "95%": 0.05}
SEV_TASKS = [f"sev_{k}_{g}" for k in ("dep", "anx") for g in ("t0", "h7", "h14", "h21")]
MIN_RELIABLE_N = 30          # subgroups with fewer test participants are flagged UNRELIABLE
SRC = {"gbt": ("results/stage3", "results/stage3c_forecast"), "gru": ("results/stage3", "results/stage3c_forecast"),
       "kan": ("results/stage3", "results/stage3c_forecast"), "kan_reg": ("results/stage3b_kan_reg/final", "results/stage3c_forecast"),
       "m1": ("results/stage4_m1", "results/stage4_m1"), "m1_static": ("results/stage4_ablation", "results/stage4_ablation")}
SEEDS = {"gbt": [0]}


# ------------------------------------------------------------------ core (unit-tested)
def conformal_q(scores, alpha):
    """finite-sample split-conformal quantile: ceil((n+1)(1-alpha))-th smallest score; +inf if it does not exist."""
    s = np.sort(np.asarray(scores, float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def make_interval(pred, q):
    return pred - q, pred + q


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def draw_origins(n, rng):
    """one origin per participant, uniform on 27..98 inclusive"""
    return rng.integers(ORIG_LO, ORIG_HI + 1, size=n)


def calibrate_and_evaluate(pred_cal, y_cal, pred_test, y_test, alpha):
    """Quantile from calibration pairs ONLY; applied to test predictions. Returns q, coverage, mean width, lo, hi."""
    q = conformal_q(np.abs(y_cal - pred_cal), alpha)
    lo, hi = make_interval(pred_test, q)
    return q, coverage(y_test, lo, hi), float(np.mean(hi - lo)), lo, hi


def pred_file(model, task, seed, split):
    d = SRC[model][0 if task.endswith("_t0") else 1]
    return os.path.join(d, "predictions", f"{model}_seed{seed}_{split}.npz")


# ------------------------------------------------------------------ pipeline
def main(out):
    os.makedirs(out, exist_ok=True)
    lab = {s: dd.load_split(s) for s in ("cal", "test")}
    pa = pd.read_csv("data/synthetic_v1/participants.csv").set_index("pid")
    pid = {s: lab[s]["pid"] for s in lab}
    assert not set(pid["cal"]) & set(pid["test"])
    n = {s: len(pid[s]) for s in pid}
    sex = {s: pa.loc[pid[s], "sex"].values for s in pid}
    age = {s: pa.loc[pid[s], "age_group"].values for s in pid}
    groups = lambda s: [("all", "all", np.ones(n[s], bool))] + [("sex", g, sex[s] == g) for g in ("F", "M")] + [("age_group", str(g), age[s] == g) for g in (0, 1, 2)]
    rng = np.random.default_rng(ORIGIN_SEED)
    t_cal, t_test = draw_origins(n["cal"], rng), draw_origins(n["test"], rng)
    rrng = np.random.default_rng(REPEAT_SEED)
    tr_cal, tr_test = rrng.integers(ORIG_LO, ORIG_HI + 1, (N_REPEATS, n["cal"])), rrng.integers(ORIG_LO, ORIG_HI + 1, (N_REPEATS, n["test"]))
    ar = {s: np.arange(n[s]) for s in n}
    overall, sub, ints = [], [], []
    for model in SRC:
        for seed in SEEDS.get(model, [0, 1, 2, 3, 4]):
            cache = {}
            for task in SEV_TASKS:
                path = {s: pred_file(model, task, seed, s) for s in ("cal", "test")}
                P = {}
                for s in path:
                    if path[s] not in cache:
                        z = np.load(path[s])
                        assert (z["pid"] == pid[s]).all()
                        cache[path[s]] = (z["pred"].astype(np.float64), list(z["heads"]))   # float64: no float32 rounding at interval edges
                    arr, heads = cache[path[s]]
                    P[s] = arr[:, :, heads.index(task)]
                Y = {s: lab[s][f"y_{task}"].astype(np.float64) for s in lab}
                yc, pc = Y["cal"][ar["cal"], t_cal], P["cal"][ar["cal"], t_cal]
                yt, pt = Y["test"][ar["test"], t_test], P["test"][ar["test"], t_test]
                assert np.isfinite(yc).all() and np.isfinite(yt).all()
                # repeated random origin draws (robustness)
                yc_r, pc_r = Y["cal"][ar["cal"][None, :], tr_cal], P["cal"][ar["cal"][None, :], tr_cal]
                yt_r, pt_r = Y["test"][ar["test"][None, :], tr_test], P["test"][ar["test"][None, :], tr_test]
                sc_r = np.sort(np.abs(yc_r - pc_r), axis=1)
                elig = np.isfinite(Y["test"])
                for lvl, alpha in ALPHAS.items():
                    q, cov, width, lo, hi = calibrate_and_evaluate(pc, yc, pt, yt, alpha)
                    k = math.ceil((n["cal"] + 1) * (1 - alpha))
                    q_r = sc_r[:, k - 1]
                    cover_r = np.abs(yt_r - pt_r) <= q_r[:, None]              # (repeats, test participants)
                    cov_r = cover_r.mean(1)
                    covered = (yt >= lo) & (yt <= hi)
                    overall.append(dict(model=model, seed=seed, task=task, level=lvl, nominal=1 - alpha, n_cal=n["cal"], n_test=n["test"], q=q, width=2 * q,
                                        width_clipped_0_27=float(np.mean(np.clip(hi, 0, 27) - np.clip(lo, 0, 27))), coverage_primary=cov,
                                        coverage_repeat_mean=float(cov_r.mean()), coverage_repeat_p2_5=float(np.percentile(cov_r, 2.5)), coverage_repeat_p97_5=float(np.percentile(cov_r, 97.5)),
                                        width_repeat_mean=float(2 * q_r.mean()), rmse_primary_test=float(np.sqrt(np.mean((yt - pt) ** 2))),
                                        rmse_all_origins_test=float(np.sqrt(np.mean((Y["test"][elig] - P["test"][elig]) ** 2)))))
                    for gt, gname, m in groups("test"):
                        ng = int(m.sum())
                        kk = int(covered[m].sum())
                        wl, wh = wilson(kk, ng)
                        sub.append(dict(model=model, seed=seed, task=task, level=lvl, group_type=gt, group=gname, n_test=ng,
                                        n_cal=int(dict((a_ + b_, c_.sum()) for a_, b_, c_ in groups("cal"))[gt + gname]), coverage_primary=kk / ng, wilson_lo=wl, wilson_hi=wh,
                                        coverage_repeat_mean=float(cover_r[:, m].mean()), diff_vs_overall_primary=kk / ng - cov, diff_vs_overall_repeat=float(cover_r[:, m].mean() - cov_r.mean()),
                                        width=2 * q, rmse_primary=float(np.sqrt(np.mean((yt[m] - pt[m]) ** 2))),
                                        rmse_all_origins=float(np.sqrt(np.mean((Y["test"][m][elig[m]] - P["test"][m][elig[m]]) ** 2))),
                                        reliability_flag="ok" if ng >= MIN_RELIABLE_N else f"UNRELIABLE (n={ng}<{MIN_RELIABLE_N})"))
                    for s_, ids, yy, pp, tt, lo_, hi_ in (("test", pid["test"], yt, pt, t_test, lo, hi), ("cal", pid["cal"], yc, pc, t_cal, pc - q, pc + q)):
                        ints.append(pd.DataFrame(dict(model=model, seed=seed, task=task, level=lvl, split=s_, pid=ids, origin_day=tt, y=yy, pred=pp, q=q, lo=lo_, hi=hi_,
                                                      covered=(yy >= lo_) & (yy <= hi_), sex=sex[s_], age_group=age[s_],
                                                      note="in-sample (used to set q)" if s_ == "cal" else "out-of-sample test")))
            print("done", model, seed, flush=True)
    ov, sb = pd.DataFrame(overall), pd.DataFrame(sub)
    pd.concat(ints).to_csv(os.path.join(out, "intervals_primary_draw.csv"), index=False)
    ov.to_csv(os.path.join(out, "coverage_width_by_seed.csv"), index=False)
    sb.to_csv(os.path.join(out, "subgroup_by_seed.csv"), index=False)
    agg = ov.groupby(["model", "task", "level"]).agg(n_seeds=("seed", "nunique"), coverage_primary_mean=("coverage_primary", "mean"), coverage_primary_sd=("coverage_primary", "std"),
                                                     coverage_repeat_mean=("coverage_repeat_mean", "mean"), coverage_repeat_p2_5=("coverage_repeat_p2_5", "mean"),
                                                     coverage_repeat_p97_5=("coverage_repeat_p97_5", "mean"), width_mean=("width", "mean"), width_sd=("width", "std"),
                                                     width_clipped_mean=("width_clipped_0_27", "mean"), rmse_all_origins=("rmse_all_origins_test", "mean")).reset_index()
    agg.to_csv(os.path.join(out, "coverage_width_summary.csv"), index=False)
    sagg = sb.groupby(["model", "task", "level", "group_type", "group"]).agg(n_test=("n_test", "first"), n_cal=("n_cal", "first"), coverage_primary_mean=("coverage_primary", "mean"),
                                                                             wilson_lo=("wilson_lo", "mean"), wilson_hi=("wilson_hi", "mean"), coverage_repeat_mean=("coverage_repeat_mean", "mean"),
                                                                             diff_vs_overall_primary=("diff_vs_overall_primary", "mean"), diff_vs_overall_repeat=("diff_vs_overall_repeat", "mean"),
                                                                             width=("width", "mean"), rmse_all_origins=("rmse_all_origins", "mean"), reliability_flag=("reliability_flag", "first")).reset_index()
    sagg.to_csv(os.path.join(out, "subgroup_summary.csv"), index=False)
    json.dump({"nonconformity_score": "absolute residual |y - yhat|", "interval": "yhat +- q (symmetric, constant width 2q)",
               "quantile": "ceil((n+1)(1-alpha))-th smallest calibration score", "levels": list(ALPHAS), "calibration_participants": n["cal"], "test_participants": n["test"],
               "origin_rule": f"one origin per participant, uniform on days {ORIG_LO}..{ORIG_HI}, seed {ORIGIN_SEED} (identical for all models/tasks)",
               "robustness": f"{N_REPEATS} further origin draws, seed {REPEAT_SEED}", "min_reliable_subgroup_n": MIN_RELIABLE_N,
               "tasks": SEV_TASKS, "models": {m: SEEDS.get(m, [0, 1, 2, 3, 4]) for m in SRC}}, open(os.path.join(out, "config.json"), "w"), indent=1)
    make_plots(out, ov, sb)
    print("done ->", out)


# ------------------------------------------------------------------ plots
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
COLS = {"gru": ("#2a78d6", "o", "GRU"), "gbt": ("#eb6834", "s", "GBT"), "m1": ("#1baf7a", "^", "M1 (proposed)")}


def _ax(ax, title):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6)


def make_plots(out, ov, sb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prim = ov.groupby(["model", "task", "level"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURF, sharey=True)
    for ax, lvl, nom in zip(axes, ("90%", "95%"), (0.90, 0.95)):
        _ax(ax, f"Empirical test coverage, nominal {lvl}")
        se = math.sqrt(nom * (1 - nom) / 200)
        ax.axvspan(nom - 1.96 * se, nom + 1.96 * se, color="#e5e4e0", alpha=0.6, lw=0)
        ax.axvline(nom, color=INK, lw=1.2)
        for i, (m, (c, mk, lab)) in enumerate(COLS.items()):
            s = prim[(prim.model == m) & (prim.level == lvl)].set_index("task").loc[SEV_TASKS]
            ax.errorbar(s.coverage_primary, np.arange(len(SEV_TASKS)) + (i - 1) * 0.2, xerr=[s.coverage_primary - s.coverage_repeat_p2_5.clip(upper=s.coverage_primary),
                        s.coverage_repeat_p97_5.clip(lower=s.coverage_primary) - s.coverage_primary], fmt=mk, color=c, ms=5, lw=0.8, capsize=2, label=lab)
        ax.set_yticks(range(len(SEV_TASKS)))
        ax.set_yticklabels(SEV_TASKS, fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlabel("test coverage (marker: pre-specified origin draw;"+chr(10)+"bars: 2.5-97.5% over 1000 random origin draws)", fontsize=7.5, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Split conformal intervals: coverage vs nominal (grey band = +-1.96 binomial SE for 200 participants). SYNTHETIC data.", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(os.path.join(out, "fig1_coverage_overall.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.4), facecolor=SURF)
    _ax(ax, "Interval width at 95% (severity points; scale 0-27)")
    for i, (m, (c, mk, lab)) in enumerate(COLS.items()):
        s = prim[(prim.model == m) & (prim.level == "95%")].set_index("task").loc[SEV_TASKS]
        ax.plot(s.width, np.arange(len(SEV_TASKS)) + (i - 1) * 0.2, mk, color=c, ms=6, label=lab)
    ax.set_yticks(range(len(SEV_TASKS)))
    ax.set_yticklabels(SEV_TASKS, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("average width = 2q", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_interval_width.png"), dpi=130)
    plt.close(fig)

    sg = sb[(sb.level == "95%")].groupby(["model", "task", "group_type", "group"]).mean(numeric_only=True).reset_index()
    order = [("sex", "F"), ("sex", "M"), ("age_group", "0"), ("age_group", "1"), ("age_group", "2")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), facecolor=SURF, sharey=True)
    for ax, task in zip(axes, ("sev_dep_t0", "sev_anx_t0", "sev_dep_h14", "sev_anx_h14")):
        _ax(ax, f"{task} (95%)")
        ax.axvline(0.95, color=INK, lw=1.2)
        for i, m in enumerate(("gru", "m1")):
            c, mk, lab = COLS[m]
            s = sg[(sg.model == m) & (sg.task == task)].set_index(["group_type", "group"]).loc[order]
            ax.errorbar(s.coverage_primary, np.arange(5) + (i - 0.5) * 0.25, xerr=[s.coverage_primary - s.wilson_lo, s.wilson_hi - s.coverage_primary], fmt=mk, color=c, ms=5, lw=0.9, capsize=2, label=lab)
        n_t = sg[(sg.model == "gru") & (sg.task == task)].set_index(["group_type", "group"]).loc[order].n_test
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{a.replace('_', ' ')} {b} (n={int(n_t.loc[(a, b)])}){' UNRELIABLE' if n_t.loc[(a, b)] < MIN_RELIABLE_N else ''}" for a, b in order], fontsize=7.5, color=INK)
        ax.set_xlim(0.55, 1.02)
        ax.set_xlabel("test coverage (bars = Wilson 95% CI)", fontsize=8, color=MUTED)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Subgroup coverage at nominal 95% (mean over seeds). Small subgroups are flagged and their estimates are unreliable. SYNTHETIC data.", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(os.path.join(out, "fig3_subgroup_coverage.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
