"""Stage 10 / P0-2: probability calibration and conformal PREDICTION SETS for the status and sustained-onset heads. SYNTHETIC. Saved predictions only; no retraining.

A. Calibration assessment on TEST (all eligible origins): Brier, ECE (equal-count 10 bins), calibration intercept/slope (logistic on logit p), mean prediction / prevalence.
   Post-hoc recalibration maps (Platt = logistic on logit p; isotonic) are FIT ON THE CALIBRATION SPLIT ONLY and applied to test (monotone Platt leaves AP/AUROC unchanged; isotonic may change ties).
B. Split-conformal prediction sets {0,1} for each binary head, score s = 1 - p_true (LAC), two variants fixed in advance:
     marginal  : one quantile q for both classes
     Mondrian  : class-conditional quantiles (q for y=1 and for y=0), which targets coverage of the RARE positive class
   Quantile = ceil((n+1)(1-alpha))-th smallest calibration score (+inf, i.e. class always included, if it does not exist). Levels 90% and 95%.
   Participant-level calibration as in Stage 6: each CALIBRATION participant contributes ONE (origin, label) pair, origin drawn uniformly among that participant's eligible
   origins in days 27..98 (label defined); the same draws are used for every model/seed of a head. Draw 0 = primary; draws 1..R = robustness repeats.
   Reported: coverage (overall / positives / negatives) and set size (mean size, share of empty / singleton / {0,1} sets, share of sets containing class 1) on
   (i) one origin per test participant (same draw) and (ii) ALL eligible test origins (lower variance, correlated within participant); for the primary draw and the mean over repeats.
   Calibration quantiles never see test labels (tested). Subgroup coverage (sex/age) for GRU and M1.
Usage: python extension_stage10/p02_calibration.py"""
import json
import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import common as C  # noqa: E402

OUT = "results/stage10_calibration"
ORIGIN_SEED, R_REPEATS = 20261116, 200
ALPHAS = {"90%": 0.10, "95%": 0.05}
LO, HI = 27, 98
SUB_MODELS = ("gru", "m1")


# ------------------------------------------------------------------ core (unit-tested)
def lac_scores(p, y):
    """nonconformity 1 - p_true for the binary head (p = P(y=1))."""
    return np.where(y == 1, 1.0 - p, p)


def _kth(sorted_rows, n, alpha):
    """finite-sample quantile per row: ceil((n+1)(1-alpha))-th smallest of the first n entries; +inf if absent. n may be an array."""
    k = np.ceil((n + 1) * (1 - alpha)).astype(int)
    ok = (k <= n) & (n > 0)
    idx = np.clip(k - 1, 0, sorted_rows.shape[1] - 1)
    q = np.take_along_axis(sorted_rows, idx[:, None], axis=1)[:, 0]
    return np.where(ok, q, np.inf)


def class_quantiles(P, Y, alpha, mondrian):
    """P, Y: (D, n) calibration probabilities / labels for D draws. Returns (q1, q0), each (D,). Uses calibration data only."""
    S = lac_scores(P, Y)
    D, n = S.shape
    if not mondrian:
        q = _kth(np.sort(S, axis=1), np.full(D, n), alpha)
        return q, q.copy()
    out = []
    for c in (1, 0):
        m = Y == c
        Sm = np.sort(np.where(m, S, np.inf), axis=1)
        out.append(_kth(Sm, m.sum(1), alpha))
    return out[0], out[1]


def set_membership(p, q1, q0):
    """p (..., m); q1/q0 broadcastable. Returns (in1, in0) boolean: class 1 in set iff 1-p <= q1; class 0 in set iff p <= q0."""
    return (1.0 - p) <= q1, p <= q0


def eval_sets(p, y, q1, q0):
    """p, y: (D, m) test arrays (or (m,) broadcast against (D,1) q). Returns dict of (D,) arrays."""
    i1, i0 = set_membership(p, q1[:, None], q0[:, None])
    cov = np.where(y == 1, i1, i0)
    pos, neg = y == 1, y == 0
    size = i1.astype(int) + i0.astype(int)
    f = lambda a, m: (a * m).sum(1) / np.maximum(m.sum(1), 1)  # noqa: E731
    return dict(coverage=cov.mean(1), cov_pos=np.where(pos.sum(1) > 0, f(cov, pos), np.nan), cov_neg=f(cov, neg), set_size=size.mean(1), frac_empty=(size == 0).mean(1),
                frac_single=(size == 1).mean(1), frac_both=(size == 2).mean(1), frac_contains1=i1.mean(1))


def draw_origins(Y, rng, n_draws):
    """Per participant with >=1 eligible origin in LO..HI: n_draws origins uniformly among eligible ones. Returns (participant idx (n,), T (n_draws, n))."""
    ids, T = [], []
    for i in range(Y.shape[0]):
        ok = np.where(~np.isnan(Y[i, LO:HI + 1]))[0] + LO
        if len(ok):
            ids.append(i)
            T.append(ok[rng.integers(0, len(ok), n_draws)])
    return np.array(ids), np.array(T).T


# ------------------------------------------------------------------ pipeline
def main():
    os.chdir(C.ROOT)
    os.makedirs(OUT, exist_ok=True)
    D = {s: C.load_split(s) for s in ("cal", "test")}
    gm = C.group_masks(D["test"])
    cal_rows, rel_rows, set_rows, sub_rows = [], [], [], []
    for ti_, task in enumerate(C.CLF):
        Yc, Yt = D["cal"][f"y_{task}"], D["test"][f"y_{task}"]
        rng = np.random.default_rng([ORIGIN_SEED, ti_])
        ids_c, Tc = draw_origins(Yc, rng, R_REPEATS + 1)
        ids_t, Tt = draw_origins(Yt, rng, R_REPEATS + 1)
        yc = Yc[ids_c[None, :], Tc]
        yt1 = Yt[ids_t[None, :], Tt]
        pi, ti = C.eligible(Yt)
        yt_all = Yt[pi, ti].astype(float)
        pic, tic = C.eligible(Yc)
        yc_all = Yc[pic, tic].astype(float)
        for model in C.MODELS:
            for seed in C.seeds_of(model):
                Pc, _ = C.load_pred(model, task, seed, "cal")
                Pt, _ = C.load_pred(model, task, seed, "test")
                pc_all, pt_all = Pc[pic, tic], Pt[pi, ti]
                # ---- A. calibration and recalibration (maps fit on calibration rows only)
                platt = LogisticRegression(C=1e6, max_iter=1000).fit(C.logit(pc_all)[:, None], yc_all)
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pc_all, yc_all)
                maps = {"raw": pt_all, "platt": platt.predict_proba(C.logit(pt_all)[:, None])[:, 1], "isotonic": iso.predict(pt_all)}
                for meth, pp in maps.items():
                    a, b = C.calib_slope_intercept(yt_all, pp)
                    cal_rows.append(dict(model=model, seed=seed, task=task, method=meth, intercept=a, slope=b, **C.clf_summary(yt_all, pp)))
                if model in SUB_MODELS:
                    o = np.argsort(pt_all, kind="stable")
                    for bi, idx in enumerate(np.array_split(o, 10)):
                        rel_rows.append(dict(model=model, seed=seed, task=task, bin=bi, n=len(idx), mean_pred=float(pt_all[idx].mean()), obs_rate=float(yt_all[idx].mean())))
                # ---- B. conformal sets
                pc1 = Pc[ids_c[None, :], Tc]
                pt1 = Pt[ids_t[None, :], Tt]
                for lvl, alpha in ALPHAS.items():
                    for meth, mond in (("marginal", False), ("mondrian", True)):
                        q1, q0 = class_quantiles(pc1, yc, alpha, mond)
                        e1 = eval_sets(pt1, yt1, q1, q0)                                   # one origin per test participant, same draw
                        ea = eval_sets(np.broadcast_to(pt_all, (R_REPEATS + 1, len(pt_all))), np.broadcast_to(yt_all, (R_REPEATS + 1, len(pt_all))), q1, q0)    # all eligible origins
                        row = dict(model=model, seed=seed, task=task, level=lvl, nominal=1 - alpha, method=meth, n_cal=len(ids_c), n_cal_pos_primary=int(yc[0].sum()),
                                   n_cal_pos_mean=float(yc.sum(1).mean()), n_test=len(ids_t), n_test_rows=len(pt_all), prevalence_test_rows=float(yt_all.mean()),
                                   q1_primary=float(q1[0]), q0_primary=float(q0[0]), frac_draws_q1_inf=float(np.isinf(q1).mean()), frac_draws_q0_inf=float(np.isinf(q0).mean()))
                        for k, v in e1.items():
                            row[f"one_origin_{k}_primary"], row[f"one_origin_{k}_repmean"] = float(v[0]), float(np.nanmean(v[1:]))
                        for k, v in ea.items():
                            row[f"all_origins_{k}_primary"], row[f"all_origins_{k}_repmean"] = float(v[0]), float(np.nanmean(v[1:]))
                        set_rows.append(row)
                        if model in SUB_MODELS:
                            i1, i0 = set_membership(pt_all[None, :], q1[:, None], q0[:, None])
                            cov = np.where(yt_all[None, :] == 1, i1, i0)
                            size = i1.astype(int) + i0.astype(int)
                            for g, m in gm.items():
                                r = m[pi]
                                pos = r & (yt_all == 1)
                                sub_rows.append(dict(model=model, seed=seed, task=task, level=lvl, method=meth, group=g, n_participants=int(m.sum()),
                                                     n_event_participants=int(len(np.unique(pi[pos]))), coverage=float(cov[:, r].mean()),
                                                     cov_pos=float(cov[:, pos].mean()) if pos.any() else np.nan, set_size=float(size[:, r].mean()),
                                                     flag=("UNRELIABLE (n<%d)" % C.MIN_N) if (int(m.sum()) < C.MIN_N and g != "all") else
                                                     ("LOW EVENTS" if len(np.unique(pi[pos])) < C.MIN_EVENT_P else "ok")))
        print("done", task, flush=True)
    ca, rl, st, sb = pd.DataFrame(cal_rows), pd.DataFrame(rel_rows), pd.DataFrame(set_rows), pd.DataFrame(sub_rows)
    ca.to_csv(os.path.join(OUT, "calibration_by_seed.csv"), index=False)
    ca.groupby(["model", "task", "method"]).mean(numeric_only=True).drop(columns=["seed"]).reset_index().to_csv(os.path.join(OUT, "calibration_summary.csv"), index=False)
    rl.groupby(["model", "task", "bin"]).agg(n=("n", "mean"), mean_pred=("mean_pred", "mean"), obs_rate=("obs_rate", "mean")).reset_index().to_csv(os.path.join(OUT, "reliability_bins.csv"), index=False)
    st.to_csv(os.path.join(OUT, "conformal_sets_by_seed.csv"), index=False)
    st.groupby(["model", "task", "level", "method"]).mean(numeric_only=True).drop(columns=["seed"]).reset_index().to_csv(os.path.join(OUT, "conformal_sets_summary.csv"), index=False)
    sb.to_csv(os.path.join(OUT, "conformal_sets_subgroup_by_seed.csv"), index=False)
    sb.groupby(["model", "task", "level", "method", "group"]).agg(n_participants=("n_participants", "first"), n_event_participants=("n_event_participants", "first"), flag=("flag", "first"),
                                                                 coverage=("coverage", "mean"), cov_pos=("cov_pos", "mean"), set_size=("set_size", "mean")).reset_index().to_csv(os.path.join(OUT, "conformal_sets_subgroup_summary.csv"), index=False)
    json.dump({"SYNTHETIC": True, "score": "1 - p_true (LAC)", "variants": ["marginal", "mondrian (class-conditional)"], "levels": list(ALPHAS), "quantile": "ceil((n+1)(1-alpha))-th smallest, +inf if absent",
               "calibration": "calibration split participants, one origin per participant uniform among eligible origins in %d..%d" % (LO, HI), "origin_seed": ORIGIN_SEED, "repeat_draws": R_REPEATS,
               "recalibration": "Platt and isotonic fit on calibration-split rows (all eligible origins), applied to test", "ece": "equal-count 10 bins", "heads": C.CLF,
               "models": {m: C.seeds_of(m) for m in C.MODELS}, "not_implemented": ["Venn-Abers", "conformalised risk thresholds", "time-series/adaptive conformal (P2-5)"]},
              open(os.path.join(OUT, "config.json"), "w"), indent=1)
    plots(ca, rl, st)
    print("done ->", OUT)


def plots(ca, rl, st):
    tasks = ["status_dep_t0", "status_anx_t0", "onset_sust_dep_h14", "onset_sust_anx_h14"]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5))
    for j, t in enumerate(tasks):
        for i, m in enumerate(SUB_MODELS):
            ax = axs[i, j]
            b = rl[(rl.model == m) & (rl.task == t)].groupby("bin").agg(x=("mean_pred", "mean"), y=("obs_rate", "mean"))
            mx = max(b.x.max(), b.y.max()) * 1.05
            ax.plot([0, mx], [0, mx], color="#999", lw=1)
            ax.plot(b.x, b.y, "o-", color="#2a78d6")
            ax.set_title(f"{m.upper()}: {t}", fontsize=9)
            ax.set_xlabel("mean predicted (test decile)")
            ax.set_ylabel("observed rate")
    fig.suptitle("Reliability (test, equal-count deciles, mean over seeds) - SYNTHETIC", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_reliability.png"), dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    s = st[(st.level == "90%") & (st.model == "gru")].groupby(["task", "method"]).mean(numeric_only=True).reset_index()
    for ax, (col, ttl) in zip(axs, [("all_origins_cov_pos_repmean", "coverage of POSITIVES (nominal 0.90)"), ("all_origins_set_size_repmean", "mean set size"), ("all_origins_frac_both_repmean", "share of {0,1} sets")]):
        for meth, c in (("marginal", "#eb6834"), ("mondrian", "#1baf7a")):
            x = s[s.method == meth]
            ax.barh(np.arange(len(x)) + (0.2 if meth == "mondrian" else -0.2), x[col], height=0.4, color=c, label=meth)
        ax.set_yticks(np.arange(len(x)))
        ax.set_yticklabels(x.task, fontsize=7)
        ax.set_title(ttl, fontsize=9)
    axs[0].axvline(0.9, color="k", lw=1)
    axs[0].legend(fontsize=8)
    fig.suptitle("GRU conformal prediction sets at 90% (all eligible test origins, mean over seeds and draws) - SYNTHETIC", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_set_coverage_size_gru.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
