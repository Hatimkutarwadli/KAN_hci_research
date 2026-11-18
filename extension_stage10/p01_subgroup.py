"""Stage 10 / P0-1: subgroup PREDICTIVE performance (sex, age group) from saved predictions. SYNTHETIC. No retraining; test split used for final reporting only.
Metrics on the TEST split at all eligible origins: severity RMSE/MAE/bias; status and sustained-onset AP, AUROC, Brier, ECE (equal-count 10 bins), mean prediction/prevalence.
Gap = subgroup minus its complement (sex F vs M; age g vs others). Participant-level bootstrap (resample participants within subgroup and complement separately; seed 0
checkpoints of GRU and M1, B=100). Flags: UNRELIABLE if <30 test participants; LOW EVENTS if <10 participants carry a positive label (classification).
Lead time by subgroup reuses src/leadtime.py with the calibration-derived thresholds (not re-fitted per subgroup).
Usage: python extension_stage10/p01_subgroup.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import common as C  # noqa: E402

OUT = "results/stage10_subgroup"
BOOT_B, BOOT_SEED, BOOT_MODELS = 100, 20261116, ("gru", "m1")


def reg_stats(y, p):
    e = p - y
    return dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), bias=float(e.mean()))


def metric_fn(task):
    if task in C.SEV:
        return {"rmse": lambda y, p: float(np.sqrt(((p - y) ** 2).mean()))}
    return {"ap": lambda y, p: float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
            "auroc": lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
            "brier": lambda y, p: float(((p - y) ** 2).mean())}


def main():
    os.chdir(C.ROOT)
    os.makedirs(OUT, exist_ok=True)
    d = C.load_split("test")
    gm = C.group_masks(d)
    rows, gaps = [], []
    for model in C.MODELS:
        for seed in C.seeds_of(model):
            for task in C.SEV + C.CLF:
                Y = d[f"y_{task}"]
                pi, ti = C.eligible(Y)
                P, pid = C.load_pred(model, task, seed, "test")
                assert np.array_equal(pid, d["pid"])
                y, p = Y[pi, ti].astype(float), P[pi, ti]
                assert np.isfinite(p).all()
                for g, m in gm.items():
                    r = m[pi]
                    n_part = int(m.sum())
                    ev_p = int(len(np.unique(pi[r & (y == 1)]))) if task in C.CLF else None
                    st = reg_stats(y[r], p[r]) if task in C.SEV else C.clf_summary(y[r], p[r])
                    flag = ("UNRELIABLE (n<%d)" % C.MIN_N) if (n_part < C.MIN_N and g != "all") else ("LOW EVENTS" if ev_p is not None and ev_p < C.MIN_EVENT_P else "ok")
                    rows.append(dict(model=model, seed=seed, task=task, group=g, n_participants=n_part, n_rows=int(r.sum()), n_event_participants=ev_p, flag=flag, **st))
                    if g != "all":
                        for name, fn in metric_fn(task).items():
                            gaps.append(dict(model=model, seed=seed, task=task, group=g, metric=name, value=fn(y[r], p[r]), complement=fn(y[~r], p[~r])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "subgroup_metrics_by_seed.csv"), index=False)
    keep = [m for m in ["rmse", "mae", "bias", "ap", "auroc", "brier", "ece", "mean_pred", "prevalence"] if m in df.columns]
    agg = df.groupby(["model", "task", "group"]).agg(n_participants=("n_participants", "first"), n_event_participants=("n_event_participants", "first"), flag=("flag", "first"),
                                                     **{f"{m}_mean": (m, "mean") for m in keep}, **{f"{m}_sd": (m, "std") for m in ("rmse", "ap", "auroc", "brier", "ece") if m in df.columns}).reset_index()
    agg.to_csv(os.path.join(OUT, "subgroup_metrics_summary.csv"), index=False)
    gp = pd.DataFrame(gaps)
    gp["gap"] = gp.value - gp.complement
    gs = gp.groupby(["model", "task", "group", "metric"]).agg(value=("value", "mean"), complement=("complement", "mean"), gap_mean=("gap", "mean"), gap_sd=("gap", "std"),
                                                             n_seeds_same_sign=("gap", lambda x: int(max((x > 0).sum(), (x < 0).sum())))).reset_index()
    gs.to_csv(os.path.join(OUT, "subgroup_gaps_summary.csv"), index=False)

    # participant-level bootstrap (seed 0; GRU and M1)
    rng = np.random.default_rng(BOOT_SEED)
    brow = []
    for model in BOOT_MODELS:
        for task in C.SEV + C.CLF:
            Y = d[f"y_{task}"]
            pi, ti = C.eligible(Y)
            P, _ = C.load_pred(model, task, 0, "test")
            y, p = Y[pi, ti].astype(float), P[pi, ti]
            byp = {i: np.where(pi == i)[0] for i in np.unique(pi)}
            for g, m in gm.items():
                if g == "all":
                    continue
                ids_g = np.array([i for i in byp if m[i]])
                ids_c = np.array([i for i in byp if not m[i]])
                cat = lambda ids: np.concatenate([byp[i] for i in ids])  # noqa: E731
                for name, fn in metric_fn(task).items():
                    pt_g, pt_c = fn(y[cat(ids_g)], p[cat(ids_g)]), fn(y[cat(ids_c)], p[cat(ids_c)])
                    vg, vgap = [], []
                    for _ in range(BOOT_B):
                        ig, ic = cat(rng.choice(ids_g, len(ids_g))), cat(rng.choice(ids_c, len(ids_c)))
                        a, b = fn(y[ig], p[ig]), fn(y[ic], p[ic])
                        vg.append(a)
                        vgap.append(a - b)
                    lo, hi = C.boot_ci(vg)
                    glo, ghi = C.boot_ci(vgap)
                    brow.append(dict(model=model, task=task, group=g, metric=name, n_participants=len(ids_g), point=pt_g, ci95_lo=lo, ci95_hi=hi, gap_point=pt_g - pt_c,
                                     gap_ci95_lo=glo, gap_ci95_hi=ghi, gap_ci_excludes_0=bool(np.isfinite(glo) and (glo > 0 or ghi < 0))))
    pd.DataFrame(brow).to_csv(os.path.join(OUT, "subgroup_bootstrap_seed0.csv"), index=False)

    # lead time by subgroup (thresholds from the calibration split, all participants)
    import leadtime as lt
    lrows = []
    for k in lt.KS:
        cal, test = lt.load_cohort("cal", k), lt.load_cohort("test", k)
        assert np.array_equal(test["pid"], d["pid"])
        for name in ("gru", "m1"):
            dirn, seeds = lt.SRC[name]
            for seed in seeds:
                Scal, Stest = lt.load_scores(dirn, name, seed, "cal", k, cal["pid"]), lt.load_scores(dirn, name, seed, "test", k, test["pid"])
                for f in lt.TARGETS:
                    _, st = lt.run_one(Scal, Stest, cal, test, f)
                    for g, m in gm.items():
                        res = lt.stat_metrics(st, np.where(m)[0])
                        lrows.append(dict(model=name, seed=seed, target=k, target_fap=f, group=g, n_onset=res["n_onset"], n_never=res["n_never"],
                                          flag="NOT ESTIMABLE (<10 onset participants)" if res["n_onset"] < 10 else "ok", det_lead1=res["det_lead1"], det_lead7=res["det_lead7"],
                                          fap=res["fap"], precision_participant=res["precision_participant"]))
    ld = pd.DataFrame(lrows)
    ld.to_csv(os.path.join(OUT, "leadtime_by_subgroup_per_seed.csv"), index=False)
    ld.groupby(["model", "target", "target_fap", "group"]).agg(n_onset=("n_onset", "first"), n_never=("n_never", "first"), flag=("flag", "first"), det_lead1=("det_lead1", "mean"),
                                                                det_lead1_sd=("det_lead1", "std"), det_lead7=("det_lead7", "mean"), fap=("fap", "mean"),
                                                                precision_participant=("precision_participant", "mean")).reset_index().to_csv(os.path.join(OUT, "leadtime_by_subgroup_summary.csv"), index=False)
    json.dump({"SYNTHETIC": True, "split": "test (final reporting only; no selection)", "models": C.MODELS,
               "bootstrap": dict(B=BOOT_B, seed=BOOT_SEED, models=list(BOOT_MODELS), resampling="participants within subgroup and within complement separately"),
               "min_participants": C.MIN_N, "min_event_participants": C.MIN_EVENT_P, "ece": "equal-count 10 bins",
               "leadtime": "thresholds from calibration split (all participants), not refitted per subgroup", "group_sizes_test": {g: int(m.sum()) for g, m in gm.items()}},
              open(os.path.join(OUT, "config.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
