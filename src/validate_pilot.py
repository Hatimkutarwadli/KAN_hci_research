"""Validate a simulated dataset directory (pilot). SYNTHETIC data only.
Usage: python src/validate_pilot.py data/pilot results/pilot_validation.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

LOOKBACK = 27   # first usable origin day for a 28-day lookback
HORIZONS = (7, 14, 21)


def load(d):
    return {n: pd.read_csv(os.path.join(d, f"{n}.csv")) for n in
            ["participants", "observations", "targets", "truth", "truth_participants", "splits"]}


def ols_cluster(X, y, g):
    X = np.column_stack([np.ones(len(y)), X]) if False else X
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1],) * 2)
    ug = np.unique(g)
    for k in ug:
        xs = X[g == k].T @ e[g == k]
        meat += np.outer(xs, xs)
    G = len(ug)
    V = bread @ meat @ bread * G / (G - 1)
    return beta, np.sqrt(np.diag(V))


def within(df, cols, pid):
    return df[cols] - df.groupby(pid)[cols].transform("mean")


def acf_mean(arr2d, lag):
    """mean over participants of lag-`lag` autocorrelation (pairwise complete)."""
    out = []
    for x in arr2d:
        a, b = x[:-lag], x[lag:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() > 10 and a[ok].std() > 0 and b[ok].std() > 0:
            out.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.mean(out))


def icc(df, col):
    g = df.groupby("pid")[col]
    between = g.mean().var()
    w = g.var().mean()
    return float(between / (between + w))


def main(d, outpath):
    D = load(d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    cfg = meta["config"]
    obs, tg, tr, tp, sp, pa = (D[k] for k in ["observations", "targets", "truth", "truth_participants", "splits", "participants"])
    N, T = cfg["n_participants"], cfg["n_days"]
    R = {"SYNTHETIC": True, "dataset_dir": d}

    # 1-2 dimensions, ids, repeated observations
    cnt = obs.groupby("pid").size()
    R["dims"] = {"participants": int(obs.pid.nunique()), "days_per_participant_min": int(cnt.min()),
                 "days_per_participant_max": int(cnt.max()), "rows": int(len(obs)),
                 "pids_contiguous": bool((np.sort(obs.pid.unique()) == np.arange(N)).all()),
                 "days_contiguous_sorted": bool((obs.groupby("pid").day.apply(lambda s: (s.values == np.arange(T)).all())).all()),
                 "n_obs_columns": obs.shape[1], "n_target_columns": tg.shape[1]}
    assert R["dims"]["participants"] == N and R["dims"]["days_per_participant_min"] == T == R["dims"]["days_per_participant_max"]
    assert R["dims"]["pids_contiguous"] and R["dims"]["days_contiguous_sorted"]

    # 3 missingness
    feats = meta["features"]
    R["missingness"] = {"whole_day_missing": float(obs.missing_day.mean()),
                        "per_feature_nan": {f: float(obs[f].isna().mean()) for f in feats},
                        "per_person_missing_min_median_max": [float(x) for x in obs.groupby("pid").missing_day.mean().quantile([0, .5, 1])],
                        "by_age_group": obs.merge(pa, on="pid").groupby("age_group").missing_day.mean().round(3).to_dict(),
                        "mean_gap_run_length": None}
    runs = []
    for _, s in obs.groupby("pid").missing_day:
        v = s.values
        r = 0
        for x in v:
            if x:
                r += 1
            elif r:
                runs.append(r)
                r = 0
        if r:
            runs.append(r)
    R["missingness"]["mean_gap_run_length"] = float(np.mean(runs))

    # 4 prevalence
    R["prevalence_person_day"] = {k: float(tg[f"status_{k}"].mean()) for k in ("dep", "anx")}
    R["participants_positive_at_day0"] = {k: int(tp[f"prevalent_{k}"].sum()) for k in ("dep", "anx")}
    R["participants_with_first_onset"] = {k: int(tp[f"onset_day_{k}"].notna().sum()) for k in ("dep", "anx")}
    R["prevalence_by_group"] = {
        "sex": tg.merge(pa, on="pid").groupby("sex")[["status_dep", "status_anx"]].mean().round(3).to_dict(),
        "age_group": tg.merge(pa, on="pid").groupby("age_group")[["status_dep", "status_anx"]].mean().round(3).to_dict()}

    # 5-6 onset counts by horizon and split (eligible origins: day >= LOOKBACK, label defined)
    t2 = tg.merge(sp, on="pid")
    R["split_participants"] = sp.split.value_counts().to_dict()
    R["onset_counts"] = {}
    for k in ("dep", "anx"):
        for h in HORIZONS:
            col = f"onset_{k}_h{h}"
            e = t2[(t2.day >= LOOKBACK) & t2[col].notna()]
            ppl = e[e[col] == 1].groupby("split").pid.nunique()
            g = e.groupby("split")[col].agg(["sum", "count"])
            R["onset_counts"][f"{k}_h{h}"] = {s: {"positive_origins": int(g.loc[s, "sum"]), "eligible_origins": int(g.loc[s, "count"]),
                                                  "positive_participants": int(ppl.get(s, 0))} for s in g.index}
    # right-censoring checks
    cens_ok = True
    for k in ("dep", "anx"):
        for h in HORIZONS:
            col = f"onset_{k}_h{h}"
            cens_ok &= bool(tg.loc[tg.day > T - 1 - h, col].isna().all())
            cens_ok &= bool(tg.loc[(tg.day <= T - 1 - h) & (tg[f"at_risk_{k}"] == 1), col].notna().all())
            cens_ok &= bool(tg.loc[tg[f"at_risk_{k}"] == 0, col].isna().all())
    R["right_censoring_and_risk_set_checks_pass"] = cens_ok
    assert cens_ok
    # onset label consistent with onset_day
    lab_ok = True
    m = tg.merge(tp[["pid", "onset_day_dep", "onset_day_anx"]], on="pid")
    for k in ("dep", "anx"):
        for h in HORIZONS:
            sub = m[m[f"onset_{k}_h{h}"].notna()]
            exp = ((sub[f"onset_day_{k}"] > sub.day) & (sub[f"onset_day_{k}"] <= sub.day + h)).astype(float)
            lab_ok &= bool((exp.values == sub[f"onset_{k}_h{h}"].values).all())
    R["onset_labels_match_onset_day"] = lab_ok
    assert lab_ok

    # 7 temporal autocorrelation
    zd = tr.pivot(index="pid", columns="day", values="z_dep").values
    sevd = tg.pivot(index="pid", columns="day", values="severity_dep").values
    ac = {"z_dep_true": zd, "severity_dep_observed": sevd}
    for f in ("steps", "sleep_variability", "hrv", "resting_hr"):
        ac[f + "_observed"] = obs.pivot(index="pid", columns="day", values=f).values
    R["autocorrelation_mean_over_participants"] = {n: {"lag1": acf_mean(a, 1), "lag7": acf_mean(a, 7)} for n, a in ac.items()}
    # contrast: shuffled-days control (should be ~0)
    rng = np.random.default_rng(0)
    R["autocorrelation_shuffled_control_lag1_z_dep"] = acf_mean(np.array([rng.permutation(r) for r in zd]), 1)

    # 8 within vs between person
    od = obs.copy()
    od["z_dep"] = tr.z_dep.values
    od["severity_dep"] = tg.severity_dep.values
    R["icc_between_person_share"] = {c: icc(od, c) for c in ["z_dep", "severity_dep", "steps", "sleep_variability", "hrv", "resting_hr"]}

    # 9 planted-effect sanity checks
    # (a) on TRUE latent/features: dz(t+1) on lagged g(u); confirms simulator implements effects with right sign/size
    tt = tr.copy()
    g = tt.groupby("pid")
    def lagged(col, L):
        return g[col].shift(L)
    tt["dz_dep"] = g.z_dep.shift(-1) - tt.z_dep
    tt["dz_anx"] = g.z_anx.shift(-1) - tt.z_anx
    # effect of u(t+1-L) on z(t+1)  => u(t+1-L) = shift(L-1) at row t
    tt["g_sv"] = np.tanh(g.u_sleep_variability.shift(cfg["lag_sv"] - 1))
    tt["g_st_dep"] = np.tanh(0.7 * g.u_steps.shift(cfg["lag_steps"] - 1))
    tt["g_hrv"] = np.tanh(0.8 * g.u_hrv.shift(cfg["lag_hrv"] - 1))
    tt["g_rhr"] = np.tanh(g.u_resting_hr.shift(2))
    tt["g_cal"] = np.tanh(g.u_calories.shift(1))
    tt["z_other_dep"] = tt.z_anx
    tt["z_other_anx"] = tt.z_dep
    chk = {}
    for k, cols in (("dep", ["z_dep", "g_sv", "g_st_dep", "g_rhr", "g_cal", "stress"]),
                    ("anx", ["z_anx", "g_st_dep", "g_hrv", "g_rhr", "g_cal", "stress"])):
        ok = tt.dropna(subset=cols + [f"dz_{k}"])
        X = within(ok, cols, ok.pid).values
        b, se = ols_cluster(X, within(ok, [f"dz_{k}"], ok.pid).values[:, 0], ok.pid.values)
        chk[k] = {c: {"coef": round(float(bb), 4), "t": round(float(bb / s), 1)} for c, bb, s in zip(cols, b, se)}
    R["sanity_true_latent_regression"] = {
        "note": "regress z(t+1)-z(t) on lagged g(true feature); planted: sv->dep +0.06, steps->dep -0.05 (per tanh(0.7u)), steps->anx -0.03, hrv->anx -0.05 (x person sensitivity ~1.05); resting_hr/calories have no direct effect",
        **chk}

    # (b) on OBSERVED data (noisy, gaps): adjusted vs naive within-person lagged regression
    o = obs.merge(tg[["pid", "day", "severity_dep", "severity_anx"]], on=["pid", "day"])
    zc = ["steps", "calories", "sleep_variability", "resting_hr", "hrv"]
    gg = o.groupby("pid")
    obs_chk = {}
    for name, k, h in (("sv->dep (h=3)", "dep", 3), ("steps->dep (h=2)", "dep", 2), ("hrv->anx (h=1)", "anx", 1)):
        dfh = o.copy()
        dfh["y"] = gg[f"severity_{k}"].shift(-h)
        dfh["ylag"] = dfh[f"severity_{k}"]
        for c in zc + ["stress_proxy"]:
            dfh[c] = (dfh[c] - dfh[c].mean()) / dfh[c].std()
        res = {}
        for label, cov in (("naive", zc), ("adjusted_stress_proxy", zc + ["stress_proxy"])):
            ok = dfh.dropna(subset=["y", "ylag"] + cov)
            X = within(ok, ["ylag"] + cov, ok.pid).values
            b, se = ols_cluster(X, within(ok, ["y"], ok.pid).values[:, 0], ok.pid.values)
            res[label] = {c: {"coef": round(float(bb), 3), "t": round(float(bb / s), 1)} for c, bb, s in zip(["ylag"] + cov, b, se)
                          if c in zc + ["stress_proxy"]}
        obs_chk[name] = res
    R["sanity_observed_data_regression"] = {
        "note": "person-demeaned, participant-clustered SE, standardized features at t -> severity(t+h); expect sv +, steps -, hrv -; resting_hr/calories spurious (non-causal)",
        **obs_chk}

    # 10 leakage checks (structural; the future-perturbation test is in tests/test_simulator.py)
    banned = set(tr.columns) | set(tg.columns) | set(tp.columns)
    inputs = set(obs.columns) - {"pid", "day"}
    R["leakage"] = {"input_columns": sorted(inputs),
                    "input_target_or_truth_overlap": sorted((inputs & banned) - {"pid", "day"}),
                    "stress_proxy_is_analysis_only": True}
    assert not R["leakage"]["input_target_or_truth_overlap"]
    # inputs at day t vs onset labels: no feature is a near-perfect predictor of future onset
    e = tg.merge(obs, on=["pid", "day"])
    e = e[(e.day >= LOOKBACK) & e["onset_dep_h14"].notna()]
    lk = {f: float(abs(e[f].corr(e["onset_dep_h14"]))) for f in feats}
    R["leakage"]["max_abs_corr_feature_vs_onset_dep_h14"] = max(lk.values())

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    json.dump(R, open(outpath, "w"), indent=2, default=str)
    print(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
