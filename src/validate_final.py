"""Validate the FINAL frozen synthetic dataset. SYNTHETIC. Read-only on the data directory.
Usage: python src/validate_final.py data/synthetic_v1 results/final_validation.json
Re-uses validate_pilot.main (dimension/censoring/label/sanity-regression checks) and audit_pilot.lag_recovery,
then adds: both onset definitions, episodes/chronic counts, split integrity, reproducibility and leakage checks.
"""
import contextlib
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import validate_pilot as vp  # noqa: E402
from audit_pilot import episode_stats, lag_recovery  # noqa: E402
from simulate import Config, simulate  # noqa: E402

LOOKBACK, HORIZONS = 27, (7, 14, 21)


def main(d, outpath):
    D = vp.load(d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    obs, tg, tr, tp, sp, pa = (D[k] for k in ["observations", "targets", "truth", "truth_participants", "splits", "participants"])
    N, T, P = cfg.n_participants, cfg.n_days, cfg.persistence_days
    R = {"SYNTHETIC": True, "dataset_dir": d, "seed": cfg.seed, "checks": {}}
    ck = R["checks"]

    # ---- reuse base validation (asserts internally); keep its JSON as a separate file ----
    base_path = os.path.join(os.path.dirname(outpath) or ".", "final_validation_base.json")
    with contextlib.redirect_stdout(io.StringIO()):
        base = vp.main(d, base_path)
    base = json.load(open(base_path))
    R["base_validation_file"] = base_path

    # 1 dimensions
    ck["dims"] = {"unique_participants": int(obs.pid.nunique()), "days_per_participant": sorted(set(obs.groupby("pid").size())),
                  "rows": int(len(obs)), "expected_rows": N * T, "pass": bool(obs.pid.nunique() == N == 1000 and len(obs) == N * T == 120000
                                                                                  and set(obs.groupby("pid").size()) == {T})}
    # 2 missingness
    hrv_only = int(((obs.hrv_missing == 1) & (obs.missing_day == 0)).sum())
    ck["missingness"] = {"missing_day": float(obs.missing_day.mean()), "hrv_missing": float(obs.hrv_missing.mean()),
                         "hrv_only_gap_days": hrv_only, "hrv_missing_equals_hrv_isna": bool((obs.hrv_missing == obs.hrv.isna().astype(int)).all()),
                         "whole_day_gap_implies_hrv_missing": bool((obs.loc[obs.missing_day == 1, "hrv_missing"] == 1).all()),
                         "other_features_nan_only_on_gap_days": bool(all((obs[f].isna() == (obs.missing_day == 1)).all() for f in meta["features"] if f != "hrv")),
                         "by_age_group": obs.merge(pa, on="pid").groupby("age_group").missing_day.mean().round(3).to_dict(),
                         "by_sex": obs.merge(pa, on="pid").groupby("sex").missing_day.mean().round(3).to_dict()}
    ck["missingness"]["pass"] = bool(0.10 <= ck["missingness"]["missing_day"] <= 0.20 and ck["missingness"]["hrv_missing_equals_hrv_isna"]
                                     and ck["missingness"]["whole_day_gap_implies_hrv_missing"] and ck["missingness"]["other_features_nan_only_on_gap_days"])

    # 3-4 prevalence (daily vs participant-level) and episodes
    ck["prevalence_and_episodes"] = {}
    for k in ("dep", "anx"):
        st = tg.pivot(index="pid", columns="day", values=f"status_{k}").values
        e = episode_stats(st, T)
        e["participants_with_first_crossing_onset"] = int(tp[f"onset_day_{k}"].notna().sum())
        e["participants_with_sustained_onset"] = int(tp[f"onset_day_sust_{k}"].notna().sum())
        e["participants_positive_at_day0(prevalent)"] = int(tp[f"prevalent_{k}"].sum())
        e["daily_prevalence_by_sex"] = tg.merge(pa, on="pid").groupby("sex")[f"status_{k}"].mean().round(3).to_dict()
        e["daily_prevalence_by_age_group"] = tg.merge(pa, on="pid").groupby("age_group")[f"status_{k}"].mean().round(3).to_dict()
        e["pass_prevalence_5_to_10pct"] = bool(0.05 <= e["daily_prevalence"] <= 0.10)
        ck["prevalence_and_episodes"][k] = e
        d_ = tp[tp[f"onset_day_sust_{k}"].notna()]
        ck["prevalence_and_episodes"][k]["delay_first_crossing_to_sustained_onset_days_mean_median_max"] = [
            float((d_[f"onset_day_sust_{k}"] - d_[f"onset_day_{k}"]).mean()), float((d_[f"onset_day_sust_{k}"] - d_[f"onset_day_{k}"]).median()),
            float((d_[f"onset_day_sust_{k}"] - d_[f"onset_day_{k}"]).max())]

    # 5 onset counts by definition / horizon / split
    t2 = tg.merge(sp, on="pid")
    oc = {}
    for prefix, label in (("onset_sust_", "PRIMARY_sustained"), ("onset_", "SECONDARY_first_crossing")):
        for k in ("dep", "anx"):
            for h in HORIZONS:
                col = f"{prefix}{k}_h{h}"
                e = t2[(t2.day >= LOOKBACK) & t2[col].notna()]
                g = e.groupby("split")[col].agg(["sum", "count"])
                pos_p = e[e[col] == 1].groupby("split").pid.nunique()
                risk_p = e.groupby("split").pid.nunique()
                oc[f"{label}|{k}|h{h}"] = {s: {"positive_origins": int(g.loc[s, "sum"]), "eligible_origins": int(g.loc[s, "count"]),
                                               "positive_participants": int(pos_p.get(s, 0)), "at_risk_participants": int(risk_p.get(s, 0))}
                                           for s in ("train", "cal", "test")}
    ck["onset_counts_origins_ge_day27"] = oc
    min_pos = min(v[s]["positive_participants"] for key, v in oc.items() if key.startswith("PRIMARY") for s in ("cal", "test"))
    ck["onset_counts_min_primary_positive_participants_cal_or_test"] = min_pos

    # 5b censoring / eligibility rules for BOTH definitions
    rc = True
    for k in ("dep", "anx"):
        for h in HORIZONS:
            c1, c2 = f"onset_{k}_h{h}", f"onset_sust_{k}_h{h}"
            rc &= bool(tg.loc[tg.day + h > T - 1, c1].isna().all())
            rc &= bool(tg.loc[tg.day + h + P - 1 > T - 1, c2].isna().all())
            rc &= bool(tg.loc[(tg.day + h + P - 1 <= T - 1) & (tg[f"at_risk_sust_{k}"] == 1), c2].notna().all())
            rc &= bool(tg.loc[tg[f"at_risk_sust_{k}"] == 0, c2].isna().all())
            rc &= bool(tg.loc[tg[f"at_risk_{k}"] == 0, c1].isna().all())
            prev_p = tp.pid[tp[f"prevalent_{k}"] == 1]
            rc &= bool(tg.loc[tg.pid.isin(prev_p), [c1, c2]].isna().all().all())
        m = tg.merge(tp[["pid", f"onset_day_sust_{k}", f"onset_day_{k}"]], on="pid")
        for h in HORIZONS:
            sub = m[m[f"onset_sust_{k}_h{h}"].notna()]
            rc &= bool((((sub[f"onset_day_sust_{k}"] > sub.day) & (sub[f"onset_day_sust_{k}"] <= sub.day + h)).astype(float).values
                        == sub[f"onset_sust_{k}_h{h}"].values).all())
        rc &= bool((tp[f"onset_day_sust_{k}"].dropna() >= tp.loc[tp[f"onset_day_sust_{k}"].notna(), f"onset_day_{k}"]).all())
        # sustained onset really is 3 consecutive positive latent days; check from truth
        z = tr.pivot(index="pid", columns="day", values=f"z_{k}").values >= getattr(cfg, f"z_thr_{k}")
        for pid, dsu in tp[f"onset_day_sust_{k}"].dropna().items():
            rc &= bool(z[pid, int(dsu):int(dsu) + P].all() and not z[pid, 0])
    ck["right_censoring_and_eligibility_rules_pass"] = rc
    assert rc

    # 6 split integrity
    s_cnt = sp.split.value_counts().to_dict()
    ci = pa.merge(sp, on="pid")
    ck["splits"] = {"counts": s_cnt, "each_pid_in_exactly_one_split": bool(sp.pid.is_unique and set(sp.pid) == set(range(N))),
                    "sizes_expected_600_200_200": bool(s_cnt == {"train": 600, "cal": 200, "test": 200}),
                    "sex_by_split": pd.crosstab(ci.split, ci.sex).to_dict("index"), "age_group_by_split": pd.crosstab(ci.split, ci.age_group).to_dict("index"),
                    "daily_prevalence_by_split": {k: tg.merge(sp, on="pid").groupby("split")[f"status_{k}"].mean().round(4).to_dict() for k in ("dep", "anx")}}
    ck["splits"]["pass"] = bool(ck["splits"]["each_pid_in_exactly_one_split"] and ck["splits"]["sizes_expected_600_200_200"]
                                and set(obs.pid) == set(sp.pid))

    # 7 causal relationships: base sanity regressions + lag recovery on the true latent
    ck["causal_sanity_true_latent"] = base["sanity_true_latent_regression"]
    ck["causal_sanity_observed"] = base["sanity_observed_data_regression"]
    lag = lag_recovery(tr)
    ck["planted_lag_recovery"] = lag
    peak = {"sleep_variability->dep": ("dep", "g_sleep_variability", 3), "steps->dep": ("dep", "g_steps", 2),
            "steps->anx": ("anx", "g_steps", 2), "hrv->anx": ("anx", "g_hrv", 1)}
    peaks = {}
    for name, (o, pre, L) in peak.items():
        coefs = {int(c.split("_L")[1]): v["coef"] for c, v in lag[o].items() if c.startswith(pre + "_L")}
        peaks[name] = {"planted_lag": L, "argmax_abs_lag": max(coefs, key=lambda x: abs(coefs[x])), "coef_at_planted_lag": coefs[L]}
    ck["planted_lag_peaks"] = peaks
    ck["planted_lag_recovery_pass"] = all(v["planted_lag"] == v["argmax_abs_lag"] for v in peaks.values())

    # 8 leakage / input-column checks
    inputs = set(obs.columns) - {"pid", "day"}
    declared = {c for c in meta["model_input_columns"] if "(static)" not in c}
    forbidden = (set(tr.columns) | set(tg.columns) | set(tp.columns) | {"stress_proxy"}) - {"pid", "day"}
    analysis_only_present = sorted(set(meta["analysis_only_columns"]) & inputs)
    ck["leakage"] = {"observation_columns": sorted(inputs), "declared_model_input_columns": sorted(declared),
                     "declared_inputs_overlap_forbidden(truth/targets/stress_proxy)": sorted(declared & forbidden),
                     "stress_proxy_in_observations_file_but_not_declared_input": analysis_only_present == ["stress_proxy"] and "stress_proxy" not in declared,
                     "declared_inputs_subset_of_observation_columns": bool(declared <= inputs),
                     "static_inputs": ["sex", "age_group"]}
    # reproducibility + future-perturbation test on the actual final config (in memory only)
    a = simulate(cfg)
    repro = True
    for name, df in a.items():
        saved = D[name]
        repro &= bool(np.allclose(df.select_dtypes("number").values, saved[df.select_dtypes("number").columns].values, equal_nan=True, rtol=1e-9, atol=1e-9))
    b = simulate(cfg, perturb_after=60)
    oa, ob_ = a["observations"], b["observations"]
    ta, tb = a["targets"], b["targets"]
    early_same = bool(np.allclose(oa[oa.day < 60].values, ob_[ob_.day < 60].values, equal_nan=True)
                      and np.allclose(ta.loc[ta.day < 60, ["severity_dep", "severity_anx", "status_dep", "status_anx"]].values,
                                      tb.loc[tb.day < 60, ["severity_dep", "severity_anx", "status_dep", "status_anx"]].values))
    late_diff = bool(not np.allclose(a["truth"].loc[a["truth"].day > 65, "z_dep"], b["truth"].loc[b["truth"].day > 65, "z_dep"]))
    ck["leakage"].update({"regenerated_in_memory_matches_saved_files(reproducible)": repro,
                          "future_perturbation_inputs_and_severity_status_before_day60_identical": early_same,
                          "future_perturbation_changes_later_latent": late_diff})
    ck["leakage"]["pass"] = bool(not ck["leakage"]["declared_inputs_overlap_forbidden(truth/targets/stress_proxy)"] and ck["leakage"]["declared_inputs_subset_of_observation_columns"]
                                 and ck["leakage"]["stress_proxy_in_observations_file_but_not_declared_input"] and repro and early_same and late_diff)

    # checksum integrity
    chk = {}
    if os.path.exists(os.path.join(d, "checksums.sha256")):
        for line in open(os.path.join(d, "checksums.sha256")):
            h, fn = line.split()
            chk[fn] = hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest() == h
    ck["checksums_match"] = chk

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    json.dump(R, open(outpath, "w"), indent=2, default=str)
    flags = {"dims": ck["dims"]["pass"], "missingness": ck["missingness"]["pass"], "splits": ck["splits"]["pass"],
             "censoring_eligibility": rc, "lag_recovery": ck["planted_lag_recovery_pass"], "leakage": ck["leakage"]["pass"],
             "prevalence_dep": ck["prevalence_and_episodes"]["dep"]["pass_prevalence_5_to_10pct"],
             "prevalence_anx": ck["prevalence_and_episodes"]["anx"]["pass_prevalence_5_to_10pct"],
             "checksums": all(chk.values()) if chk else None}
    print(json.dumps(flags, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
