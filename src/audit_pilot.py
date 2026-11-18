"""Targeted audits of a simulated dataset dir (read-only on the data). SYNTHETIC.
Usage: python src/audit_pilot.py data/pilot results/pilot_audit.json
Covers: prevalence reporting, positive-episode durations (+ candidate-adjustment evidence),
causal-implementation audit (lag recovery, ablations, correlation vs cause).
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from simulate import Config, simulate  # noqa: E402
from validate_pilot import ols_cluster, within, load  # noqa: E402


def runs(v):
    out, r, start = [], 0, 0
    for i, x in enumerate(v):
        if x:
            if r == 0:
                start = i
            r += 1
        elif r:
            out.append((start, r)); r = 0
    if r:
        out.append((start, r))
    return out


def episode_stats(status, D):
    """status: N x D array of 0/1."""
    N = status.shape[0]
    frac = status.mean(axis=1)
    lens, lens_complete, n_ep = [], [], []
    for v in status:
        rr = runs(v)
        n_ep.append(len(rr))
        for s, l in rr:
            lens.append(l)
            if s > 0 and s + l < D:  # neither left- nor right-censored
                lens_complete.append(l)
    q = lambda a: [float(x) for x in np.percentile(a, [10, 25, 50, 75, 90, 100])] if len(a) else []
    pos_days = status.sum()
    return {
        "daily_prevalence": float(status.mean()),
        "participants_with_any_positive_day": float((status.any(axis=1)).mean()),
        "participants_positive_day0": float(status[:, 0].mean()),
        "n_episodes_total": len(lens), "episodes_per_participant_among_ever_positive": float(np.mean([n for n in n_ep if n > 0])),
        "episode_length_pctiles_10_25_50_75_90_max_ALL": q(lens),
        "episode_length_pctiles_10_25_50_75_90_max_UNCENSORED": q(lens_complete), "n_uncensored_episodes": len(lens_complete),
        "participants_positive_ge_50pct_of_days": int((frac >= .5).sum()),
        "participants_positive_ge_90pct_of_days": int((frac >= .9).sum()),
        "participants_positive_ge_95pct_of_days": int((frac >= .95).sum()),
        "share_of_all_positive_days_from_participants_pos_ge_50pct": float(status[frac >= .5].sum() / max(pos_days, 1)),
        "positive_days_in_participants_with_ge_2_episodes_share": float(status[np.array(n_ep) >= 2].sum() / max(pos_days, 1)),
    }


def thresholds_for(cfg, q=0.925):
    o = simulate(cfg)
    return float(o["truth"].z_dep.quantile(q)), float(o["truth"].z_anx.quantile(q))


def candidate(cfg_kwargs, N=200, seed=2026):
    c0 = Config(n_participants=N, seed=seed, **cfg_kwargs)
    td, ta = thresholds_for(c0)
    c = replace(c0, z_thr_dep=td, z_thr_anx=ta)
    o = simulate(c)
    D = c.n_days
    res = {"thr_dep": round(td, 2), "thr_anx": round(ta, 2)}
    for k in ("dep", "anx"):
        st = o["targets"].pivot(index="pid", columns="day", values=f"status_{k}").values
        e = episode_stats(st, D)
        sp = o["splits"].set_index("pid").split
        tp = o["truth_participants"]
        has = tp[f"onset_day_{k}"].notna()
        res[k] = {"prevalence": round(e["daily_prevalence"], 3),
                  "ever_positive": round(e["participants_with_any_positive_day"], 3),
                  "pos_ge_50pct": e["participants_positive_ge_50pct_of_days"],
                  "pos_ge_90pct": e["participants_positive_ge_90pct_of_days"],
                  "median_len_uncensored": e["episode_length_pctiles_10_25_50_75_90_max_UNCENSORED"][2] if e["n_uncensored_episodes"] else None,
                  "p90_len_uncensored": e["episode_length_pctiles_10_25_50_75_90_max_UNCENSORED"][4] if e["n_uncensored_episodes"] else None,
                  "n_uncensored_episodes": e["n_uncensored_episodes"],
                  "participants_with_first_onset": int(has.sum()),
                  "onset_participants_by_split": {s: int((has & (sp == s)).sum()) for s in ("train", "cal", "test")}}
    return res


def lag_recovery(tr):
    t = tr.copy()
    gb = t.groupby("pid")
    t["dz_dep"] = gb.z_dep.shift(-1) - t.z_dep
    t["dz_anx"] = gb.z_anx.shift(-1) - t.z_anx
    t["zo_dep"] = t.z_anx - 0   # other-state deviation proxy (level; person demeaning removes m)
    t["zo_anx"] = t.z_dep
    lag = {}
    for outcome, feats, own in (("dep", [("sleep_variability", 1.0), ("steps", 0.7)], "z_dep"),
                                ("anx", [("steps", 0.7), ("hrv", 0.8)], "z_anx")):
        cols = []
        for f, sl in feats:
            for L in range(1, 6):   # effect of feature day (t+1-L) on z(t+1)
                c = f"g_{f}_L{L}"
                t[c] = np.tanh(sl * gb[f"u_{f}"].shift(L - 1))
                cols.append(c)
        ctrl = [own, "stress", f"zo_{outcome}"]
        use = cols + ctrl
        ok = t.dropna(subset=use + [f"dz_{outcome}"])
        X = within(ok, use, ok.pid).values
        b, se = ols_cluster(X, within(ok, [f"dz_{outcome}"], ok.pid).values[:, 0], ok.pid.values)
        lag[outcome] = {c: {"coef": round(float(bb), 4), "t": round(float(bb / s), 1)} for c, bb, s in zip(use, b, se) if c in cols}
    return lag


def main(d, outpath):
    D = load(d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    tg, tr, tp, sp, obs = D["targets"], D["truth"], D["truth_participants"], D["splits"], D["observations"]
    N, T = cfg.n_participants, cfg.n_days
    R = {"SYNTHETIC": True, "dataset_dir": d}

    # ---- 2/3: prevalence + episodes on the existing pilot ----
    R["pilot_prevalence_and_episodes"] = {}
    for k in ("dep", "anx"):
        st = tg.pivot(index="pid", columns="day", values=f"status_{k}").values
        R["pilot_prevalence_and_episodes"][k] = episode_stats(st, T)
    # why: participant-level mean latent state vs threshold, and what drives it
    zdep = tr.pivot(index="pid", columns="day", values="z_dep").values
    ev = {}
    for k, thr in (("dep", cfg.z_thr_dep), ("anx", cfg.z_thr_anx)):
        z = tr.pivot(index="pid", columns="day", values=f"z_{k}").values
        st = (z >= thr)
        frac = st.mean(axis=1)
        pm = z.mean(axis=1)
        wsd = z.std(axis=1)
        persistent = frac >= 0.5
        mcol = tp[f"m_{k}"].values
        # decomposition of the between-person variance of mean latent state
        ev[k] = {
            "sd_of_person_mean_z": float(pm.std()), "sd_of_person_baseline_m": float(mcol.std()),
            "mean_within_person_sd_z": float(wsd.mean()), "threshold": thr,
            "corr_person_mean_z_vs_fraction_positive": float(np.corrcoef(pm, frac)[0, 1]),
            "persistent_participants(>=50% positive)": {
                "n": int(persistent.sum()), "mean_of_person_mean_z": float(pm[persistent].mean()) if persistent.any() else None,
                "mean_baseline_m": float(mcol[persistent].mean()) if persistent.any() else None,
                "all_have_person_mean_z_above_threshold": bool((pm[persistent] >= thr).all()) if persistent.any() else None},
            "participants_whose_mean_z_exceeds_threshold": int((pm >= thr).sum()),
            "R2_person_mean_z_on_baseline_m_alone": float(np.corrcoef(pm, mcol)[0, 1] ** 2)}
    # contribution of planted features to the person-level equilibrium (dep)
    g = tr.groupby("pid")
    sv = np.tanh(tr.u_sleep_variability).groupby(tr.pid).mean().values
    st_ = np.tanh(0.7 * tr.u_steps).groupby(tr.pid).mean().values
    stress = g.stress.mean().values
    eq_effects = (cfg.eff_sv_dep * tp.sens_sv.values * sv + cfg.eff_steps_dep * tp.sens_steps.values * st_
                  + cfg.eff_stress_direct * stress) / cfg.a
    ev["dep_equilibrium_shift_from_effects"] = {
        "sd_of_(effects/a)": float(eq_effects.std()), "mean": float(eq_effects.mean()),
        "corr_with_baseline_m": float(np.corrcoef(eq_effects, tp.m_dep.values)[0, 1]),
        "sd_of_m": float(tp.m_dep.std()),
        "note": "person mean z ~= m + effects/a + drift episodes; both m (sd 0.5) and feature-baseline-driven effects/a vary between people, so some people sit above the threshold on average"}
    R["why_persistent_positive_participants"] = ev
    # candidate adjustment evidence (in-memory only; nothing saved; same seed, thresholds re-tuned to 7.5% prevalence)
    R["candidate_adjustments_N200_same_seed"] = {
        "A_current(m_sd=.5,base_sd=.6)": candidate({}),
        "B(m_sd=.35,base_sd=.4)": candidate({"m_sd": 0.35, "base_sd": 0.4}),
        "C(m_sd=.35,base_sd=.4,a=.12)": candidate({"m_sd": 0.35, "base_sd": 0.4, "a": 0.12})}
    R["candidate_adjustments_N1000_seed99"] = {
        "A_current": candidate({}, N=1000, seed=99),
        "B(m_sd=.35,base_sd=.4)": candidate({"m_sd": 0.35, "base_sd": 0.4}, N=1000, seed=99)}

    # ---- 4: causal audit (empirical) ----
    lag = lag_recovery(tr)
    R["lag_recovery_all_lags_1to5_jointly"] = {
        "note": "planted lags: sleep_variability->dep L=3; steps->dep and anx L=2; hrv->anx L=1. Coef at other lags should be ~0.", **lag}

    # ablations: same seed/config, switch one mechanism off, see which latent series change
    base = simulate(cfg)["truth"]
    abl = {}
    for name, kw in (("sleep_variability->dep OFF", {"eff_sv_dep": 0.0}), ("steps->dep OFF", {"eff_steps_dep": 0.0}),
                     ("steps->anx OFF", {"eff_steps_anx": 0.0}), ("hrv->anx OFF", {"eff_hrv_anx": 0.0}),
                     ("stress direct path OFF", {"eff_stress_direct": 0.0}), ("dep<->anx coupling OFF", {"coupling": 0.0}),
                     ("reverse state->steps OFF", {"rev_steps": 0.0}), ("reverse state->hrv OFF", {"rev_hrv": 0.0})):
        o2 = simulate(replace(cfg, **kw))["truth"]
        abl[name] = {"mean_abs_change_z_dep": round(float((o2.z_dep - base.z_dep).abs().mean()), 3),
                     "mean_abs_change_z_anx": round(float((o2.z_anx - base.z_anx).abs().mean()), 3),
                     "mean_signed_change_z_dep": round(float((o2.z_dep - base.z_dep).mean()), 3),
                     "mean_signed_change_z_anx": round(float((o2.z_anx - base.z_anx).mean()), 3)}
    R["ablation_same_seed"] = abl

    # correlation vs cause: within-person corr of true feature(t) with z(t+L), raw and given true stress
    out = {}
    tt = t.copy()
    for f, L, outc in (("sleep_variability", 3, "dep"), ("steps", 2, "dep"), ("hrv", 1, "anx"),
                       ("resting_hr", 3, "dep"), ("calories", 2, "dep"), ("sleep_duration", 3, "dep"),
                       ("circadian_regularity", 3, "dep")):
        tt["y"] = tt.groupby("pid")[f"z_{outc}"].shift(-L)
        ok = tt.dropna(subset=["y", f"u_{f}", "stress"])
        w = within(ok, ["y", f"u_{f}", "stress"], ok.pid)
        raw = np.corrcoef(w.y, w[f"u_{f}"])[0, 1]
        # partial out true stress (and for calories also true steps)
        ctrl = [w.stress.values] if f != "calories" else [w.stress.values, within(ok.assign(s_=ok.u_steps), ["s_"], ok.pid).s_.values]
        Z = np.column_stack(ctrl)
        ry = w.y.values - Z @ np.linalg.lstsq(Z, w.y.values, rcond=None)[0]
        rx = w[f"u_{f}"].values - Z @ np.linalg.lstsq(Z, w[f"u_{f}"].values, rcond=None)[0]
        out[f"{f} -> z_{outc}(t+{L})"] = {"raw_within_corr": round(float(raw), 3),
                                          "partial_corr_given_true_stress(+steps for calories)": round(float(np.corrcoef(ry, rx)[0, 1]), 3)}
    R["correlation_vs_cause_true_values"] = out

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    json.dump(R, open(outpath, "w"), indent=2, default=str)
    print(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
