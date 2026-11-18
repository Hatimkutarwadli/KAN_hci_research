"""P3 Phase 0 feasibility (SYNTHETIC). NO model training, NO test split, NO file outside results/p3_small_data/phase0 is written.
Counterfactual simulations run IN MEMORY only (nothing saved as a dataset).
  python extension_p3/phase0_feasibility.py
Parts
  A  scale mapping observed x <-> latent u; truth curves for the two response-curve interventions ('set' window to v, 'shift' window by delta)
  B  detectability: held-out (cal) R2 gain of a distributed-lag spline over a distributed-lag linear term (true u / observed x; target z / severity)
  C  estimand check: fitted associational partial-dependence curves vs truth; planted share via CRN counterfactuals (planted effect off; reverse+confounding off)
  D  severity clipping
  E  RQ1: subset scaler vs frozen 600-participant scaler (exactness + size of the difference), stratified nested subsets, onset counts"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "extension_p22"))
import common_p22 as C  # noqa: E402
import gen_p22 as G  # noqa: E402
import preprocess as pp  # noqa: E402

OUT = os.path.join(ROOT, "results", "p3_small_data", "phase0")
DATASETS = ["v1", "R1", "S-EFF"]
PAIRS = {("sleep_variability", "dep"): ("lag_sv", "eff_sv_dep", 1.0), ("steps", "dep"): ("lag_steps", "eff_steps_dep", 0.7),
         ("steps", "anx"): ("lag_steps", "eff_steps_anx", 0.7), ("hrv", "anx"): ("lag_hrv", "eff_hrv_anx", 0.8)}
T0 = 27                                    # first origin with a 28-day lookback
SET_GRID = np.linspace(-2.5, 2.5, 21)
SHIFT_GRID = np.linspace(-2.0, 2.0, 17)


def dcfg(ds):
    return {"seed": 2027, "overrides": {}, "rho_scale": 1.0} if ds == "v1" else C.ds_config(ds)


def arrays(out, feats):
    """dict of DataFrames from sim.simulate -> arrays (N, D, ...) sorted by pid/day."""
    ob = out["observations"].sort_values(["pid", "day"])
    tr = out["truth"].sort_values(["pid", "day"])
    tg = out["targets"].sort_values(["pid", "day"])
    N = out["splits"].shape[0]
    D = len(ob) // N
    rs = lambda a: a.reshape(N, D, -1)  # noqa: E731
    return dict(raw=rs(ob[feats].values.astype(float)), masks=rs(ob[["missing_day", "hrv_missing"]].values.astype(float)),
                u=rs(tr[[f"u_{f}" for f in feats]].values), z=rs(tr[["z_dep", "z_anx"]].values),
                sev=rs(tg[["severity_dep", "severity_anx"]].values), split=out["splits"].sort_values("pid").split.values,
                part=out["participants"].sort_values("pid"), N=N, D=D)


def simulate(ds, eff_zero=None, clean=False):
    d = json.loads(json.dumps(dcfg(ds)))
    ov = dict(d.get("overrides", {}))
    if eff_zero:
        ov[eff_zero] = 0.0
    if clean:                                                        # remove reverse paths and the stress -> z path
        ov.update(rev_steps=0.0, rev_hrv=0.0, eff_stress_direct=0.0)
    d["overrides"] = ov
    cfg = G.build_config(d)
    out, _ = G.simulate_with(cfg, d.get("rho_scale", 1.0))
    return out, cfg


def ew_weights(L, a, K):
    return (1 - a) ** np.arange(K - L)                               # day t-L-k, k = 0..K-L-1 (inside the 28-day window)


def dl_features(V, L, a, basis=None):
    """Distributed-lag features at every origin t >= T0: sum_k w_k f(V[:, t-L-k]) / sum_k w_k over non-missing days.
    V (N, D) with NaN = missing. basis: callable (n,) -> (n, nb) or None (identity)."""
    N, D = V.shape
    w = ew_weights(L, a, 28)
    F = V[..., None] if basis is None else np.where(np.isnan(V)[..., None], np.nan, basis(np.nan_to_num(V).reshape(-1, 1)).reshape(N, D, -1))
    num = np.zeros((N, D - T0, F.shape[-1]))
    den = np.zeros((N, D - T0, 1))
    for k, wk in enumerate(w):
        sl = F[:, T0 - L - k: D - L - k]
        ok = ~np.isnan(sl[..., :1])
        num += wk * np.nan_to_num(sl)
        den += wk * ok
    return num / np.maximum(den, 1e-12)


def r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def design(lin_feats, planted, static):
    return np.concatenate([np.ones(planted.shape[:-1] + (1,)), planted, lin_feats, static], -1)


def wline_curv(xg, yg, w=None):
    """curvature share = 1 - weighted R2 of the best straight line."""
    w = np.ones_like(xg) if w is None else w
    A = np.column_stack([np.ones_like(xg), xg])
    b = np.linalg.lstsq(A * np.sqrt(w)[:, None], yg * np.sqrt(w), rcond=None)[0]
    res = yg - A @ b
    m = np.average(yg, weights=w)
    return float(np.average(res ** 2, weights=w) / max(np.average((yg - m) ** 2, weights=w), 1e-30))


def affine_err(f, T, w):
    """min_{alpha,beta} ||f - alpha T - beta||_w^2 / ||f - mean f||_w^2  (shape mismatch of f relative to T, scale/offset free)."""
    A = np.column_stack([np.ones_like(T), T])
    b = np.linalg.lstsq(A * np.sqrt(w)[:, None], f * np.sqrt(w), rcond=None)[0]
    res = f - A @ b
    m = np.average(f, weights=w)
    return float(np.average(res ** 2, weights=w) / max(np.average((f - m) ** 2, weights=w), 1e-30))


def analyse_pair(A_, ds, pair, cfg, scaler, feats, tag):
    (feat, k), (lag_key, eff_key, c) = pair, PAIRS[pair]
    L, a = int(cfg[lag_key]), float(cfg["a"])
    fi, ki = feats.index(feat), {"dep": 0, "anx": 1}[k]
    tr, ca = A_["split"] == "train", A_["split"] == "cal"
    x = (A_["raw"][..., fi] - scaler["mean"][fi]) / scaler["std"][fi]                 # observed, scaled, NaN = missing
    u = A_["u"][..., fi]
    # A: mapping x = alpha + beta*u (observed days, train)
    ok = tr[:, None] & ~np.isnan(x)
    beta, alpha = np.polyfit(u[ok], x[ok], 1)
    rel = float(np.corrcoef(u[ok], x[ok])[0, 1] ** 2)
    g = lambda uu: np.tanh(c * uu)  # noqa: E731
    u_of_v = (SET_GRID - alpha) / beta
    T_set = g(u_of_v)
    uu = u[tr][:, T0 - 27:].ravel()
    T_shift = np.array([g(uu + d / beta).mean() for d in SHIFT_GRID]) - g(uu).mean()
    dens = np.histogram(x[ok], bins=np.r_[-np.inf, (SET_GRID[1:] + SET_GRID[:-1]) / 2, np.inf])[0].astype(float)
    dens /= dens.sum()
    # B/C: distributed-lag regressions (planted channel: lag L, others: lag 1, linear), static covariates
    sp = SplineTransformer(n_knots=6, degree=3, extrapolation="linear").fit(np.clip(x[ok], -3, 3).reshape(-1, 1))
    spu = SplineTransformer(n_knots=6, degree=3, extrapolation="linear").fit(u[ok].reshape(-1, 1))
    others = [j for j in range(len(feats)) if j != fi]
    lin_x = np.concatenate([dl_features((A_["raw"][..., j] - scaler["mean"][j]) / scaler["std"][j], 1, a) for j in others], -1)
    lin_u = np.concatenate([dl_features(A_["u"][..., j], 1, a) for j in others], -1)
    stat = np.column_stack([(A_["part"].sex == "F").values] + [(A_["part"].age_group == q).values for q in (1, 2)]).astype(float)
    stat = np.repeat(stat[:, None, :], A_["D"] - T0, 1)
    res = {}
    for src, V, basis, LIN in (("observed_x", x, sp.transform, lin_x), ("true_u", u, spu.transform, lin_u)):
        P_lin = dl_features(V, L, a)
        P_spl = dl_features(V, L, a, basis)
        for tgt in ("severity", "latent_z"):
            Y = (A_["sev"] if tgt == "severity" else A_["z"])[:, T0:, ki]
            out = {}
            for name, P in (("linear", P_lin), ("spline", P_spl)):
                Xd = design(LIN, P, stat)
                okr = ~np.isnan(Xd).any(-1)
                trm, cam = tr[:, None] & okr, ca[:, None] & okr
                b = np.linalg.lstsq(Xd[trm], Y[trm], rcond=None)[0]
                out[name] = (r2(Y[cam], Xd[cam] @ b), b)
            Xn = np.concatenate([np.ones(LIN.shape[:-1] + (1,)), LIN, stat], -1)                # planted channel removed
            okn = ~np.isnan(design(LIN, P_spl, stat)).any(-1)
            bn = np.linalg.lstsq(Xn[tr[:, None] & okn], Y[tr[:, None] & okn], rcond=None)[0]
            r2_without = r2(Y[ca[:, None] & okn], Xn[ca[:, None] & okn] @ bn)
            res[(src, tgt)] = dict(r2_without_planted=r2_without, r2_linear=out["linear"][0], r2_spline=out["spline"][0],
                                   planted_linear_gain=out["linear"][0] - r2_without, delta_r2=out["spline"][0] - out["linear"][0])
            if src == "observed_x" and tgt == "severity":
                bs = out["spline"][1][1:1 + P_spl.shape[-1]]
                pd_set = sp.transform(SET_GRID.reshape(-1, 1)) @ bs                            # 'set whole window to v' partial dependence
                xs = x[tr][:, T0 - 27:]
                xs = xs[~np.isnan(xs)]
                base = (sp.transform(xs.reshape(-1, 1)) @ bs).mean()
                pd_shift = np.array([(sp.transform((xs + d).reshape(-1, 1)) @ bs).mean() for d in SHIFT_GRID]) - base
                res["curves"] = dict(pd_set=pd_set, pd_shift=pd_shift, bs=bs)
    return dict(ds=ds, tag=tag, feature=feat, outcome=k, lag=L, a=a, map_alpha=float(alpha), map_beta=float(beta), reliability_day=rel,
                frac_obs_beyond_2sd=float(np.mean(np.abs(x[ok]) > 2)), T_set=T_set, T_shift=T_shift, dens=dens, res=res)


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.listdir(OUT):
        raise SystemExit(f"{OUT} not empty; refusing to overwrite")
    t0 = time.time()
    rows_truth, rows_det, rows_est, rows_clip, curves = [], [], [], [], {}
    for ds in DATASETS:
        feats = json.load(open(os.path.join(ROOT, C.raw_dir(ds), "meta.json")))["features"]
        pm = json.load(open(os.path.join(ROOT, C.proc_dir(ds), "preprocessing_meta.json")))
        scaler = {"mean": np.array([pm["scaler_mean"][f] for f in feats]), "std": np.array([pm["scaler_std"][f] for f in feats])}
        fac, cfg = simulate(ds)
        cfgd = cfg.__dict__
        Af = arrays(fac, feats)
        # check the in-memory factual simulation equals the saved dataset (train+cal severity)
        tg = pd.read_csv(os.path.join(ROOT, C.raw_dir(ds), "targets.csv"), usecols=["pid", "day", "severity_dep"]).sort_values(["pid", "day"])
        sv = tg.severity_dep.values.reshape(Af["N"], Af["D"])
        keep = np.isin(Af["split"], ["train", "cal"])
        assert np.abs(sv[keep] - Af["sev"][keep][..., 0]).max() < 1e-9, f"{ds}: in-memory simulation differs from saved data"   # CSV round-off ~1e-15
        tr = Af["split"] == "train"
        s_ = Af["sev"][tr][:, T0:]
        rows_clip.append(dict(ds=ds, frac_sev_at_0=float(np.mean(s_ <= 0)), frac_sev_at_27=float(np.mean(s_ >= 27)),
                              frac_8plus3z_below_0=float(np.mean(8 + 3 * Af["z"][tr][:, T0:] < 0))))
        clean, _ = simulate(ds, clean=True)
        Ac = arrays(clean, feats)
        for pair in PAIRS:
            r = analyse_pair(Af, ds, pair, cfgd, scaler, feats, "factual")
            cf, _ = simulate(ds, eff_zero=PAIRS[pair][1])
            r0 = analyse_pair(arrays(cf, feats), ds, pair, cfgd, scaler, feats, "planted_off")
            rc = analyse_pair(Ac, ds, pair, cfgd, scaler, feats, "reverse_confound_off")
            name = f"{ds}|{pair[0]}->{pair[1]}"
            w = r["dens"]
            wu = np.ones_like(SHIFT_GRID)
            F_set, F0_set, Fc_set = r["res"]["curves"]["pd_set"], r0["res"]["curves"]["pd_set"], rc["res"]["curves"]["pd_set"]
            F_sh, F0_sh = r["res"]["curves"]["pd_shift"], r0["res"]["curves"]["pd_shift"]
            Pl_set, Pl_sh = F_set - F0_set, F_sh - F0_sh                                     # planted contribution (CRN difference)
            wv = lambda f, ww: float(np.average((f - np.average(f, weights=ww)) ** 2, weights=ww))  # noqa: E731
            rows_truth.append(dict(pair=name, lag=r["lag"], map_slope_x_per_u=r["map_beta"], map_intercept=r["map_alpha"], day_reliability_R2=r["reliability_day"],
                                   frac_obs_beyond_2sd=r["frac_obs_beyond_2sd"], curvature_T_set=wline_curv(SET_GRID, r["T_set"], w),
                                   curvature_T_set_unweighted=wline_curv(SET_GRID, r["T_set"]), curvature_T_shift=wline_curv(SHIFT_GRID, r["T_shift"])))
            for key, v in r["res"].items():
                if key != "curves":
                    rows_det.append(dict(pair=name, features=key[0], target=key[1], **v))
            rows_est.append(dict(pair=name,
                                 curvature_assoc_set=wline_curv(SET_GRID, F_set, w), curvature_assoc_shift=wline_curv(SHIFT_GRID, F_sh),
                                 curvature_planted_part_set=wline_curv(SET_GRID, Pl_set, w),
                                 planted_share_of_assoc_set=wv(Pl_set, w) / max(wv(F_set, w), 1e-30),
                                 planted_share_of_assoc_shift=wv(Pl_sh, wu) / max(wv(F_sh, wu), 1e-30),
                                 contamination_share_set=wv(F_set - Fc_set, w) / max(wv(F_set, w), 1e-30),
                                 assoc_vs_Tset_shape_err=affine_err(F_set, r["T_set"], w), planted_vs_Tset_shape_err=affine_err(Pl_set, r["T_set"], w),
                                 assoc_shift_vs_Tshift_shape_err=affine_err(F_sh, r["T_shift"], wu), assoc_set_line_err=wline_curv(SET_GRID, F_set, w),
                                 slope_sign_assoc=float(np.sign(np.polyfit(SET_GRID, F_set, 1, w=np.sqrt(w))[0])),
                                 slope_sign_planted=float(np.sign(np.polyfit(SET_GRID, Pl_set, 1, w=np.sqrt(w))[0]))))
            curves[name] = dict(set_grid=SET_GRID, shift_grid=SHIFT_GRID, density=w, T_set=r["T_set"], T_shift=r["T_shift"], assoc_set=F_set, assoc_shift=F_sh,
                                planted_off_set=F0_set, clean_set=Fc_set, planted_part_set=Pl_set)
            print(name, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows_truth).to_csv(os.path.join(OUT, "A_truth_curves_and_mapping.csv"), index=False)
    pd.DataFrame(rows_det).to_csv(os.path.join(OUT, "B_detectability.csv"), index=False)
    pd.DataFrame(rows_est).to_csv(os.path.join(OUT, "C_estimand_check.csv"), index=False)
    pd.DataFrame(rows_clip).to_csv(os.path.join(OUT, "D_severity_clipping.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "curves.npz"), **{f"{k}::{kk}": vv for k, v in curves.items() for kk, vv in v.items()})
    subset_part(feats)
    json.dump({"SYNTHETIC": True, "training_run": False, "test_split_read": False, "in_memory_simulations_only": True, "runtime_s": time.time() - t0},
              open(os.path.join(OUT, "run_info.json"), "w"), indent=1)


def strat_order(keys, rng):
    """Nested stratified order: within each stratum a random order; global order by (i + U) / n_stratum."""
    pos = np.zeros(len(keys))
    for kv in np.unique(keys):
        idx = np.where(keys == kv)[0]
        pos[rng.permutation(idx)] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    return np.argsort(pos, kind="stable")


def subset_part(_feats):
    rows, exact = [], []
    for ds in ("v1", "R1"):
        raw = os.path.join(ROOT, C.raw_dir(ds))
        meta = json.load(open(os.path.join(raw, "meta.json")))
        feats = meta["features"]
        ob = pd.read_csv(os.path.join(raw, "observations.csv"), usecols=["pid", "day"] + feats + ["missing_day", "hrv_missing"]).sort_values(["pid", "day"])
        N, D = meta["config"]["n_participants"], meta["config"]["n_days"]
        values = ob[feats].values.reshape(N, D, -1).astype(float)
        masks = ob[["missing_day", "hrv_missing"]].values.reshape(N, D, 2).astype(float)
        split = pd.read_csv(os.path.join(raw, "splits.csv")).sort_values("pid").split.values
        part = pd.read_csv(os.path.join(raw, "participants.csv")).sort_values("pid")
        tr_ids, ca_ids = np.where(split == "train")[0], np.where(split == "cal")[0]
        full = pp.fit_scaler(values[tr_ids])
        # exactness: the pure preprocessing functions reproduce the frozen processed arrays (train, cal) -> subset scaling is feasible in memory
        for sname, ids in (("train", tr_ids), ("cal", ca_ids)):
            Xp = np.load(os.path.join(ROOT, C.proc_dir(ds), f"{sname}.npz"))
            assert np.array_equal(Xp["pid"], ids)
            Xr = pp.transform(values[ids], masks[ids], full)
            exact.append(dict(ds=ds, split=sname, max_abs_diff=float(np.abs(Xr - Xp["X"]).max())))
        trl = np.load(os.path.join(ROOT, C.proc_dir(ds), "train.npz"))
        ons = {k: np.nan_to_num(np.stack([trl[f"y_onset_sust_{k}_h{h}"] for h in (7, 14, 21)], -1)).max((1, 2)) > 0 for k in ("dep", "anx")}
        stratum = (part.sex.values[tr_ids] == "F").astype(int) * 100 + part.age_group.values[tr_ids] * 10 + (ons["dep"] | ons["anx"]).astype(int)
        Vc = values[ca_ids]
        for draw in range(5):
            order = strat_order(stratum, np.random.default_rng(1000 + draw))
            for n in (50, 100, 200, 400, 600):
                sel = order[:n]
                sc = pp.fit_scaler(values[tr_ids[sel]])
                zc_sub = (Vc - sc["mean"]) / sc["std"]
                zc_full = (Vc - full["mean"]) / full["std"]
                rows.append(dict(ds=ds, draw=draw, n=n, max_abs_mean_shift_in_full_sd=float(np.max(np.abs(sc["mean"] - full["mean"]) / full["std"])),
                                 max_abs_std_ratio_minus_1=float(np.max(np.abs(sc["std"] / full["std"] - 1))),
                                 cal_frac_outside_kan_grid_subset=float(np.nanmean(np.abs(zc_sub) > 3)), cal_frac_outside_kan_grid_full=float(np.nanmean(np.abs(zc_full) > 3)),
                                 max_abs_cal_input_change=float(np.nanmax(np.abs(zc_sub - zc_full))),
                                 onset_participants_dep=int(ons["dep"][sel].sum()), onset_participants_anx=int(ons["anx"][sel].sum()),
                                 age_group_2=int((part.age_group.values[tr_ids[sel]] == 2).sum()), female=int((part.sex.values[tr_ids[sel]] == "F").sum())))
    pd.DataFrame(exact).to_csv(os.path.join(OUT, "E_scaler_exactness.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E_subsets_and_scaler.csv"), index=False)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
