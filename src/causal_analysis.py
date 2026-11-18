"""Stage 5: planted-effect and lag recovery on the SYNTHETIC dataset.  Observational association, not proof of causation.

Observational estimates use ONLY observable columns: 5 wearable features (steps, sleep_variability, hrv, resting_hr,
calories), `stress_proxy`, and the measured severity scores (observations.csv + the severity columns of targets.csv).
Ground truth (truth.csv, truth_participants.csv, config effects) is used ONLY for (a) benchmark values and (b) variants
explicitly flagged uses_ground_truth=True (oracle / partly-oracle regressions that decompose the sources of bias).

Estimator ("distributed-lag increment model", person fixed effects, participant-clustered SE):
    severity(tau) - severity(tau-1) = a * severity(tau-1) + sum_{L=1..5} b_{f,L} * x_f(tau-L) [+ sum_L c_L * stress(tau-L)] + person effect + e
Lag L means the feature on day tau-L is associated with the day-tau change; planted lags: sleep_variability->dep L=3,
steps->dep/anx L=2, hrv->anx L=1. Units: feature in simulator nominal units (u), outcome in severity points (= 3 x latent).
Usage: python src/causal_analysis.py data/synthetic_v1 results/stage5_causal
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from simulate import UNITS, RHO, F as FIDX  # noqa: E402  (unit constants; RHO = generating AR persistence, used only for the benchmark)

FEATS = ["sleep_variability", "steps", "hrv", "resting_hr", "calories"]
D = 120
LAGS = (1, 2, 3, 4, 5)
SCALE = 3.0   # severity = 8 + 3 z + noise: latent -> severity points
PLANTED = {("sleep_variability", "dep"): 3, ("steps", "dep"): 2, ("steps", "anx"): 2, ("hrv", "anx"): 1}
RNG_SEED = 0


# ---------------------------------------------------------------------------------------------- loading
def load_observational(raw):
    """Only observation.csv and the two severity columns of targets.csv are read (no truth files)."""
    obs = pd.read_csv(os.path.join(raw, "observations.csv")).sort_values(["pid", "day"])
    tg = pd.read_csv(os.path.join(raw, "targets.csv"), usecols=["pid", "day", "severity_dep", "severity_anx"]).sort_values(["pid", "day"])
    N = obs.pid.nunique()
    A = {}
    for f in FEATS:
        mu, sd, _, _ = UNITS[f]
        A[f] = ((obs[f].values - mu) / sd).reshape(N, D)          # nominal unit conversion, not an effect-related quantity
    A["stress_proxy"] = obs["stress_proxy"].values.reshape(N, D)
    A["sev_dep"], A["sev_anx"] = tg.severity_dep.values.reshape(N, D), tg.severity_anx.values.reshape(N, D)
    return A


def load_truth(raw):
    """EVALUATION ONLY."""
    tr = pd.read_csv(os.path.join(raw, "truth.csv")).sort_values(["pid", "day"])
    N = tr.pid.nunique()
    T = {f: tr[f"u_{f}"].values.reshape(N, D) for f in FEATS}
    T["stress"] = tr["stress"].values.reshape(N, D)
    T["sev_dep"], T["sev_anx"] = SCALE * tr.z_dep.values.reshape(N, D), SCALE * tr.z_anx.values.reshape(N, D)
    tp = pd.read_csv(os.path.join(raw, "truth_participants.csv")).sort_values("pid")
    with open(os.path.join(raw, "meta.json")) as fh:
        cfg = json.load(fh)["config"]
    return T, tp, cfg


def benchmarks(T, tp, cfg):
    """Planted per-day increment effect on severity points per 1 unit of the TRUE feature (population average slope of the
    planted tanh effect): 3 * eff * E[sensitivity * g'(u)]."""
    def avg(sens, slope, u):
        return float(np.mean(sens.values[:, None] * slope * (1 - np.tanh(slope * u) ** 2)))
    return {("sleep_variability", "dep"): SCALE * cfg["eff_sv_dep"] * avg(tp.sens_sv, 1.0, T["sleep_variability"]),
            ("steps", "dep"): SCALE * cfg["eff_steps_dep"] * avg(tp.sens_steps, 0.7, T["steps"]),
            ("steps", "anx"): SCALE * cfg["eff_steps_anx"] * avg(tp.sens_steps_anx, 0.7, T["steps"]),
            ("hrv", "anx"): SCALE * cfg["eff_hrv_anx"] * avg(tp.sens_hrv, 0.8, T["hrv"])}


# ---------------------------------------------------------------------------------------------- estimation
def design(Y, regs, diff=True, tau_lo=5):
    """dependent day tau in [tau_lo, D-1]: Y[tau] - Y[tau-1] (diff) or Y[tau]; regressors (name, array, lag L): value at tau-L."""
    N = Y.shape[0]
    ts = np.arange(tau_lo, D)
    y = Y[:, ts] - (Y[:, ts - 1] if diff else 0.0)
    X = np.stack([A[:, ts - L] for _, A, L in regs], -1)
    ok = np.isfinite(y) & np.isfinite(X).all(-1)
    pid = np.repeat(np.arange(N)[:, None], len(ts), 1)
    names = [n for n, _, _ in regs]
    return names, X[ok], y[ok], pid[ok]


def within(X, y, pid, min_rows=10):
    cnt = np.bincount(pid)
    keep = cnt[pid] >= min_rows
    X, y, pid = X[keep], y[keep], pid[keep]
    cnt = np.bincount(pid)

    def dm(v):
        s = np.bincount(pid, weights=v)
        return v - (s / np.maximum(cnt, 1))[pid]
    Xd = np.column_stack([dm(X[:, k]) for k in range(X.shape[1])])
    return Xd, dm(y), pid


def ols_cr(X, y, pid):
    """OLS + CR1 participant-clustered standard errors (rows sorted by participant)."""
    n, k = X.shape
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ beta
    starts = np.flatnonzero(np.diff(pid, prepend=-1))
    G = np.add.reduceat(X * e[:, None], starts, axis=0)
    inv = np.linalg.inv(XtX)
    g = len(starts)
    V = inv @ (G.T @ G) @ inv * (g / (g - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(V)), g


def fit(Y, regs, diff=True):
    names, X, y, pid = design(Y, regs, diff)
    Xd, yd, pidd = within(X, y, pid)
    beta, se, g = ols_cr(Xd, yd, pidd)
    return names, beta, se, len(yd), g, (Xd, yd, pidd)


def boot_ci(Xd, yd, pid, B=300, seed=RNG_SEED):
    """Participant bootstrap of the fixed-effects slopes via per-participant sufficient statistics."""
    starts = np.flatnonzero(np.diff(pid, prepend=-1))
    ends = np.append(starts[1:], len(pid))
    XtX = np.stack([Xd[a:b].T @ Xd[a:b] for a, b in zip(starts, ends)])
    Xty = np.stack([Xd[a:b].T @ yd[a:b] for a, b in zip(starts, ends)])
    rng = np.random.default_rng(seed)
    G = len(starts)
    out = []
    for _ in range(B):
        w = rng.multinomial(G, np.ones(G) / G).astype(float)
        out.append(np.linalg.solve(np.tensordot(w, XtX, 1), np.tensordot(w, Xty, 1)))
    out = np.array(out)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def mask_like(true_arr, obs_arr):
    """oracle value only where the observed one exists, so every variant uses the same complete-case rows."""
    return np.where(np.isfinite(obs_arr), true_arr, np.nan)


def dl_regs(Y_name, feats, extra=(), ar=True, lags=LAGS):
    regs = []
    if ar:
        regs.append(("AR", Y_name, 1))
    for f in FEATS:
        for L in lags:
            regs.append((f"{f}|L{L}", feats[f], L))
    for nm, arr in extra:
        for L in lags:
            regs.append((f"{nm}|L{L}", arr, L))
    return regs


def rows_from(variant, outcome, names, beta, se, n, g, bench, uses_truth):
    out = []
    for nm, b, s in zip(names, beta, se):
        if "|L" not in nm:
            continue
        f, L = nm.split("|L")
        L = int(L)
        bm = bench.get((f, outcome)) if (f, outcome) in PLANTED and PLANTED[(f, outcome)] == L else (0.0 if f in FEATS else np.nan)
        if f in ("stress_proxy", "stress"):
            bm = SCALE * 0.03 if L == 1 else 0.0
        out.append(dict(variant=variant, uses_ground_truth=uses_truth, outcome=outcome, feature=f, lag=L, coef=b, se=s,
                        ci_lo=b - 1.96 * s, ci_hi=b + 1.96 * s, t=b / s, n_rows=n, n_participants=g, benchmark=bm))
    return out


# ---------------------------------------------------------------------------------------------- analyses
def main(raw, out):
    os.makedirs(out, exist_ok=True)
    A = load_observational(raw)
    T, tp, cfg = load_truth(raw)
    bench = benchmarks(T, tp, cfg)
    ST = mask_like(T["stress"], A["stress_proxy"])
    TO = {f: mask_like(T[f], A[f]) for f in FEATS}
    rows, boots, notes = [], [], {}
    fits = {}
    for k in ("dep", "anx"):
        Yo, Yt = A[f"sev_{k}"], T[f"sev_{k}"]
        specs = [("V1_adjusted_for_other_features", False, Yo, dl_regs(Yo, A, ()), None),
                 ("V2_plus_stress_proxy", False, Yo, dl_regs(Yo, A, (("stress", A["stress_proxy"]),)), None),
                 ("E1_observed_features_true_stress", True, Yo, dl_regs(Yo, A, (("stress", ST),)), None),
                 ("E2_oracle_no_stress", True, Yt, dl_regs(Yt, TO, ()), None),
                 ("E3_oracle_true_stress", True, Yt, dl_regs(Yt, TO, (("stress", ST),)), None),
                 ("E4_true_features_true_stress_noisy_outcome", True, Yo, dl_regs(Yo, TO, (("stress", ST),)), None),
                 ("E5_noisy_features_true_stress_true_outcome", True, Yt, dl_regs(Yt, A, (("stress", ST),)), None)]
        for vname, truth_flag, Y, regs, _ in specs:
            # regs refer to arrays directly; "AR" regressor uses its own outcome array
            regs = [(n_, (Y if n_ == "AR" else a_), L) for n_, a_, L in regs]
            names, beta, se, n, g, pack = fit(Y, regs)
            rows += rows_from(vname, k, names, beta, se, n, g, bench, truth_flag)
            fits[(vname, k)] = (names, beta, se, n, g)
            if vname == "V2_plus_stress_proxy":
                lo, hi = boot_ci(*pack)
                for nm, l_, h_, b_ in zip(names, lo, hi, beta):
                    if "|L" in nm:
                        boots.append(dict(outcome=k, term=nm, coef=b_, boot_lo=l_, boot_hi=h_))
    eff = pd.DataFrame(rows)
    eff.to_csv(os.path.join(out, "effect_lag_table.csv"), index=False)
    pd.DataFrame(boots).to_csv(os.path.join(out, "bootstrap_ci_V2.csv"), index=False)

    # -------- naive lag associations (univariate, lags 1-5): pooled and within-person
    nv = []
    for k in ("dep", "anx"):
        for f in FEATS:
            for h in LAGS:
                x, y = A[f][:, :D - h], A[f"sev_{k}"][:, h:]
                ok = np.isfinite(x) & np.isfinite(y)
                pid = np.repeat(np.arange(x.shape[0])[:, None], x.shape[1], 1)[ok]
                xv, yv = x[ok], y[ok]
                Xd, yd, pidd = within(xv[:, None], yv, pid)
                bw, sw, g = ols_cr(Xd, yd, pidd)
                Xp, yp = (xv - xv.mean())[:, None], yv - yv.mean()
                bp, sp, _ = ols_cr(Xp, yp, pid)
                nv += [dict(variant="V0a_pooled_univariate", outcome=k, feature=f, lag=h, coef=bp[0], se=sp[0], ci_lo=bp[0] - 1.96 * sp[0], ci_hi=bp[0] + 1.96 * sp[0], n_rows=len(yp)),
                       dict(variant="V0b_within_person_univariate", outcome=k, feature=f, lag=h, coef=bw[0], se=sw[0], ci_lo=bw[0] - 1.96 * sw[0], ci_hi=bw[0] + 1.96 * sw[0], n_rows=len(yd))]
    pd.DataFrame(nv).to_csv(os.path.join(out, "naive_lag_association.csv"), index=False)

    # -------- recovery summary for the four planted relationships (+ stress direct path)
    rec = []
    variants = ["V1_adjusted_for_other_features", "V2_plus_stress_proxy", "E1_observed_features_true_stress", "E4_true_features_true_stress_noisy_outcome",
                "E5_noisy_features_true_stress_true_outcome", "E2_oracle_no_stress", "E3_oracle_true_stress"]
    for (f, k), Ls in PLANTED.items():
        for v in variants:
            sub = eff[(eff.variant == v) & (eff.outcome == k) & (eff.feature == f)].sort_values("lag")
            pk = sub.loc[sub.coef.abs().idxmax()]
            at = sub[sub.lag == Ls].iloc[0]
            others = sub[sub.lag != Ls]
            rec.append(dict(relationship=f"{f} -> {k}", planted_lag=Ls, benchmark=bench[(f, k)], variant=v, uses_ground_truth=bool(at.uses_ground_truth),
                            coef_at_planted_lag=at.coef, ci_lo=at.ci_lo, ci_hi=at.ci_hi, t_at_planted_lag=at.t, ratio_to_benchmark=at.coef / bench[(f, k)],
                            ci_covers_benchmark=bool(at.ci_lo <= bench[(f, k)] <= at.ci_hi), sign_correct=bool(np.sign(at.coef) == np.sign(bench[(f, k)])),
                            peak_lag_by_abs_coef=int(pk.lag), peak_is_planted_lag=bool(pk.lag == Ls),
                            n_other_lags_significant=int((others.t.abs() > 1.96).sum()), max_abs_t_other_lags=float(others.t.abs().max())))
    pd.DataFrame(rec).to_csv(os.path.join(out, "recovery_summary.csv"), index=False)

    # -------- non-causal / not-planted pairs
    nc = []
    planted_pairs = set(PLANTED)
    for k in ("dep", "anx"):
        for f in FEATS:
            if (f, k) in planted_pairs:
                continue
            for v in variants:
                sub = eff[(eff.variant == v) & (eff.outcome == k) & (eff.feature == f)]
                nc.append(dict(pair=f"{f} -> {k}", status="no planted effect", variant=v, uses_ground_truth=bool(sub.uses_ground_truth.iloc[0]),
                               max_abs_coef=float(sub.coef.abs().max()), max_abs_t=float(sub.t.abs().max()), n_lags_significant=int((sub.t.abs() > 1.96).sum()),
                               sum_of_lag_coefs=float(sub.coef.sum())))
    pd.DataFrame(nc).to_csv(os.path.join(out, "noncausal_summary.csv"), index=False)

    # -------- stress: contamination of features and proxy quality (evaluation-side statistics)
    sc = []
    loads = {"sleep_variability": 0.5, "steps": -0.2, "hrv": -0.2, "resting_hr": 0.6, "calories": -0.14}
    Zs = lambda a: (a - np.nanmean(a, 1, keepdims=True))
    for f in FEATS:
        a, s = Zs(A[f]), Zs(ST)
        ok = np.isfinite(a) & np.isfinite(s)
        sc.append(dict(feature=f, planted_loading_on_stress_in_u_units=loads[f], within_person_corr_observed_feature_vs_TRUE_stress=float(np.corrcoef(a[ok], s[ok])[0, 1]),
                       within_person_corr_observed_feature_vs_stress_proxy=float(np.corrcoef(a[ok & np.isfinite(Zs(A["stress_proxy"]))], Zs(A["stress_proxy"])[ok & np.isfinite(Zs(A["stress_proxy"]))])[0, 1])))
    okp = np.isfinite(A["stress_proxy"])
    reliab = float(np.corrcoef(A["stress_proxy"][okp], T["stress"][okp])[0, 1] ** 2)
    pd.DataFrame(sc).to_csv(os.path.join(out, "stress_confounding.csv"), index=False)
    notes["stress_proxy_reliability_R2_with_true_stress"] = reliab
    notes["stress_proxy_corr_with_true_stress"] = float(np.sqrt(reliab))

    # -------- proxy reliability sensitivity (evaluation-only: proxies built from TRUE stress plus noise)
    rng = np.random.default_rng(RNG_SEED)
    ps = []
    sdS = float(np.nanstd(ST))
    for sd_noise in (0.0, 0.4, 0.8, 1.6, 3.2, None):
        if sd_noise is None:
            extra, rel = (), 0.0
        else:
            extra = (("stress", np.where(np.isfinite(ST), ST + sd_noise * rng.standard_normal(ST.shape), np.nan)),)
            rel = sdS ** 2 / (sdS ** 2 + sd_noise ** 2)
        Yo = A["sev_dep"]
        names, beta, se, n, g, _ = fit(Yo, [(n_, (Yo if n_ == "AR" else a_), L) for n_, a_, L in dl_regs(Yo, A, extra)])
        d_ = dict(zip(names, beta))
        ps.append(dict(proxy_noise_sd=sd_noise if sd_noise is not None else "no stress adjustment", proxy_reliability=rel,
                       resting_hr_sum_lag_coefs=float(sum(d_[f"resting_hr|L{L}"] for L in LAGS)),
                       sleep_variability_L3=float(d_["sleep_variability|L3"]), steps_L2=float(d_["steps|L2"]),
                       benchmark_sleep_variability_L3=bench[("sleep_variability", "dep")], benchmark_steps_L2=bench[("steps", "dep")]))
    pd.DataFrame(ps).to_csv(os.path.join(out, "proxy_sensitivity.csv"), index=False)

    # -------- reverse paths: feature(tau) on severity(tau-j), j=0..4, with own lag and stress (proxy) control
    rv = []
    rev_bench = {("steps", "dep"): -0.15 / SCALE, ("hrv", "anx"): -0.10 / SCALE}
    for k in ("dep", "anx"):
        for f in FEATS:
            for vname, uses_t, X_f, S_k, st in (("observed", False, A[f], A[f"sev_{k}"], A["stress_proxy"]),
                                                  ("oracle(true feature, true latent, true stress)", True, TO[f], T[f"sev_{k}"], ST)):
                regs = [("AR", X_f, 1)] + [(f"sev|L{j}", S_k, j) for j in range(0, 5)] + [("stress|L0", st, 0)]
                names, beta, se, n, g, _ = fit(X_f, regs, diff=False)
                for nm, b, s in zip(names, beta, se):
                    if nm.startswith("sev|L"):
                        j = int(nm.split("L")[1])
                        rev = rev_bench.get((f, k), 0.0)        # AR(1)-filtered benchmark: lag0 = rev, lag1 = -rev * rho_feature
                        bm = rev if j == 0 else (-rev * float(RHO[FIDX[f]]) if j == 1 else 0.0)
                        rv.append(dict(feature=f, from_outcome=k, variant=vname, uses_ground_truth=uses_t, outcome_lag_days=j, coef=b, se=s, ci_lo=b - 1.96 * s, ci_hi=b + 1.96 * s, t=b / s,
                                       planted_benchmark_u_per_severity_point=bm))
    pd.DataFrame(rv).to_csv(os.path.join(out, "reverse_paths.csv"), index=False)

    # -------- coupling dep <-> anx on the increment of the other
    cp = []
    for tgt, oth in (("dep", "anx"), ("anx", "dep")):
        Yo, Yt = A[f"sev_{tgt}"], T[f"sev_{tgt}"]
        for vname, uses_t, Y, Oth, feats, st in (("naive (AR + other severity only)", False, Yo, A[f"sev_{oth}"], None, None),
                                                  ("adjusted: features + stress_proxy", False, Yo, A[f"sev_{oth}"], A, A["stress_proxy"]),
                                                  ("oracle, no stress", True, Yt, T[f"sev_{oth}"], TO, None),
                                                  ("oracle, true stress", True, Yt, T[f"sev_{oth}"], TO, ST)):
            regs = [("AR", Y, 1)] + [(f"other|L{L}", Oth, L) for L in LAGS]
            if feats is not None:
                regs += [(f"{f}|L{L}", feats[f], L) for f in FEATS for L in LAGS]
            if st is not None:
                regs += [(f"stress|L{L}", st, L) for L in LAGS]
            names, beta, se, n, g, _ = fit(Y, regs)
            d_ = dict(zip(names, zip(beta, se)))
            b, s = d_["other|L1"]
            cp.append(dict(target=tgt, source=oth, variant=vname, uses_ground_truth=uses_t, lag=1, coef=b, se=s, ci_lo=b - 1.96 * s, ci_hi=b + 1.96 * s,
                           planted_benchmark=cfg["coupling"], t=b / s, n_rows=n))
    pd.DataFrame(cp).to_csv(os.path.join(out, "coupling.csv"), index=False)

    json.dump({"benchmarks_severity_points_per_u_per_day": {f"{f}->{k}": v for (f, k), v in bench.items()}, "planted_lags": {f"{f}->{k}": L for (f, k), L in PLANTED.items()},
               "reverse_path_benchmarks_u_per_severity_point": {f"{k[1]}->{k[0]}": v for k, v in rev_bench.items()}, "coupling_benchmark": cfg["coupling"],
               "stress_direct_benchmark_severity_points_per_stress_unit_lag1": SCALE * cfg["eff_stress_direct"], "units": "feature: simulator nominal unit u; outcome: severity points (3 x latent)",
               **notes}, open(os.path.join(out, "analysis_config.json"), "w"), indent=1)
    make_plots(out, eff, pd.DataFrame(ps), pd.DataFrame(rv), pd.DataFrame(cp), bench, cfg)
    print("done ->", out)


# ---------------------------------------------------------------------------------------------- plots
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
COL = {"V1": "#eb6834", "V2": "#2a78d6", "E3": "#1baf7a"}
MARK = {"V1": "s", "V2": "o", "E3": "^"}
LABEL = {"V1": "adjusted for other features, no stress adjustment", "V2": "+ noisy stress_proxy (what an analyst can do)", "E3": "oracle (true features, latent, stress; evaluation only)"}


def style(ax, title, zero=True):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    if zero:
        ax.axhline(0, color=MUTED, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)


def make_plots(out, eff, ps, rv, cp, bench, cfg):
    short = {"V1": "V1_adjusted_for_other_features", "V2": "V2_plus_stress_proxy", "E3": "E3_oracle_true_stress"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURF)
    for ax, ((f, k), Ls) in zip(axes.ravel(), PLANTED.items()):
        style(ax, f"{f} -> {k}  (planted lag {Ls}, sign {'+' if bench[(f, k)] > 0 else '-'})")
        for i, key in enumerate(("V1", "V2", "E3")):
            s = eff[(eff.variant == short[key]) & (eff.outcome == k) & (eff.feature == f)].sort_values("lag")
            x = s.lag.values + (i - 1) * 0.14
            ax.errorbar(x, s.coef, yerr=1.96 * s.se, fmt=MARK[key] + "-", color=COL[key], ms=5, lw=1.2, capsize=2, label=LABEL[key])
        ax.plot([Ls], [bench[(f, k)]], marker="D", ms=9, color=INK, ls="none", label="planted effect (ground truth)", zorder=5)
        ax.set_xticks(LAGS)
        ax.set_xlabel("lag L (days between feature and outcome change)", fontsize=8, color=MUTED)
        ax.set_ylabel("severity points per unit, per day", fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Lag recovery of planted effects (SYNTHETIC data; observational estimates, 95% participant-clustered CI)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(out, "fig1_lag_recovery_planted.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=SURF)
    for ax, (f, k) in zip(axes.ravel(), (("resting_hr", "dep"), ("calories", "dep"), ("resting_hr", "anx"), ("calories", "anx"))):
        style(ax, f"{f} -> {k}  (NO planted effect)")
        for i, key in enumerate(("V1", "V2", "E3")):
            s = eff[(eff.variant == short[key]) & (eff.outcome == k) & (eff.feature == f)].sort_values("lag")
            ax.errorbar(s.lag.values + (i - 1) * 0.14, s.coef, yerr=1.96 * s.se, fmt=MARK[key] + "-", color=COL[key], ms=5, lw=1.2, capsize=2, label=LABEL[key])
        ax.set_xticks(LAGS)
        ax.set_xlabel("lag L (days)", fontsize=8, color=MUTED)
        ax.set_ylabel("severity points per unit, per day", fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Non-causal variables: spurious association and what adjustment removes (benchmark = 0 at every lag)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(out, "fig2_noncausal_resting_hr_calories.png"), dpi=130)
    plt.close(fig)

    rec = pd.read_csv(os.path.join(out, "recovery_summary.csv"))
    order = ["V1_adjusted_for_other_features", "V2_plus_stress_proxy", "E1_observed_features_true_stress", "E4_true_features_true_stress_noisy_outcome",
             "E5_noisy_features_true_stress_true_outcome", "E2_oracle_no_stress", "E3_oracle_true_stress"]
    nice = {"V1_adjusted_for_other_features": "V1 observed, no stress adj.", "V2_plus_stress_proxy": "V2 observed + stress proxy",
            "E1_observed_features_true_stress": "E1 + TRUE stress", "E4_true_features_true_stress_noisy_outcome": "E4 true features, noisy outcome",
            "E5_noisy_features_true_stress_true_outcome": "E5 noisy features, true outcome", "E2_oracle_no_stress": "E2 oracle, no stress",
            "E3_oracle_true_stress": "E3 oracle + true stress"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), facecolor=SURF, sharey=True)
    for ax, ((f, k), Ls) in zip(axes, PLANTED.items()):
        style(ax, f"{f} -> {k} (lag {Ls})", zero=False)
        sub = rec[rec.relationship == f"{f} -> {k}"].set_index("variant").loc[order]
        bm = sub.benchmark.iloc[0]
        for i, (v, r) in enumerate(sub.iterrows()):
            c = COL["E3"] if r.uses_ground_truth else COL["V2"]
            ax.errorbar([r.coef_at_planted_lag / bm], [i], xerr=[[(r.coef_at_planted_lag - r.ci_lo) / abs(bm)], [(r.ci_hi - r.coef_at_planted_lag) / abs(bm)]],
                        fmt="o" if not r.uses_ground_truth else "^", color=c, ms=6, capsize=2, lw=1.2)
        ax.axvline(1.0, color=INK, lw=1.2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([nice[o] for o in order], fontsize=7.5, color=INK)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e5e4e0", lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("estimate / planted effect (1 = exact)", fontsize=8, color=MUTED)
    fig.suptitle("Why observed estimates differ from the planted effect (circles: observable analysis; triangles: evaluation-only variants using ground truth)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "fig5_bias_decomposition.png"), dpi=130)
    plt.close(fig)

    p = ps[ps.proxy_reliability > 0].copy()
    none = ps[ps.proxy_reliability == 0].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURF)
    for ax, col, bm, ttl in ((axes[0], "resting_hr_sum_lag_coefs", 0.0, "resting_hr: sum of lag 1-5 coefficients (true 0)"),
                             (axes[1], "sleep_variability_L3", bench[("sleep_variability", "dep")], "sleep_variability, lag 3 (planted +)"),
                             (axes[2], "steps_L2", bench[("steps", "dep")], "steps, lag 2 (planted -)")):
        style(ax, ttl)
        ax.plot(p.proxy_reliability, p[col], "o-", color=COL["V2"], ms=5, label="adjusted, proxy = true stress + noise")
        ax.axhline(none[col], color=COL["V1"], ls="--", lw=1.2, label="no stress adjustment")
        ax.axhline(bm, color=INK, lw=1.2, label="planted effect" if bm != 0 else "truth (0)")
        ax.set_xlabel("proxy reliability (share of proxy variance that is true stress)", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Residual confounding shrinks as the stress proxy gets more reliable (evaluation-only sensitivity, depression outcome)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "fig3_stress_proxy_reliability.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), facecolor=SURF)
    for ax, (f, k) in zip(axes[:2], (("steps", "dep"), ("hrv", "anx"))):
        style(ax, f"reverse path {k} -> {f}  (planted same-day, negative)")
        for i, (v, c, m) in enumerate((("observed", COL["V2"], "o"), ("oracle(true feature, true latent, true stress)", COL["E3"], "^"))):
            s = rv[(rv.feature == f) & (rv.from_outcome == k) & (rv.variant == v)].sort_values("outcome_lag_days")
            ax.errorbar(s.outcome_lag_days + (i - 0.5) * 0.15, s.coef, yerr=1.96 * s.se, fmt=m + "-", color=c, ms=5, lw=1.2, capsize=2, label="observed" if i == 0 else "oracle (evaluation only)")
        rv0 = -0.15 / SCALE if (f, k) == ("steps", "dep") else -0.10 / SCALE
        ax.plot([0, 1], [rv0, -rv0 * float(RHO[FIDX[f]])], "D", color=INK, ms=9, label="planted effect (AR-filtered benchmark)", ls="none")
        ax.set_xticks(range(5))
        ax.set_xlabel("days between severity and feature (0 = same day)", fontsize=8, color=MUTED)
        ax.set_ylabel("feature units per severity point", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[2]
    style(ax, "dep <-> anx coupling (lag 1), planted 0.03")
    c = cp[cp.target.isin(["dep", "anx"])]
    labels = list(dict.fromkeys(c.variant))
    xs = np.arange(len(labels))
    for j, (tgt, mk, col) in enumerate((("dep", "o", COL["V2"]), ("anx", "s", COL["V1"]))):
        s = c[c.target == tgt].set_index("variant").loc[labels]
        ax.errorbar(xs + (j - 0.5) * 0.15, s.coef, yerr=1.96 * s.se, fmt=mk, color=col, ms=6, capsize=2, label=f"anx -> dep" if tgt == "dep" else "dep -> anx")
    ax.axhline(cfg["coupling"], color=INK, lw=1.2, label="planted 0.03")
    ax.set_xticks(xs)
    ax.set_xticklabels(["naive", "adjusted", "oracle,\nno stress", "oracle,\ntrue stress"], fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Reverse paths (state -> feature) and depression-anxiety coupling (SYNTHETIC, 95% CI)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "fig4_reverse_paths_and_coupling.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
