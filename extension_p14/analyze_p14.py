"""P1-4 analysis (SYNTHETIC; EXTENSION): lag-recovery metrics, controls, Stage 5 comparison, seed stability and figures from results/p1_lag_recovery/raw/*.npz (read-only).
Writes results/p1_lag_recovery/analysis/{lag_recovery_tables.md, metrics_per_seed.csv, controls.csv, stage5_comparison.csv, ig_summary.csv, fig_*.png}.
Everything here describes what a PREDICTOR learned from observational data (autocorrelated, reverse-path-contaminated, collinear channels), not causal identification; the largest attribution
or the peak lag is never taken to be the true causal lag (peak lag, increment lag, pre-lag mass and spike ratio are reported separately and compared with analytic truth and with null models)."""
import glob
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p14 as C  # noqa: E402

RAW, OUT = "results/p1_lag_recovery/raw", "results/p1_lag_recovery/analysis"
MODELS = ["gru", "m1", "mlp_m1", "kangru"]
COL = {"gru": "#444444", "m1": "#c51b7d", "mlp_m1": "#1b9e77", "kangru": "#7570b3", "truth": "black"}
PAIRS = list(C.PLANTED)                                    # (feature, outcome)
HK = {"dep": 0, "anx": 1}
NL = len(C.NEAR)


def load(label, untrained=False):
    out = {}
    for f in sorted(glob.glob(f"{RAW}/{label}{'_untrained' if untrained else ''}_seed*.npz")):
        out[int(f.split("seed")[-1][:-4])] = dict(np.load(f))
    return out


def profile(r, feat, outcome, kind="pert", half=0):
    """level profile over lags 0..10 (severity points per +1 standardised unit); kind pert | ar | coupled"""
    a = r[kind]
    ci = {"pert": C.CH.index(feat), "ar": ["steps", "sleep_variability", "hrv"].index(feat) if feat in ("steps", "sleep_variability", "hrv") else None,
          "coupled": ["steps", "sleep_variability"].index(feat) if feat in ("steps", "sleep_variability") else None}[kind]
    return a[ci, :NL, HK[outcome], half].astype(float)


def truth(feat, outcome, rho=None, kind="pert"):
    L, b = C.PLANTED[(feat, outcome)], C.benchmarks()[(feat, outcome)] * C.std_per_unit(feat)      # per +1 standardised unit
    return C.truth_level(b, L) if kind != "ar" else C.truth_ar(b, L, rho)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {m: load(m) for m in MODELS}
    N = {m: load(m, True) for m in MODELS}
    R = {m: v for m, v in R.items() if v}
    rho_true = {"steps": 0.75, "sleep_variability": 0.85, "hrv": 0.75}                              # simulator AR persistence (ground truth, used only for the analytic AR-aware truth)
    L_ = ["# P1-4 model-based lag recovery (SYNTHETIC; EXTENSION). Calibration split, 1,000 fixed origins. Predictive attribution of what the models learned; NOT causal identification.\n"]
    rows = []
    # ------------------------------------------------------------------ 1. metrics per model/seed/pair (single-day impulse and AR-propagated impulse)
    for kind in ("pert", "ar"):
        for pop, src in (("trained", R), ("untrained", N)):
            if kind == "ar" and pop == "untrained":
                continue
            for m, runs in src.items():
                for sd, r in runs.items():
                    for (f, k) in PAIRS:
                        P = profile(r, f, k, kind)
                        mt = C.lag_metrics(P, C.PLANTED[(f, k)], C.SIGN[(f, k)])
                        T = truth(f, k, rho_true[f] if kind == "ar" else None, kind)
                        pa, pb = profile(r, f, k, kind, 1), profile(r, f, k, kind, 2)
                        rows.append(dict(kind=kind, population=pop, model=m, seed=sd, feature=f, outcome=k, planted_lag=C.PLANTED[(f, k)], **mt, r_with_truth=C.corr(P, T),
                                         magnitude_vs_truth=float(P.sum() / T.sum()), split_half_r=C.corr(pa, pb)))
    M = pd.DataFrame(rows)
    M.to_csv(f"{OUT}/metrics_per_seed.csv", index=False)
    # analytic truth metrics (reference rows)
    tr = []
    for kind in ("pert", "ar"):
        for (f, k) in PAIRS:
            T = truth(f, k, rho_true[f] if kind == "ar" else None, kind)
            tr.append(dict(kind=kind, feature=f, outcome=k, planted_lag=C.PLANTED[(f, k)], **C.lag_metrics(T, C.PLANTED[(f, k)], C.SIGN[(f, k)])))
    TR = pd.DataFrame(tr)

    def summ(df):
        g = df.groupby(["population", "model", "feature", "outcome"], sort=False)
        s = g.agg(planted_lag=("planted_lag", "first"), peak_level=("peak_lag_level", lambda x: sorted(x)), peak_incr=("peak_lag_increment", lambda x: sorted(x)), incr15=("peak_lag_increment_1to5", lambda x: sorted(x)), onset=("onset_lag50", lambda x: sorted(x)),
                  exact_level=("peak_lag_level", "size"), prelag=("prelag_mass", "mean"), prelag_sd=("prelag_mass", "std"), spike=("spike_ratio", "mean"), r_truth=("r_with_truth", "mean"),
                  r_truth_sd=("r_with_truth", "std"), mag=("magnitude_vs_truth", "mean"), shr=("split_half_r", "mean"), sign=("sign_ok", "mean"), n=("seed", "count")).reset_index()
        s["exact_level"] = [int(sum(1 for v in sub.peak_lag_level if v == sub.planted_lag.iloc[0])) for _, sub in df.groupby(["population", "model", "feature", "outcome"], sort=False)]
        s["exact_incr15"] = [int(sum(1 for v in sub.peak_lag_increment_1to5 if v == sub.planted_lag.iloc[0])) for _, sub in df.groupby(["population", "model", "feature", "outcome"], sort=False)]
        s["exact_onset"] = [int(sum(1 for v in sub.onset_lag50 if v == sub.planted_lag.iloc[0])) for _, sub in df.groupby(["population", "model", "feature", "outcome"], sort=False)]
        s["exact_incr"] = [int(sum(1 for v in sub.peak_lag_increment if v == sub.planted_lag.iloc[0])) for _, sub in df.groupby(["population", "model", "feature", "outcome"], sort=False)]
        return s

    for kind, title in (("pert", "single-day +1 SD impulse"), ("ar", "impulse persisting as the channel's own AR(1)")):
        S = summ(M[M.kind == kind])
        TRk = TR[TR.kind == kind].set_index(["feature", "outcome"])
        L_ += [f"\n## 1{'a' if kind == 'pert' else 'b'}. Lag-recovery metrics, {title} (profile over lags 0-10; mean over seeds; peak lags listed per seed)\n",
               "Truth row = analytic response of the planted simulator. exact = seeds whose peak equals the planted lag. pre-lag mass = share of |profile| at lags before the planted lag (truth 0). "
               "spike = |increment at planted lag| / mean |other increments| (truth large). r = Pearson with the analytic truth profile; magnitude = profile sum / truth sum (sensor noise and the model's own "
               "attenuation not removed); split-half r = correlation between two disjoint participant halves.\n",
               "| outcome | feature | planted lag | model | level-peak lag per seed (exact) | increment-peak lag per seed, lags 0-10 (exact) | increment-peak lag per seed, lags 1-5 as in Stage 5 (exact) | onset lag = first lag reaching 50% of max (exact) | pre-lag mass | spike ratio | r with truth | magnitude vs truth | split-half r | sign ok |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (f, k) in PAIRS:
            t = TRk.loc[(f, k)]
            L_.append(f"| {k} | {f} | {C.PLANTED[(f, k)]} | **truth** | {t.peak_lag_level} | {t.peak_lag_increment} | {t.peak_lag_increment_1to5} | {t.onset_lag50} | {t.prelag_mass:.2f} | {t.spike_ratio:.1f} | 1.00 | 1.00 | | |")
            for pop in ("trained", "untrained"):
                for m in MODELS:
                    x = S[(S.population == pop) & (S.model == m) & (S.feature == f) & (S.outcome == k)]
                    if x.empty:
                        continue
                    x = x.iloc[0]
                    L_.append(f"| {k} | {f} | {x.planted_lag} | {m}{' (untrained null)' if pop == 'untrained' else ''} | {x.peak_level} ({x.exact_level}/{x.n}) | {x.peak_incr} ({x.exact_incr}/{x.n}) | {x.incr15} ({x.exact_incr15}/{x.n}) | {x.onset} ({x.exact_onset}/{x.n}) | "
                              f"{x.prelag:.2f} ±{x.prelag_sd:.2f} | {x.spike:.1f} | {x.r_truth:+.2f} ±{x.r_truth_sd:.2f} | {x.mag:.2f} | {x.shr:+.2f} | {x['sign']:.0%} |")
    L_ += ["\n## 1c. Pooled lag-hit rates over all model-seeds and the 4 planted pairs (single-day impulse). Exact = estimated lag equals the planted lag; within 1 day = |error| <= 1. Chance is for guessing uniformly among the estimator's candidate lags (11 for lags 0-10, 5 for lags 1-5). Untrained nulls use the same architectures with random weights.\n",
           "| estimator | population | exact hits | within 1 day | n | chance (exact) |", "|---|---|---|---|---|---|"]
    for est, col, ch in (("level peak (lags 0-10)", "peak_lag_level", 1 / 11), ("increment peak (lags 0-10)", "peak_lag_increment", 1 / 11), ("increment peak (lags 1-5, as Stage 5)", "peak_lag_increment_1to5", 1 / 5), ("onset lag (50% of max)", "onset_lag50", 1 / 11)):
        for pop in ("trained", "untrained"):
            x = M[(M.kind == "pert") & (M.population == pop)]
            err = (x[col] - x.planted_lag).abs()
            L_.append(f"| {est} | {pop} | {int((err == 0).sum())}/{len(x)} ({(err == 0).mean():.0%}) | {int((err <= 1).sum())}/{len(x)} ({(err <= 1).mean():.0%}) | {len(x)} | {ch:.0%} |")
    L_ += ["\nMean signed error (estimated minus planted lag), trained models, level peak: " + ", ".join(f"{m}: {(M[(M.kind == 'pert') & (M.population == 'trained') & (M.model == m)].peak_lag_level - M[(M.kind == 'pert') & (M.population == 'trained') & (M.model == m)].planted_lag).mean():+.1f}" for m in MODELS) + "\n"]
    L_ += ["\n### 1d. Ordering test (does the estimate change with the planted lag?). Pairs of planted pairs with the SAME outcome and different planted lags: hrv->anx (1) vs steps->anx (2); steps->dep (2) vs sleep_variability->dep (3). Counts of model-seeds where the estimated lags are in the planted order (<), tied (=) or reversed (>). The planted lags are clustered at 1-3, so a recency-weighted profile peaking at 1-2 earns partial exact hits without carrying lag information; the ordering test removes that advantage.", "",
           "| estimator | population | hrv<steps (anx) | hrv=steps | hrv>steps | steps<sv (dep) | steps=sv | steps>sv |", "|---|---|---|---|---|---|---|---|"]
    for est, col in (("level peak", "peak_lag_level"), ("increment peak 1-5", "peak_lag_increment_1to5"), ("onset lag", "onset_lag50")):
        for pop in ("trained", "untrained"):
            x = M[(M.kind == "pert") & (M.population == pop)].pivot_table(index=["model", "seed"], columns=["feature", "outcome"], values=col)
            a, b = x[("hrv", "anx")] - x[("steps", "anx")], x[("steps", "dep")] - x[("sleep_variability", "dep")]
            L_.append(f"| {est} | {pop} | {int((a < 0).sum())} | {int((a == 0).sum())} | {int((a > 0).sum())} | {int((b < 0).sum())} | {int((b == 0).sum())} | {int((b > 0).sum())} |")
    L_ += [""]
    # ------------------------------------------------------------------ 2. controls: all channels, per outcome
    cr = []
    for m, runs in R.items():
        for sd, r in runs.items():
            for k in ("dep", "anx"):
                for c, name in enumerate(C.CH):
                    P = r["pert"][c, :NL, HK[k], 0].astype(float)
                    D = C.increments(P)
                    cr.append(dict(model=m, seed=sd, outcome=k, channel=name, total_abs=float(np.abs(P).sum()), net=float(P.sum()), peak_lag_level=int(np.argmax(np.abs(P))),
                                   peak_lag_increment_1to5=int(1 + np.argmax(np.abs(D[1:6]))), prelag_mass_L2=float(np.abs(P[:2]).sum() / (np.abs(P).sum() + 1e-12)), spike_at_2=float(abs(D[2]) / (np.delete(np.abs(D), 2).mean() + 1e-12)),
                                   spike_at_3=float(abs(D[3]) / (np.delete(np.abs(D), 3).mean() + 1e-12)), planted=(name, k) in C.PLANTED))
    CT = pd.DataFrame(cr)
    CT.to_csv(f"{OUT}/controls.csv", index=False)
    L_ += ["\n## 2. Controls across all 8 channels (single-day profile; mean over seeds). total |P| = sum over lags 0-10 of |change in severity points per +1 SD|. Planted channels marked *. "
           "Negative controls: channels with no planted effect on that outcome. Reverse-path/confounder channels: HRV and resting HR for depression (driven by anxiety state / stress), calories (= 0.7 x steps) and circadian regularity (= -0.3 x sleep variability) are collinear correlates.\n"]
    for k in ("dep", "anx"):
        L_ += [f"\n### outcome: {k}\n", "| model | " + " | ".join(f"{'*' if (c, k) in C.PLANTED else ''}{c}" for c in C.CH) + " |", "|---|" + "---|" * 8]
        for m in MODELS:
            x = CT[(CT.model == m) & (CT.outcome == k)].groupby("channel").agg(t=("total_abs", "mean"), p=("peak_lag_increment_1to5", lambda v: int(pd.Series(v).mode().iloc[0])))
            if x.empty:
                continue
            L_.append(f"| {m} | " + " | ".join(f"{x.t[c]:.2f} (incr. peak lag {x.p[c]})" for c in C.CH) + " |")
    L_ += ["\nIncrement-peak lag is the lag in 1-5 with the largest change of the profile (mode over seeds). Planted lags: " + ", ".join(f"{f}->{k}: {L}" for (f, k), L in C.PLANTED.items()) +
           ". A channel with no planted lag still has a peak somewhere: its position and magnitude are the false-positive reference.\n"]
    # coupled control
    L_ += ["\n### Correlated-channel control: steps (resp. sleep variability) moved together with the channel the generator couples to it (calories +0.7 x steps; circadian regularity -0.3 x sleep variability)\n",
           "| model | pair | single-channel total |P| | coupled total |P| | ratio | increment-peak lag single -> coupled |", "|---|---|---|---|---|---|"]
    for m, runs in R.items():
        for ci, (f, k) in enumerate([("steps", "dep"), ("steps", "anx"), ("sleep_variability", "dep")]):
            a, b, pk = [], [], []
            for sd, r in runs.items():
                P, Q = profile(r, f, k), profile(r, f, k, "coupled")
                a.append(np.abs(P).sum())
                b.append(np.abs(Q).sum())
                pk.append((int(np.argmax(np.abs(C.increments(P)))), int(np.argmax(np.abs(C.increments(Q))))))
            L_.append(f"| {m} | {f}->{k} | {np.mean(a):.2f} | {np.mean(b):.2f} | {np.mean(b) / np.mean(a):.2f} | {pd.Series([p[0] for p in pk]).mode().iloc[0]} -> {pd.Series([p[1] for p in pk]).mode().iloc[0]} |")
    # ------------------------------------------------------------------ 3. Stage 5 comparison
    e = pd.read_csv("results/stage5_causal/effect_lag_table.csv")
    cmp_rows = []
    L_ += ["\n## 3. Comparison with the Stage 5 distributed-lag regression (increment of severity on lagged feature, lags 1-5, coefficients as multiples of the planted benchmark; "
           "V1 = observed features adjusted for other features, V2 = + stress proxy, E2 = oracle true features without stress). Model value = increment of the single-day profile at lag j, converted to simulator units (divided by standardised-units-per-u) and by the benchmark.\n",
           "| pair | planted lag | source | normalised coefficient at lags 1,2,3,4,5 | peak lag (1-5) | r with truth-increment profile |", "|---|---|---|---|---|---|"]
    for (f, k) in PAIRS:
        L, bench = C.PLANTED[(f, k)], C.benchmarks()[(f, k)]
        ti = C.increments(C.truth_level(1.0, L))[1:6]
        L_.append(f"| {f}->{k} | {L} | analytic truth | " + ", ".join(f"{v:+.2f}" for v in ti) + f" | {1 + int(np.argmax(np.abs(ti)))} | 1.00 |")
        for v in ("V1_adjusted_for_other_features", "V2_plus_stress_proxy", "E2_oracle_no_stress"):
            s = e[(e.variant == v) & (e.outcome == k) & (e.feature == f)].sort_values("lag")
            co = (s.coef.values / bench)
            L_.append(f"| {f}->{k} | {L} | Stage 5 {v.split('_')[0]} | " + ", ".join(f"{u:+.2f}" for u in co) + f" | {1 + int(np.argmax(np.abs(co)))} | {C.corr(co, ti):+.2f} |")
            cmp_rows.append(dict(pair=f"{f}->{k}", source=v, peak=1 + int(np.argmax(np.abs(co))), r_truth=C.corr(co, ti)))
        for m, runs in R.items():
            arr = np.array([C.increments(profile(r, f, k))[1:6] / C.std_per_unit(f) / bench for r in runs.values()])
            mu = arr.mean(0)
            pk = [1 + int(np.argmax(np.abs(a))) for a in arr]
            L_.append(f"| {f}->{k} | {L} | model {m} (mean of {len(arr)} seeds) | " + ", ".join(f"{u:+.2f}" for u in mu) + f" | {pk} | {C.corr(mu, ti):+.2f} |")
            cmp_rows.append(dict(pair=f"{f}->{k}", source=m, peak=str(pk), r_truth=C.corr(mu, ti)))
    pd.DataFrame(cmp_rows).to_csv(f"{OUT}/stage5_comparison.csv", index=False)
    # ------------------------------------------------------------------ 4. Integrated gradients
    ig_rows = []
    L_ += ["\n## 4. Integrated-gradients attribution (baseline = all-zero standardised window, 16 midpoint steps, 600 origins; mean |IG| by lag and channel; mean over seeds)\n",
           "| model | pair | planted lag | mean-|IG| peak lag (per seed) | pre-lag share of |IG| over lags 0-10 | planted channel's rank among 8 by total |IG| (per seed) | seed-to-seed r of the day profile | r(|IG| profile, |perturbation| profile) | mean signed IG sign ok | completeness error (pts) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, runs in R.items():
        for (f, k) in PAIRS:
            ci, hk, L = C.CH.index(f), HK[k], C.PLANTED[(f, k)]
            prof, ranks, pks, pre, sg, pr = [], [], [], [], [], []
            for sd, r in runs.items():
                ia = r["ig_abs"][hk, :11, ci].astype(float)
                prof.append(ia / (ia.sum() + 1e-12))
                tot = r["ig_abs"][hk, :, :8].sum(0)
                ranks.append(int(pd.Series(-tot).rank().iloc[ci]))
                pks.append(int(np.argmax(ia)))
                pre.append(float(ia[:L].sum() / (ia.sum() + 1e-12)))
                sg.append(float(np.sign(r["ig_signed"][hk, :, ci].sum()) == C.SIGN[(f, k)]))
                pr.append(C.corr(ia, np.abs(profile(r, f, k))))
            pw = [C.corr(prof[i], prof[j]) for i in range(len(prof)) for j in range(i + 1, len(prof))]
            comp = np.mean([float(r["ig_completeness_err"]) for r in runs.values()])
            ig_rows.append(dict(model=m, pair=f"{f}->{k}", peak_lags=pks, prelag=np.mean(pre), rank=ranks, seed_r=np.mean(pw), r_pert=np.mean(pr), sign_ok=np.mean(sg), completeness_err=comp))
            L_.append(f"| {m} | {f}->{k} | {L} | {pks} | {np.mean(pre):.2f} | {ranks} | {np.mean(pw):+.2f} | {np.mean(pr):+.2f} | {np.mean(sg):.0%} | {comp:.3f} |")
    pd.DataFrame(ig_rows).to_csv(f"{OUT}/ig_summary.csv", index=False)
    # ------------------------------------------------------------------ 5. stability across seeds and models
    L_ += ["\n## 5. Stability\n", "Seed-to-seed: mean pairwise Pearson correlation between seeds of the single-day level profile (lags 0-10), and SD of the increment-peak lag over seeds. Cross-model: correlation between the seed-mean profiles of two architectures.\n",
           "| model | pair | seed-to-seed r | increment-peak lag SD over seeds | split-half r (within seed) |", "|---|---|---|---|---|"]
    for m, runs in R.items():
        for (f, k) in PAIRS:
            ps = [profile(r, f, k) for r in runs.values()]
            pw = [C.corr(ps[i], ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps))]
            pk = [int(np.argmax(np.abs(C.increments(p)))) for p in ps]
            sh = M[(M.kind == "pert") & (M.population == "trained") & (M.model == m) & (M.feature == f) & (M.outcome == k)].split_half_r.mean()
            L_.append(f"| {m} | {f}->{k} | {np.mean(pw):+.2f} | {np.std(pk, ddof=1):.2f} | {sh:+.2f} |")
    L_ += ["\n| pair | " + " | ".join(f"{a} vs {b}" for i, a in enumerate(R) for b in list(R)[i + 1:]) + " |", "|---|" + "---|" * (len(R) * (len(R) - 1) // 2)]
    for (f, k) in PAIRS:
        mp = {m: np.mean([profile(r, f, k) for r in runs.values()], 0) for m, runs in R.items()}
        L_.append(f"| {f}->{k} | " + " | ".join(f"{C.corr(mp[a], mp[b]):+.2f}" for i, a in enumerate(R) for b in list(R)[i + 1:]) + " |")
    open(f"{OUT}/lag_recovery_tables.md", "w", encoding="utf-8").write("\n".join(L_) + "\n")

    # ------------------------------------------------------------------ figures
    lags = np.array(C.NEAR)
    for kind, fname, ttl in (("pert", "fig_level_profiles.png", "single-day +1 SD impulse"), ("ar", "fig_ar_profiles.png", "impulse persisting as the channel's AR(1)")):
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        for a, (f, k) in zip(ax.ravel(), PAIRS):
            for m, runs in R.items():
                ps = np.array([profile(r, f, k, kind) for r in runs.values()])
                a.plot(lags, ps.mean(0), color=COL[m], label=m, lw=1.8)
                a.fill_between(lags, ps.min(0), ps.max(0), color=COL[m], alpha=0.12)
            a.plot(lags, truth(f, k, rho_true[f] if kind == "ar" else None, kind), color="black", ls=":", lw=2.4, label="analytic truth (planted)")
            a.axvline(C.PLANTED[(f, k)], color="green", ls="--", lw=1, label=f"planted lag {C.PLANTED[(f, k)]}")
            a.axhline(0, color="#bbbbbb", lw=0.6)
            a.set_title(f"{f} -> {k} severity", fontsize=9)
            a.set_xlabel("lag j (days before the origin)")
            a.set_ylabel("severity points per +1 SD")
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f"SYNTHETIC data: model response profiles, {ttl} (mean over 5 seeds, band = min-max). Predictive attribution, not causal identification", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{fname}", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, (f, k) in zip(ax.ravel(), PAIRS):
        L, bench = C.PLANTED[(f, k)], C.benchmarks()[(f, k)]
        a.plot(range(1, 6), C.increments(C.truth_level(1.0, L))[1:6], color="black", ls=":", lw=2.4, marker="o", label="analytic truth")
        for v, mk, c in (("V1_adjusted_for_other_features", "s", "#e6ab02"), ("V2_plus_stress_proxy", "^", "#d95f02"), ("E2_oracle_no_stress", "D", "#1f78b4")):
            s = e[(e.variant == v) & (e.outcome == k) & (e.feature == f)].sort_values("lag")
            a.plot(s.lag, s.coef / bench, marker=mk, color=c, lw=1, ls="--", label=f"Stage 5 {v.split('_')[0]}")
        for m, runs in R.items():
            arr = np.array([C.increments(profile(r, f, k))[1:6] / C.std_per_unit(f) / bench for r in runs.values()])
            a.errorbar(np.arange(1, 6) + 0.06 * list(R).index(m), arr.mean(0), yerr=arr.std(0), color=COL[m], lw=1.4, capsize=2, label=f"model {m}")
        a.axvline(L, color="green", ls="--", lw=1)
        a.axhline(0, color="#bbbbbb", lw=0.6)
        a.set_title(f"{f} -> {k} (planted lag {L})", fontsize=9)
        a.set_xlabel("lag L")
        a.set_ylabel("increment response / benchmark")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle("SYNTHETIC data: increment (lag-specific) response, models vs Stage 5 regression vs analytic truth", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_increment_vs_stage5.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(len(R), 2, figsize=(11, 2.9 * len(R)))
    ax = np.atleast_2d(ax)
    for i, (m, runs) in enumerate(R.items()):
        for j, k in enumerate(("dep", "anx")):
            Mx = np.mean([r["ig_abs"][HK[k], :15, :8] for r in runs.values()], 0)
            Mx = Mx / Mx.max()
            a = ax[i, j]
            a.imshow(Mx, aspect="auto", cmap="viridis")
            for (f, kk), L in C.PLANTED.items():
                if kk == k:
                    a.add_patch(plt.Rectangle((C.CH.index(f) - 0.5, L - 0.5), 1, 1, fill=False, ec="red", lw=2))
            a.set_xticks(range(8))
            a.set_xticklabels(C.CH, rotation=35, ha="right", fontsize=7)
            a.set_ylabel("lag (days)")
            a.set_title(f"{m}: mean |IG|, {k} severity (red box = planted lag)", fontsize=8)
    fig.suptitle("SYNTHETIC data: integrated-gradients day x channel attribution (mean over 5 seeds, scaled to max = 1). Attribution is not causation", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ig_heatmaps.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, k in zip(ax, ("dep", "anx")):
        w = 0.8 / len(R)
        for i, m in enumerate(R):
            v = CT[(CT.model == m) & (CT.outcome == k)].groupby("channel").total_abs.mean()[C.CH]
            a.bar(np.arange(8) + i * w, v.values, w, color=COL[m], label=m)
        a.set_xticks(np.arange(8) + 0.4 - w / 2)
        a.set_xticklabels([("* " if (c, k) in C.PLANTED else "") + c for c in C.CH], rotation=35, ha="right", fontsize=8)
        a.set_title(f"{k} severity: total |response| over lags 0-10 (* planted)", fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("SYNTHETIC data: planted vs unplanted channels (negative / reverse-path / correlated-channel controls)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_controls.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for a, (f, k) in zip(ax, PAIRS):
        for i, m in enumerate(R):
            for sd, r in R[m].items():
                a.scatter(i + 0.06 * sd - 0.12, int(np.argmax(np.abs(C.increments(profile(r, f, k))))), color=COL[m], s=22)
        a.axhline(C.PLANTED[(f, k)], color="green", ls="--")
        a.set_xticks(range(len(R)))
        a.set_xticklabels(list(R), fontsize=8)
        a.set_title(f"{f} -> {k} (planted {C.PLANTED[(f, k)]})", fontsize=9)
    ax[0].set_ylabel("increment-peak lag per seed")
    fig.suptitle("SYNTHETIC data: stability of the recovered lag across seeds (dots = seeds; green = planted lag)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stability.png", dpi=130)
    plt.close(fig)
    print("analysis written to", OUT)


if __name__ == "__main__":
    os.chdir(C.ROOT)
    main()
