"""Stage 10 / P0-4: oracle ceiling - how much headroom do the observable-data models leave? SYNTHETIC. ANALYSIS-ONLY: true simulator quantities (truth.csv) are NEVER model inputs
elsewhere; here they are used post hoc to estimate an upper bound on prediction quality.
Information rule: the features for origin t use ONLY truth values of days <= t (tested by perturbing days > t) plus time-invariant person parameters.
  Tier A "state oracle": true latent z_dep, z_anx at t, differences over 1/3/7 days, 7-day mean, current status flag (z above the frozen threshold) and run length, sex, age group.
  Tier B "full-truth oracle": Tier A + true stress (t..t-3) + noise-free features u (8 channels, lags 0..3) + person parameters (m_dep, m_anx, sens_sv, sens_steps, sens_hrv, sens_steps_anx).
  NEVER used: onset_day_*, prevalent_* (outcome-derived), any label or label-derived column, any information from days > t.
Predictor: sklearn HistGradientBoosting (deterministic, random_state 0), trained on TRAIN participants' eligible origins (same rows as the models: label defined, t>=27),
n_iter in {100, 300} chosen on the CALIBRATION split (MSE / log-loss); TEST used only for reporting. 22 heads: severity, status (t0,h7,h14,h21 x dep,anx), sustained onset (h7,h14,h21).
Residual randomness (future noise, future drift episodes, severity noise sd 1.0, not-yet-drawn shocks) is irreducible even for these oracles, so they bound, not equal, the Bayes limit
(a boosted-tree fit on ~56k rows can also under-reach it): the comparison is a headroom ESTIMATE.
Lead time: oracle h21 onset scores through src/leadtime.py with calibration-split thresholds, same protocol as Stage 7.
Usage: python extension_stage10/p04_oracle.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import common as C  # noqa: E402

OUT = "results/stage10_oracle"
HEADS = C.SEV + C.STATUS + C.ONSET
GRID = [100, 300]
U = ["steps", "calories", "sleep_duration", "sleep_efficiency", "sleep_variability", "resting_hr", "hrv", "circadian_regularity"]
TRAITS = ["m_dep", "m_anx", "sens_sv", "sens_steps", "sens_hrv", "sens_steps_anx"]
BOOT_B, BOOT_SEED = 100, 20261117


# ------------------------------------------------------------------ features (unit-tested for causality)
def shift(a, k):
    """value at day t-k placed at day t (NaN for t<k). a: (N, T, ...)."""
    out = np.full_like(a, np.nan, dtype=float)
    if k == 0:
        return a.astype(float)
    out[:, k:] = a[:, :-k]
    return out


def run_length(flag, cap=14):
    """consecutive days up to and including t with flag==1 (capped)."""
    N, T = flag.shape
    out, prev = np.zeros((N, T)), np.zeros(N)
    for t in range(T):
        prev = (prev + 1) * flag[:, t]
        out[:, t] = prev
    return np.minimum(out, cap)


def build_features(tr, static, tier, thr):
    """tr: dict z_dep, z_anx, stress (N,T); u (N,T,8); traits (N,6). Returns (N, T, F) using days <= t only."""
    z = np.stack([tr["z_dep"], tr["z_anx"]], -1)                       # (N, T, 2)
    N, T, _ = z.shape
    f = [z, z - shift(z, 1), z - shift(z, 3), z - shift(z, 7)]
    f.append(np.mean(np.stack([shift(z, k) for k in range(7)]), 0))            # trailing 7-day mean (days t-6..t); NaN for t<6
    flags = (z > np.array(thr)[None, None, :]).astype(float)
    f += [flags, np.stack([run_length(flags[:, :, 0]), run_length(flags[:, :, 1])], -1)]
    f.append(np.repeat(static[:, None, :], T, 1))
    if tier == "B":
        f += [shift(tr["stress"][:, :, None], k) for k in range(4)]
        f += [shift(tr["u"], k) for k in range(4)]
        f.append(np.repeat(tr["traits"][:, None, :], T, 1))
    return np.concatenate(f, -1)


def load_truth(split_pid):
    t = pd.read_csv("data/synthetic_v1/truth.csv")
    tp = pd.read_csv("data/synthetic_v1/truth_participants.csv").set_index("pid")
    N, T = len(split_pid), 120
    t = t.set_index(["pid", "day"])
    idx = pd.MultiIndex.from_product([split_pid, range(T)], names=["pid", "day"])
    t = t.loc[idx]
    arr = lambda c: t[c].values.reshape(N, T)  # noqa: E731
    return dict(z_dep=arr("z_dep"), z_anx=arr("z_anx"), stress=arr("stress"), u=np.stack([arr("u_" + c) for c in U], -1), traits=tp.loc[split_pid, TRAITS].values.astype(float))


def fit_predict(head, Xtr, ytr, Xca, yca, Xall):
    """grid over n_iter chosen on calibration; returns predictions for every row of Xall (one model per split list) and chosen n_iter."""
    best = None
    for it in GRID:
        if head in C.SEV:
            m = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
            score = float(np.mean((np.clip(m.predict(Xca), 0, 27) - yca) ** 2))
        else:
            m = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
            score = float(log_loss(yca, np.clip(m.predict_proba(Xca)[:, 1], 1e-6, 1 - 1e-6)))
        if best is None or score < best[0]:
            best = (score, it, m)
    _, it, m = best
    pred = [np.clip(m.predict(X), 0, 27) if head in C.SEV else m.predict_proba(X)[:, 1] for X in Xall]
    return pred, it


def main():
    os.chdir(C.ROOT)
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    cfg = json.load(open("configs/frozen_config_v1.json"))
    thr = (cfg["z_thr_dep"], cfg["z_thr_anx"])
    D = {s: C.load_split(s) for s in ("train", "cal", "test")}
    F = {}
    for tier in ("A", "B"):
        for s in D:
            F[(tier, s)] = build_features(load_truth(D[s]["pid"]), D[s]["static"].astype(float), tier, thr)
    rows, preds = [], {(t, s): np.full((len(D[s]["pid"]), 120, len(HEADS)), np.nan, np.float32) for t in ("A", "B") for s in ("cal", "test")}
    for hi_, head in enumerate(HEADS):
        sel = {s: C.eligible(D[s][f"y_{head}"]) for s in D}
        y = {s: D[s][f"y_{head}"][sel[s]].astype(float) for s in D}
        for tier in ("A", "B"):
            X = {s: F[(tier, s)][sel[s]] for s in D}
            # prediction for every day >= 27 of cal/test (lead-time needs scores at origins without labels too)
            Xfull = {s: F[(tier, s)][:, 27:].reshape(-1, F[(tier, s)].shape[-1]) for s in ("cal", "test")}
            (pc, pt), it = fit_predict(head, X["train"], y["train"], X["cal"], y["cal"], [Xfull["cal"], Xfull["test"]])
            for s, p in (("cal", pc), ("test", pt)):
                preds[(tier, s)][:, 27:, hi_] = p.reshape(len(D[s]["pid"]), 93)
            p_test = preds[(tier, "test")][:, :, hi_][sel["test"]].astype(float)
            st = {"rmse": float(np.sqrt(np.mean((p_test - y["test"]) ** 2)))} if head in C.SEV else C.clf_summary(y["test"], p_test)
            rows.append(dict(model=f"oracle_{tier}", head=head, n_iter=it, n_test_rows=len(y["test"]), **st))
        print("done", head, flush=True)
    for (tier, s), a in preds.items():
        np.savez_compressed(os.path.join(OUT, "predictions", f"oracle_{tier}_{s}.npz"), pred=a, pid=D[s]["pid"], heads=np.array(HEADS))
    # reference models from saved predictions (mean over seeds)
    ref = []
    for model in ("gru", "gbt", "m1"):
        for head in HEADS:
            Y = D["test"][f"y_{head}"]
            pi, ti = C.eligible(Y)
            vals = []
            for seed in C.seeds_of(model):
                P, _ = C.load_pred(model, head, seed, "test")
                y, p = Y[pi, ti].astype(float), P[pi, ti]
                vals.append({"rmse": float(np.sqrt(np.mean((p - y) ** 2)))} if head in C.SEV else C.clf_summary(y, p))
            ref.append(dict(model=model, head=head, n_test_rows=len(pi), **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}))
    allr = pd.DataFrame(rows + ref)
    allr.to_csv(os.path.join(OUT, "oracle_vs_models_test.csv"), index=False)
    # trivial references
    triv = []
    for head in HEADS:
        ytr = D["train"][f"y_{head}"][C.eligible(D["train"][f"y_{head}"])].astype(float)
        yte = D["test"][f"y_{head}"][C.eligible(D["test"][f"y_{head}"])].astype(float)
        triv.append(dict(head=head, null_rmse=float(np.sqrt(np.mean((yte - ytr.mean()) ** 2))) if head in C.SEV else np.nan, null_ap=float(yte.mean()) if head not in C.SEV else np.nan))
    pd.DataFrame(triv).to_csv(os.path.join(OUT, "null_references_test.csv"), index=False)
    # headroom table
    hr = []
    tv = pd.DataFrame(triv).set_index("head")
    for head in HEADS:
        g = allr[(allr["head"] == head)].set_index("model")
        if head in C.SEV:
            null, o, gr = tv.loc[head, "null_rmse"], g.loc["oracle_B", "rmse"], g.loc["gru", "rmse"]
            hr.append(dict(head=head, metric="rmse", null=null, gru=gr, gbt=g.loc["gbt", "rmse"], m1=g.loc["m1", "rmse"], oracle_A=g.loc["oracle_A", "rmse"], oracle_B=o,
                           share_of_null_to_oracleB_gap_closed_by_gru=(null - gr) / (null - o)))
        else:
            for met in ("ap", "auroc"):
                null = tv.loc[head, "null_ap"] if met == "ap" else 0.5
                o, gr = g.loc["oracle_B", met], g.loc["gru", met]
                hr.append(dict(head=head, metric=met, null=null, gru=gr, gbt=g.loc["gbt", met], m1=g.loc["m1", met], oracle_A=g.loc["oracle_A", met], oracle_B=o,
                               share_of_null_to_oracleB_gap_closed_by_gru=(gr - null) / (o - null)))
    pd.DataFrame(hr).to_csv(os.path.join(OUT, "headroom_summary.csv"), index=False)
    # participant bootstrap of oracle_B - GRU(seed 0)
    rng = np.random.default_rng(BOOT_SEED)
    brow = []
    for head in HEADS:
        Y = D["test"][f"y_{head}"]
        pi, ti = C.eligible(Y)
        y = Y[pi, ti].astype(float)
        pg = C.load_pred("gru", head, 0, "test")[0][pi, ti]
        po = preds[("B", "test")][:, :, HEADS.index(head)][pi, ti].astype(float)
        byp = {i: np.where(pi == i)[0] for i in np.unique(pi)}
        ids = np.array(list(byp))
        fns = {"rmse": lambda a, b: float(np.sqrt(np.mean((b - a) ** 2)))} if head in C.SEV else {
            "ap": lambda a, b: float(average_precision_score(a, b)) if 0 < a.sum() < len(a) else np.nan,
            "auroc": lambda a, b: float(roc_auc_score(a, b)) if 0 < a.sum() < len(a) else np.nan}
        for name, fn in fns.items():
            d = []
            for _ in range(BOOT_B):
                ix = np.concatenate([byp[i] for i in rng.choice(ids, len(ids))])
                d.append(fn(y[ix], po[ix]) - fn(y[ix], pg[ix]))
            lo, hi = C.boot_ci(d)
            brow.append(dict(head=head, metric=name, oracleB_minus_gru_seed0=fn(y, po) - fn(y, pg), ci95_lo=lo, ci95_hi=hi))
    pd.DataFrame(brow).to_csv(os.path.join(OUT, "oracle_minus_gru_bootstrap.csv"), index=False)
    # lead time (Stage 7 protocol)
    import leadtime as lt
    lrows = []
    for k in lt.KS:
        cal, test = lt.load_cohort("cal", k), lt.load_cohort("test", k)
        j = HEADS.index(f"onset_sust_{k}_h21")
        for tier in ("A", "B"):
            Scal, Stest = preds[(tier, "cal")][:, :, j].astype(float), preds[(tier, "test")][:, :, j].astype(float)
            for f in lt.TARGETS:
                res, _ = lt.run_one(Scal, Stest, cal, test, f)
                lrows.append(dict(model=f"oracle_{tier}", target=k, target_fap=f, **{c: res[c] for c in ("n_onset", "n_never", "det_lead1", "det_lead3", "det_lead7", "det_lead14", "det_lead21", "fap",
                                                                                                 "precision_participant", "median_earliest_lead_detected")}))
    ld = pd.DataFrame(lrows)
    s7 = pd.read_csv("results/stage7_leadtime/metrics_per_seed.csv")
    s7 = s7[s7.model.isin(["gru", "gbt", "m1", "random_null"])].groupby(["model", "target", "target_fap"]).mean(numeric_only=True).reset_index()
    pd.concat([ld, s7[ld.columns.intersection(s7.columns)]], ignore_index=True).to_csv(os.path.join(OUT, "leadtime_oracle_vs_models.csv"), index=False)
    json.dump({"SYNTHETIC": True, "analysis_only": True, "tiers": {"A": "true latent state + dynamics + status run length + static", "B": "A + true stress, noise-free features (lags 0-3), person parameters"},
               "excluded_columns": ["onset_day_*", "onset_day_sust_*", "prevalent_*", "any label"], "model": "HistGradientBoosting lr 0.05, 15 leaves, min_leaf 50, l2 1, n_iter chosen on calibration from %s" % GRID,
               "heads": HEADS, "bootstrap": dict(B=BOOT_B, seed=BOOT_SEED, unit="participant"), "split_use": "train fit; calibration selection and thresholds; test report only"},
              open(os.path.join(OUT, "config.json"), "w"), indent=1)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
