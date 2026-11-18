"""Stage 7 (additional post-hoc, PRE-SPECIFIED analysis): early-warning lead time before SUSTAINED onset. SYNTHETIC data only.
Post-processing of SAVED predictions: nothing is retrained, no thresholds or models are tuned on test data.

Protocol (fixed before any result was seen):
  * Score = saved onset_sust_{dep,anx}_h21 probability; window W = 21 days; origins t = 27..96 (last origin with a fully observed
    21-day outcome + 3-day confirmation). Alarm on day t  <=>  score_t > tau   (STRICT).
  * Cohort: onset participants = at risk on day 27 and sustained onset day o >= 28 (o from y_at_risk_sust: first day >= 28 with at_risk 0);
    never-onset = at risk on day 27 and no sustained onset in the record; everyone else (prevalent / onset <= day 27) EXCLUDED.
  * Threshold tau (CALIBRATION split only): m_i = max score over days 27..96 of calibration never-onset participant i; for target false-alarm
    proportion f and n such participants, k = floor(f*n), tau = (k+1)-th LARGEST m_i. Hence calibration FAP <= f (ties make it conservative).
  * Onset participant: window t in [max(27,o-21), min(o-1,96)]; lead = o - t; detected at lead >= l iff the earliest alarm in the window has
    lead >= l. Alarms earlier than 21 days before onset or after o-1 are not evaluated. Never-onset: window 27..96 (70 days).
  * False alarm proportion (FAP) = share of never-onset participants with >= 1 alarm. Episodes: alarm days separated by >= 7 non-alarm days start a
    new episode; rate per 100 participant-days; burden = alarm days / participant-days (never-onset only).
  * Participant precision = TP/(TP+FP), TP = onset participants detected (lead >= 1), FP = never-onset participants with >= 1 alarm.
    Day-level precision/recall use the existing h21 labels at the same tau.
  * Random-score null: i.i.d. U(0,1) per participant-day through the IDENTICAL functions (calibration and test scores are separate draws).
Usage: python src/leadtime.py results/stage7_leadtime
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROC = "data/processed_v1"
T0, T1, W = 27, 96, 21
LEADS = (1, 3, 7, 14, 21)
TARGETS = (0.10, 0.20)
KS = ("dep", "anx")
EP_GAP = 7
N_BOOT, BOOT_SEED = 200, 20261115
NULL_DRAWS, NULL_SEED = 5, 7000
SRC = {"gbt": ("results/stage3", [0]), "gru": ("results/stage3", range(5)), "kan": ("results/stage3", range(5)),
       "kan_reg": ("results/stage3b_kan_reg/final", range(5)), "m1": ("results/stage4_m1", range(5)),
       "m1_static": ("results/stage4_ablation", range(5)), "kangru": ("results/ext_kangru", range(5)),
       "kangru_fixed": ("results/ext_kangru", range(5))}


# ------------------------------------------------------------------ core (unit-tested)
def cohort_from_at_risk(at_risk):
    """at_risk (N,120) (NaN for t<27). Returns onset day (N,) (-1 none) and group array: 'onset' / 'never' / 'excl'."""
    N = at_risk.shape[0]
    onset = np.full(N, -1)
    group = np.array(["excl"] * N, dtype=object)
    for i in range(N):
        if at_risk[i, T0] != 1:
            continue
        z = np.where(at_risk[i, T0 + 1:] == 0)[0]
        if len(z):
            onset[i], group[i] = T0 + 1 + z[0], "onset"
        else:
            group[i] = "never"
    return onset, group


def tau_from_calibration(m_never, f):
    """(k+1)-th largest of the calibration never-onset maxima, k = floor(f*n); alarm iff score > tau."""
    m = np.sort(np.asarray(m_never, float))[::-1]
    k = int(math.floor(f * len(m) + 1e-9))
    return float(m[k])


def window_max(S, onset, group):
    """Per participant maximum score in the evaluation window (onset: pre-onset window; never: days 27..96). NaN for excluded."""
    M = np.full(len(group), np.nan)
    for i, g in enumerate(group):
        if g == "never":
            M[i] = S[i, T0:T1 + 1].max()
        elif g == "onset":
            lo, hi = max(T0, onset[i] - W), min(onset[i] - 1, T1)
            M[i] = S[i, lo:hi + 1].max()
    return M


def participant_stats(S, tau, onset, group):
    """Per participant: first_lead (onset; 0 = no alarm in window), n_days / n_ep (never-onset alarm days / episodes)."""
    N = len(group)
    first_lead = np.zeros(N, int)
    n_days = np.zeros(N, int)
    n_ep = np.zeros(N, int)
    for i, g in enumerate(group):
        if g == "onset":
            lo, hi = max(T0, onset[i] - W), min(onset[i] - 1, T1)
            a = np.where(S[i, lo:hi + 1] > tau)[0]
            first_lead[i] = onset[i] - (lo + a[0]) if len(a) else 0
        elif g == "never":
            a = np.where(S[i, T0:T1 + 1] > tau)[0]
            n_days[i] = len(a)
            n_ep[i] = 0 if not len(a) else 1 + int((np.diff(a) - 1 >= EP_GAP).sum())
    return dict(group=group, onset=onset, first_lead=first_lead, n_days=n_days, n_ep=n_ep)


def achievable(o, l):
    """Is an alarm with lead >= l possible inside the window for onset day o?"""
    lo, hi = max(T0, o - W), min(o - 1, T1)
    return lo <= min(hi, o - l)


def stat_metrics(st, idx=None):
    g, fl, nd, ne, on = st["group"], st["first_lead"], st["n_days"], st["n_ep"], st["onset"]
    if idx is not None:
        g, fl, nd, ne, on = g[idx], fl[idx], nd[idx], ne[idx], on[idx]
    onm, nvm = g == "onset", g == "never"
    n_on, n_nv = int(onm.sum()), int(nvm.sum())
    out = {"n_onset": n_on, "n_never": n_nv}
    for l in LEADS:
        out[f"det_lead{l}"] = float((fl[onm] >= l).mean()) if n_on else np.nan
        ach = np.array([achievable(o, l) for o in on[onm]], bool)
        out[f"n_achievable_lead{l}"] = int(ach.sum())
        out[f"det_lead{l}_achievable"] = float((fl[onm][ach] >= l).mean()) if ach.sum() else np.nan
    det = fl[onm] >= 1
    out["median_earliest_lead_detected"] = float(np.median(fl[onm][det])) if det.any() else np.nan
    fp = int((nd[nvm] > 0).sum())
    out["fap"] = fp / n_nv if n_nv else np.nan
    out["episodes_per_100_participant_days"] = float(ne[nvm].sum()) / (n_nv * (T1 - T0 + 1)) * 100 if n_nv else np.nan
    out["alarm_burden"] = float(nd[nvm].sum()) / (n_nv * (T1 - T0 + 1)) if n_nv else np.nan
    tp = int(det.sum())
    out["precision_participant"] = tp / (tp + fp) if tp + fp else np.nan
    out["recall_participant"] = out["det_lead1"]
    return out


def day_level(S, tau, Y):
    """Y: existing h21 labels (N,120), NaN = undefined. Precision/recall at tau over defined origins within days 27..96."""
    ok = ~np.isnan(Y)
    ok[:, :T0] = False
    ok[:, T1 + 1:] = False
    a, y = S[ok] > tau, Y[ok] == 1
    tp = int((a & y).sum())
    return {"day_precision": tp / a.sum() if a.sum() else np.nan, "day_recall": tp / y.sum() if y.sum() else np.nan,
            "day_n_origins": int(ok.sum()), "day_n_pos": int(y.sum())}


def participant_auc(S, onset, group):
    """Area under the participant-level detection-vs-FAP curve = P(M_onset > m_never) (+0.5 ties)."""
    M = window_max(S, onset, group)
    keep = group != "excl"
    return float(roc_auc_score(group[keep] == "onset", M[keep])), M


# ------------------------------------------------------------------ data
def load_cohort(split, k):
    d = np.load(os.path.join(PROC, f"{split}.npz"))
    onset, group = cohort_from_at_risk(d[f"y_at_risk_sust_{k}"])
    return dict(pid=d["pid"], onset=onset, group=group, Y=d[f"y_onset_sust_{k}_h21"])


def load_scores(dirn, name, seed, split, k, pid):
    z = np.load(os.path.join(dirn, "predictions", f"{name}_seed{seed}_{split}.npz"))
    assert np.array_equal(z["pid"], pid), "participant order mismatch"
    j = list(z["heads"]).index(f"onset_sust_{k}_h21")
    S = z["pred"][:, :, j].astype(float)
    assert np.isfinite(S[:, T0:T1 + 1]).all()
    return S


def null_scores(draw, shape=(200, 120)):
    rng = np.random.default_rng(NULL_SEED + draw)
    return rng.random(shape), rng.random(shape)           # (calibration, test): separate draws, fixed seed


def run_one(Scal, Stest, cal, test, f):
    """The ONE evaluation path used by every model and by the null. tau is a function of CALIBRATION data only."""
    m = window_max(Scal, cal["onset"], cal["group"])
    tau = tau_from_calibration(m[cal["group"] == "never"], f)
    st = participant_stats(Stest, tau, test["onset"], test["group"])
    cal_st = participant_stats(Scal, tau, cal["onset"], cal["group"])
    res = stat_metrics(st)
    res.update(day_level(Stest, tau, test["Y"]))
    res["tau"], res["cal_fap_realised"] = tau, stat_metrics(cal_st)["fap"]
    return res, st


# ------------------------------------------------------------------ driver
def main(out):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    rows, leads, curves, bstats = [], [], {}, {}
    for k in KS:
        cal, test = load_cohort("cal", k), load_cohort("test", k)
        runs = [(name, seed, lambda n=name, s=seed, d=d: (load_scores(d, n, s, "cal", k, cal["pid"]), load_scores(d, n, s, "test", k, test["pid"])))
                for name, (d, seeds) in SRC.items() for seed in seeds]
        runs += [("random_null", s, lambda s=s: null_scores(s)) for s in range(NULL_DRAWS)]
        for name, seed, get in runs:
            Scal, Stest = get()
            auc, M = participant_auc(Stest, test["onset"], test["group"])
            keep = test["group"] != "excl"
            fpr, tpr, _ = roc_curve(test["group"][keep] == "onset", M[keep])
            if seed == 0:
                curves[(k, name)] = (fpr, tpr)
            for f in TARGETS:
                res, st = run_one(Scal, Stest, cal, test, f)
                res.update(model=name, seed=seed, target=k, target_fap=f, auc_participant=auc)
                rows.append(res)
                if seed == 0:
                    bstats[(k, name, f)] = st
                for i in np.where(st["group"] == "onset")[0]:
                    leads.append(dict(model=name, seed=seed, target=k, target_fap=f, pid=int(test["pid"][i]), onset_day=int(st["onset"][i]),
                                      first_alarm_lead=int(st["first_lead"][i]) or None))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metrics_per_seed.csv"), index=False)
    pd.DataFrame(leads).to_csv(os.path.join(out, "onset_participant_leads.csv"), index=False)

    # participant bootstrap (first seed of each model / first null draw), tau fixed; paired differences vs GRU
    rng = np.random.default_rng(BOOT_SEED)
    brow = []
    for k in KS:
        n_test = len(bstats[(k, "gru", TARGETS[0])]["group"])
        draws = [rng.integers(0, n_test, n_test) for _ in range(N_BOOT)]
        for f in TARGETS:
            vals = {name: [stat_metrics(bstats[(k, name, f)], ix) for ix in draws] for name in list(SRC) + ["random_null"]}
            for name in vals:
                for met in ("det_lead1", "det_lead7", "fap", "precision_participant"):
                    v = np.array([r[met] for r in vals[name]], float)
                    point = df[(df.model == name) & (df.seed == 0) & (df.target == k) & (df.target_fap == f)][met].iloc[0]
                    row = dict(model=name, target=k, target_fap=f, metric=met, point=point, ci95_lo=np.nanpercentile(v, 2.5), ci95_hi=np.nanpercentile(v, 97.5))
                    if name != "gru":
                        d = v - np.array([r[met] for r in vals["gru"]], float)
                        row.update(diff_vs_gru_lo=np.nanpercentile(d, 2.5), diff_vs_gru_hi=np.nanpercentile(d, 97.5))
                    brow.append(row)
    pd.DataFrame(brow).to_csv(os.path.join(out, "bootstrap_ci_seed0.csv"), index=False)
    write_tables(df, out)
    write_figures(df, curves, out)
    json.dump({"SYNTHETIC": True, "post_hoc_pre_specified": True, "window_W": W, "origins": [T0, T1], "targets": TARGETS, "leads": LEADS, "episode_gap": EP_GAP,
               "tau_rule": "(floor(f*n)+1)-th largest calibration never-onset window max; alarm iff score > tau", "alarm_rule": "score > tau",
               "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "null": {"draws": NULL_DRAWS, "seed_base": NULL_SEED, "dist": "U(0,1) iid per participant-day"},
               "sources": {k: [v[0], list(v[1])] for k, v in SRC.items()}}, open(os.path.join(out, "config.json"), "w"), indent=1)
    print("done ->", out)


ORDER = ["gbt", "gru", "kan", "kan_reg", "m1_static", "m1", "kangru_fixed", "kangru", "random_null"]
COLS = ["fap", "cal_fap_realised", "det_lead1", "det_lead3", "det_lead7", "det_lead14", "det_lead21", "median_earliest_lead_detected",
        "precision_participant", "day_precision", "day_recall", "episodes_per_100_participant_days", "alarm_burden", "auc_participant"]


def write_tables(df, out):
    L = ["# Stage 7 (additional, post-hoc, pre-specified): early-warning lead time. SYNTHETIC data; test predictions were already evaluated in earlier stages.",
         "Cells: mean +-SD over seeds (gbt: one deterministic run; random_null: 5 random draws). NO model-selection or superiority claim is made; see INTERPRETATION.md.\n"]
    P = ["# Per-seed results (every seed / draw separately)\n"]
    for k in KS:
        for f in TARGETS:
            sub = df[(df.target == k) & (df.target_fap == f)]
            r0 = sub.iloc[0]
            L.append(f"\n## {k.upper()} | calibration-set target false-alarm proportion {f:.0%} | test: {int(r0.n_onset)} onset participants, {int(r0.n_never)} never-onset\n")
            L.append("| model | n seeds | " + " | ".join(COLS) + " |\n|---|---|" + "---|" * len(COLS))
            for m in ORDER:
                g = sub[sub.model == m]
                L.append(f"| {m} | {len(g)} | " + " | ".join(f"{g[c].mean():.3f}" + (f" +-{g[c].std():.3f}" if len(g) > 1 else "") for c in COLS) + " |")
            P.append(f"\n## {k.upper()} | target FAP {f:.0%}\n")
            P.append("| model | seed | tau | " + " | ".join(COLS) + " |\n|---|---|---|" + "---|" * len(COLS))
            for m in ORDER:
                for _, r in sub[sub.model == m].iterrows():
                    P.append(f"| {r.model} | {int(r.seed)} | {r.tau:.4f} | " + " | ".join(f"{r[c]:.3f}" for c in COLS) + " |")
    L.append("\n## Achievable-lead denominators (GRU seed-mean; target FAP 10%; test)\n")
    L.append("| target | l | n onset participants for whom lead >= l is possible (of all) | detection rate: all onset vs achievable-only |\n|---|---|---|---|")
    for k in KS:
        g = df[(df.target == k) & (df.target_fap == TARGETS[0]) & (df.model == "gru")]
        for l in LEADS:
            L.append(f"| {k} | {l} | {int(g[f'n_achievable_lead{l}'].iloc[0])} of {int(g.n_onset.iloc[0])} | {g[f'det_lead{l}'].mean():.3f} vs {g[f'det_lead{l}_achievable'].mean():.3f} |")
    open(os.path.join(out, "summary_tables.md"), "w", encoding="utf-8").write("\n".join(L))
    open(os.path.join(out, "per_seed_tables.md"), "w", encoding="utf-8").write("\n".join(P))


def write_figures(df, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = dict(zip(ORDER, ["#8c564b", "#1f77b4", "#2ca02c", "#98df8a", "#ff7f0e", "#d62728", "#9467bd", "#e377c2", "#7f7f7f"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, k in zip(ax, KS):
        for m in ORDER:
            fpr, tpr = curves[(k, m)]
            a.plot(fpr, tpr, color=cols[m], ls="--" if m == "random_null" else "-", lw=1.4, label=m)
        a.set(title=f"{k}: detection vs false-alarm proportion (test, seed 0)", xlabel="false-alarm proportion (never-onset)", ylabel="detection (lead >= 1)")
    ax[0].legend(fontsize=7)
    fig.suptitle("SYNTHETIC data; 32 onset participants per target", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig1_detection_vs_far.png"), dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for i, k in enumerate(KS):
        for j, f in enumerate(TARGETS):
            sub = df[(df.target == k) & (df.target_fap == f)]
            for m in ORDER:
                g = sub[sub.model == m]
                ax[i, j].plot(LEADS, [g[f"det_lead{l}"].mean() for l in LEADS], marker="o", ms=3, color=cols[m], ls="--" if m == "random_null" else "-", label=m)
            ax[i, j].set(title=f"{k} | target FAP {f:.0%}", xlabel="lead time l (days before onset)", ylabel="detection rate (alarm with lead >= l)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Detection vs lead time (mean over seeds, test; SYNTHETIC; all onset participants as denominator)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig2_detection_vs_lead.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    outdir = sys.argv[1]
    if os.path.isdir(outdir) and os.listdir(outdir):
        raise SystemExit(f"{outdir} exists and is not empty; refusing to overwrite.")
    main(outdir)
