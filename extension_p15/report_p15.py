"""P1-5 report (SYNTHETIC; EXTENSION): tables and figures from results/p1_early_warning/{sust,cross}/*.csv (read-only). Writes results/p1_early_warning/analysis/*.md|png.
All numbers describe a simulator's latent threshold, not clinical warning performance. Intervals: bootstrap = stratified participant bootstrap (thresholds fixed, conditional on the trained
seeds); the shuffle-null range is a NULL REFERENCE RANGE (2.5-97.5 percentile of 200 trajectory shuffles), not a confidence interval of the model.
Usage: python extension_p15/report_p15.py"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ew15 as ew  # noqa: E402

BASE = "results/p1_early_warning"
OUT = f"{BASE}/analysis"
MAIN = ["gru", "transformer", "cnnlstm", "m1", "mlp_m1", "kangru"]
OTHER = ["kan", "kan_reg", "mlp_b2", "mlp_b2_reg", "kan_nospline", "m1_static", "kangru_fixed", "gbt", "st_task", "st_task_matched", "family"]
COL = {"gru": "#444444", "transformer": "#1f78b4", "cnnlstm": "#e6ab02", "m1": "#c51b7d", "mlp_m1": "#1b9e77", "kangru": "#7570b3"}


def fmt(p, lo, hi, d=2):
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if np.isfinite(p) else "n/a"


def load(kind):
    return (pd.read_csv(f"{BASE}/{kind}/per_seed_metrics.csv"), pd.read_csv(f"{BASE}/{kind}/bootstrap_ci.csv"), pd.read_csv(f"{BASE}/{kind}/shuffle_null.csv"))


def main():
    os.chdir(ew.ROOT)
    os.makedirs(OUT, exist_ok=True)
    P = {kind: load(kind) for kind in ("sust", "cross")}
    # ------------------------------------------------------------------ 1. primary contrast
    per, bt, nl = P["sust"]
    L = ["# P1-5 PRIMARY contrast (SYNTHETIC; EXTENSION): sustained onset, h21 (W = 21), 20% false-alarm target, detection at lead >= 7 days\n",
         "Overall rate = share of ALL evaluable onset participants alarmed >= 7 days before onset (primary); achievable rate = among participants for whom a lead of 7 days was possible inside their window. "
         "Values are seed means; [ ] = stratified participant bootstrap 95% interval (B = 1000, thresholds fixed, conditional on the trained seeds). Difference = model minus GRU, same resampled participants. "
         "Null columns: shuffle null (complete score trajectories permuted across participants, 200 shared permutations; mean [2.5-97.5 percentile range], not a CI) and the iid U(0,1) null of Stage 7.\n"]
    for k in ew.KS:
        sel = per[(per.outcome == k) & (per.h == 21) & (per.target_fap == 0.2)]
        n_on, n_nv = int(sel.n_onset.iloc[0]), int(sel.n_never.iloc[0])
        L += [f"\n## {k}: {n_on} onset participants, {n_nv} never-onset participants (test); lead 7 achievable for {int(sel[sel.model == 'gru'].n_achievable_lead7.iloc[0])} of {n_on}\n",
              "| model | overall det >= 7 d | seed SD | achievable-lead det >= 7 d | diff vs GRU (overall) | resamples better than GRU | shuffle null | detection >= 1 d | FAP (target 0.20) |", "|---|---|---|---|---|---|---|---|---|"]
        for m in MAIN + OTHER + ["random_null_iid"]:
            b = bt[(bt.outcome == k) & (bt.h == 21) & (bt.target_fap == 0.2) & (bt.model == m)]
            if b.empty:
                continue
            g = lambda met: b[b.metric == met].iloc[0]      # noqa: E731
            d7, da, d1, fp = g("det_lead7"), g("det_lead7_achievable"), g("det_lead1"), g("fap")
            sd = sel[sel.model == m].det_lead7.std() if (sel.model == m).sum() > 1 else np.nan
            n7 = nl[(nl.outcome == k) & (nl.h == 21) & (nl.target_fap == 0.2) & (nl.model == m) & (nl.metric == "det_lead7")]
            nulltxt = f"{n7.null_mean.mean():.2f} [{n7.null_lo.mean():.2f}, {n7.null_hi.mean():.2f}]; obs >= null in {1 - n7.frac_null_ge_observed.mean():.0%} of shuffles" if len(n7) else "n/a"
            diff = fmt(d7.get("diff_vs_gru_point", np.nan), d7.get("diff_vs_gru_lo", np.nan), d7.get("diff_vs_gru_hi", np.nan)) if m != "gru" else "(reference)"
            fb = f"{d7.frac_resamples_better_than_gru:.0%}" if m != "gru" and "frac_resamples_better_than_gru" in d7 and np.isfinite(d7.get("frac_resamples_better_than_gru", np.nan)) else ""
            L.append(f"| {m}{' *(secondary group)*' if m in OTHER else ''} | {fmt(d7.point_seed_mean, d7.ci95_lo, d7.ci95_hi)} | {sd:.2f} | {fmt(da.point_seed_mean, da.ci95_lo, da.ci95_hi)} | {diff} | {fb} | {nulltxt} | "
                     f"{fmt(d1.point_seed_mean, d1.ci95_lo, d1.ci95_hi)} | {fmt(fp.point_seed_mean, fp.ci95_lo, fp.ci95_hi)} |")
    L += ["\nOnly this contrast is primary; everything else in the other files is descriptive. 'Resamples better than GRU' is the share of bootstrap resamples with a positive difference; it is not a p-value. "
          "With 32 onset participants each participant moves a rate by 3.1 points, and several models are compared with the GRU, so intervals that exclude 0 should be read as indicative only.\n"]
    open(f"{OUT}/primary_contrast.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ------------------------------------------------------------------ 2. full tables
    for kind, title in (("sust", "PRIMARY analysis: sustained onset"), ("cross", "SECONDARY analysis (kept separate): first-crossing onset (models with first-crossing heads only)")):
        per, bt, nl = P[kind]
        T = [f"# P1-5 {title} (SYNTHETIC; EXTENSION)\n", "Seed means with stratified participant bootstrap 95% intervals (thresholds fixed). Rates are over evaluable onset participants (detection) or never-onset participants (FAP). "
             "Lead cells longer than the window (W = h) are undefined and omitted. FAP target = calibration target; the realised test FAP is shown. Burden = alarm days / never-onset participant-days; episodes per 100 participant-days.\n"]
        for h in ew.HORIZONS:
            for k in ew.KS:
                for f in ew.TARGETS:
                    sel = per[(per.outcome == k) & (per.h == h) & (per.target_fap == f)]
                    if sel.empty:
                        continue
                    leads = [l for l in ew.LEADS if l <= h]
                    T += [f"\n## {k}, h{h} (W = {h}, origins 27-{ew.t1_of(kind, h)}), false-alarm target {f:.0%}: {int(sel.n_onset.iloc[0])} onset / {int(sel.n_never.iloc[0])} never-onset participants\n",
                          "| model | " + " | ".join(f"det >= {l} d (overall)" for l in leads) + " | det >= 7 d among achievable (n)" * (h >= 7) + " | realised FAP | burden % | episodes /100 pd | participant precision | participant AUC (null) | day precision / recall |",
                          "|---|" + "---|" * (len(leads) + 6 + (h >= 7))]
                    for m in MAIN + OTHER + ["random_null_iid"]:
                        b = bt[(bt.outcome == k) & (bt.h == h) & (bt.target_fap == f) & (bt.model == m)]
                        s = sel[sel.model == m]
                        if b.empty:
                            continue
                        g = lambda met: b[b.metric == met].iloc[0]      # noqa: E731
                        cells = [fmt(g(f"det_lead{l}").point_seed_mean, g(f"det_lead{l}").ci95_lo, g(f"det_lead{l}").ci95_hi) for l in leads]
                        ach = fmt(g("det_lead7_achievable").point_seed_mean, g("det_lead7_achievable").ci95_lo, g("det_lead7_achievable").ci95_hi) + f" ({int(s.n_achievable_lead7.iloc[0])})" if h >= 7 else ""
                        nauc = nl[(nl.outcome == k) & (nl.h == h) & (nl.target_fap == f) & (nl.model == m) & (nl.metric == "auc_participant")]
                        aucn = f"{s.auc_participant.mean():.2f} ({nauc.null_mean.mean():.2f})" if len(nauc) else f"{s.auc_participant.mean():.2f}"
                        T.append(f"| {m} | " + " | ".join(cells) + (f" | {ach}" if h >= 7 else "") + f" | {fmt(g('fap').point_seed_mean, g('fap').ci95_lo, g('fap').ci95_hi)} | {100 * g('alarm_burden').point_seed_mean:.1f} | "
                                 f"{g('episodes_per_100_participant_days').point_seed_mean:.2f} | {fmt(g('precision_participant').point_seed_mean, g('precision_participant').ci95_lo, g('precision_participant').ci95_hi)} | {aucn} | "
                                 f"{s.day_precision.mean():.2f} / {s.day_recall.mean():.2f} |")
        open(f"{OUT}/tables_{kind}.md", "w", encoding="utf-8").write("\n".join(T) + "\n")

    # ------------------------------------------------------------------ 3. null comparison (shuffle) and exposure-matched
    per, bt, nl = P["sust"]
    N = ["# P1-5 null comparison: observed vs participant-trajectory shuffle null (SYNTHETIC; EXTENSION)\n", "Each model's real test scores vs the same model's score trajectories assigned to randomly chosen other participants (200 shared permutations; calendar alignment and temporal ordering kept; "
         "thresholds from the real calibration scores). Null range = 2.5-97.5 percentile of the shuffles (reference range, not a CI); 'obs above null' = share of shuffles strictly below the observed value (seed mean). "
         "Metrics that depend only on the alarm rule (FAP, burden) are expected to sit near the target under the null and are shown for context.\n"]
    for h in ew.HORIZONS:
        for k in ew.KS:
            N += [f"\n## {k}, h{h}, 20% target\n", "| model | det >= 1 d: observed | null mean [range] | obs above null | det >= 7 d: observed | null | obs above null | participant AUC: observed | null | obs above null | precision: observed | null |", "|---|" + "---|" * 11]
            for m in MAIN + OTHER:
                row = []
                for met in ("det_lead1", "det_lead7", "auc_participant", "precision_participant"):
                    x = nl[(nl.outcome == k) & (nl.h == h) & (nl.target_fap == 0.2) & (nl.model == m) & (nl.metric == met)]
                    if x.empty or (met == "det_lead7" and h < 7):
                        row += ["", "", ""] if met != "precision_participant" else ["", ""]
                        continue
                    base = [f"{x.observed.mean():.2f}", f"{x.null_mean.mean():.2f} [{x.null_lo.mean():.2f}, {x.null_hi.mean():.2f}]"]
                    row += base + ([f"{1 - x.frac_null_ge_observed.mean():.0%}"] if met != "precision_participant" else [])
                if row:
                    N.append(f"| {m} | " + " | ".join(row) + " |")
    open(f"{OUT}/null_comparison.md", "w", encoding="utf-8").write("\n".join(N) + "\n")

    M = ["# P1-5 SUPPLEMENTARY: exposure-matched analysis (SYNTHETIC; EXTENSION)\n",
         "Never-onset windows (days 27..T1) are cut into COMPLETE non-overlapping W-day blocks; leftover days at the end of the window are NOT used (h21: 70 days = 3 blocks + 7 unused days; h14: 77 days = 5 blocks + 7 unused; "
         "h7: 84 days = 12 blocks, none unused). Block FAP = share of blocks with >= 1 alarm (same calibration threshold as the primary analysis). Full-window onsets = onset participants whose complete W-day pre-onset "
         "window lies inside the origin range (o - W >= 27 and o - 1 <= T1); their detection at lead >= l (all leads <= W achievable) is compared with the block FAP for the same window length. "
         "Matched AUC = P(onset window max > a never-onset block max), chance 0.5 (the unmatched participant AUC has chance well below 0.5). Seed means; SD over seeds; shuffle-null block FAP in the last column.\n"]
    for h in ew.HORIZONS:
        for k in ew.KS:
            sel = per[(per.outcome == k) & (per.h == h) & (per.target_fap == 0.2)]
            M += [f"\n## {k}, h{h}, 20% target: {int(sel.matched_n_full_window_onsets.iloc[0])} full-window onsets, {int(sel.matched_n_blocks.iloc[0])} never-onset blocks ({int(sel.matched_leftover_days.iloc[0])} leftover days unused)\n",
                  "| model | block FAP (SD) | full-window det >= 1 d (SD) | full-window det >= 7 d | matched AUC (SD) | unmatched participant AUC | participant FAP | block FAP under shuffle null |", "|---|---|---|---|---|---|---|---|"]
            for m in MAIN + OTHER + ["random_null_iid"]:
                s = sel[sel.model == m]
                if s.empty:
                    continue
                x = nl[(nl.outcome == k) & (nl.h == h) & (nl.target_fap == 0.2) & (nl.model == m) & (nl.metric == "matched_block_fap")]
                d7 = f"{s.matched_det_lead7.mean():.2f}" if h >= 7 else "n/a"
                M.append(f"| {m} | {s.matched_block_fap.mean():.3f} ({s.matched_block_fap.std():.3f}) | {s.matched_det_lead1.mean():.2f} ({s.matched_det_lead1.std():.2f}) | {d7} | {s.matched_auc_vs_blocks.mean():.2f} ({s.matched_auc_vs_blocks.std():.2f}) | "
                         f"{s.auc_participant.mean():.2f} | {s.fap.mean():.2f} | {x.null_mean.mean() if len(x) else float('nan'):.3f} |")
    open(f"{OUT}/matched_exposure.md", "w", encoding="utf-8").write("\n".join(M) + "\n")

    # ------------------------------------------------------------------ 4. threshold uncertainty
    thr = pd.read_csv(f"{BASE}/sust/threshold_bootstrap_draws.csv")
    Tt = ["# P1-5 threshold uncertainty (SYNTHETIC; EXTENSION)\n", "Seed 0 only. The calibration never-onset participants are resampled (B = 300), the threshold is re-derived by the Stage 7 rule, and the SAME test scores are re-evaluated. "
          "Median and 2.5-97.5 percentiles of the resulting test metrics: this isolates the effect of the finite calibration sample on the operating point. It is NOT combined with the participant bootstrap.\n"]
    for k in ew.KS:
        Tt += [f"\n## {k}, 20% target\n", "| model | h | FAP median [2.5, 97.5] | det >= 1 d | det >= 7 d | precision |", "|---|---|---|---|---|---|"]
        for m in MAIN + OTHER:
            for h in ew.HORIZONS:
                x = thr[(thr.outcome == k) & (thr.model == m) & (thr.h == h) & (thr.target_fap == 0.2)]
                if x.empty:
                    continue
                q = lambda c: f"{x[c].median():.2f} [{x[c].quantile(.025):.2f}, {x[c].quantile(.975):.2f}]" if x[c].notna().any() else "n/a"      # noqa: E731
                Tt.append(f"| {m} | {h} | {q('fap')} | {q('det_lead1')} | {q('det_lead7')} | {q('precision_participant')} |")
    open(f"{OUT}/threshold_uncertainty.md", "w", encoding="utf-8").write("\n".join(Tt) + "\n")

    # ------------------------------------------------------------------ figures
    per, bt, nl = P["sust"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, k in zip(ax, ew.KS):
        ms = [m for m in MAIN + OTHER if m != "gru"]
        y = np.arange(len(ms))
        for i, m in enumerate(ms):
            r = bt[(bt.outcome == k) & (bt.h == 21) & (bt.target_fap == 0.2) & (bt.model == m) & (bt.metric == "det_lead7")].iloc[0]
            a.errorbar(r.diff_vs_gru_point, i, xerr=[[r.diff_vs_gru_point - r.diff_vs_gru_lo], [r.diff_vs_gru_hi - r.diff_vs_gru_point]], fmt="o", color=COL.get(m, "#888888"), capsize=3)
        a.axvline(0, color="black", lw=0.8)
        a.set_yticks(y)
        a.set_yticklabels(ms, fontsize=8)
        a.set_title(f"{k}: detection >= 7 d, h21, 20% FAP target\ndifference vs GRU (overall rate), paired bootstrap 95%", fontsize=9)
        a.invert_yaxis()
    fig.suptitle("SYNTHETIC data: primary contrast. Intervals reflect 32 onset participants per outcome; exploratory, no clinical meaning", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_primary_contrast.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for r, k in enumerate(ew.KS):
        for c, h in enumerate(ew.HORIZONS):
            a = ax[r, c]
            leads = [l for l in ew.LEADS if l <= h]
            for m in MAIN:
                p = [bt[(bt.outcome == k) & (bt.h == h) & (bt.target_fap == 0.2) & (bt.model == m) & (bt.metric == f"det_lead{l}")].iloc[0] for l in leads]
                a.plot(leads, [q.point_seed_mean for q in p], marker="o", color=COL[m], label=m, lw=1.5)
                a.fill_between(leads, [q.ci95_lo for q in p], [q.ci95_hi for q in p], color=COL[m], alpha=0.07)
            nn = [nl[(nl.outcome == k) & (nl.h == h) & (nl.target_fap == 0.2) & (nl.model == "gru") & (nl.metric == f"det_lead{l}")] for l in leads]
            a.fill_between(leads, [x.null_lo.mean() for x in nn], [x.null_hi.mean() for x in nn], color="#bbbbbb", alpha=0.5, label="shuffle null range (GRU scores)")
            a.set_title(f"{k}, h{h}", fontsize=9)
            a.set_xlabel("minimum lead (days)")
            a.set_ylim(0, None)
            if c == 0:
                a.set_ylabel("detection rate (overall)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("SYNTHETIC data: detection vs required lead, 20% false-alarm target (seed means; bands = bootstrap 95%; grey = shuffle-null range)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_detection_by_lead.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, k in zip(ax, ew.KS):
        for m in MAIN:
            y = [bt[(bt.outcome == k) & (bt.h == h) & (bt.target_fap == 0.2) & (bt.model == m) & (bt.metric == "det_lead1")].iloc[0].point_seed_mean for h in ew.HORIZONS]
            a.plot(ew.HORIZONS, y, marker="o", color=COL[m], label=m)
        a.set_xticks(ew.HORIZONS)
        a.set_xlabel("horizon h (= window W)")
        a.set_ylabel("detection >= 1 day (overall)")
        a.set_title(f"{k}: detection vs horizon, 20% target", fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("SYNTHETIC data: horizons differ in window length and never-onset exposure; not directly comparable without the exposure-matched supplement", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_horizon.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, k in zip(ax, ew.KS):
        sel = per[(per.outcome == k) & (per.target_fap == 0.2)]
        for m in MAIN:
            y = [sel[(sel.model == m) & (sel.h == h)].matched_auc_vs_blocks.mean() for h in ew.HORIZONS]
            a.plot(ew.HORIZONS, y, marker="o", color=COL[m], label=m)
        a.axhline(0.5, color="black", ls=":", lw=1)
        a.set_xticks(ew.HORIZONS)
        a.set_xlabel("horizon h (= window W)")
        a.set_ylabel("exposure-matched AUC (chance 0.5)")
        a.set_title(f"{k}: onset window vs never-onset W-day blocks", fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("SYNTHETIC data, supplementary: exposure-matched discrimination", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_matched_auc.png", dpi=130)
    plt.close(fig)
    print("report written to", OUT)


if __name__ == "__main__":
    main()
