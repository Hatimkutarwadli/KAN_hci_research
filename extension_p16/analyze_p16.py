"""P1-6 step 3 (SYNTHETIC; EXTENSION): statistical analysis of the Stage 9 test results from the participant-cluster bootstrap arrays (results/p1_stage9_stats/boot). Reads only; writes
results/p1_stage9_stats/analysis/*. Participant-level uncertainty (bootstrap over the 200 test participants, CONDITIONAL on the five trained seeds) is kept separate from seed variability
(per-seed estimates, SD, n seeds better) and from the supplementary two-level (participants + seeds) bootstrap.

Confirmatory family (fixed in the approved plan): 5 families x 5 contrasts = 25 panel-mean contrasts; two-sided centred-bootstrap p-values (see stats16.two_sided_centred_p), Holm correction,
Bonferroni-simultaneous intervals. Exploratory: per-scenario cells (5 x 5 x 11 = 275) with Benjamini-Hochberg (q = 0.10); risk-inflation contrasts (5 x 10 = 50) with BH. Secondary descriptive (no
confirmatory claim, no p-values): clean non-inferiority, panel-mean risk inflation, participant consistency, subgroups, concentration, supplementary two-level bootstrap.
worse% convention (Stage 9): NEGATIVE = the new configuration is better."""
import json
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "extension_v2"), os.path.join(ROOT, "extension_p16")]
import stats16 as st  # noqa: E402

OUT = "results/p1_stage9_stats"
PANEL = ["S1_r40", "S2_r40_L7", "S3_k3", "S3_k7", "S3_k14", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S5_rand40", "S5_blk40"]
ALLS = ["C0"] + PANEL
CON, FAMS = st.CONTRASTS, st.FAMS
SEEDS = range(5)
LOSS_OF = {"severity (nowcast)": "sev_nowcast", "status (nowcast)": "status_main", "onset": "onset", "forecast severity": "sev_forecast", "forecast status": "status_forecast"}
PART_OF = {f: v[0] for f, v in FAMS.items()}
LAB = lambda c: "orig GRU" if c == "baseline" else c      # noqa: E731
SCEN_GROUP = {"random whole-day": ["S1_r40", "S5_rand40"], "7-day blocks": ["S2_r40_L7", "S5_blk40"], "trailing outage": ["S3_k3", "S3_k7", "S3_k14"], "sensor-group absent": ["S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent"]}
SEED_TWO_LEVEL, B_NOTE = 20261121, None


class Data:
    def __init__(self):
        z = np.load(f"{OUT}/boot/boot_values.npz")
        self.v = {k: z[k] for k in z.files}
        pl = np.load(f"{OUT}/boot/participant_loss.npz")
        self.pl = {k: pl[k] for k in pl.files}
        self.C = np.load(f"{OUT}/boot/counts_C.npy").astype(float)
        self.K = self.C.shape[0]
        self.cfg = json.load(open(f"{OUT}/boot/config.json"))

    def val(self, c, p, sid, seed, grp):
        return self.v[f"{c}|{p}|{sid}|{seed}|{grp}"]

    def w(self, new, ref, fam, sid, seed):
        """worse% over the groups of the family for one scenario and seed (K,) - mean over the family's groups (Stage 9 convention)."""
        p, groups = FAMS[fam]
        return np.mean([st.worse_pct(g, self.val(new, p, sid, seed, g), self.val(ref, p, sid, seed, g)) for g in groups], 0)

    def per_seed(self, new, ref, fam, scen):
        """(5, K): per-seed family worse%, equal-weight mean over the scenario set."""
        return np.stack([np.mean([self.w(new, ref, fam, s, sd) for s in scen], 0) for sd in SEEDS])


def summarise(Ts, alpha_bonf=None, rng=None):
    """Ts (5, K) per-seed statistics. Returns point, participant-bootstrap CI (conditional on the 5 seeds), p, per-seed stats, two-level CI."""
    T = Ts.mean(0)
    tb = T[1:]
    r = dict(point=T[0], lo=np.percentile(tb, 2.5), hi=np.percentile(tb, 97.5), p=st.two_sided_centred_p(T), seed_sd=Ts[:, 0].std(ddof=1), seed_min=Ts[:, 0].min(), seed_max=Ts[:, 0].max(),
             n_seeds_better=int((Ts[:, 0] < 0).sum()), var_participant=float(tb.var(ddof=1)), var_seedmean=float(Ts[:, 0].var(ddof=1) / 5))
    if alpha_bonf is not None:
        r["lo_bonf"], r["hi_bonf"] = np.percentile(tb, 100 * alpha_bonf / 2), np.percentile(tb, 100 * (1 - alpha_bonf / 2))
    draws = rng.integers(0, 5, (Ts.shape[1] - 1, 5))                                       # supplementary: participants AND seeds resampled
    two = np.array([Ts[draws[b], b + 1].mean() for b in range(Ts.shape[1] - 1)])
    r["lo_two"], r["hi_two"] = np.percentile(two, 2.5), np.percentile(two, 97.5)
    r["seed_ci_lo"] = [float(np.percentile(Ts[s, 1:], 2.5)) for s in range(5)]
    r["seed_ci_hi"] = [float(np.percentile(Ts[s, 1:], 97.5)) for s in range(5)]
    return r


def parse_report():
    """panel-mean and per-scenario cells of the existing Stage 9 final test report (for the consistency check): {fam: {scenario: [5 values]}}."""
    out, fam = {}, None
    for line in open("results/stage9_v2/analysis/final_test_report.md", encoding="utf-8"):
        m = re.match(r"## (.+?) \(calibration-selected", line)
        if m:
            fam = m.group(1)
            out[fam] = {}
        elif fam and line.startswith("| ") and not line.startswith("| scenario") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 6 and re.match(r"^[+-]?\d", cells[1]):
                out[fam][cells[0].replace("**", "")] = [float(re.match(r"([+-]?\d+\.\d+)", c).group(1)) for c in cells[1:]]
    return out


def clean_margins():
    """Stage 9 clean non-inferiority margin = original-GRU seed SD as % of its mean on the clean control, computed from the CALIBRATION Stage 8 metrics exactly as decide_v2.py did (fixed before any test result)."""
    import decide_v2 as D
    base = D.read([f for f in __import__("glob").glob("results/stage8_missing/cal/metrics/*.csv")])
    out = {}
    for fam, (part, groups) in FAMS.items():
        o = D.values(base[(base.model == "gru") & (base.part == part)])
        for g in groups:
            x = o.xs("C0", level="scenario").xs(g, level="group")
            out[(fam, g)] = float(x.std() / x.mean() * 100)
    return out


def main():
    os.chdir(ROOT)
    od = f"{OUT}/analysis"
    os.makedirs(od, exist_ok=True)
    D = Data()
    rng = np.random.default_rng(SEED_TWO_LEVEL)
    N = D.C.shape[1]
    meta = json.load(open(f"{OUT}/regen_meta.json"))
    dec = json.load(open("results/stage9_v2/decision/decision_cal.json"))

    # ---------------------------------------------------------------- 0. consistency with the existing Stage 9 final test report
    rep = parse_report()
    maxdev, ncell = 0.0, 0
    for fam in FAMS:
        for sc in ALLS + ["panel mean"]:
            scen = PANEL if sc == "panel mean" else [sc]
            for ci, (new, ref) in enumerate(CON):
                mine = D.per_seed(new, ref, fam, scen)[:, 0].mean()
                maxdev = max(maxdev, abs(mine - rep[fam][sc][ci]))
                ncell += 1
    json.dump(dict(n_cells=ncell, max_abs_dev_pct_points=maxdev, note="point estimates (row 0) vs the cells printed in results/stage9_v2/analysis/final_test_report.md (2 decimals)"), open(f"{od}/report_consistency.json", "w"), indent=1)
    assert maxdev < 0.0051, f"point estimates disagree with the Stage 9 final report by {maxdev}"

    # ---------------------------------------------------------------- 1. confirmatory (25) panel-mean contrasts
    rows, store = [], {}
    n_conf = len(FAMS) * len(CON)
    for fam in FAMS:
        for new, ref in CON:
            Ts = D.per_seed(new, ref, fam, PANEL)
            r = summarise(Ts, 0.05 / n_conf, rng)
            store[(fam, new, ref)] = Ts
            rows.append(dict(family=fam, new=new, ref=ref, **r))
    conf = pd.DataFrame(rows)
    conf["p_holm"] = st.holm(conf.p.values)
    conf["holm_reject_05"] = conf.p_holm < 0.05
    conf["bonf_interval_excludes_0"] = (conf.lo_bonf > 0) | (conf.hi_bonf < 0)
    conf["unadjusted_interval_excludes_0"] = (conf.lo > 0) | (conf.hi < 0)
    conf.drop(columns=["seed_ci_lo", "seed_ci_hi"]).to_csv(f"{od}/confirmatory_contrasts.csv", index=False)
    pd.DataFrame([dict(family=r.family, new=r.new, ref=r.ref, seed=s, point=store[(r.family, r.new, r.ref)][s, 0], ci_lo=r.seed_ci_lo[s], ci_hi=r.seed_ci_hi[s]) for r in conf.itertuples() for s in SEEDS]).to_csv(f"{od}/per_seed_panel_contrasts.csv", index=False)
    L = ["# P1-6 CONFIRMATORY panel-mean contrasts (SYNTHETIC; EXTENSION). Test split, 200 participants, 10 pre-specified panel scenarios, equal weights\n",
         "worse% (NEGATIVE = new configuration better). Point = mean over the 5 seeds of the per-seed panel-mean worse% (the Stage 9 report cell). **Participant-level 95% interval** = percentile interval of the "
         "participant-cluster bootstrap (B = 2000, shared resamples), CONDITIONAL on the five trained seeds; **Bonferroni interval** = simultaneous over the 25 contrasts (99.8%). **p** = two-sided centred "
         "bootstrap test of H0: effect = 0; **Holm** = adjusted over the 25 confirmatory contrasts. Seed variability (separate): SD, range and n seeds better of the per-seed estimates. "
         "Supplementary two-level interval resamples participants and seeds together (5 seeds: unstable, shown for orientation only).\n",
         "| family | contrast | point | participant 95% interval | Bonferroni (25) | p | Holm p | seeds: SD, [min, max], n better | two-level 95% (suppl.) | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in conf.itertuples():
        verdict = "Holm-significant" if r.holm_reject_05 else ("unadjusted interval excludes 0 only" if r.unadjusted_interval_excludes_0 else "not distinguishable from 0")
        L.append(f"| {r.family} | {LAB(r.new)} vs {LAB(r.ref)} | {r.point:+.2f} | [{r.lo:+.2f}, {r.hi:+.2f}] | [{r.lo_bonf:+.2f}, {r.hi_bonf:+.2f}] | {r.p:.4f} | {r.p_holm:.4f} | {r.seed_sd:.2f}, [{r.seed_min:+.2f}, {r.seed_max:+.2f}], {r.n_seeds_better}/5 | "
                 f"[{r.lo_two:+.2f}, {r.hi_two:+.2f}] | {verdict} |")
    L += [f"\n{int(conf.holm_reject_05.sum())} of {len(conf)} confirmatory contrasts are Holm-significant at 0.05; {int(conf.unadjusted_interval_excludes_0.sum())} have an unadjusted 95% interval excluding 0. "
          "Variance comparison (participant sampling vs seeds): see `confirmatory_contrasts.csv` columns var_participant (bootstrap variance of the seed-mean statistic) and var_seedmean (between-seed variance / 5).\n"]
    open(f"{od}/confirmatory_contrasts.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- 2. exploratory per-scenario cells (BH)
    rows = []
    for fam in FAMS:
        for new, ref in CON:
            for sid in ALLS:
                Ts = D.per_seed(new, ref, fam, [sid])
                r = summarise(Ts, None, rng)
                rows.append(dict(family=fam, new=new, ref=ref, scenario=sid, **{k: v for k, v in r.items() if k not in ("seed_ci_lo", "seed_ci_hi")}))
    ex = pd.DataFrame(rows)
    ex["p_bh"] = st.benjamini_hochberg(ex.p.values)
    ex["bh_q10"] = ex.p_bh < 0.10
    ex.to_csv(f"{od}/exploratory_scenario_cells.csv", index=False)
    sg = []
    for fam in FAMS:
        for new, ref in CON:
            for gname, scs in SCEN_GROUP.items():
                Ts = D.per_seed(new, ref, fam, scs)
                r = summarise(Ts, None, rng)
                sg.append(dict(family=fam, new=new, ref=ref, scenario_family=gname, **{k: v for k, v in r.items() if k not in ("seed_ci_lo", "seed_ci_hi")}))
    sgd = pd.DataFrame(sg)
    sgd["p_bh"] = st.benjamini_hochberg(sgd.p.values)
    sgd.to_csv(f"{od}/exploratory_scenario_families.csv", index=False)
    L = ["# P1-6 EXPLORATORY per-scenario contrasts (SYNTHETIC; EXTENSION)\n", f"{len(ex)} cells (5 families x 5 contrasts x 11 scenarios). Intervals are UNADJUSTED participant-bootstrap 95% intervals (conditional on the 5 seeds); "
         f"flags use Benjamini-Hochberg FDR q = 0.10 over all {len(ex)} cells. At q = 0.10 about {0.1 * ex.bh_q10.sum():.1f} of the {int(ex.bh_q10.sum())} flagged cells could be false discoveries. worse% NEGATIVE = better. "
         "n better = seeds with the new configuration better.\n"]
    for fam in FAMS:
        L += [f"\n## {fam}\n", "| scenario | " + " | ".join(f"{LAB(n)} vs {LAB(r)}" for n, r in CON) + " |", "|---|" + "---|" * len(CON)]
        for sid in ALLS:
            cells = []
            for n, r in CON:
                x = ex[(ex.family == fam) & (ex.new == n) & (ex.ref == r) & (ex.scenario == sid)].iloc[0]
                cells.append(f"{x.point:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] ({x.n_seeds_better}/5){' *' if x.bh_q10 else ''}")
            L.append(f"| {sid} | " + " | ".join(cells) + " |")
    L += ["\n`*` = BH q <= 0.10 across all exploratory cells. Scenario-family aggregates (random whole-day, 7-day blocks, trailing outage, sensor-group absent): `exploratory_scenario_families.csv`.\n"]
    open(f"{od}/exploratory_scenarios.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- 3. secondary descriptive: clean non-inferiority
    marg = clean_margins()
    sel = [("severity (nowcast)", "R0O2T1"), ("status (nowcast)", "R0O3T1"), ("onset", "R0O0T1"), ("forecast severity", "R0O2T1"), ("forecast severity", "R0O3T1"), ("forecast status", "R0O3T1")]
    L = ["# P1-6 SECONDARY DESCRIPTIVE: clean-control non-inferiority vs the original GRU (SYNTHETIC; EXTENSION)\n",
         "Not part of the confirmatory family and not adjusted. Margin = Stage 9's pre-specified rule: the original GRU's seed SD as a percentage of its mean on the clean control, computed from the CALIBRATION "
         "Stage 8 metrics before any test result (value fixed, not tuned). Selected finalists are those chosen by the calibration decision file (forecast severity: both R0O2T1 and R0O3T1, 'not separable'). "
         "worse% > 0 = worse than the original. 'Upper 95%' = 95th percentile of the one-sided participant-bootstrap distribution (conditional on the 5 seeds).\n",
         "| family | finalist | group | clean worse% (Stage 9 point rule: <= margin) | margin | participant 95% interval | upper 95% bound | upper bound <= margin? | seeds worse than original |", "|---|---|---|---|---|---|---|---|---|"]
    ni = []
    for fam, c in sel:
        p, groups = FAMS[fam]
        for g in groups:
            Ts = np.stack([st.worse_pct(g, D.val(c, p, "C0", s, g), D.val("baseline", p, "C0", s, g)) for s in SEEDS])
            T = Ts.mean(0)
            up = float(np.percentile(T[1:], 95))
            ni.append(dict(family=fam, finalist=c, group=g, point=T[0], margin=marg[(fam, g)], lo=np.percentile(T[1:], 2.5), hi=np.percentile(T[1:], 97.5), upper95=up, n_seeds_worse=int((Ts[:, 0] > 0).sum())))
            n = ni[-1]
            L.append(f"| {fam} | {c} | {g} | {n['point']:+.2f} ({'pass' if n['point'] <= n['margin'] else 'above margin'}) | {n['margin']:.2f} | [{n['lo']:+.2f}, {n['hi']:+.2f}] | {n['upper95']:+.2f} | {'yes' if up <= n['margin'] else 'no'} | {n['n_seeds_worse']}/5 |")
    pd.DataFrame(ni).to_csv(f"{od}/secondary_clean_noninferiority.csv", index=False)
    open(f"{od}/secondary_clean_noninferiority.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- 4. risk inflation
    def ratio(c, sid, seed):
        return D.val(c, "main", sid, seed, "onset_meanpred") / D.val(c, "main", "C0", seed, "onset_meanpred")

    models = ["baseline"] + [c for c in meta["configs"] if c != "baseline"]
    ri, rows = [], []
    for c in models:
        for sid in PANEL:
            T = np.mean([ratio(c, sid, s) for s in SEEDS], 0)
            ri.append(dict(model=c, scenario=sid, ratio=T[0], lo=np.percentile(T[1:], 2.5), hi=np.percentile(T[1:], 97.5)))
    pd.DataFrame(ri).to_csv(f"{od}/risk_inflation_ratios.csv", index=False)
    for new, ref in CON:
        for sid in PANEL:
            Ts = np.stack([ratio(new, sid, s) - ratio(ref, sid, s) for s in SEEDS])
            r = summarise(Ts, None, rng)
            rows.append(dict(new=new, ref=ref, scenario=sid, **{k: v for k, v in r.items() if k not in ("seed_ci_lo", "seed_ci_hi")}))
    rid = pd.DataFrame(rows)
    rid["p_bh"] = st.benjamini_hochberg(rid.p.values)
    rid.to_csv(f"{od}/risk_inflation_contrasts.csv", index=False)
    L = ["# P1-6 risk inflation (SYNTHETIC; EXTENSION): mean predicted sustained-onset probability relative to the SAME model's clean control\n",
         "Ratio 1.00 = unchanged; > 1 = the model predicts more onset risk under the missingness scenario. Seed mean with participant-bootstrap 95% interval (conditional on the 5 seeds). A caveat metric in Stage 9 (not a selection "
         "criterion); no calibration claim.\n", "| scenario | " + " | ".join(LAB(c) for c in models) + " |", "|---|" + "---|" * len(models)]
    for sid in PANEL:
        cells = []
        for c in models:
            x = [r for r in ri if r["model"] == c and r["scenario"] == sid][0]
            cells.append(f"{x['ratio']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]")
        L.append(f"| {sid} | " + " | ".join(cells) + " |")
    L += ["\n## Difference in inflation ratio (new minus reference; NEGATIVE = less inflation); EXPLORATORY, BH over the 50 cells\n", "| scenario | " + " | ".join(f"{LAB(n)} vs {LAB(r)}" for n, r in CON) + " |", "|---|" + "---|" * len(CON)]
    for sid in PANEL:
        cells = []
        for n, r in CON:
            x = rid[(rid.new == n) & (rid.ref == r) & (rid.scenario == sid)].iloc[0]
            cells.append(f"{x.point:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}]{' *' if x.p_bh < 0.10 else ''}")
        L.append(f"| {sid} | " + " | ".join(cells) + " |")
    L += ["\n`*` = BH q <= 0.10 across the 50 risk-inflation cells.\n", "\n## Panel mean (10 scenarios) - secondary descriptive\n", "| contrast | difference in ratio | participant 95% interval | seeds with less inflation |", "|---|---|---|---|"]
    for new, ref in CON:
        Ts = np.stack([np.mean([ratio(new, s_, sd) - ratio(ref, s_, sd) for s_ in PANEL], 0) for sd in SEEDS])
        T = Ts.mean(0)
        L.append(f"| {LAB(new)} vs {LAB(ref)} | {T[0]:+.3f} | [{np.percentile(T[1:], 2.5):+.3f}, {np.percentile(T[1:], 97.5):+.3f}] | {int((Ts[:, 0] < 0).sum())}/5 |")
    open(f"{od}/risk_inflation.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- 5. participant-level consistency (descriptive)
    import regen_p16 as rg
    d, pi, ti, *_ = rg.setup()
    Yd, Ya = d["y_sev_dep_t0"][pi, ti], d["y_sev_anx_t0"][pi, ti]
    sev = (np.bincount(pi, weights=Yd, minlength=N) + np.bincount(pi, weights=Ya, minlength=N)) / (2 * np.bincount(pi, minlength=N))     # participant mean TRUE nowcast severity (labels only)
    grp = st.tertile_groups(sev)
    cuts = [float(np.sort(sev)[i]) for i in (N // 3, 2 * N // 3)]
    GN = ["lowest third", "middle third", "highest third"]
    L = ["# P1-6 participant-level consistency (SYNTHETIC; EXTENSION) - DESCRIPTIVE\n",
         "Per participant: loss difference (new - reference), loss = squared error (severity) or Brier score (status/onset), averaged over seeds, replicates, tasks and origins; negative = new better for that participant. "
         "'Improved' = participants with a negative difference. Shares carry participant-bootstrap 95% intervals (shared resamples). 'Top-10% share' = fraction of the total net improvement contributed by the 10% of participants "
         "with the largest improvement (descriptive; a large value means the mean gain is concentrated in few participants; not a test). Subgroups = thirds of the participants' mean TRUE nowcast severity "
         f"(labels only, fixed before any model comparison; cut points {cuts[0]:.2f} and {cuts[1]:.2f} severity points). No p-values.\n"]
    prow = []
    for fam in FAMS:
        lg, p = LOSS_OF[fam], PART_OF[fam]
        L += [f"\n## {fam}\n", "| contrast | scenarios | improved (share) | median difference | top-10% share | improved: " + " / ".join(GN) + " |", "|---|---|---|---|---|---|"]
        for new, ref in CON:
            for label, scs in (("panel mean (10)", PANEL), ("S3 trailing outages", SCEN_GROUP["trailing outage"]), ("clean C0", ["C0"])):
                dlt = np.mean([D.pl[f"{new}|{p}|{s}|{lg}"] - D.pl[f"{ref}|{p}|{s}|{lg}"] for s in scs], 0)
                sh = st.weighted_share(dlt < 0, D.C)
                subs = []
                for gi in range(3):
                    mk = (grp == gi)
                    s_ = (D.C[:, mk] @ (dlt[mk] < 0).astype(float)) / D.C[:, mk].sum(1)
                    subs.append(f"{s_[0]:.2f} [{np.percentile(s_[1:], 2.5):.2f}, {np.percentile(s_[1:], 97.5):.2f}]")
                prow.append(dict(family=fam, new=new, ref=ref, scenarios=label, share_improved=sh[0], lo=np.percentile(sh[1:], 2.5), hi=np.percentile(sh[1:], 97.5), median_diff=float(np.median(dlt)), mean_diff=float(dlt.mean()), top10_share=st.top_share(-dlt),
                                 share_low=float(((dlt[grp == 0]) < 0).mean()), share_mid=float(((dlt[grp == 1]) < 0).mean()), share_high=float(((dlt[grp == 2]) < 0).mean())))
                x = prow[-1]
                L.append(f"| {LAB(new)} vs {LAB(ref)} | {label} | {x['share_improved']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}] | {x['median_diff']:+.4f} | {x['top10_share']:.2f} | " + " / ".join(subs) + " |")
    pd.DataFrame(prow).to_csv(f"{od}/participant_consistency.csv", index=False)
    open(f"{od}/participant_consistency.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- 6. event counts, NaN resamples
    ec = pd.read_csv(f"{OUT}/boot/event_counts.csv")
    nanr = pd.read_csv(f"{OUT}/boot/nan_resamples.csv") if os.path.getsize(f"{OUT}/boot/nan_resamples.csv") > 2 else pd.DataFrame(columns=["n_nan_resample_values"])
    L = ["# P1-6 event counts and degenerate resamples (SYNTHETIC; EXTENSION)\n", "Test split, 200 participants, origins 27,29,...,119; labels are identical across scenarios (only the inputs change). "
         "Tasks with fewer than 10 participants having any event are flagged: AP/AUROC for them is unstable.\n", "| task | origins with defined label | positive origins | prevalence | participants with an event | flag |", "|---|---|---|---|---|---|"]
    for r in ec.itertuples():
        L.append(f"| {r.task} | {r.n_origins} | {r.n_positive_origins} | {r.prevalence:.3f} | {r.participants_with_event} | {'FEW EVENTS' if r.participants_with_event < 10 else ''} |")
    L += [f"\nBootstrap resamples in which an AP/AUROC value was undefined (no positives or no negatives): {int(nanr.n_nan_resample_values.sum()) if len(nanr) else 0} values in total.\n"]
    open(f"{od}/event_counts.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- 7. figures
    fig, ax = plt.subplots(1, 5, figsize=(18, 5.5), sharex=False)
    for a, fam in zip(ax, FAMS):
        x = conf[conf.family == fam].reset_index(drop=True)
        y = np.arange(len(x))
        a.errorbar(x.point, y, xerr=[x.point - x.lo_bonf, x.hi_bonf - x.point], fmt="none", ecolor="#bbbbbb", lw=4, capsize=0, label="Bonferroni (25)")
        a.errorbar(x.point, y, xerr=[x.point - x.lo, x.hi - x.point], fmt="o", color="#1f78b4", capsize=3, label="participant 95%")
        for i, r in x.iterrows():
            a.plot([r.seed_min, r.seed_max], [i, i], color="#d95f02", lw=1.2, marker="|", label="seed range" if i == 0 else None)
        a.axvline(0, color="black", lw=0.8)
        a.set_yticks(y)
        a.set_yticklabels([f"{LAB(r.new)} vs {LAB(r.ref)}" + (" *" if r.holm_reject_05 else "") for r in x.itertuples()], fontsize=7)
        a.set_title(fam, fontsize=9)
        a.invert_yaxis()
        a.set_xlabel("panel-mean worse% (negative = better)", fontsize=8)
    ax[0].legend(fontsize=7)
    fig.suptitle("SYNTHETIC data: 25 confirmatory contrasts (* = Holm-significant). Blue = participant-level uncertainty (conditional on 5 seeds); orange = seed range", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{od}/fig_confirmatory_forest.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 5, figsize=(18, 4.8), sharey=True)
    for a, fam in zip(ax, FAMS):
        M, S = np.zeros((len(ALLS), len(CON))), np.zeros((len(ALLS), len(CON)), bool)
        for i, sid in enumerate(ALLS):
            for j, (n, r) in enumerate(CON):
                x = ex[(ex.family == fam) & (ex.new == n) & (ex.ref == r) & (ex.scenario == sid)].iloc[0]
                M[i, j], S[i, j] = x.point, x.bh_q10
        vmax = np.abs(M).max()
        a.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:+.1f}{'*' if S[i, j] else ''}", ha="center", va="center", fontsize=6)
        a.set_xticks(range(len(CON)))
        a.set_xticklabels([f"{n[2:]} v\n{'orig' if r == 'baseline' else r[2:]}" for n, r in CON], fontsize=6)
        a.set_title(fam, fontsize=9)
    ax[0].set_yticks(range(len(ALLS)))
    ax[0].set_yticklabels(ALLS, fontsize=7)
    fig.suptitle("SYNTHETIC data: per-scenario worse% (blue = new better; * = BH q <= 0.10, exploratory)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{od}/fig_scenario_heatmap.png", dpi=130)
    plt.close(fig)
    print("analysis written;", int(conf.holm_reject_05.sum()), "Holm-significant of", len(conf), "; report consistency max dev", maxdev)


if __name__ == "__main__":
    main()
