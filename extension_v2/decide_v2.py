"""Stage 9 final selection on CALIBRATION data only (SYNTHETIC). Rule fixed before forecast results were viewed (approved plan).
Families (assessed separately, no overall winner): severity (nowcast), status (nowcast), onset (AP and AUROC), forecast severity, forecast status.
Reference = R0O0T1 (augmentation baseline, simplest fill). A finalist is 'better' in a family if: (1) clean non-inferiority vs the ORIGINAL GRU
(worse% on C0 <= original GRU seed SD as % of its mean, every group of the family); (2) panel-mean worse% vs R0O0T1 < 0 with >=4/5 seeds negative.
Panel (equal weights): S1_r40, S2_r40_L7, S3_k3, S3_k7, S3_k14, S4_{activity,sleep,cardiac}_absent, S5_rand40, S5_blk40. S3_k14 alone is never decisive.
Risk-inflation ratios are reported as a caveat and do NOT disqualify (user decision).
Writes results/stage9_v2/decision/decision_cal.json and analysis/final_cal_decision.md. Usage: python extension_v2/decide_v2.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "src")]
import analyze_missing as am  # noqa: E402

OUT = "results/stage9_v2"
FIN = ["R0O0T1", "R0O2T1", "R0O3T1"]
PANEL = ["S1_r40", "S2_r40_L7", "S3_k3", "S3_k7", "S3_k14", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S5_rand40", "S5_blk40"]
FAMS = {"severity (nowcast)": ("main", ["sev_nowcast_rmse"]), "status (nowcast)": ("main", ["status_ap"]), "onset": ("main", ["onset_ap", "onset_auroc"]),
        "forecast severity": ("forecast", ["sev_forecast_rmse"]), "forecast status": ("forecast", ["status_ap"])}


def read(files):
    return pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True)


def values(df):
    d = df.copy()
    d["group"] = [am.group_of(t, k) for t, k in zip(d.task, d.metric)]
    d = d[d.group.notna()]
    v = d.groupby(["scenario", "seed", "group", "task", "metric"]).value.mean().reset_index()
    return v.groupby(["scenario", "seed", "group"]).value.mean().reset_index().set_index(["scenario", "seed", "group"]).value


def worse(g, new, ref):
    return np.where(g.str.contains("rmse"), (new - ref) / ref * 100, (ref - new) / ref * 100)


def main():
    os.chdir(ROOT)
    V = {}
    for part in ("main", "forecast"):
        b = read(glob.glob("results/stage8_missing/cal/metrics/*.csv"))
        V[("orig", part)] = values(b[(b.model == "gru") & (b.part == part)])
        for c in FIN:
            fs = glob.glob(os.path.join(OUT, "cal", f"metrics_{c}{'' if part == 'main' else '_forecast'}", "*.csv"))
            assert len(fs) == 59, (c, part, len(fs))
            V[(c, part)] = values(read(fs))

    def pair(c, ref, part, groups, scen):
        d = pd.concat([V[(c, part)].rename("n"), V[(ref, part)].rename("r")], axis=1).reset_index()
        d = d[d.group.isin(groups) & d.scenario.isin(scen)]
        d["w"] = worse(d.group, d.n, d.r)
        return d

    dec, L = {}, ["# Stage 9 final selection - CALIBRATION only (SYNTHETIC; 5 paired seeds). Reference = R0O0T1. No overall winner is declared.\n",
                  "worse% = (new-ref)/ref (RMSE) or (ref-new)/ref (AP/AUROC); NEGATIVE = better. n = seeds (of 5) better.\n"]
    for fam, (part, groups) in FAMS.items():
        o = V[("orig", part)]
        sd = {g: o.xs("C0", level="scenario").xs(g, level="group").std() / o.xs("C0", level="scenario").xs(g, level="group").mean() * 100 for g in groups}
        L += [f"\n## {fam}  (part={part}; clean margins: " + ", ".join(f"{g} {sd[g]:.2f}%" for g in groups) + ")\n",
              "| config | clean worse% vs original (margin test) | panel worse% vs R0O0T1 (n better) | S3_k14 worse% vs R0O0T1 | S3_k14 worse% vs original | clean worse% R0O0T1-ref |", "|---|---|---|---|---|---|"]
        info = {}
        for c in FIN:
            cl = pair(c, "orig", part, groups, ["C0"])
            clean_ok = all(cl[cl.group == g].w.mean() <= sd[g] for g in groups)
            if c == "R0O0T1":
                pn, k14, k14o = None, None, None
                pn_txt = "(reference)"
            else:
                p = pair(c, "R0O0T1", part, groups, PANEL)
                s = p.groupby("seed").w.mean()
                pn = (s.mean(), int((s < 0).sum()))
                pn_txt = f"{pn[0]:+.2f} ({pn[1]}/5)"
            k14 = pair(c, "R0O0T1", part, groups, ["S3_k14"]).w.mean() if c != "R0O0T1" else float("nan")
            k14o = pair(c, "orig", part, groups, ["S3_k14"]).w.mean()
            info[c] = dict(clean_ok=clean_ok, panel=pn)
            L.append(f"| {c} | " + ", ".join(f"{g} {cl[cl.group == g].w.mean():+.2f}" for g in groups) + f" -> {'pass' if clean_ok else 'FAIL'} | {pn_txt} | {k14:+.2f} | {k14o:+.2f} | |")
        best = "R0O0T1"
        cands = [c for c in FIN[1:] if info[c]["clean_ok"] and info[c]["panel"][0] < 0 and info[c]["panel"][1] >= 4]
        if len(cands) == 1:
            best = cands[0]
        elif len(cands) == 2:                                          # both beat the reference: separate them by the same >=4/5 rule
            sab = pair(cands[0], cands[1], part, groups, PANEL).groupby("seed").w.mean()
            if sab.mean() < 0 and (sab < 0).sum() >= 4:
                best = cands[0]
            elif sab.mean() > 0 and (sab > 0).sum() >= 4:
                best = cands[1]
            else:
                best = "R0O2T1 or R0O3T1 (not separable; both beat R0O0T1)"
        dec[fam] = dict(part=part, selected=best, clean_ok={c: bool(info[c]["clean_ok"]) for c in FIN},
                        panel_vs_ref={c: (None if info[c]["panel"] is None else dict(mean=round(float(info[c]["panel"][0]), 3), n_better=info[c]["panel"][1])) for c in FIN})
        L.append(f"\n**Selected for {fam}: {best}**" + (" (default: no finalist met the pre-specified improvement rule)" if not cands else f" (met rule vs R0O0T1: {cands})"))
    # risk inflation caveat (main part, onset, own control)
    L += ["\n## Caveat (not a selection criterion): onset risk inflation, ratio of mean predicted onset probability to own clean control\n", "| scenario | original GRU | " + " | ".join(FIN) + " |", "|---|---|" + "---|" * len(FIN)]
    mp = {}
    for c in ["orig"] + FIN:
        df = read(glob.glob("results/stage8_missing/cal/metrics/*.csv")) if c == "orig" else read(glob.glob(os.path.join(OUT, "cal", f"metrics_{c}", "*.csv")))
        if c == "orig":
            df = df[(df.model == "gru") & (df.part == "main")]
        t = df[(df.metric == "mean_pred") & df.task.str.startswith("onset_sust")]
        mp[c] = t.groupby(["scenario", "seed"]).value.mean()
    for sc in ("S1_r40", "S2_r40_L7", "S3_k7", "S3_k14", "S5_blk40"):
        L.append(f"| {sc} | " + " | ".join(f"{(mp[c].xs(sc, level='scenario') / mp[c].xs('C0', level='scenario')).mean():.2f}" for c in ["orig"] + FIN) + " |")
    os.makedirs(os.path.join(OUT, "decision"), exist_ok=True)
    dec["_rule"] = "see decide_v2.py docstring; calibration data only; risk guard not used for disqualification"
    p = os.path.join(OUT, "decision", "decision_cal.json")
    json.dump(dec, open(p, "w"), indent=1, sort_keys=True)
    open(os.path.join(OUT, "analysis", "final_cal_decision.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nSHA256", hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
