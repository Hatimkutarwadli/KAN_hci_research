"""Stage 9 factor comparison on CALIBRATION data (SYNTHETIC): original GRU (R0.O0.T0, Stage 8 metrics) vs all trained configurations R x O x T, identical Stage 8 scenarios.
Reads CSVs only. Seed s of every configuration is paired with seed s of the baseline (same init stream for R0, same batch order for all).
Delta% = worse% of the configuration relative to a REFERENCE on the same scenario and seed: rmse (new-ref)/ref*100, ap/auroc (ref-new)/ref*100; NEGATIVE = better.
Usage: python extension_v2/compare_factors.py results/stage9_v2
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "src")]
import analyze_missing as am  # noqa: E402
import inject as ij  # noqa: E402

GROUPS = ["sev_nowcast_rmse", "status_ap", "onset_ap", "onset_auroc"]
CFGS = ["R0O0T1", "R1O0T0", "R1O0T1", "R0O1T0", "R0O1T1", "R0O2T0", "R0O2T1", "R0O3T0", "R0O3T1"]
KEY = ["S1_r40", "S2_r40_L7", "S3_k3", "S3_k7", "S3_k14", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S4_circadian_absent", "S5_rand40", "S5_blk40", "Z0"]
SCEN = [s["id"] for s in ij.registry()]


def read(files):
    return pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True)


def values(df):
    d = df.copy()
    d["group"] = [am.group_of(t, k) for t, k in zip(d.task, d.metric)]
    d = d[d.group.notna()]
    v = d.groupby(["scenario", "seed", "group", "task", "metric"]).value.mean().reset_index()
    return v.groupby(["scenario", "seed", "group"]).value.mean().reset_index()


def mean_pred(df, prefix):
    t = df[(df.metric == "mean_pred") & df.task.str.startswith(prefix)]
    return t.groupby(["scenario", "seed"]).value.mean()


def worse(group, new, ref):
    return np.where(group == "sev_nowcast_rmse", (new - ref) / ref * 100, (ref - new) / ref * 100)


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def main(root):
    raw = {"R0O0T0": read(glob.glob("results/stage8_missing/cal/metrics/*.csv"))}
    raw["R0O0T0"] = raw["R0O0T0"][(raw["R0O0T0"].model == "gru") & (raw["R0O0T0"].part == "main")]
    for c in CFGS:
        fs = glob.glob(os.path.join(root, "cal", f"metrics_{c}", "*.csv"))
        assert len(fs) == 59, (c, len(fs))
        raw[c] = read(fs)
    V = {c: values(df).set_index(["scenario", "seed", "group"]).value for c, df in raw.items()}
    allc = ["R0O0T0"] + CFGS

    def delta(c, ref):
        d = pd.concat([V[c].rename("n"), V[ref].rename("r")], axis=1).reset_index()
        d["w"] = worse(d.group, d.n, d.r)
        return d

    def cell(d, sc, g):
        x = d[(d.scenario == sc) & (d.group == g)].w
        return f"{x.mean():+.1f} ({(x < 0).sum()}/{len(x)})"

    L = ["# Stage 9 steps R and O - factorial comparison, CALIBRATION split (SYNTHETIC; origins 27,29,...,119; 5 paired seeds)\n",
         "Configurations: R0/R1 = 2 frozen flags / 8 per-signal flags; O0 frozen fill, O1 unlimited carry-forward, O2 exponential decay to the train mean (tau 7 d), O3 forward-fill 3 d then in-window running mean; T0/T1 = no/with training augmentation. R0O0T0 is the original GRU (not retrained).\n",
         "Cells `x (n/5)`: mean worse% vs the stated reference over seeds (negative = better) and the number of seeds where the configuration is better.\n"]
    runs = [json.loads(l) for l in open(os.path.join(root, "train_runs.jsonl"))]
    tr = pd.DataFrame(runs).groupby("config").agg(params=("n_params", "first"), wall=("wall_s", "mean"), epochs=("best_epoch", "mean"))
    L += ["\n## Training cost\n", md_table(["config", "params", "mean wall s/seed", "mean best epoch"], [[c, str(int(r.params)), f"{r.wall:.0f}", f"{r.epochs:.1f}"] for c, r in tr.iterrows()])]
    # clean data
    L += ["\n## Clean data (C0): absolute value mean ±SD over seeds, and worse% vs the original GRU\n"]
    rows = []
    for g in GROUPS:
        for c in allc:
            x = V[c].xs("C0", level="scenario").xs(g, level="group")
            d = delta(c, "R0O0T0") if c != "R0O0T0" else None
            rows.append([g, c, f"{x.mean():.4f} ±{x.std():.4f}", "" if d is None else cell(d, "C0", g)])
    L.append(md_table(["group", "config", "value", "worse% vs original (n better)"], rows))
    # scenarios vs baseline
    for g in GROUPS:
        L += [f"\n## {g}: worse% vs the original GRU under each scenario (negative = better; n/5 seeds better)\n"]
        D = {c: delta(c, "R0O0T0") for c in CFGS}
        L.append(md_table(["scenario"] + CFGS, [[sc] + [cell(D[c], sc, g) for c in CFGS] for sc in KEY]))
    # absolute under prolonged outage
    L += ["\n## Absolute metric under prolonged trailing outage (mean over seeds)\n"]
    rows = []
    for g in GROUPS:
        for sc in ("C0", "S3_k7", "S3_k14"):
            rows.append([g, sc] + [f"{V[c].xs(sc, level='scenario').xs(g, level='group').mean():.4f}" for c in allc])
    L.append(md_table(["group", "scenario"] + allc, rows))
    # degradation vs own control
    L += ["\n## Degradation vs each configuration's OWN clean control (worse%, + = worse; mean over seeds)\n"]
    rows = []
    for g in GROUPS:
        for sc in ("S1_r40", "S2_r40_L7", "S3_k7", "S3_k14"):
            r = [g, sc]
            for c in allc:
                x = V[c].xs(g, level="group")
                cur, ctl = x.xs(sc, level="scenario"), x.xs("C0", level="scenario")
                r.append(f"{worse(np.array([g] * len(cur)), cur.values, ctl.reindex(cur.index).values).mean():+.1f}")
            rows.append(r)
    L.append(md_table(["group", "scenario"] + allc, rows))
    # risk inflation
    L += ["\n## Risk inflation: mean predicted onset probability, ratio to the configuration's own clean control (1.00 = unchanged); severity shift in points\n"]
    mp = {c: mean_pred(df, "onset_sust") for c, df in raw.items()}
    ms = {c: mean_pred(df, "sev_") for c, df in raw.items()}
    rows = []
    for sc in ("S1_r20", "S1_r40", "S2_r40_L7", "S3_k7", "S3_k14", "S5_blk40", "S1_r40_O", "Z0"):
        rows.append([sc, "onset ratio"] + [f"{(mp[c].xs(sc, level='scenario') / mp[c].xs('C0', level='scenario')).mean():.2f}" for c in allc])
        rows.append([sc, "severity shift"] + [f"{(ms[c].xs(sc, level='scenario') - ms[c].xs('C0', level='scenario')).mean():+.2f}" for c in allc])
    L.append(md_table(["scenario", "quantity"] + allc, rows))
    prev = raw["R0O0T0"]
    prev = prev[(prev.scenario == "C0") & (prev.metric == "prevalence") & prev.task.str.startswith("onset_sust")].value.mean()
    L.append(f"\nClean-control mean predicted onset probability (observed prevalence {prev:.4f}): " + ", ".join(f"{c} {mp[c].xs('C0', level='scenario').mean():.4f}" for c in allc))
    # factor effects
    L += ["\n## Factor effects (paired, same T): worse% vs the matched reference R0O0 at the same T (R0O0T0 = original GRU; R0O0T1 = augmentation alone)\n"]
    for g in GROUPS:
        L += [f"\n### {g}\n"]
        effs = [("A: R1O0T0 vs R0O0T0", "R1O0T0", "R0O0T0"), ("A: R1O0T1 vs R0O0T1", "R1O0T1", "R0O0T1")] + \
               [(f"B: R0{o}T{t} vs R0O0T{t}", f"R0{o}T{t}", f"R0O0T{t}") for o in ("O1", "O2", "O3") for t in (0, 1)]
        D = {n: delta(c, r) for n, c, r in effs}
        L.append(md_table(["scenario"] + [n for n, *_ in effs], [[sc] + [cell(D[n], sc, g) for n, *_ in effs] for sc in ["C0"] + KEY]))
    # screening
    L += ["\n## Pre-specified screening (calibration; descriptive, no configuration is declared the winner)\n",
          "Rule from the approved design: (a) clean non-inferiority: worse% vs the original GRU on C0 must not exceed the original GRU's own seed SD (as % of its mean) in any of the 4 groups; (b) improvement counts only if >=4 of 5 seeds agree; ranking by S3_k7/S3_k14 metrics and the S1_r40 / S2_r40_L7 risk ratios.\n"]
    sd = {g: V["R0O0T0"].xs("C0", level="scenario").xs(g, level="group").std() / V["R0O0T0"].xs("C0", level="scenario").xs(g, level="group").mean() * 100 for g in GROUPS}
    L.append("Margins (original GRU seed SD, % of mean): " + ", ".join(f"{g} {sd[g]:.2f}%" for g in GROUPS) + "\n")
    rows = []
    for c in CFGS:
        d = delta(c, "R0O0T0")
        clean = [d[(d.scenario == "C0") & (d.group == g)].w.mean() for g in GROUPS]
        ok = all(w <= sd[g] for w, g in zip(clean, GROUPS))
        r = [c, "pass" if ok else "FAIL (" + ", ".join(g for w, g in zip(clean, GROUPS) if w > sd[g]) + ")"]
        for g in ("sev_nowcast_rmse", "onset_ap", "onset_auroc"):
            x = d[d.scenario.isin(["S3_k7", "S3_k14"]) & (d.group == g)].groupby("seed").w.mean()
            r.append(f"{x.mean():+.1f} ({(x < 0).sum()}/5)")
        inf = [(mp[c].xs(s, level="scenario") / mp[c].xs("C0", level="scenario")).mean() for s in ("S1_r40", "S2_r40_L7")]
        r.append(f"{inf[0]:.2f} / {inf[1]:.2f}")
        rows.append(r)
    b = [(mp["R0O0T0"].xs(s, level="scenario") / mp["R0O0T0"].xs("C0", level="scenario")).mean() for s in ("S1_r40", "S2_r40_L7")]
    L.append(md_table(["config", "clean non-inferiority", "S3 k7/k14 severity RMSE", "S3 onset AP", "S3 onset AUROC", "risk ratio S1_r40 / S2_r40_L7 (original GRU %.2f / %.2f)" % tuple(b)], rows))
    open(os.path.join(root, "analysis", "step3_cal_factors.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", len(L))


if __name__ == "__main__":
    main(sys.argv[1])
