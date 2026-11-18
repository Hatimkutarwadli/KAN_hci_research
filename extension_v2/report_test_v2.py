"""Stage 9 final TEST report (SYNTHETIC), descriptive; per family; no overall winner. Reads CSVs only. Original GRU = Stage 8 test metrics (gru)."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import decide_v2 as D
os.chdir(D.ROOT)
FIN, PANEL, FAMS, OUT = D.FIN, D.PANEL, D.FAMS, D.OUT
V, base = {}, D.read(glob.glob("results/stage8_missing/test/metrics/*.csv"))
for part in ("main", "forecast"):
    V[("orig", part)] = D.values(base[(base.model == "gru") & (base.part == part)])
    for c in FIN:
        fs = glob.glob(f"{OUT}/test/metrics_{c}{'' if part == 'main' else '_forecast'}/*.csv"); assert len(fs) == 59
        V[(c, part)] = D.values(D.read(fs))
def pair(c, ref, part, groups, scen):
    d = pd.concat([V[(c, part)].rename("n"), V[(ref, part)].rename("r")], axis=1).reset_index()
    d = d[d.group.isin(groups) & d.scenario.isin(scen)].copy(); d["w"] = D.worse(d.group, d.n, d.r); return d
def cell(d):
    s = d.groupby("seed").w.mean(); return f"{s.mean():+.2f} ({(s < 0).sum()}/5)"
L = ["# Stage 9 final evaluation - TEST split (SYNTHETIC; evaluated once after the calibration decision file; 5 paired seeds)\n",
     "worse% (NEGATIVE = better) = (new-ref)/ref for RMSE, (ref-new)/ref for AP/AUROC; cell = mean over seeds (n seeds better). Panel = mean over the 10 pre-specified scenarios. Descriptive: no overall winner.\n"]
dec = __import__("json").load(open(f"{OUT}/decision/decision_cal.json"))
for fam, (part, groups) in FAMS.items():
    L += [f"\n## {fam} (calibration-selected: {dec[fam]['selected']})\n", "Absolute (mean over seeds) clean C0 / S3_k7 / S3_k14: " + "; ".join(f"{c}: " + " / ".join(f"{'+'.join(f'{V[(c, part)].xs(s, level=chr(115)+chr(99)+chr(101)+chr(110)+chr(97)+chr(114)+chr(105)+chr(111)).xs(g, level=chr(103)+chr(114)+chr(111)+chr(117)+chr(112)).mean():.4f}' for g in groups)}" for s in ('C0', 'S3_k7', 'S3_k14')) for c in ['orig'] + FIN) + "\n",
          "| scenario | " + " | ".join(f"{c} vs orig" for c in FIN) + " | R0O2T1 vs R0O0T1 | R0O3T1 vs R0O0T1 |", "|---|" + "---|" * 5]
    for sc in ["C0"] + PANEL:
        L.append(f"| {sc} | " + " | ".join(cell(pair(c, "orig", part, groups, [sc])) for c in FIN) + " | " + " | ".join(cell(pair(c, "R0O0T1", part, groups, [sc])) for c in FIN[1:]) + " |")
    L.append("| **panel mean** | " + " | ".join(cell(pair(c, "orig", part, groups, PANEL)) for c in FIN) + " | " + " | ".join(cell(pair(c, "R0O0T1", part, groups, PANEL)) for c in FIN[1:]) + " |")
L += ["\n## Caveat (not a selection criterion): onset risk inflation on test, ratio to own clean control\n", "| scenario | original GRU | " + " | ".join(FIN) + " |", "|---|---|---|---|---|"]
mp = {}
for c in ["orig"] + FIN:
    df = base[(base.model == "gru") & (base.part == "main")] if c == "orig" else D.read(glob.glob(f"{OUT}/test/metrics_{c}/*.csv"))
    mp[c] = df[(df.metric == "mean_pred") & df.task.str.startswith("onset_sust")].groupby(["scenario", "seed"]).value.mean()
for sc in ("S1_r40", "S2_r40_L7", "S3_k7", "S3_k14", "S5_blk40"):
    L.append(f"| {sc} | " + " | ".join(f"{(mp[c].xs(sc, level='scenario') / mp[c].xs('C0', level='scenario')).mean():.2f}" for c in ["orig"] + FIN) + " |")
open(f"{OUT}/analysis/final_test_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
