"""P3 pilot report (SYNTHETIC; calibration only): runtime, stability checks B1-B3, runtime projection for the main Standard experiment.
  python extension_p3/pilot_report.py   -> results/p3_small_data/pilot/{pilot_jobs.csv, pilot_families.csv, PILOT_REPORT.md}"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p3 as P  # noqa: E402
import analyze_p22 as A  # noqa: E402

PRIMARY = ("sev_now", "sev_fc", "status_now", "status_fc")
PIL = os.path.join(P.RES, "pilot")


def families(metrics):
    out = {}
    for fam, (met, tpl) in A.FAMS.items():
        hs = [t.format(k=k) for k in A.KS for t in tpl]
        if not all(h in metrics for h in hs):
            continue
        out[fam] = dict(metric=met, value=float(np.mean([metrics[h]["value"] for h in hs])),
                        reference=float(np.mean([metrics[h]["const_rmse" if met == "rmse" else "prevalence"] for h in hs])))
    return out


def main():
    rows, frows = [], []
    for d in sorted(glob.glob(os.path.join(PIL, "R4", "*"))):
        if d.endswith(".tmp") or not os.path.isdir(d):
            continue
        r = json.load(open(os.path.join(d, "run.json")))
        rows.append(dict(id=r["id"], model=r["model"], ts=r["ts"], n=r["n"], cpu_s=r["cpu_s"], wall_s=r["wall_s"], finite=r["finite"],
                         best_epoch=r.get("best_epoch"), epochs_run=len(r["history"]) if r.get("history") else None))
        for fam, v in families(json.load(open(os.path.join(d, "metrics.json")))).items():
            worse = (v["value"] >= v["reference"]) if v["metric"] == "rmse" else (v["value"] <= v["reference"])
            frows.append(dict(model=r["model"], ts=r["ts"], n=r["n"], family=fam, primary=fam in PRIMARY, **v, worse_than_constant=bool(worse)))
    J, F = pd.DataFrame(rows), pd.DataFrame(frows)
    J.to_csv(os.path.join(PIL, "pilot_jobs.csv"), index=False)
    F.to_csv(os.path.join(PIL, "pilot_families.csv"), index=False)
    q = open(os.path.join(PIL, "queue_log.txt")).read()
    el = [int(x) for x in re.findall(r"end set=pilot .* elapsed=(\d+)s", q)]
    pilot_elapsed = el[-1] if el else float("nan")
    # stop checks
    b1 = bool((~J.finite).any())
    b2 = F[(F.n == 50) & F.primary].groupby(["model", "ts"]).worse_than_constant.sum()
    b2_fail = b2[b2 >= 2]
    b3 = J[J.best_epoch.notna() & (J.best_epoch <= 1)]
    # projection: per (model, ts) linear CPU model in n through the n=50 and n=400 pilot points
    coef = {}
    for (m, ts), g in J.groupby(["model", "ts"]):
        g = g.sort_values("n")
        b = (g.cpu_s.iloc[-1] - g.cpu_s.iloc[0]) / (g.n.iloc[-1] - g.n.iloc[0])
        coef[(m, ts)] = (g.cpu_s.iloc[0] - b * g.n.iloc[0], b)
    jobs = [j for ds in P.DATASETS for j in P.jobs_main(ds)]
    cpu_main = sum(max(coef[(j["model"], j["ts"])][0] + coef[(j["model"], j["ts"])][1] * j["n"], 1.0) for j in jobs)
    eff_pilot = J.cpu_s.sum() / pilot_elapsed
    eff_p22 = 9.3
    proj = {"main_jobs": len(jobs), "main_cpu_h": cpu_main / 3600, "pilot_cpu_s": float(J.cpu_s.sum()), "pilot_elapsed_s": pilot_elapsed,
            "cpu_per_elapsed_pilot": eff_pilot, "cpu_per_elapsed_p22_phaseB": eff_p22,
            "main_elapsed_h_at_pilot_efficiency": cpu_main / 3600 / eff_pilot, "main_elapsed_h_at_p22_efficiency": cpu_main / 3600 / eff_p22,
            "by_model_cpu_h": {f"{m}_{ts}": sum(max(coef[(m, ts)][0] + coef[(m, ts)][1] * j["n"], 1.0) for j in jobs if (j["model"], j["ts"]) == (m, ts)) / 3600 for m, ts in coef}}
    sens = {v: sum(max(coef[(j["model"], j["ts"])][0] + coef[(j["model"], j["ts"])][1] * j["n"], 1.0) for j in P.jobs_sens(v)) / 3600 for v in ("S2", "S3", "S1")}
    proj["sens_cpu_h"] = sens
    json.dump({"B1_nonfinite": b1, "B2_models_failing": {f"{k[0]}_{k[1]}": int(v) for k, v in b2_fail.items()}, "B3_best_epoch_le1": b3.id.tolist(),
               "projection": proj}, open(os.path.join(PIL, "pilot_checks.json"), "w"), indent=1)
    print(J.round(1).to_string(index=False))
    print(F[F.n == 50].round(4).to_string(index=False))
    print(json.dumps({"B1": b1, "B2_fail": b2_fail.to_dict(), "B3": b3.id.tolist()}, default=str))
    print(json.dumps(proj, indent=1, default=str))


if __name__ == "__main__":
    os.chdir(P.ROOT)
    main()
