"""P3 locked TEST analysis (SYNTHETIC) implementing P3_TEST_EVALUATION_PLAN.md: the calibration analysis rules applied unchanged to test metrics.
  python extension_p3/analyze_test_p3.py                     test metrics (results/p3_small_data/test/metrics) -> results/p3_small_data/analysis_test/
  python extension_p3/analyze_test_p3.py --dry-run-cal OUT   pre-lock check: same code on the CALIBRATION metrics must reproduce analysis_cal exactly
Uses analyze_p3 functions unchanged (families, units, compare, rq1a [amendment 2], rq1b, all_pairs). No model, threshold or rule is changed."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p3 as P  # noqa: E402
import analyze_p3 as AN  # noqa: E402


def collect_metrics(source):
    rows = []
    for ds in P.DATASETS:
        for j in P.jobs_main(ds):
            f = (os.path.join(P.RES, "test", "metrics", ds, j["id"] + ".json") if source == "test"
                 else os.path.join(P.RES, "runs", ds, j["id"], "metrics.json"))
            for fam, (met, v) in AN.families(json.load(open(f))).items():
                rows.append(dict(ds=ds, variant="primary", model=j["model"], n=j["n"], draw=j["draw"], seed=j["seed"], family=fam, metric=met, value=v))
    return pd.DataFrame(rows)


def analyse(F, out, split):
    os.makedirs(out, exist_ok=True)
    F.to_csv(os.path.join(out, "family_values.csv"), index=False)
    lc = []
    for (ds, model, fam, n), _ in F.groupby(["ds", "model", "family", "n"]):
        u = AN.units(F, ds, n, model, fam)
        lc.append(dict(ds=ds, model=model, family=fam, n=n, mean=float(u.mean()), sd=float(u.std(ddof=1)) if len(u) > 1 else np.nan, n_units=len(u)))
    pd.DataFrame(lc).to_csv(os.path.join(out, "learning_curves.csv"), index=False)
    Pr = AN.all_pairs(F, P.DATASETS, P.SIZES)
    Pr.to_csv(os.path.join(out, "paired.csv"), index=False)
    res = {"SYNTHETIC": True, "split": split}
    for n in (50, 100):
        la, ia, Ta = AN.rq1a(Pr, n)
        lb, ib, Tb = AN.rq1b(Pr, n)
        res[f"RQ1a_n{n}"] = dict(label=la, **ia)
        res[f"RQ1b_n{n}"] = dict(label=lb, **ib)
        Ta.to_csv(os.path.join(out, f"RQ1a_n{n}.csv"), index=False)
        Tb.to_csv(os.path.join(out, f"RQ1b_n{n}.csv"), index=False)
    res["secondary_counts"] = {n: dict(RQ1a_counts=AN.rq1a(Pr, n)[1], RQ1b_counts=AN.rq1b(Pr, n)[1]) for n in (200, 400, 600)}
    return res, Pr


def cal_vs_test(Pc, Pt, out):
    k = ["ds", "n", "A", "B", "family"]
    M = Pc[k + ["mean_diff", "verdict", "consistent_better"]].merge(Pt[k + ["mean_diff", "verdict", "consistent_better"]], on=k, suffixes=("_cal", "_test"))
    M["same_sign"] = np.sign(M.mean_diff_cal) == np.sign(M.mean_diff_test)
    M["same_better_label"] = M.consistent_better_cal.fillna("") == M.consistent_better_test.fillna("")
    M["direction_reversal"] = ((M.consistent_better_cal == "A") & (M.consistent_better_test == "B")) | ((M.consistent_better_cal == "B") & (M.consistent_better_test == "A"))
    M.to_csv(os.path.join(out, "cal_vs_test_paired.csv"), index=False)
    s = {}
    for n, g in M.groupby("n"):
        s[int(n)] = dict(comparisons=len(g), same_sign=int(g.same_sign.sum()), same_better_label=int(g.same_better_label.sum()),
                         direction_reversals=int(g.direction_reversal.sum()))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-cal", default=None)
    a = ap.parse_args()
    os.chdir(P.ROOT)
    if a.dry_run_cal:
        res, Pr = analyse(collect_metrics("cal"), a.dry_run_cal, "cal")
        ref = json.load(open(os.path.join(P.RES, "analysis_cal", "conclusions_cal.json")))
        refP = pd.read_csv(os.path.join(P.RES, "analysis_cal", "paired.csv"))
        ok = all(json.loads(json.dumps(res[k], default=str)) == ref[k] for k in ("RQ1a_n50", "RQ1a_n100", "RQ1b_n50", "RQ1b_n100"))
        ok &= json.loads(json.dumps(res["secondary_counts"], default=str)) == ref["secondary_counts"]
        k = ["ds", "n", "A", "B", "family"]
        m = refP.merge(Pr, on=k, suffixes=("_r", "_n"))
        ok &= len(m) == len(refP) == len(Pr) and bool(np.allclose(m.mean_diff_r, m.mean_diff_n, atol=1e-12, rtol=0))
        ok &= bool((m.consistent_better_r.fillna("") == m.consistent_better_n.fillna("")).all())
        print("DRY RUN reproduces analysis_cal:", ok)
        assert ok
        return
    out = os.path.join(P.RES, "analysis_test")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} not empty; refusing to overwrite")
    log = json.load(open(os.path.join(P.RES, "test", "test_access_log.json")))
    assert log.get("n_jobs") == 416 and log.get("all_finite"), "test evaluation incomplete or non-finite"
    res, Pt = analyse(collect_metrics("test"), out, "test")
    Pc = pd.read_csv(os.path.join(P.RES, "analysis_cal", "paired.csv"))
    res["cal_vs_test"] = cal_vs_test(Pc, Pt, out)
    cal = json.load(open(os.path.join(P.RES, "analysis_cal", "conclusions_cal.json")))
    res["labels_cal_vs_test"] = {k: dict(cal=cal[k]["label"], test=res[k]["label"]) for k in ("RQ1a_n50", "RQ1a_n100", "RQ1b_n50", "RQ1b_n100")}
    res["test_access"] = log
    json.dump(res, open(os.path.join(out, "conclusions_test.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("RQ1", "labels", "cal_vs"))}, indent=1, default=str))


if __name__ == "__main__":
    main()
