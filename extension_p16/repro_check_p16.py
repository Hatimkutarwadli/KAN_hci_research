"""P1-6 reproduction gate (SYNTHETIC; EXTENSION): metrics recomputed from the regenerated per-origin predictions (pooled over all test participants, exactly as Stage 8/9 did) must reproduce the
EXISTING saved CSVs: Stage 8 test metrics (original GRU) and Stage 9 test metrics (R0O0T1, R0O2T1, R0O3T1), for every scenario/replicate/seed/task/metric of the 11 scenarios used here.
Exit code 1 (and a discrepancy report) if any value deviates by more than TOL; nothing downstream may run in that case. Usage: python extension_p16/repro_check_p16.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "extension_p16")]
import inject as ij  # noqa: E402
import regen_p16 as rg  # noqa: E402
import run_missing as rm  # noqa: E402

TOL = 1e-5
OUT = "results/p1_stage9_stats"


def saved_metrics(c, sid, rep):
    out = {}
    for p in rg.PARTS:
        if c == "baseline":
            df = pd.read_csv(f"results/stage8_missing/test/metrics/{sid}_rep{rep}.csv")
            df = df[(df.model == "gru") & (df.part == p)]
        else:
            df = pd.read_csv(f"results/stage9_v2/test/metrics_{c}{'' if p == 'main' else '_forecast'}/{sid}_rep{rep}.csv")
            df = df[df.part == p]
        out[p] = df
    return out


def recompute(pred, meta, Y, c, p, seed, sc, rep):
    u = dict(fam=c, part=p, seed=seed, heads=meta["heads"][f"{c}|{p}"])
    return pd.DataFrame(rm.evaluate(pred[f"{c}|{p}|{seed}"], u, Y, dict(split="test", scenario=sc["id"], family=sc["family"], rep=rep)))


def main():
    os.chdir(ROOT)
    meta = json.load(open(os.path.join(OUT, "regen_meta.json")))
    d, pi, ti, *_ = rg.setup()
    heads = sorted({h for v in meta["heads"].values() for h in v})
    Y = {h: d["y_" + h][pi, ti] for h in heads}
    reg = {s["id"]: s for s in ij.registry()}
    worst, n, bad = [], 0, []
    for sid in meta["scenarios"]:
        sc = reg[sid]
        for rep in sc["reps"]:
            pred = np.load(os.path.join(OUT, "predictions", f"{sid}_rep{rep}.npz"))
            assert np.array_equal(pred["pi"], pi) and np.array_equal(pred["ti"], ti) and np.array_equal(pred["pid"], d["pid"])
            for c in meta["configs"]:
                sv = saved_metrics(c, sid, rep)
                for p in rg.PARTS:
                    for seed in rg.SEEDS:
                        new = recompute(pred, meta, Y, c, p, seed, sc, rep)
                        old = sv[p][sv[p].seed == seed]
                        m = new.merge(old, on=["task", "metric"], suffixes=("_new", "_old"), how="outer", indicator=True)
                        if (m["_merge"] != "both").any():
                            bad.append(dict(scenario=sid, rep=rep, config=c, part=p, seed=seed, issue="row mismatch", n=int((m["_merge"] != "both").sum())))
                            continue
                        diff = (m.value_new - m.value_old).abs()
                        n += len(m)
                        k = int(np.argmax(diff.values))
                        worst.append((float(diff.max()), sid, rep, c, p, seed, m.task.iloc[k], m.metric.iloc[k], float(m.value_new.iloc[k]), float(m.value_old.iloc[k])))
                        if diff.max() > TOL:
                            bad.append(dict(scenario=sid, rep=rep, config=c, part=p, seed=seed, issue="value", max_abs=float(diff.max()), task=m.task.iloc[k], metric=m.metric.iloc[k]))
    worst.sort(reverse=True)
    res = dict(n_values_compared=n, tolerance=TOL, max_abs_diff=worst[0][0] if worst else None, worst5=worst[:5], n_failures=len(bad), failures=bad[:50])
    os.makedirs(os.path.join(OUT, "repro"), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "repro", "repro_check.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_values_compared", "tolerance", "max_abs_diff", "n_failures")}))
    if bad:
        print("REPRODUCTION FAILED - stop and report the discrepancies (see repro_check.json).")
        sys.exit(1)
    print("REPRODUCTION OK")


if __name__ == "__main__":
    main()
