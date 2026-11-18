"""Stage 9: evaluate GRU checkpoints on the Stage 8 scenarios (same windows, origins 27,29,...,119, metrics, replicates). SYNTHETIC. Inference only.
The TEST split is locked: this script refuses split=test until the user approves final evaluation.
  python extension_v2/eval_v2.py --split cal --configs R0O0T1 R1O0T0 R0O2T1 [--scenarios C0 S1_r20 ...]
  config 'baseline' = existing frozen GRU main checkpoints (R0O0T0) through the same code (equivalence check vs Stage 8)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "extension_v2")]
import data as dd  # noqa: E402
import inject as ij  # noqa: E402
import models as md  # noqa: E402
import run_missing as rm  # noqa: E402  (Stage 8 predict/evaluate, read-only import)
import v2  # noqa: E402


def load_units(config, seeds=range(5), taskset="main"):
    d = ("results/stage3" if taskset == "main" else "results/stage3c_forecast") if config == "baseline" else "results/stage9_v2"
    tag = "" if taskset == "main" else "_forecast"
    units = []
    for s in seeds:
        fn = f"gru_seed{s}.pt" if config == "baseline" else f"gru_{config}{tag}_seed{s}.pt"
        p = os.path.join(d, "checkpoints", fn)
        if not os.path.exists(p):
            continue
        ck = torch.load(p, weights_only=False)
        m = md.MODELS["gru"](n_out=len(ck["heads"]), n_ch=ck.get("n_ch", 10))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        units.append(dict(fam="gru_" + config, part=taskset, seed=s, model=m, heads=list(ck["heads"]), ymean=ck["ymean"], ystd=ck["ystd"], n_reg=sum(h.startswith("sev_") for h in ck["heads"])))
    return units


def repr_of(config):
    return ("R0", "O0") if config == "baseline" else v2.parse_config(config)[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["cal", "test"])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["all"])
    ap.add_argument("--out", default="results/stage9_v2")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--taskset", default="main", choices=["main", "forecast"])
    ap.add_argument("--decision-sha", default=None, help="sha256 of results/stage9_v2/decision/decision_cal.json; required for split=test")
    a = ap.parse_args()
    if a.split == "test":
        import hashlib
        dp = os.path.join(ROOT, "results/stage9_v2/decision/decision_cal.json")
        if not (os.path.exists(dp) and a.decision_sha == hashlib.sha256(open(dp, "rb").read()).hexdigest()):
            raise SystemExit("STOP: the test split is locked until the calibration decision file exists and its sha256 is passed via --decision-sha.")
    mtag = "" if a.taskset == "main" else "_forecast"
    assert a.stride == 2, "Stage 8 origin set is every second day"
    os.chdir(ROOT)
    torch.set_num_threads(a.threads)
    reg = ij.registry()
    scen = reg if a.scenarios == ["all"] else [s for s in reg if s["id"] in a.scenarios]
    d = dd.load_split(a.split)
    X0, pid, static = d["X"], d["pid"], d["static"]
    pi, ti = np.meshgrid(np.arange(len(pid)), np.arange(27, 120, a.stride), indexing="ij")
    pi, ti = pi.ravel(), ti.ravel()
    S_static = static[pi].astype(np.float32)
    R = ij.reconstruct_raw(X0)
    Rp = v2.pad_raw(R)
    units = {c: load_units(c, taskset=a.taskset) for c in a.configs}
    assert all(units.values()), {c: len(u) for c, u in units.items()}
    heads_all = sorted({h for us in units.values() for u in us for h in u["heads"]})
    Y = {h: d["y_" + h][pi, ti] for h in heads_all}
    for c in a.configs:
        os.makedirs(os.path.join(a.out, a.split, f"metrics_{c}{mtag}"), exist_ok=True)
    for sc in scen:
        for rep in sc["reps"]:
            todo = [c for c in a.configs if not os.path.exists(os.path.join(a.out, a.split, f"metrics_{c}{mtag}", f"{sc['id']}_rep{rep}.csv"))]
            if not todo:
                continue
            t0 = time.perf_counter()
            S, S_orig = v2.scenario_slices(sc, rep, a.split, R, Rp, pi, ti)
            Wc = {}
            for c in todo:
                key = repr_of(c)
                if key not in Wc:
                    Wc[key] = v2.represent(S, key[0], key[1], sc["flags"], S_orig)
                rows = []
                with threadpool_limits(a.threads):
                    for u in units[c]:
                        P = rm.predict(u, Wc[key], S_static)
                        rows += rm.evaluate(P, u, Y, dict(split=a.split, scenario=sc["id"], family=sc["family"], rep=rep))
                pd.DataFrame(rows).to_csv(os.path.join(a.out, a.split, f"metrics_{c}{mtag}", f"{sc['id']}_rep{rep}.csv"), index=False)
            with open(os.path.join(a.out, a.split, "timing_eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(scenario=sc["id"], rep=rep, configs=todo, total_s=time.perf_counter() - t0)) + "\n")
            print(f"{sc['id']} rep{rep}: {len(todo)} configs {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
