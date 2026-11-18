"""P1-3 EXTENSION run: matched-parameter MLP controls (and the no-spline ablation) trained on the frozen processed SYNTHETIC data with the frozen protocol
(baselines.fit_nn defaults; epoch chosen on calibration loss; 5 seeds; no tuning). Each control uses the SAME training configuration as the KAN model it controls:
  B2 default      : python extension_p13/run_p13.py --taskset main --models mlp_b2 kan_nospline --out results/p1_kan_mlp_control/b2_main
  B2 regularised  : ... --models mlp_b2:wd=1e-3:drop=0.3:label=mlp_b2_reg   (= the Stage 3b kan_reg configuration, fixed there, not re-selected)
  M1              : python extension_p13/run_p13.py --taskset all --models mlp_m1:wd=1e-3:drop=0.3 --out results/p1_kan_mlp_control/m1_all   (= the Stage 4 M1 configuration)
Refuses to write into a non-empty output directory."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p13")]
import baselines as bl  # noqa: E402
import metrics as mt  # noqa: E402
import models as md  # noqa: E402
import models_p13 as m13  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", default="main", choices=["main", "forecast", "all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty; refusing to overwrite.")
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    bl.configure(a.taskset)
    m13.register()
    torch.set_num_threads(bl.THREADS)
    logf = open(os.path.join(a.out, "train_log.txt"), "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    data = bl.load_data("data/processed_v1", {"train": 80, "cal": 30, "test": 30} if a.quick else None)
    bl.counts_table(data).to_csv(os.path.join(a.out, "counts.csv"), index=False)
    rows, runlog, allP = [], [], {}
    for spec in a.models:
        base, *opts = spec.split(":")
        o = dict(x.split("=") for x in opts)
        name = o.get("label", base)
        mk = {"dropout": float(o["drop"])} if "drop" in o else {}
        if base.endswith("m1"):
            mk["heads"] = list(bl.HEADS)
        for seed in a.seeds:
            t0, c0 = time.perf_counter(), time.process_time()
            model, ym, ys, info = bl.fit_nn(base, seed, data["train"], data["cal"], epochs=2 if a.quick else 25, wd=float(o.get("wd", 1e-5)), model_kwargs=mk or None, log=log)
            P = {s: bl.predict_nn(model, bl.to_t(data[s]), ym, ys) for s in data}
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            torch.save({"state_dict": model.state_dict(), "ymean": ym, "ystd": ys, "heads": bl.HEADS, "seed": seed, "best_epoch": info["best_epoch"]},
                       os.path.join(a.out, "checkpoints", f"{name}_seed{seed}.pt"))
            r, thr = bl.evaluate(name, seed, P, data)
            rows += r
            bl.save_preds(a.out, name, seed, P, data)
            allP[(name, seed)] = P
            runlog.append(dict(model=name, seed=seed, wall_s=wall, cpu_s=cpu, torch_threads=bl.THREADS, n_params=md.n_params(model), n_trainable=m13.n_trainable(model),
                               thresholds_from_cal=thr, **info))
            log(f"{name} seed{seed}: wall {wall:.0f}s cpu {cpu:.0f}s params {md.n_params(model)} trainable {m13.n_trainable(model)} best_epoch {info['best_epoch']} epochs_run {info['epochs_run']}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "metrics_long.csv"), index=False)
    brow = []                                           # participant bootstrap CIs, seed 0, test, non-first-crossing tasks (as Stage 4 / P1-1)
    for (name, seed), P in allP.items():
        if seed != a.seeds[0]:
            continue
        for j, task in enumerate(bl.HEADS):
            if "cross" in task:
                continue
            y = data["test"]["Y"][:, j]
            ok = ~np.isnan(y)
            yy, pp, pid = y[ok], P["test"][ok, j], data["test"]["pid_o"][ok]
            fns = {"rmse": lambda u, v: mt.reg_metrics(u, v)["rmse"]} if task in bl.REG else \
                {"ap": lambda u, v: mt.clf_metrics(u, v)["ap"], "auroc": lambda u, v: mt.clf_metrics(u, v)["auroc"]}
            for mname, fn in fns.items():
                lo, hi = mt.participant_bootstrap(yy, pp, pid, fn, 100 if a.quick else 200)
                brow.append(dict(model=name, seed=seed, task=task, metric=mname, point=fn(yy, pp), ci95_lo=lo, ci95_hi=hi))
    pd.DataFrame(brow).to_csv(os.path.join(a.out, "bootstrap_ci_test.csv"), index=False)
    json.dump({"SYNTHETIC": True, "EXTENSION": "P1-3 matched-MLP control / no-spline ablation", "taskset": a.taskset, "heads": bl.HEADS, "args": vars(a), "runs": runlog,
               "torch": torch.__version__}, open(os.path.join(a.out, "run_log.json"), "w"), indent=1, default=str)
    logf.close()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
