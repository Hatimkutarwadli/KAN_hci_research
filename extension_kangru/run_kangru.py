"""EXTENSION run: KANGRUNet (adaptive) and KANGRU-fixed on the frozen processed SYNTHETIC data, all 28 heads, frozen protocol
(baselines.fit_nn: train-only fitting, epoch selected on calibration loss, thresholds from calibration, test predicted after selection).
Identical training config for both models (defaults of fit_nn; no tuning). Usage (from project root):
  python extension_kangru/run_kangru.py --seeds 0 1 2 3 4 --out results/ext_kangru"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_kangru")]
import baselines as bl  # noqa: E402
import metrics as mt  # noqa: E402
import models as md  # noqa: E402
import models_kangru as mk  # noqa: E402

CFG = dict(epochs=25, patience=4, bs=256, lr=2e-3, wd=1e-5, epoch_frac=0.5)       # = frozen fit_nn defaults, same for both models


@torch.no_grad()
def mean_gate(model, d):
    T = bl.to_t(d)
    model.eval()
    g = []
    for a in range(0, len(T["pi"]), 2048):
        x, st, _ = bl.batch(T, torch.arange(a, min(a + 2048, len(T["pi"]))))
        g.append(model(x, st, return_gate=True)[1])
    g = torch.cat(g)                                  # (M, 28, 3)
    return {"mean_per_scale": g.mean((0, 1)).tolist(), "mean_last_day": g[:, -1].mean(0).tolist(),
            "std_over_origins_last_day": g[:, -1].std(0).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=["kangru", "kangru_fixed"])
    ap.add_argument("--out", default="results/ext_kangru")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty; refusing to overwrite.")
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    bl.configure("all")
    mk.register()
    torch.set_num_threads(bl.THREADS)
    logf = open(os.path.join(a.out, "train_log.txt"), "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    data = bl.load_data("data/processed_v1", {"train": 80, "cal": 30, "test": 30} if a.quick else None)
    cfg = dict(CFG, epochs=2 if a.quick else CFG["epochs"])
    rows, runlog, allP, gates = [], [], {}, {}
    for name in a.models:
        for seed in a.seeds:
            t0 = time.perf_counter()
            model, ym, ys, info = bl.fit_nn(name, seed, data["train"], data["cal"], model_kwargs={"heads": list(bl.HEADS)}, log=log, **cfg)
            P = {s: bl.predict_nn(model, bl.to_t(data[s]), ym, ys) for s in data}
            torch.save({"state_dict": model.state_dict(), "ymean": ym, "ystd": ys, "heads": bl.HEADS, "seed": seed,
                        "best_epoch": info["best_epoch"]}, os.path.join(a.out, "checkpoints", f"{name}_seed{seed}.pt"))
            r, thr = bl.evaluate(name, seed, P, data)
            rows += r
            bl.save_preds(a.out, name, seed, P, data)
            allP[(name, seed)] = P
            if name == "kangru":
                gates[f"seed{seed}"] = {"cal": mean_gate(model, data["cal"])}
            runlog.append(dict(model=name, seed=seed, wall_s=time.perf_counter() - t0, n_params=md.n_params(model), thresholds_from_cal=thr, **info))
            log(f"{name} seed{seed}: wall {runlog[-1]['wall_s']:.0f}s best_epoch {info['best_epoch']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "metrics_long.csv"), index=False)
    brow = []                                           # participant bootstrap CIs, seed 0, test, non-first-crossing tasks (as Stage 4)
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
    json.dump({"SYNTHETIC": True, "EXTENSION": "KAN-GRU", "heads": bl.HEADS, "training_config": cfg, "args": vars(a), "runs": runlog,
               "gate_weights_adaptive": gates, "torch": torch.__version__}, open(os.path.join(a.out, "run_log.json"), "w"), indent=1, default=str)
    logf.close()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
