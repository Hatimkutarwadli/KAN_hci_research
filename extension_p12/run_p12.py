"""P1-2 run: single-task / single-family / budget-matched GRUs on the frozen processed SYNTHETIC data (frozen fit_nn protocol, 5 seeds, no tuning).
Usage (project root): python extension_p12/run_p12.py --design st_task --part main --out results/p1_multitask_ablation/st_task_main
Designs: st_task | st_task_matched | family. Refuses a non-empty output directory. Test predictions are produced once after the model is final."""
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import common_p12 as C  # noqa: E402

bl, md = C.bl, C.md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=["st_task", "st_task_matched", "family"])
    ap.add_argument("--part", required=True, choices=["main", "forecast"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.chdir(C.ROOT)
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty; refusing to overwrite.")
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    bl.configure("all")
    torch.set_num_threads(bl.THREADS)
    logf = open(os.path.join(a.out, "train_log.txt"), "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    data_all = bl.load_data("data/processed_v1", {"train": 80, "cal": 30, "test": 30} if a.quick else None)
    P = C.PARTS[a.part]
    units = {h: [h] for h in P["st"]} if a.design != "family" else {f"family_{k}": v for k, v in P["family"].items()}
    mt_ep = C.mt_best_epochs(a.part)
    cfg = dict(C.CFG, epochs=2 if a.quick else C.CFG["epochs"])
    rows, runlog = [], []
    for uname, uheads in units.items():
        heads = C.configure_unit(uheads)
        data = C.unit_data(data_all, heads)
        for seed in a.seeds:
            t0, c0 = time.perf_counter(), time.process_time()
            if a.design == "st_task_matched":
                model, ym, ys, info = C.fit_fixed(seed, data["train"], data["cal"], 2 if a.quick else mt_ep[seed], log=log)
            else:
                model, ym, ys, info = bl.fit_nn("gru", seed, data["train"], data["cal"], log=log, **cfg)
            Pr = {s: bl.predict_nn(model, bl.to_t(data[s]), ym, ys) for s in data}
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            torch.save({"state_dict": model.state_dict(), "ymean": ym, "ystd": ys, "heads": heads, "seed": seed, "best_epoch": info["best_epoch"]},
                       os.path.join(a.out, "checkpoints", f"gru_{uname}_seed{seed}.pt"))
            r, thr = bl.evaluate(a.design, seed, Pr, data)
            rows += [dict(x, unit=uname) for x in r]
            bl.save_preds(a.out, f"{uname}", seed, Pr, data)
            runlog.append(dict(design=a.design, unit=uname, heads=heads, seed=seed, wall_s=wall, cpu_s=cpu, n_params=md.n_params(model), torch_threads=bl.THREADS,
                               steps_per_epoch=-(-int(cfg["epoch_frac"] * len(data["train"]["pi"])) // cfg["bs"]), mt_best_epoch=mt_ep.get(seed), **info))
            log(f"{a.design} {uname} seed{seed}: wall {wall:.0f}s cpu {cpu:.0f}s params {md.n_params(model)} best_epoch {info['best_epoch']} epochs_run {info['epochs_run']}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "metrics_long.csv"), index=False)
    json.dump({"SYNTHETIC": True, "EXTENSION": "P1-2 multitask vs single-task", "design": a.design, "part": a.part, "training_config": cfg, "units": units, "args": vars(a),
               "runs": runlog, "torch": torch.__version__}, open(os.path.join(a.out, "run_log.json"), "w"), indent=1, default=str)
    logf.close()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
