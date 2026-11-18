"""P2-2: train the approved core panel (+ the pre-specified Stage 9 R0O0T1 augmentation GRU) on one versioned SYNTHETIC dataset, model seeds 0-2.
Frozen protocol (baselines.fit_nn / run_gbt / Stage 9 fit_aug), unchanged hyperparameters. TRAIN and CALIBRATION splits only: the test split is never loaded here.
  python extension_p22/run_p22.py --ds R1 [--models gbt gru kan_reg mlp_b2_reg m1 mlp_m1 gru_aug] [--out-root results/p2_datasets]
Each (model, task set) writes <out-root>/<ds>/<model>_<taskset>/ (refuses if non-empty): checkpoints, calibration predictions, run_log.json."""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p22 as C  # noqa: E402
import baselines as bl  # noqa: E402
import models as md  # noqa: E402
import models_p13  # noqa: E402

models_p13.register()


def save_cal(out, name, seed, P, d):
    os.makedirs(os.path.join(out, "predictions"), exist_ok=True)
    N = len(d["pid"])
    full = np.full((N, 120, len(bl.HEADS)), np.nan, np.float32)
    full[d["pi"], d["ti"]] = P
    np.savez_compressed(os.path.join(out, "predictions", f"{name}_seed{seed}_cal.npz"), pred=full, pid=d["pid"], heads=np.array(bl.HEADS))


def train_one(ds, model, taskset, out_root, seeds):
    out = os.path.join(out_root, ds, f"{model}_{taskset}")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite.")
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    bl.configure(taskset)
    data = bl.load_data(C.proc_dir(ds), None, splits=("train", "cal"))
    assert set(data) == {"train", "cal"}
    logf = open(os.path.join(out, "train_log.txt"), "w", encoding="utf-8")

    def log(m):
        logf.write(m + "\n")
        logf.flush()

    runs = []
    for seed in ([0] if model == "gbt" else seeds):           # GBT is deterministic here (as in Stage 3)
        t0, c0 = time.perf_counter(), time.process_time()
        info, n_par = {}, None
        if model == "gbt":
            with threadpool_limits(bl.THREADS):
                P, fitted, sel = bl.run_gbt(data, seed, (3, 6), 200, log)
            joblib.dump(fitted, os.path.join(out, "checkpoints", f"gbt_seed{seed}.joblib"))
            Pc, info = P["cal"], {"selection": sel}
        elif model == C.AUG:
            import inject as ij
            import train_v2 as tv
            import v2
            raw_train = ij.reconstruct_raw(data["train"]["X"])
            m, ym, ys, info = tv.fit_aug(seed, data["train"], data["cal"], raw_train, R="R0", O="O0", p_aug=v2.P_AUG, log=log)
            Pc = bl.predict_nn(m, bl.to_t(data["cal"]), ym, ys)
            torch.save({"state_dict": m.state_dict(), "ymean": ym, "ystd": ys, "heads": bl.HEADS, "seed": seed, "best_epoch": info["best_epoch"], "config": "R0O0T1", "n_ch": 10},
                       os.path.join(out, "checkpoints", f"{model}_seed{seed}.pt"))
            n_par = md.n_params(m)
        else:
            spec = C.CORE[model]
            mk = {"dropout": spec["dropout"]} if spec["dropout"] > 0 else {}
            if spec["arch"].endswith("m1"):
                mk["heads"] = list(bl.HEADS)
            m, ym, ys, info = bl.fit_nn(spec["arch"], seed, data["train"], data["cal"], wd=spec["wd"], model_kwargs=mk or None, log=log)
            Pc = bl.predict_nn(m, bl.to_t(data["cal"]), ym, ys)
            torch.save({"state_dict": m.state_dict(), "ymean": ym, "ystd": ys, "heads": bl.HEADS, "seed": seed, "best_epoch": info["best_epoch"]},
                       os.path.join(out, "checkpoints", f"{model}_seed{seed}.pt"))
            n_par = md.n_params(m)
        save_cal(out, model, seed, Pc, data["cal"])
        runs.append(dict(model=model, taskset=taskset, seed=seed, wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0, n_params=n_par,
                         best_cal_loss=min(h["cal_loss"] for h in info["history"]) if "history" in info else None,
                         **{k: v for k, v in info.items() if k not in ("history",)}, history=info.get("history")))
        print(f"{ds} {model} {taskset} seed{seed}: {runs[-1]['wall_s']:.0f}s", flush=True)
    json.dump({"SYNTHETIC": True, "dataset": ds, "test_split_loaded": False, "heads": bl.HEADS, "runs": runs, "torch_threads": bl.THREADS, "torch": torch.__version__},
              open(os.path.join(out, "run_log.json"), "w"), indent=1, default=str)
    logf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--models", nargs="+", default=list(C.CORE) + [C.AUG])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.SEEDS))
    ap.add_argument("--out-root", default=C.RES)
    ap.add_argument("--allow-gate-fail", action="store_true")
    a = ap.parse_args()
    os.chdir(C.ROOT)
    torch.set_num_threads(bl.THREADS)
    if a.ds == "v1" and a.out_root == C.RES:
        raise SystemExit("v1 results already exist; retraining v1 is only allowed into a separate --out-root (reproduction test)")
    if a.ds != "v1":
        g = json.load(open(os.path.join(C.RES, a.ds, "validation.json")))["gates"]
        if not g["all_pass"]:                        # approved Phase B policy: a failing dataset is retained, flagged and reported, never reseeded/retuned/excluded
            assert a.allow_gate_fail, "validation gate failed; rerun with --allow-gate-fail (dataset is then flagged as a limitation)"
            os.makedirs(os.path.join(C.RES, a.ds), exist_ok=True)
            json.dump({"dataset": a.ds, "gates": g, "policy": "retained and flagged; not reseeded, retuned or excluded"},
                      open(os.path.join(C.RES, a.ds, "GATE_FAIL_FLAG.json"), "w"), indent=1)
    for model in a.models:
        for ts in (("main",) if model == C.AUG else C.CORE[model]["tasksets"]):
            train_one(a.ds, model, ts, a.out_root, a.seeds)


if __name__ == "__main__":
    main()
