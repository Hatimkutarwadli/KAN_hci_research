"""P1-6 step 1 (SYNTHETIC; EXTENSION): forward-only regeneration of per-origin predictions on the TEST split for the clean control and the 10 pre-specified Stage 9 panel scenarios, for the
original (frozen) GRU and the three Stage 9 finalists (R0O0T1, R0O2T1, R0O3T1), 5 seeds, main and forecast task sets. NO retraining; existing checkpoints and results are only READ.
Uses the existing Stage 8/9 code paths unchanged (eval_v2.load_units, v2.scenario_slices/represent, run_missing.predict). The scenario masks come from the fixed Stage 8 seeds.
Writes results/p1_stage9_stats/predictions/{scenario}_rep{rep}.npz (new directory; refuses to overwrite) and a record of the decision-file hash.
Usage: python extension_p16/regen_p16.py"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "extension_v2")]
import data as dd  # noqa: E402
import eval_v2 as ev  # noqa: E402  (read-only reuse: load_units, repr_of)
import inject as ij  # noqa: E402
import run_missing as rm  # noqa: E402
import v2  # noqa: E402

OUT = "results/p1_stage9_stats"
PANEL = ["S1_r40", "S2_r40_L7", "S3_k3", "S3_k7", "S3_k14", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S5_rand40", "S5_blk40"]
SCEN = ["C0"] + PANEL
CONFIGS = ["baseline", "R0O0T1", "R0O2T1", "R0O3T1"]
PARTS = ("main", "forecast")
SEEDS = range(5)


def setup(split="test", stride=2):
    d = dd.load_split(split)
    pi, ti = np.meshgrid(np.arange(len(d["pid"])), np.arange(27, 120, stride), indexing="ij")
    pi, ti = pi.ravel(), ti.ravel()
    R = ij.reconstruct_raw(d["X"])
    return d, pi, ti, R, v2.pad_raw(R), d["static"][pi].astype(np.float32)


def main():
    os.chdir(ROOT)
    pred_dir = os.path.join(OUT, "predictions")
    if os.path.isdir(pred_dir) and os.listdir(pred_dir):
        raise SystemExit(f"{pred_dir} exists and is not empty; refusing to overwrite.")
    os.makedirs(pred_dir, exist_ok=True)
    dec = os.path.join(ROOT, "results/stage9_v2/decision/decision_cal.json")
    assert os.path.exists(dec), "calibration decision file must exist before the test split is used"
    sha = hashlib.sha256(open(dec, "rb").read()).hexdigest()
    torch.set_num_threads(6)
    d, pi, ti, R, Rp, S_static = setup()
    units = {(c, p): ev.load_units(c, taskset=p) for c in CONFIGS for p in PARTS}
    assert all(len(u) == 5 for u in units.values()), {k: len(u) for k, u in units.items()}
    reg = {s["id"]: s for s in ij.registry()}
    meta = {"decision_sha256": sha, "heads": {f"{c}|{p}": units[(c, p)][0]["heads"] for c in CONFIGS for p in PARTS}, "scenarios": SCEN, "configs": CONFIGS, "stride": 2}
    t_all = time.time()
    for sid in SCEN:
        sc = reg[sid]
        for rep in sc["reps"]:
            t0 = time.time()
            S, S_orig = v2.scenario_slices(sc, rep, "test", R, Rp, pi, ti)
            W, store = {}, {"pi": pi, "ti": ti, "pid": d["pid"]}
            for c in CONFIGS:
                key = ev.repr_of(c)
                if key not in W:
                    W[key] = v2.represent(S, key[0], key[1], sc["flags"], S_orig)
                for p in PARTS:
                    for u in units[(c, p)]:
                        with threadpool_limits(6):
                            store[f"{c}|{p}|{u['seed']}"] = rm.predict(u, W[key], S_static).astype(np.float32)
            np.savez_compressed(os.path.join(pred_dir, f"{sid}_rep{rep}.npz"), **store)
            print(f"{sid} rep{rep}: {time.time() - t0:.1f}s", flush=True)
    json.dump(meta, open(os.path.join(OUT, "regen_meta.json"), "w"), indent=1)
    print(f"done {time.time() - t_all:.0f}s; decision sha256 {sha}")


if __name__ == "__main__":
    main()
