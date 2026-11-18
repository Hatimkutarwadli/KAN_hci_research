"""P2-2 C5: evaluate GRU and GRU+R0O0T1 (Stage 9 augmentation, pre-specified) on the Stage 8 scenarios for one dataset (SYNTHETIC; CALIBRATION split; inference only).
Same scenario registry, windows (origins 27,29,...,119), mask replicates and metric code as Stage 8/9 (extension_missing.run_missing.evaluate).
  python extension_p22/aug_p22.py --ds R1   -> results/p2_datasets/<ds>/augmentation/cal_metrics.csv"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p22 as C  # noqa: E402
import data as dd  # noqa: E402
import inject as ij  # noqa: E402
import models as md  # noqa: E402
import run_missing as rm  # noqa: E402
import v2  # noqa: E402

SCEN = ["C0", "S1_r40", "S2_r40_L7", "S3_k3", "S3_k7", "S3_k14", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S5_rand40", "S5_blk40"]    # P1-6 panel + clean control


def units(ds):
    us = []
    for fam, model in (("gru", "gru"), ("gru_aug", C.AUG)):
        for seed in C.SEEDS:
            ck = torch.load(os.path.join(C.ROOT, C.ckpt_path(ds, model, "main", seed)), weights_only=False)
            m = md.MODELS["gru"](n_out=len(ck["heads"]), n_ch=ck.get("n_ch", 10))
            m.load_state_dict(ck["state_dict"])
            us.append(dict(fam=fam, part="main", seed=seed, model=m.eval(), heads=list(ck["heads"]), ymean=ck["ymean"], ystd=ck["ystd"],
                           n_reg=sum(h.startswith("sev_") for h in ck["heads"])))
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    a = ap.parse_args()
    os.chdir(C.ROOT)
    torch.set_num_threads(4)
    out = os.path.join(C.RES, a.ds, "augmentation")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} not empty; refusing to overwrite")
    os.makedirs(out, exist_ok=True)
    d = dd.load_split("cal", C.proc_dir(a.ds))
    X0, pid, static = d["X"], d["pid"], d["static"]
    pi, ti = np.meshgrid(np.arange(len(pid)), np.arange(27, 120, 2), indexing="ij")
    pi, ti = pi.ravel(), ti.ravel()
    S_static = static[pi].astype(np.float32)
    R = ij.reconstruct_raw(X0)
    Rp = v2.pad_raw(R)
    us = units(a.ds)
    Y = {h: d["y_" + h][pi, ti] for h in us[0]["heads"]}
    reg = {s["id"]: s for s in ij.registry()}
    rows = []
    with torch.no_grad():
        for sid in SCEN:
            sc = reg[sid]
            for rep in sc["reps"]:
                S, S_orig = v2.scenario_slices(sc, rep, "cal", R, Rp, pi, ti)
                W = v2.represent(S, "R0", "O0", sc["flags"], S_orig)
                for u in us:
                    rows += rm.evaluate(rm.predict(u, W, S_static), u, Y, dict(dataset=a.ds, split="cal", scenario=sid, family=sc["family"], rep=rep))
            print(a.ds, sid, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "cal_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
