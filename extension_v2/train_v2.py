"""Stage 9: GRU trained under a factorial configuration R (flags) x O (outage fill) x T (augmentation), e.g. R1O0T0, R0O2T1. SYNTHETIC. TRAIN and CALIBRATION splits only; the test split is never loaded.
Same as frozen baselines.fit_nn (seed, init, batch order, optimizer, epoch selection on CLEAN calibration loss) except that each training window is
rebuilt from the raw standardized series with augmentation. With p_aug=0 it reproduces fit_nn exactly (tested).
Usage: python extension_v2/train_v2.py --configs R1O0T0 R0O1T1 --seeds 0 1 2 3 4 --out results/stage9_v2
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "extension_v2")]
import baselines as bl  # noqa: E402
import models as md  # noqa: E402
import inject as ij  # noqa: E402
import v2  # noqa: E402

CFG = dict(epochs=25, patience=4, bs=256, lr=2e-3, wd=1e-5, epoch_frac=0.5)      # = frozen fit_nn defaults


def fit_aug(seed, tr, ca, raw_train, R="R0", O="O0", p_aug=v2.P_AUG, rmax=v2.RMAX, epochs=CFG["epochs"], patience=CFG["patience"], bs=CFG["bs"], lr=CFG["lr"], wd=CFG["wd"],
            epoch_frac=CFG["epoch_frac"], log=print):
    """tr, ca: dicts from baselines.load_data (train / cal). raw_train: (N_train,120,8) standardized values with NaN (reconstructed from the frozen X)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = md.MODELS["gru"](n_out=len(bl.HEADS), n_ch=v2.n_channels(R))
    y = tr["Y"][:, :bl.N_REG]
    ymean = torch.tensor(np.nanmean(y, 0), dtype=torch.float32)
    ystd = torch.tensor(np.nanstd(y, 0), dtype=torch.float32)
    Tt = bl.to_t(tr)
    # clean calibration windows under THIS configuration's representation (R, O); for R0.O0 identical to the frozen windows (tested)
    Wc = v2.represent(v2.slice31(v2.pad_raw(ij.reconstruct_raw(ca["X"])), ca["pi"], ca["ti"]), R, O)
    Tc = {"X": torch.from_numpy(Wc), "S": torch.from_numpy(ca["static"][ca["pi"]]), "pi": torch.arange(len(Wc)), "ti": torch.full((len(Wc),), 27), "Y": torch.from_numpy(ca["Y"])}
    Rp = v2.pad_raw(raw_train)
    arng = np.random.default_rng([seed, 9109])                  # augmentation stream is separate from torch's (perm / init identical to baseline)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    best, hist, bad, n = float("inf"), [], 0, len(tr["pi"])
    best_state, best_ep = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)[:int(epoch_frac * n)]
        m_, run = len(perm), 0.0
        for a in range(0, m_, bs):
            idx = perm[a:a + bs]
            S = v2.slice31(Rp, tr["pi"][idx.numpy()], tr["ti"][idx.numpy()])
            if p_aug > 0:
                S = v2.augment(S, arng, p_aug, rmax)
            x = torch.from_numpy(v2.represent(S, R, O))
            loss = bl.masked_loss(model(x, Tt["S"][Tt["pi"][idx]]), Tt["Y"][idx], ymean, ystd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(idx)
        cl = bl.eval_loss(model, Tc, ymean, ystd)
        hist.append({"epoch": ep, "train_loss": run / m_, "cal_loss": cl})
        log(f"  gru {R}{O}T{int(p_aug > 0)} seed{seed} ep{ep} train {run / m_:.4f} cal {cl:.4f}")
        if cl < best - 1e-5:
            best, best_state, best_ep, bad = cl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, ymean, ystd, {"best_epoch": best_ep, "epochs_run": len(hist), "history": hist,
                                "hparams": dict(epochs=epochs, patience=patience, batch_size=bs, lr=lr, weight_decay=wd, epoch_frac=epoch_frac, p_aug=p_aug, rmax=rmax, R=R, O=O)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["R0O0T1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="results/stage9_v2")
    ap.add_argument("--taskset", default="main", choices=["main", "forecast"])
    a = ap.parse_args()
    os.chdir(ROOT)
    bl.configure(a.taskset)
    tag = "" if a.taskset == "main" else "_forecast"
    torch.set_num_threads(bl.THREADS)
    ck = os.path.join(a.out, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    data = bl.load_data("data/processed_v1", splits=("train", "cal"))          # test is never loaded
    assert set(data) == {"train", "cal"}
    raw_train = ij.reconstruct_raw(data["train"]["X"])
    for cfg in a.configs:
        R, O, T = v2.parse_config(cfg)
        logf = open(os.path.join(a.out, f"train_log_{cfg}{tag}.txt"), "a", encoding="utf-8")

        def log(m):
            print(m, flush=True)
            logf.write(m + "\n")
            logf.flush()

        for seed in a.seeds:
            path = os.path.join(ck, f"gru_{cfg}{tag}_seed{seed}.pt")
            if os.path.exists(path):
                log(f"skip {cfg} seed {seed}: checkpoint exists")
                continue
            t0 = time.perf_counter()
            model, ym, ys, info = fit_aug(seed, data["train"], data["cal"], raw_train, R=R, O=O, p_aug=v2.P_AUG if T else 0.0, log=log)
            wall = time.perf_counter() - t0
            torch.save({"state_dict": model.state_dict(), "ymean": ym, "ystd": ys, "heads": bl.HEADS, "seed": seed, "best_epoch": info["best_epoch"], "config": cfg, "n_ch": v2.n_channels(R)}, path)
            rec = dict(config=cfg, taskset=a.taskset, seed=seed, wall_s=wall, n_params=md.n_params(model), threads=bl.THREADS, **info)
            with open(os.path.join(a.out, "train_runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"{cfg} seed{seed}: wall {wall:.0f}s best_epoch {info['best_epoch']} epochs_run {info['epochs_run']}")


if __name__ == "__main__":
    main()
