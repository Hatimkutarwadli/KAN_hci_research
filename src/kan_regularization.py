"""Limited regularization pass for B2 (static multi-scale KAN). TRAIN + CALIBRATION ONLY: the test split is never loaded.

Pre-declared grid (original config wd=1e-5, dropout=0 is reused from the Stage 3 logs, not rerun):
    weight decay {1e-5, 1e-3, 1e-2} x dropout {0, 0.3}  ->  5 new configs, selection seeds {0, 1}
Selection key: mean over the selection seeds of the best CALIBRATION loss (the same masked multitask loss
used for early stopping, all 16 heads equally weighted). Calibration AP/RMSE are recorded for information.

Run one config:  python src/kan_regularization.py run --wd 1e-3 --drop 0.3 --seed 0 --out results/stage3b_kan_reg/selection
Aggregate:       python src/kan_regularization.py select --out results/stage3b_kan_reg/selection
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import baselines as bl
import metrics as mt

GRID = [(1e-3, 0.0), (1e-2, 0.0), (1e-5, 0.3), (1e-3, 0.3), (1e-2, 0.3)]
SEEDS = (0, 1)


def run(wd, drop, seed, out, threads):
    torch.set_num_threads(threads)
    data = bl.load_data(splits=("train", "cal"))          # test split deliberately NOT loaded
    assert set(data) == {"train", "cal"}
    mk = {"dropout": drop} if drop > 0 else None
    model, ym, ys, info = bl.fit_nn("kan", seed, data["train"], data["cal"], wd=wd, model_kwargs=mk, log=lambda *_: None)
    P = bl.predict_nn(model, bl.to_t(data["cal"]), ym, ys)
    Y = data["cal"]["Y"]
    res = {"wd": wd, "dropout": drop, "seed": seed, "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
           "best_cal_loss": min(h["cal_loss"] for h in info["history"]), "history": info["history"]}

    def mean_metric(names, fn):
        v = []
        for t in names:
            j = bl.HEADS.index(t)
            ok = ~np.isnan(Y[:, j])
            v.append(fn(Y[ok, j], P[ok, j]))
        return float(np.mean(v))
    ap = lambda y, p: mt.clf_metrics(y, p)["ap"]
    res["cal_ap_status_mean"] = mean_metric(bl.PRIMARY[2:4], ap)
    res["cal_ap_onset_sust_mean"] = mean_metric(bl.PRIMARY[4:], ap)
    res["cal_rmse_sev_mean"] = mean_metric(bl.PRIMARY[:2], lambda y, p: mt.reg_metrics(y, p)["rmse"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"kan_wd{wd:g}_drop{drop:g}_seed{seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("done", wd, drop, seed, res["best_epoch"], round(res["best_cal_loss"], 4))


def select(out, orig_log="results/stage3/run_log.json"):
    rows = []
    for fn in sorted(glob.glob(os.path.join(out, "kan_wd*_seed*.json"))):
        with open(fn) as f:
            rows.append(json.load(f))
    with open(orig_log) as f:
        orig = [r for r in json.load(f)["runs"] if r["model"] == "kan" and r["seed"] in SEEDS]
    cal = bl.load_data(splits=("cal",))["cal"]                      # calibration labels only
    for r in orig:   # original config, reused from Stage 3 logs + its saved CALIBRATION predictions (test file never read)
        pred = np.load(f"results/stage3/predictions/kan_seed{r['seed']}_cal.npz")["pred"][cal["pi"], cal["ti"]]
        ap = lambda t: float(np.mean([mt.clf_metrics(cal["Y"][~np.isnan(cal["Y"][:, bl.HEADS.index(x)]), bl.HEADS.index(x)],
                                                      pred[~np.isnan(cal["Y"][:, bl.HEADS.index(x)]), bl.HEADS.index(x)])["ap"] for x in t]))
        rows.append({"wd": 1e-5, "dropout": 0.0, "seed": r["seed"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                     "best_cal_loss": min(h["cal_loss"] for h in r["history"]), "history": r["history"], "source": "stage3 log",
                     "cal_ap_status_mean": ap(bl.PRIMARY[2:4]), "cal_ap_onset_sust_mean": ap(bl.PRIMARY[4:])})
    cfgs = {}
    for r in rows:
        cfgs.setdefault((r["wd"], r["dropout"]), []).append(r)
    table = []
    for (wd, dr), rs in sorted(cfgs.items()):
        table.append({"wd": wd, "dropout": dr, "n_seeds": len(rs), "mean_best_cal_loss": float(np.mean([r["best_cal_loss"] for r in rs])),
                      "mean_best_epoch": float(np.mean([r["best_epoch"] for r in rs])),
                      "final_epoch_train_minus_cal_loss": float(np.mean([r["history"][-1]["train_loss"] - r["history"][-1]["cal_loss"] for r in rs])),
                      "cal_ap_status": float(np.mean([r["cal_ap_status_mean"] for r in rs if "cal_ap_status_mean" in r])) if any("cal_ap_status_mean" in r for r in rs) else None,
                      "cal_ap_onset_sust": float(np.mean([r["cal_ap_onset_sust_mean"] for r in rs if "cal_ap_onset_sust_mean" in r])) if any("cal_ap_onset_sust_mean" in r for r in rs) else None})
    best = min(table, key=lambda r: r["mean_best_cal_loss"])
    with open(os.path.join(out, "selection_table.json"), "w") as f:
        json.dump({"criterion": "lowest mean best calibration loss over seeds 0,1", "table": table, "chosen": best,
                   "curves": {f"wd{r['wd']:g}_drop{r['dropout']:g}_seed{r['seed']}": r["history"] for r in rows}}, f, indent=1)
    for r in table:
        print(r)
    print("CHOSEN:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "select"])
    ap.add_argument("--wd", type=float)
    ap.add_argument("--drop", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.wd, a.drop, a.seed, a.out, a.threads) if a.mode == "run" else select(a.out)
