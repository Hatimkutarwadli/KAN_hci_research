"""P3 runner and queue (SYNTHETIC; TRAIN + CALIBRATION only; the test split is sealed until G3).
  python extension_p3/run_p3.py --set pilot|main|S1|S2|S3 [--workers 3]
Each job trains ONE model on ONE nested train subset with the frozen protocol (baselines.fit_nn / run_gbt, unchanged), selects the epoch on
calibration, and stores: checkpoint, scaler.json (scaler + subset pids), metrics.json (per-head calibration metrics + constant-predictor
references), run.json. Writes go to '<id>.tmp' and are renamed atomically; completed jobs are skipped (stop condition D2)."""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p3 as P  # noqa: E402

ROOTS = {"pilot": "pilot", "main": "runs", "S1": "sens_S1", "S2": "sens_S2", "S3": "sens_S3"}


def train_eval(ds, model, ts, seed, train_pids, cal_pids, scaler_mode, wd_override=None, log=print):
    """Train one model and return calibration predictions on the FULL calibration set. scaler_mode: 'subset' (primary) or 'frozen' (S1).
    cal_pids: participants used for epoch/GBT selection (full cal, or the S3 reduced set)."""
    import torch
    from threadpoolctl import threadpool_limits
    import baselines as bl
    import common_p22 as C
    import models as md
    import models_p13
    models_p13.register()
    torch.set_num_threads(bl.THREADS)
    bl.configure(ts)
    scaler = P.fit_subset_scaler(ds, train_pids) if scaler_mode == "subset" else None
    tr = P.split_data(ds, "train", train_pids, scaler)
    full_cal = P.all_pids(ds, "cal")
    ca_sel = P.split_data(ds, "cal", cal_pids, scaler)
    ca_eval = ca_sel if list(cal_pids) == full_cal else P.split_data(ds, "cal", full_cal, scaler)
    info, ck, n_par = {}, None, None
    if model == "gbt":
        with threadpool_limits(bl.THREADS):
            Pd, fitted, sel = bl.run_gbt({"train": tr, "cal": ca_eval} if ca_sel is ca_eval else {"train": tr, "cal": ca_sel}, seed, (3, 6), 200, log)
            if ca_sel is ca_eval:
                Pc = Pd["cal"]
            else:
                F_ = bl.gbt_features(ca_eval)
                Pc = np.stack([fitted[t].predict(F_) if t in bl.REG else fitted[t].predict_proba(F_)[:, 1] for t in bl.HEADS], 1).astype(np.float32)
        ck, info = fitted, {"selection": sel}
    else:
        spec = C.CORE[model]
        mk = {"dropout": spec["dropout"]} if spec["dropout"] > 0 else {}
        if spec["arch"].endswith("m1"):
            mk["heads"] = list(bl.HEADS)
        wd = spec["wd"] if wd_override is None else wd_override
        m, ym, ys, info = bl.fit_nn(spec["arch"], seed, tr, ca_sel, wd=wd, model_kwargs=mk or None, log=log)
        Pc = bl.predict_nn(m, bl.to_t(ca_eval), ym, ys)
        ck = {"state_dict": m.state_dict(), "ymean": ym, "ystd": ys, "heads": bl.HEADS, "seed": seed, "best_epoch": info["best_epoch"], "wd": wd}
        n_par = md.n_params(m)
    return dict(P=Pc, ca_eval=ca_eval, tr=tr, info=info, ck=ck, n_params=n_par, scaler=scaler, heads=list(bl.HEADS))


def head_metrics(out):
    import baselines as bl
    import metrics as mt
    ca, tr = out["ca_eval"], out["tr"]
    res = {}
    for j, h in enumerate(out["heads"]):
        y, p = ca["Y"][:, j], out["P"][:, j]
        ok = ~np.isnan(y)
        if j < bl.N_REG:
            const = float(np.nanmean(tr["Y"][:, j]))
            res[h] = dict(metric="rmse", value=float(mt.reg_metrics(y[ok], p[ok])["rmse"]), const_rmse=float(np.sqrt(np.mean((y[ok] - const) ** 2))), n=int(ok.sum()))
        else:
            res[h] = dict(metric="ap", value=float(mt.clf_metrics(y[ok], p[ok])["ap"]), prevalence=float(y[ok].mean()), n=int(ok.sum()), n_pos=int(y[ok].sum()))
    return res


def run_job(jb, out_root):
    import joblib
    import torch
    final = os.path.join(out_root, jb["ds"], jb["id"])
    if os.path.isdir(final):
        return dict(id=jb["id"], status="skipped")
    tmp = final + ".tmp"
    partial = os.path.isdir(tmp)
    if partial:
        shutil.rmtree(tmp)
    os.makedirs(os.path.join(tmp, "checkpoints"))
    logf = open(os.path.join(tmp, "train_log.txt"), "w", encoding="utf-8")
    t0, c0 = time.perf_counter(), time.process_time()
    sub = P.subsets(jb["ds"])[str(jb["draw"])][str(jb["n"])]
    cal = P.cal_subset_S3(jb["ds"], jb["draw"]) if jb["variant"] == "S3" else P.all_pids(jb["ds"], "cal")
    out = train_eval(jb["ds"], jb["model"], jb["ts"], jb["seed"], sub, cal, "frozen" if jb["variant"] == "S1" else "subset",
                     wd_override=1e-3 if jb["variant"] == "S2" else None, log=lambda m: (logf.write(m + "\n"), logf.flush()))
    logf.close()
    if jb["model"] == "gbt":
        joblib.dump(out["ck"], os.path.join(tmp, "checkpoints", f"gbt_seed{jb['seed']}.joblib"))
    else:
        torch.save(out["ck"], os.path.join(tmp, "checkpoints", f"{jb['model']}_seed{jb['seed']}.pt"))
    finite = bool(np.isfinite(out["P"]).all())
    hist = out["info"].get("history")
    if hist:
        finite = finite and all(np.isfinite(h.get("cal_loss", 0.0)) and np.isfinite(h.get("train_loss", 0.0)) for h in hist)
    sc = out["scaler"]
    json.dump({"train_pids": [int(x) for x in sub], "cal_selection_pids": [int(x) for x in cal], "scaler_mode": "frozen" if sc is None else "subset",
               "mean": None if sc is None else sc["mean"].tolist(), "std": None if sc is None else sc["std"].tolist()}, open(os.path.join(tmp, "scaler.json"), "w"))
    json.dump(head_metrics(out), open(os.path.join(tmp, "metrics.json"), "w"), indent=1)
    info = {k: v for k, v in out["info"].items() if k not in ("history", "selection")}
    json.dump(dict(SYNTHETIC=True, test_split_loaded=False, **jb, wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0, finite=finite, n_params=out["n_params"],
                   partial_rerun=partial, history=hist, **info, finished=datetime.datetime.now().isoformat()), open(os.path.join(tmp, "run.json"), "w"), indent=1, default=str)
    os.replace(tmp, final)
    return dict(id=jb["id"], ds=jb["ds"], status="done", finite=finite, partial_rerun=partial, cpu_s=time.process_time() - c0, wall_s=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(ROOTS))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.chdir(P.ROOT)
    if shutil.disk_usage(P.ROOT).free < 5e9:
        raise SystemExit("STOP D3: free disk < 5 GB")
    jobs = {"pilot": P.jobs_pilot, "main": lambda: [j for ds in P.DATASETS for j in P.jobs_main(ds)],
            "S1": lambda: P.jobs_sens("S1"), "S2": lambda: P.jobs_sens("S2"), "S3": lambda: P.jobs_sens("S3")}[a.set]()
    jobs.sort(key=lambda j: -j["cost"])
    out_root = os.path.join(P.RES, ROOTS[a.set])
    for ds in {j["ds"] for j in jobs}:
        os.makedirs(os.path.join(out_root, ds), exist_ok=True)
        P.subsets(ds)                                                   # create once (read-only) before workers start
    qlog = open(os.path.join(out_root, "queue_log.txt"), "a", encoding="utf-8")
    t0 = time.time()
    qlog.write(f"{datetime.datetime.now().isoformat()} start set={a.set} jobs={len(jobs)} workers={a.workers}\n")
    done, stop = 0, False
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_job, j, out_root): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            qlog.write(f"{datetime.datetime.now().isoformat()} [{done}/{len(jobs)}] {json.dumps(r)} elapsed={time.time() - t0:.0f}s\n")
            qlog.flush()
            if r.get("status") == "done" and not r["finite"]:
                qlog.write("STOP B1: non-finite loss/prediction; cancelling pending jobs\n")
                stop = True
                ex.shutdown(cancel_futures=True)
                break
    qlog.write(f"{datetime.datetime.now().isoformat()} end set={a.set} done={done} stop={stop} elapsed={time.time() - t0:.0f}s\n")
    qlog.close()
    print("done", done, "stop", stop, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
