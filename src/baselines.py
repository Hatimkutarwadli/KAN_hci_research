"""Stage 3 baselines on the processed SYNTHETIC dataset: B1 GBT, B2 static multi-scale KAN (MTLNet-lite), B3 GRU.

Protocol: models are fit on TRAIN participants only; CALIBRATION participants are used for model selection
(GBT depth/iterations, NN epoch) and for F1 thresholds; TEST participants are only predicted after selection.
Usage: python src/baselines.py --models gbt kan gru --seeds 0 1 2 --out results/stage3
"""
import argparse
import copy
import ctypes
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import data as dd
import features as ft
import metrics as mt
import models as md

OFF = np.arange(-27, 1)
KS, HZ = ("dep", "anx"), (7, 14, 21)
PRIMARY = ([f"sev_{k}_t0" for k in KS] + [f"status_{k}_t0" for k in KS] +
           [f"onset_sust_{k}_h{h}" for k in KS for h in HZ])
SECONDARY = [f"onset_cross_{k}_h{h}" for k in KS for h in HZ]   # first-crossing: separate comparison only
MAIN_HEADS = PRIMARY + SECONDARY
FORECAST = ([f"sev_{k}_h{h}" for k in KS for h in HZ] + [f"status_{k}_h{h}" for k in KS for h in HZ])
_TAGS = ("t0", "h7", "h14", "h21")
ALL_HEADS = ([f"sev_{k}_{g}" for k in KS for g in _TAGS] + [f"status_{k}_{g}" for k in KS for g in _TAGS] +
             [f"onset_sust_{k}_h{h}" for k in KS for h in HZ] + [f"onset_cross_{k}_h{h}" for k in KS for h in HZ])   # 8 regression + 20 classification
TASKSETS = {"main": (MAIN_HEADS, 2), "forecast": (FORECAST, 6), "all": (ALL_HEADS, 8)}      # (heads, number of leading regression heads)
HEADS, N_REG = list(MAIN_HEADS), 2
REG = HEADS[:N_REG]


def configure(taskset):
    """Select the task set (module-level state): 'main' = Stage 3 tasks, 'forecast' = severity/status at t+h."""
    global HEADS, N_REG, REG
    heads, n_reg = TASKSETS[taskset]
    HEADS, N_REG = list(heads), n_reg
    REG = HEADS[:N_REG]
THREADS = 4


# ------------------------------------------------------------------ data
def load_data(proc="data/processed_v1", n_sub=None, splits=("train", "cal", "test")):
    out = {}
    for s in splits:
        d = dd.load_split(s, proc)
        if n_sub:
            k = min(n_sub[s], len(d["pid"]))
            d = {a: b[:k] for a, b in d.items()}
        N = d["X"].shape[0]
        pi, ti = np.meshgrid(np.arange(N), np.arange(27, 120), indexing="ij")
        pi, ti = pi.ravel(), ti.ravel()
        Y = np.stack([d["y_" + h][pi, ti] for h in HEADS], 1).astype(np.float32)
        out[s] = dict(X=d["X"], static=d["static"], pid=d["pid"], pi=pi, ti=ti, Y=Y, pid_o=d["pid"][pi])
    return out


def gbt_features(d, chunk=8192):
    parts = []
    for a in range(0, len(d["pi"]), chunk):
        pi, ti = d["pi"][a:a + chunk], d["ti"][a:a + chunk]
        parts.append(ft.window_features(d["X"][pi[:, None], ti[:, None] + OFF], d["static"][pi]))
    return np.vstack(parts)


# ------------------------------------------------------------------ B1: gradient boosting
def fit_gbt_task(task, Xtr, ytr, Xca, yca, seed=0, depths=(3, 6), max_iter=200, step=20):
    reg = task in REG

    def mk(depth, it):
        cls = HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier
        return cls(learning_rate=0.05, max_depth=depth, max_iter=it, min_samples_leaf=50,
                   l2_regularization=1.0, early_stopping=False, random_state=seed)

    best = None
    for depth in depths:
        m = mk(depth, max_iter).fit(Xtr, ytr if reg else ytr.astype(int))
        stages = m.staged_predict(Xca) if reg else (p[:, 1] for p in m.staged_predict_proba(Xca))
        for i, p in enumerate(stages, 1):
            if i % step and i != max_iter:
                continue
            sc = -mt.reg_metrics(yca, p)["rmse"] if reg else mt.clf_metrics(yca, p)["ap"]
            if np.isfinite(sc) and (best is None or sc > best[0]):
                best = (sc, depth, i)
    sc, depth, it = best
    final = mk(depth, it).fit(Xtr, ytr if reg else ytr.astype(int))
    return final, {"depth": depth, "n_iter": it, "cal_score": float(sc), "metric": "-RMSE" if reg else "AP"}


def run_gbt(data, seed, depths, max_iter, log):
    F_ = {s: gbt_features(data[s]) for s in data}
    P = {s: np.full((len(data[s]["pi"]), len(HEADS)), np.nan, np.float32) for s in data}
    fitted, sel = {}, {}
    for j, task in enumerate(HEADS):
        tr, ca = data["train"]["Y"][:, j], data["cal"]["Y"][:, j]
        mtr, mca = ~np.isnan(tr), ~np.isnan(ca)
        m, info = fit_gbt_task(task, F_[ "train"][mtr], tr[mtr], F_["cal"][mca], ca[mca], seed, depths, max_iter)
        fitted[task], sel[task] = m, info
        for s in data:                                     # test is predicted only AFTER selection is final
            P[s][:, j] = m.predict(F_[s]) if task in REG else m.predict_proba(F_[s])[:, 1]
        log(f"  gbt {task}: {info}")
    return P, fitted, sel


# ------------------------------------------------------------------ B2 / B3: neural nets
def to_t(d):
    return {"X": torch.from_numpy(d["X"]), "S": torch.from_numpy(d["static"]), "pi": torch.from_numpy(d["pi"]),
            "ti": torch.from_numpy(d["ti"]), "Y": torch.from_numpy(d["Y"])}


OFF_T = torch.from_numpy(OFF)


def batch(T, idx):
    pi, ti = T["pi"][idx], T["ti"][idx]
    return T["X"][pi[:, None], ti[:, None] + OFF_T], T["S"][pi], T["Y"][idx]


def head_loss_terms(out, y, ymean, ystd):
    valid = ~torch.isnan(y)
    y0 = torch.where(valid, y, torch.zeros_like(y))
    l_reg = (out[:, :N_REG] - (y0[:, :N_REG] - ymean) / ystd) ** 2
    l_clf = F.binary_cross_entropy_with_logits(out[:, N_REG:], y0[:, N_REG:], reduction="none")
    return torch.cat([l_reg, l_clf], 1) * valid, valid


def masked_loss(out, y, ymean, ystd):
    l, v = head_loss_terms(out, y, ymean, ystd)
    return (l.sum(0) / v.sum(0).clamp(min=1)).mean()


@torch.no_grad()
def eval_loss(model, T, ymean, ystd, bs=2048):
    model.eval()
    s, c = 0.0, 0.0
    for a in range(0, len(T["pi"]), bs):
        x, st, y = batch(T, torch.arange(a, min(a + bs, len(T["pi"]))))
        l, v = head_loss_terms(model(x, st), y, ymean, ystd)
        s, c = s + l.sum(0), c + v.sum(0)
    return float((s / c.clamp(min=1)).mean())


@torch.no_grad()
def predict_nn(model, T, ymean, ystd, bs=2048):
    model.eval()
    outs = []
    for a in range(0, len(T["pi"]), bs):
        x, st, _ = batch(T, torch.arange(a, min(a + bs, len(T["pi"]))))
        o = model(x, st)
        outs.append(torch.cat([o[:, :N_REG] * ystd + ymean, torch.sigmoid(o[:, N_REG:])], 1))
    return torch.cat(outs).numpy()


def fit_nn(name, seed, tr, ca, epochs=25, patience=4, bs=256, lr=2e-3, wd=1e-5, epoch_frac=0.5, model_kwargs=None, log=print):
    """Train on TRAIN, select the epoch on CALIBRATION loss. The test split is never passed in."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = md.MODELS[name](n_out=len(HEADS), **(model_kwargs or {}))
    y = tr["Y"][:, :N_REG]
    ymean = torch.tensor(np.nanmean(y, 0), dtype=torch.float32)
    ystd = torch.tensor(np.nanstd(y, 0), dtype=torch.float32)
    Tt, Tc = to_t(tr), to_t(ca)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    best, hist, bad, n = (float("inf"), None, 0), [], 0, len(tr["pi"])
    best_state, best_ep = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)[:int(epoch_frac * n)]   # origins overlap heavily: half per 'epoch'
        m_ = len(perm)
        run = 0.0
        for a in range(0, m_, bs):
            x, st, yb = batch(Tt, perm[a:a + bs])
            loss = masked_loss(model(x, st), yb, ymean, ystd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(yb)
        cl = eval_loss(model, Tc, ymean, ystd)
        hist.append({"epoch": ep, "train_loss": run / m_, "cal_loss": cl})
        log(f"  {name} seed{seed} ep{ep} train {run / m_:.4f} cal {cl:.4f}")
        if cl < best[0] - 1e-5:
            best, best_state, best_ep, bad = (cl, None, 0), copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, ymean, ystd, {"best_epoch": best_ep, "epochs_run": len(hist), "history": hist,
                                "hparams": dict(epochs=epochs, patience=patience, batch_size=bs, lr=lr, weight_decay=wd, epoch_frac=epoch_frac, model_kwargs=model_kwargs or {})}


# ------------------------------------------------------------------ evaluation
def evaluate(model, seed, P, data):
    rows, thr = [], {}
    for j, task in enumerate(HEADS):
        reg = task in REG
        if not reg:
            y = data["cal"]["Y"][:, j]
            ok = ~np.isnan(y)
            thr[task] = mt.best_f1_threshold(y[ok], P["cal"][ok, j])     # threshold from CALIBRATION only
        for s in ("train", "cal", "test"):
            y = data[s]["Y"][:, j]
            ok = ~np.isnan(y)
            m = mt.reg_metrics(y[ok], P[s][ok, j]) if reg else mt.clf_metrics(y[ok], P[s][ok, j], thr[task])
            rows += [dict(model=model, seed=seed, task=task, split=s, metric=k, value=v) for k, v in m.items()]
    return rows, thr


def counts_table(data):
    rows = []
    for j, task in enumerate(HEADS):
        for s in data:
            y, pid = data[s]["Y"][:, j], data[s]["pid_o"]
            ok = ~np.isnan(y)
            reg = task in REG
            rows.append(dict(task=task, split=s, eligible_origins=int(ok.sum()), eligible_participants=int(len(np.unique(pid[ok]))),
                             positive_origins=None if reg else int(y[ok].sum()),
                             positive_participants=None if reg else int(len(np.unique(pid[ok & (y == 1)])))))
    return pd.DataFrame(rows)


def reference_preds(data):
    """B0: train-mean for severity, train prevalence for classification (sanity floor, not a learned model)."""
    P = {s: np.zeros((len(data[s]["pi"]), len(HEADS)), np.float32) for s in data}
    for j in range(len(HEADS)):
        v = np.nanmean(data["train"]["Y"][:, j])
        for s in data:
            P[s][:, j] = v
    return P


def peak_ws_mb():
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 1e6
    except Exception:
        return None


def save_preds(out, name, seed, P, data):
    os.makedirs(os.path.join(out, "predictions"), exist_ok=True)
    for s in ("cal", "test"):
        N = len(data[s]["pid"])
        full = np.full((N, 120, len(HEADS)), np.nan, np.float32)
        full[data[s]["pi"], data[s]["ti"]] = P[s]
        np.savez_compressed(os.path.join(out, "predictions", f"{name}_seed{seed}_{s}.npz"), pred=full, pid=data[s]["pid"],
                            heads=np.array(HEADS))


# ------------------------------------------------------------------ reporting
def md_table(piv):
    cols = list(piv.columns)
    lines = ["| task | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, r in piv.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def summary_groups():
    clf = ["ap", "auroc", "f1", "precision", "recall"]
    if HEADS == ALL_HEADS:
        return (("ALL: severity, nowcast + forecast (test)", ALL_HEADS[:8], ["rmse", "mae", "r2"]),
                ("ALL: status, nowcast + forecast (test)", ALL_HEADS[8:16], clf),
                ("ALL: sustained onset (test)", ALL_HEADS[16:22], clf),
                ("ALL: first-crossing onset (test, separate)", ALL_HEADS[22:], clf))
    if HEADS == FORECAST:
        return (("FORECAST: severity at t+h (test)", FORECAST[:6], ["rmse", "mae", "r2"]),
                ("FORECAST: status at t+h (test)", FORECAST[6:], clf))
    return (("PRIMARY: severity (nowcast, test)", PRIMARY[:2], ["rmse", "mae", "r2"]),
            ("PRIMARY: status (nowcast, test)", PRIMARY[2:4], clf),
            ("PRIMARY: sustained onset (test)", PRIMARY[4:], clf),
            ("SECONDARY: first-crossing onset (test, separate comparison)", SECONDARY, clf))


def summarize(df, out):
    t = df[df.split == "test"]
    lines = []
    for title, tasks, metrics in summary_groups():
        lines.append(f"\n### {title}\n")
        for met in metrics:
            sub = t[(t.task.isin(tasks)) & (t.metric == met)]
            g = sub.groupby(["task", "model"]).value.agg(["mean", "std", "count"]).reset_index()
            piv = g.assign(cell=lambda d: d.apply(lambda r: f"{r['mean']:.3f}" + (f" ± {r['std']:.3f}" if r["count"] > 1 else ""), axis=1)).pivot(index="task", columns="model", values="cell")
            piv = piv.reindex([x for x in tasks if x in piv.index])
            lines.append(f"**{met}**\n\n" + md_table(piv) + "\n")
    open(os.path.join(out, "comparison_test.md"), "w", encoding="utf-8").write("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gbt", "kan", "gru"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results/stage3")
    ap.add_argument("--proc", default="data/processed_v1")
    ap.add_argument("--quick", action="store_true", help="smoke test: few participants, few epochs")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--taskset", default="main", choices=list(TASKSETS))
    a = ap.parse_args()
    configure(a.taskset)
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty; refusing to overwrite.")
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    torch.set_num_threads(THREADS)
    logf = open(os.path.join(a.out, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = load_data(a.proc, {"train": 80, "cal": 30, "test": 30} if a.quick else None)
    counts = counts_table(data)
    counts.to_csv(os.path.join(a.out, "counts.csv"), index=False)
    rows, runlog, allP = [], [], {}
    r0, _ = evaluate("B0_reference", 0, reference_preds(data), data)
    rows += r0
    epochs = 2 if a.quick else a.epochs
    for spec in a.models:      # e.g. "kan" or "kan:wd=1e-3:drop=0.3:label=kan_reg"
        base, *opts = spec.split(":")
        o = dict(x.split("=") for x in opts)
        name = o.get("label", base)
        seeds = [a.seeds[0]] if base == "gbt" else a.seeds   # GBT has no stochastic component here
        for seed in seeds:
            t0, c0 = time.perf_counter(), time.process_time()
            info = {}
            if base == "gbt":
                with threadpool_limits(THREADS):
                    P, fitted, sel = run_gbt(data, seed, (3,) if a.quick else (3, 6), 40 if a.quick else 200, log)
                joblib.dump(fitted, os.path.join(a.out, "checkpoints", f"gbt_seed{seed}.joblib"))
                info = {"selection": sel, "n_features": len(ft.NAMES), "feature_names": ft.NAMES}
                n_par = None
            else:
                tr, ca = data["train"], data["cal"]
                mk = {"dropout": float(o["drop"])} if "drop" in o else {}
                if base.startswith("m1"):
                    mk["heads"] = list(HEADS)
                mk = mk or None
                model, ym, ys, info = fit_nn(base, seed, tr, ca, epochs=epochs, wd=float(o.get("wd", 1e-5)), model_kwargs=mk, log=log)
                P = {s: predict_nn(model, to_t(data[s]), ym, ys) for s in data}   # test predicted after selection
                torch.save({"state_dict": model.state_dict(), "ymean": ym, "ystd": ys, "heads": HEADS, "seed": seed,
                            "best_epoch": info["best_epoch"]}, os.path.join(a.out, "checkpoints", f"{name}_seed{seed}.pt"))
                n_par = md.n_params(model)
            r, thr = evaluate(name, seed, P, data)
            rows += r
            save_preds(a.out, name, seed, P, data)
            allP[(name, seed)] = P
            runlog.append(dict(model=name, seed=seed, wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0,
                               peak_working_set_mb_process_so_far=peak_ws_mb(), torch_threads=THREADS, n_params=n_par,
                               thresholds_from_cal=thr, **{k: v for k, v in info.items() if k != "feature_names"}))
            log(f"{name} seed{seed}: wall {runlog[-1]['wall_s']:.0f}s cpu {runlog[-1]['cpu_s']:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "metrics_long.csv"), index=False)

    if not a.no_bootstrap:   # participant-level bootstrap CIs for the first seed of each model, test split, primary tasks
        brow = []
        for (name, seed), P in allP.items():
            if seed != a.seeds[0]:
                continue
            btasks = [(j, t_) for j, t_ in enumerate(HEADS) if "cross" not in t_] if a.taskset != "main" else list(enumerate(PRIMARY))
            for j, task in btasks:
                y = data["test"]["Y"][:, j]
                ok = ~np.isnan(y)
                yy, pp, pid = y[ok], P["test"][ok, j], data["test"]["pid_o"][ok]
                fns = {"rmse": lambda u, v: mt.reg_metrics(u, v)["rmse"]} if task in REG else \
                    {"ap": lambda u, v: mt.clf_metrics(u, v)["ap"], "auroc": lambda u, v: mt.clf_metrics(u, v)["auroc"]}
                for mname, fn in fns.items():
                    lo, hi = mt.participant_bootstrap(yy, pp, pid, fn, 100 if a.quick else 200)
                    brow.append(dict(model=name, seed=seed, task=task, metric=mname, point=fn(yy, pp), ci95_lo=lo, ci95_hi=hi))
        pd.DataFrame(brow).to_csv(os.path.join(a.out, "bootstrap_ci_test.csv"), index=False)
    summarize(df, a.out)
    with open(os.path.join(a.out, "run_log.json"), "w") as f:
        json.dump({"SYNTHETIC": True, "heads": HEADS, "args": vars(a), "runs": runlog, "torch": torch.__version__}, f, indent=1, default=str)
    logf.close()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
