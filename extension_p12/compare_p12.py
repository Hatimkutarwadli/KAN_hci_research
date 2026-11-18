"""P1-2 comparison (SYNTHETIC; EXTENSION): existing multitask GRU (MT; read-only) vs single-task (ST), budget-matched single-task (ST-matched) and per-family multitask (FAM) GRUs.
Reads saved metrics/predictions only. Writes results/p1_multitask_ablation/comparison/*.
'worse%' of A relative to B = (A-B)/B*100 for RMSE, (B-A)/B*100 for AP/AUROC (NEGATIVE = A better). Pairing is by seed (same initialisation stream and batch order for all conditions).
Usage: python extension_p12/compare_p12.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import common_p12 as C  # noqa: E402

os.chdir(C.ROOT)
R = "results/p1_multitask_ablation"
OUT = f"{R}/comparison"
CONDS = {"MT": None, "ST": "st_task", "ST-matched": "st_task_matched", "FAM": "family"}
GROUPS = [("Severity nowcast", "main", lambda h: h.startswith("sev_"), ["rmse"]), ("Status nowcast", "main", lambda h: h.startswith("status_"), ["ap", "auroc"]),
          ("Sustained onset", "main", lambda h: h.startswith("onset_sust"), ["ap", "auroc"]), ("Severity forecast", "forecast", lambda h: h.startswith("sev_"), ["rmse"]),
          ("Status forecast", "forecast", lambda h: h.startswith("status_"), ["ap", "auroc"])]
LOWER = {"rmse"}
BOOT_B, BOOT_SEED = 200, 20261118


def load_metrics():
    parts = []
    for part in ("main", "forecast"):
        mt = pd.read_csv(os.path.join(C.PARTS[part]["mt_dir"], "metrics_long.csv"))
        mt = mt[(mt.model == "gru") & mt.task.isin(C.PARTS[part]["st"])].assign(cond="MT", part=part)
        parts.append(mt)
        for cond, design in CONDS.items():
            if design is None:
                continue
            p = os.path.join(R, f"{design}_{part}", "metrics_long.csv")
            if os.path.exists(p):
                parts.append(pd.read_csv(p).assign(cond=cond, part=part))
    return pd.concat(parts, ignore_index=True)


def worse(metric, a, b):
    return (a - b) / b * 100 if metric in LOWER else (b - a) / b * 100


def pred(cond, part, task, seed, split):
    if cond == "MT":
        z = np.load(os.path.join(C.PARTS[part]["mt_dir"], "predictions", f"gru_seed{seed}_{split}.npz"))
        return z["pred"][:, :, list(z["heads"]).index(task)].astype(float)
    design = CONDS[cond]
    unit = task if design != "family" else "family_" + C.family_of(task)
    z = np.load(os.path.join(R, f"{design}_{part}", "predictions", f"{unit}_seed{seed}_{split}.npz"))
    return z["pred"][:, :, list(z["heads"]).index(task)].astype(float)


def metric_value(m, y, p):
    if m == "rmse":
        return float(np.sqrt(np.mean((p - y) ** 2)))
    if not 0 < y.sum() < len(y):
        return np.nan
    return float(average_precision_score(y, p) if m == "ap" else roc_auc_score(y, p))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_metrics()
    conds = [c for c in CONDS if c in set(df.cond)]
    L = ["# P1-2: multitask vs single-task GRU ablation (SYNTHETIC; EXTENSION, not part of the original six stages)\n",
         "MT = existing multitask GRU (16 main heads / 12 forecast heads; not retrained). ST = one GRU per head (n_out=1), frozen early stopping on its own calibration loss. ST-matched = one GRU per head trained for exactly "
         "the epochs the same-seed MT run selected (no early stopping; equal gradient steps). FAM = one GRU per task family (heads of one family share a representation). Same data, splits, protocol, 5 seeds, seed-paired.\n"
         "Cell `x ±sd`: mean ±SD over 5 seeds. worse% of A vs B: NEGATIVE = A better; `(n/5)` = seeds where A is better.\n"]
    # ---- per task
    rows, pt_rows = [], []
    for part in ("main", "forecast"):
        for task in C.PARTS[part]["st"]:
            for split in ("cal", "test"):
                for m in (["rmse"] if task.startswith("sev_") else ["ap", "auroc"]):
                    v = df[(df.part == part) & (df.task == task) & (df.split == split) & (df.metric == m)].pivot_table(index="seed", columns="cond", values="value")
                    row = dict(part=part, task=task, split=split, metric=m, **{c: v[c].mean() for c in conds if c in v}, **{c + "_sd": v[c].std() for c in conds if c in v})
                    for c in conds:
                        if c != "MT" and c in v and "MT" in v:
                            w = worse(m, v[c], v["MT"])
                            row[f"{c}_vs_MT_worse_pct"], row[f"{c}_vs_MT_n_better"] = w.mean(), int((w < 0).sum())
                    rows.append(row)
    pt = pd.DataFrame(rows)
    pt.to_csv(os.path.join(OUT, "per_task_paired.csv"), index=False)
    L.append("\n## Per-task test results (mean ±SD over seeds) and paired worse% vs MT\n")
    L.append("| task | metric | " + " | ".join(conds) + " | " + " | ".join(f"{c} vs MT" for c in conds if c != "MT") + " |")
    L.append("|---|---|" + "---|" * (len(conds) + len(conds) - 1))
    for d in pt[pt.split == "test"].to_dict("records"):
        L.append(f"| {d['task']} | {d['metric']} | " + " | ".join(f"{d[c]:.3f} ±{d[c + '_sd']:.3f}" if c in d else "" for c in conds) + " | " +
                 " | ".join(f"{d[c + '_vs_MT_worse_pct']:+.1f} ({d[c + '_vs_MT_n_better']}/5)" for c in conds if c != "MT") + " |")
    # ---- group level paired comparisons (mean over the group's tasks per seed)
    gr = []
    pairs = [("ST", "MT", "single-task vs multitask (total effect)"), ("ST-matched", "MT", "single-task at MT's epochs vs multitask (equal budget)"),
             ("ST", "ST-matched", "ST selection advantage (own early stopping vs MT's epochs)"), ("FAM", "MT", "family-only multitask vs full multitask (cross-family sharing)"),
             ("ST", "FAM", "single-task vs family multitask (within-family sharing)")]
    for gname, part, sel, ms in GROUPS:
        tasks = [t for t in C.PARTS[part]["st"] if sel(t)]
        for m in ms:
            for split in ("cal", "test"):
                v = df[(df.part == part) & df.task.isin(tasks) & (df.split == split) & (df.metric == m)].groupby(["cond", "seed", "task"]).value.mean().reset_index().groupby(["cond", "seed"]).value.mean().unstack(0)
                for a, b, desc in pairs:
                    if a in v and b in v:
                        w = worse(m, v[a], v[b])
                        gr.append(dict(group=gname, metric=m, split=split, A=a, B=b, description=desc, A_mean=v[a].mean(), B_mean=v[b].mean(), worse_pct_mean=w.mean(), worse_pct_sd=w.std(),
                                       n_seeds_A_better=int((w < 0).sum()), n_seeds=len(w)))
    gdf = pd.DataFrame(gr)
    gdf.to_csv(os.path.join(OUT, "group_paired.csv"), index=False)
    for split in ("test", "cal"):
        L.append(f"\n## Group-level paired comparisons, {split} split (mean over the group's tasks; worse% of A vs B, NEGATIVE = A better; seeds where A better)\n")
        L.append("| group | metric | " + " | ".join(f"{a} vs {b}" for a, b, _ in pairs) + " |")
        L.append("|---|---|" + "---|" * len(pairs))
        for (g, m), x in gdf[gdf.split == split].groupby(["group", "metric"], sort=False):
            cells = []
            for a, b, _ in pairs:
                r = x[(x.A == a) & (x.B == b)]
                cells.append("" if r.empty else f"{r.iloc[0].worse_pct_mean:+.2f} ±{r.iloc[0].worse_pct_sd:.2f} ({r.iloc[0].n_seeds_A_better}/{r.iloc[0].n_seeds})")
            L.append(f"| {g} | {m} | " + " | ".join(cells) + " |")
    L.append("\nPair meanings: " + "; ".join(f"**{a} vs {b}** = {d}" for a, b, d in pairs) + ".")
    # ---- participant bootstrap (seed 0, test): group-level paired difference
    pid = np.load("data/processed_v1/test.npz")
    Yt = {t: pid[f"y_{t}"] for t in C.PARTS["main"]["st"] + C.PARTS["forecast"]["st"]}
    rng = np.random.default_rng(BOOT_SEED)
    N = len(pid["pid"])
    draws = [rng.integers(0, N, N) for _ in range(BOOT_B)]
    brow = []
    for gname, part, sel, ms in GROUPS:
        tasks = [t for t in C.PARTS[part]["st"] if sel(t)]
        for m in ms:
            data = {}
            for t in tasks:
                pi, ti = np.where(~np.isnan(Yt[t]))
                data[t] = (pi, Yt[t][pi, ti].astype(float), {c: pred(c, part, t, 0, "test")[pi, ti] for c in conds})
            byp = {t: [np.where(data[t][0] == i)[0] for i in range(N)] for t in tasks}

            def agg(c, ix):
                vals = []
                for t in tasks:
                    pi, y, P = data[t]
                    idx = np.concatenate([byp[t][i] for i in ix])
                    vals.append(metric_value(m, y[idx], P[c][idx]))
                return float(np.nanmean(vals))

            pts = {c: agg(c, np.arange(N)) for c in conds}
            boots = {c: np.array([agg(c, ix) for ix in draws]) for c in conds}
            for a, b, desc in pairs:
                if a in conds and b in conds:
                    d = np.array([worse(m, boots[a][k], boots[b][k]) for k in range(BOOT_B)])
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    brow.append(dict(group=gname, metric=m, A=a, B=b, worse_pct_point=worse(m, pts[a], pts[b]), ci95_lo=lo, ci95_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
    bdf = pd.DataFrame(brow)
    bdf.to_csv(os.path.join(OUT, "group_bootstrap_seed0.csv"), index=False)
    L += ["\n## Seed-0 participant-bootstrap 95% intervals (test, B=200, paired resamples) for the group-level worse% of A vs B; `*` = interval excludes 0\n",
          "| group | metric | " + " | ".join(f"{a} vs {b}" for a, b, _ in pairs) + " |", "|---|---|" + "---|" * len(pairs)]
    for (g, m), x in bdf.groupby(["group", "metric"], sort=False):
        cells = []
        for a, b, _ in pairs:
            r = x[(x.A == a) & (x.B == b)]
            cells.append("" if r.empty else f"{r.iloc[0].worse_pct_point:+.2f} [{r.iloc[0].ci95_lo:+.2f}, {r.iloc[0].ci95_hi:+.2f}]" + ("*" if r.iloc[0].ci_excludes_0 else ""))
        L.append(f"| {g} | {m} | " + " | ".join(cells) + " |")
    # ---- cost and convergence
    cr = []
    for part in ("main", "forecast"):
        for run in json.load(open(os.path.join(C.PARTS[part]["mt_dir"], "run_log.json")))["runs"]:
            if run["model"] == "gru":
                h = run["history"]
                cr.append(dict(part=part, cond="MT", unit="multitask (all heads)", seed=run["seed"], n_params=run["n_params"], wall_s=run["wall_s"], cpu_s=run.get("cpu_s"), best_epoch=run["best_epoch"],
                               epochs_run=run["epochs_run"], cal_rise_after_best=h[-1]["cal_loss"] - h[run["best_epoch"] - 1]["cal_loss"], n_heads=16 if part == "main" else 12))
        for cond, design in CONDS.items():
            p = os.path.join(R, f"{design}_{part}", "run_log.json") if design else None
            if p and os.path.exists(p):
                for run in json.load(open(p))["runs"]:
                    h = run["history"]
                    cr.append(dict(part=part, cond=cond, unit=run["unit"], seed=run["seed"], n_params=run["n_params"], wall_s=run["wall_s"], cpu_s=run["cpu_s"], best_epoch=run["best_epoch"],
                                   epochs_run=run["epochs_run"], cal_rise_after_best=h[-1]["cal_loss"] - h[run["best_epoch"] - 1]["cal_loss"], n_heads=len(run["heads"])))
    cdf = pd.DataFrame(cr)
    cdf.to_csv(os.path.join(OUT, "cost_convergence.csv"), index=False)
    s = cdf.groupby(["part", "cond"]).agg(trainings_per_seed=("unit", lambda x: x.nunique()), params_per_model=("n_params", "mean"), cpu_s_per_model=("cpu_s", "mean"), wall_s_per_model=("wall_s", "median"),
                                         cpu_s_total_per_seed=("cpu_s", lambda x: x.sum() / 5), best_epoch=("best_epoch", "mean"), epochs_run=("epochs_run", "mean"),
                                         share_best_le2=("best_epoch", lambda x: (x <= 2).mean()), cal_rise=("cal_rise_after_best", "mean")).reset_index()
    L += ["\n## Training cost and convergence (CPU, 4 torch threads per process; 2 processes ran concurrently and the laptop throttled and hibernated at times, so wall seconds are unreliable; CPU seconds are the fairer cost figure but also vary with load)\n",
          "| part | condition | trainings per seed | params per model | CPU s per model | median wall s per model (one run includes a laptop-hibernation gap, so the median is used) | CPU s for ALL heads per seed | mean best epoch | share best epoch <= 2 | mean cal-loss rise after best |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        L.append(f"| {r.part} | {r.cond} | {r.trainings_per_seed} | {r.params_per_model:.0f} | {r.cpu_s_per_model:.0f} | {r.wall_s_per_model:.0f} | {r.cpu_s_total_per_seed:.0f} | {r.best_epoch:.1f} | {r.share_best_le2:.2f} | {r.cal_rise:+.4f} |")
    # supervised label coverage per head (budget context)
    cnt = pd.concat([pd.read_csv(os.path.join(C.PARTS[p]["mt_dir"], "counts.csv")) for p in ("main", "forecast")]).drop_duplicates(["task", "split"])
    tr = cnt[cnt.split == "train"].set_index("task").eligible_origins / (600 * 93)
    L += ["\n## Supervision context: share of the 55,800 training origins with a defined label, per head (a single-head model gets gradient from only this share of each batch; the multitask model gets gradient from every row through at least one head)\n",
          "| head | labelled share of training origins |", "|---|---|"]
    for t in C.PARTS["main"]["st"] + C.PARTS["forecast"]["st"]:
        L.append(f"| {t} | {tr[t]:.2f} |")
    open(os.path.join(OUT, "p12_comparison.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
