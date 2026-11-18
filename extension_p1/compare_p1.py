"""P1-1 comparison (SYNTHETIC; EXTENSION): existing results (read-only) + results/p1_baselines/{main,forecast}. Retrains nothing. Reuses src/compare_stage3.py helpers.
Writes results/p1_baselines/comparison/{p1_comparison.md, paired_vs_gru.csv, cost_and_convergence.csv}.
Usage: python extension_p1/compare_p1.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]
os.chdir(ROOT)
import baselines as bl  # noqa: E402
import compare_stage3 as cs  # noqa: E402

OUT = "results/p1_baselines/comparison"
DIRS = ["results/stage3", "results/stage3b_kan_reg/final", "results/stage3c_forecast", "results/stage4_m1", "results/stage4_ablation", "results/ext_kangru",
        "results/p1_baselines/main", "results/p1_baselines/forecast"]
MODELS = ["B0_reference", "gbt", "gru", "kan", "kan_reg", "m1_static", "m1", "kangru_fixed", "kangru", "transformer", "cnnlstm"]
NEW = ["transformer", "cnnlstm"]
GROUPS = [("NOWCAST severity (RMSE)", bl.PRIMARY[:2], "rmse"), ("NOWCAST status (AP)", bl.PRIMARY[2:4], "ap"), ("NOWCAST status (AUROC)", bl.PRIMARY[2:4], "auroc"),
          ("PRIMARY sustained onset (AP)", bl.PRIMARY[4:], "ap"), ("PRIMARY sustained onset (AUROC)", bl.PRIMARY[4:], "auroc"),
          ("SECONDARY first-crossing onset (AP)", bl.SECONDARY, "ap"), ("FORECAST severity at t+h (RMSE)", bl.FORECAST[:6], "rmse"),
          ("FORECAST status at t+h (AP)", bl.FORECAST[6:], "ap"), ("FORECAST status at t+h (AUROC)", bl.FORECAST[6:], "auroc")]
RANK_GROUPS = [("nowcast severity", bl.PRIMARY[:2], "rmse"), ("nowcast status", bl.PRIMARY[2:4], "ap"), ("sustained onset", bl.PRIMARY[4:], "ap"),
               ("first-crossing onset", bl.SECONDARY, "ap"), ("forecast severity", bl.FORECAST[:6], "rmse"), ("forecast status", bl.FORECAST[6:], "ap")]


def paired(df):
    """per seed: mean over the group's tasks of the metric, new model minus GRU of the same seed; worse% = relative degradation (positive = worse than GRU)."""
    rows = []
    for gname, tasks, key in GROUPS:
        for split in ("cal", "test"):
            sub = df[(df.split == split) & df.task.isin(tasks) & (df.metric == key)]
            per = sub.groupby(["model", "seed", "task"]).value.mean().reset_index().groupby(["model", "seed"]).value.mean().unstack(0)
            for m in NEW:
                if m not in per or "gru" not in per:
                    continue
                d = per[m] - per["gru"]
                worse = d / per["gru"] * 100 * (1 if key in cs.LOWER_BETTER else -1)
                rows.append(dict(group=gname, metric=key, split=split, model=m, gru_mean=per["gru"].mean(), model_mean=per[m].mean(), worse_pct_mean=worse.mean(), worse_pct_sd=worse.std(),
                                 n_seeds_better_than_gru=int((worse < 0).sum()), n_seeds=len(worse)))
    return pd.DataFrame(rows)


def cost_table():
    rows = []
    for src, part in (("results/stage3", "main"), ("results/stage3c_forecast", "forecast"), ("results/p1_baselines/main", "main"), ("results/p1_baselines/forecast", "forecast")):
        p = os.path.join(src, "run_log.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["runs"]:
            if r["model"] not in ("gru", "transformer", "cnnlstm"):
                continue
            h = r["history"]
            b = h[r["best_epoch"] - 1]
            rows.append(dict(part=part, model=r["model"], seed=r["seed"], n_params=r["n_params"], wall_s=r["wall_s"], cpu_s=r.get("cpu_s"), peak_mb_process=r.get("peak_working_set_mb_process_so_far"),
                             best_epoch=r["best_epoch"], epochs_run=r["epochs_run"], train_loss_at_best=b["train_loss"], cal_loss_at_best=b["cal_loss"],
                             train_loss_last=h[-1]["train_loss"], cal_loss_last=h[-1]["cal_loss"], cal_loss_rise_after_best=h[-1]["cal_loss"] - b["cal_loss"],
                             hit_max_epochs=r["epochs_run"] >= 25))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    dirs = [d for d in DIRS if os.path.exists(os.path.join(d, "metrics_long.csv"))]
    df = cs.load(dirs)
    cs.counts = pd.concat([pd.read_csv(os.path.join(d, "counts.csv")) for d in dirs if os.path.exists(os.path.join(d, "counts.csv"))]).drop_duplicates(["task", "split"])
    models = [m for m in MODELS if m in set(df.model)]
    L = ["# P1-1: Transformer-encoder and CNN-LSTM baselines vs existing models (SYNTHETIC; EXTENSION, not part of the original six stages)\n",
         "Frozen protocol, 5 seeds (gbt: one deterministic run), no tuning. Cells: mean ±SD over seeds. Test = final evaluation only; CAL = decision-grade. Blank = that model was not trained on that task set "
         "(e.g. the GRU/KAN/GBT forecast heads come from the 12-head forecast runs; first-crossing heads exist only in the 16-head main runs and in M1/KAN-GRU 28-head runs).\n"]
    for title, tasks, key in GROUPS:
        L += [f"\n## {title}\n", cs.table(df, tasks, key, models), ""]
    L.append("\n## Mean rank over each group's tasks (excluding B0; lower = better; ties not broken)\n")
    L.append("| group | CAL ranking | TEST ranking |\n|---|---|---|")
    for gname, tasks, key in RANK_GROUPS:
        res = []
        for split in ("cal", "test"):
            ranks = []
            for t in tasks:
                v = cs.agg(df, split, t, key)
                v = v[v.index != "B0_reference"]["mean"]
                ranks.append(v.rank(ascending=key in cs.LOWER_BETTER))
            res.append(pd.concat(ranks, axis=1).mean(axis=1).sort_values())
        L.append(f"| {gname} | " + ", ".join(f"{m} ({r:.2f})" for m, r in res[0].items()) + " | " + ", ".join(f"{m} ({r:.2f})" for m, r in res[1].items()) + " |")
    pr = paired(df)
    pr.to_csv(os.path.join(OUT, "paired_vs_gru.csv"), index=False)
    L += ["\n## Paired comparison with the GRU, same seed (mean over the group's tasks; worse% = degradation relative to the GRU, NEGATIVE = better than GRU; n = seeds better)\n",
          "| group | split | " + " | ".join(f"{m}: worse% ±SD (n better)" for m in NEW) + " |", "|---|---|" + "---|" * len(NEW)]
    for (g, sp), x in pr.groupby(["group", "split"], sort=False):
        L.append(f"| {g} | {sp} | " + " | ".join((lambda r: f"{r.worse_pct_mean:+.1f} ±{r.worse_pct_sd:.1f} ({r.n_seeds_better_than_gru}/{r.n_seeds})")(x[x.model == m].iloc[0]) for m in NEW) + " |")
    ct = cost_table()
    ct.to_csv(os.path.join(OUT, "cost_and_convergence.csv"), index=False)
    s = ct.groupby(["part", "model"]).agg(params=("n_params", "first"), wall_s=("wall_s", "mean"), cpu_s=("cpu_s", "mean"), best_epoch=("best_epoch", "mean"), epochs_run=("epochs_run", "mean"),
                                         n_best_epoch_le2=("best_epoch", lambda x: int((x <= 2).sum())), max_epochs_hit=("hit_max_epochs", "sum"), train_at_best=("train_loss_at_best", "mean"),
                                         cal_at_best=("cal_loss_at_best", "mean"), cal_rise=("cal_loss_rise_after_best", "mean"), n=("seed", "count")).reset_index()
    L += ["\n## Computational cost and convergence (CPU, 4 torch threads; GRU values from the original runs)\n",
          "| part | model | params | mean wall s/seed | mean CPU s/seed | mean best epoch | seeds with best epoch <= 2 | runs hitting 25 epochs | train loss at best | cal loss at best | mean cal-loss rise after best | n |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.part} | {r.model} | {int(r.params)} | {r.wall_s:.0f} | {r.cpu_s if pd.notna(r.cpu_s) else float('nan'):.0f} | {r.best_epoch:.1f} | {int(r.n_best_epoch_le2)}/{int(r.n)} | {int(r.max_epochs_hit)} | {r.train_at_best:.4f} | {r.cal_at_best:.4f} | {r.cal_rise:+.4f} | {int(r.n)} |")
    for part, d in (("main", "results/p1_baselines/main"), ("forecast", "results/p1_baselines/forecast")):
        p = os.path.join(d, "bootstrap_ci_test.csv")
        if not os.path.exists(p):
            continue
        b = pd.read_csv(p)
        ref = pd.read_csv("results/stage3/bootstrap_ci_test.csv" if part == "main" else "results/stage3c_forecast/bootstrap_ci_test.csv")
        ref = ref[ref.model == "gru"]
        b = pd.concat([b, ref], ignore_index=True)
        b = b[b.seed == 0]
        L += [f"\n## Seed-0 participant-bootstrap 95% CIs, test ({part} task set; point [lo, hi]); overlap with the GRU interval means the difference is not resolvable\n",
              "| task | metric | gru | transformer | cnnlstm |", "|---|---|---|---|---|"]
        for (t, k), x in b.groupby(["task", "metric"], sort=False):
            c = {r.model: f"{r.point:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]" for r in x.itertuples()}
            L.append(f"| {t} | {k} | {c.get('gru', '')} | {c.get('transformer', '')} | {c.get('cnnlstm', '')} |")
    open(os.path.join(OUT, "p1_comparison.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-60:]))


if __name__ == "__main__":
    main()
