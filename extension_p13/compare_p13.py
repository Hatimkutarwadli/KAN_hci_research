"""P1-3 predictive comparison (SYNTHETIC; EXTENSION): matched-parameter MLP controls and no-spline ablation vs the existing KAN models and the GRU. Existing results are read-only;
nothing is retrained. Writes results/p1_kan_mlp_control/comparison/{p13_comparison.md, paired.csv, cost_and_convergence.csv}.
Usage: python extension_p13/compare_p13.py"""
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

BASE = "results/p1_kan_mlp_control"
OUT = f"{BASE}/comparison"
DIRS = ["results/stage3", "results/stage3b_kan_reg/final", "results/stage3c_forecast", "results/stage4_m1", "results/stage4_ablation", f"{BASE}/b2_main", f"{BASE}/m1_all"]
MODELS = ["B0_reference", "gru", "kan", "kan_nospline", "mlp_b2", "kan_reg", "mlp_b2_reg", "m1_static", "m1", "mlp_m1"]
# (new model, reference model, label): MLP control vs the KAN it controls; no-spline ablation vs full KAN; each new model vs GRU
PAIRS = [("mlp_b2", "kan", "B2: matched MLP vs KAN"), ("mlp_b2_reg", "kan_reg", "B2 regularised: matched MLP vs KAN"), ("kan_nospline", "kan", "B2: no-spline ablation vs KAN"),
         ("mlp_m1", "m1", "M1: matched MLP vs KAN"), ("mlp_b2", "gru", "B2 MLP vs GRU"), ("kan", "gru", "B2 KAN vs GRU (existing)"), ("mlp_m1", "gru", "M1 MLP vs GRU"),
         ("m1", "gru", "M1 KAN vs GRU (existing)")]
GROUPS = [("NOWCAST severity (RMSE)", bl.PRIMARY[:2], "rmse"), ("NOWCAST status (AP)", bl.PRIMARY[2:4], "ap"), ("NOWCAST status (AUROC)", bl.PRIMARY[2:4], "auroc"),
          ("PRIMARY sustained onset (AP)", bl.PRIMARY[4:], "ap"), ("PRIMARY sustained onset (AUROC)", bl.PRIMARY[4:], "auroc"),
          ("SECONDARY first-crossing onset (AP)", bl.SECONDARY, "ap"), ("FORECAST severity at t+h (RMSE)", bl.FORECAST[:6], "rmse"),
          ("FORECAST status at t+h (AP)", bl.FORECAST[6:], "ap")]


def paired(df):
    rows = []
    for gname, tasks, key in GROUPS:
        for split in ("cal", "test"):
            sub = df[(df.split == split) & df.task.isin(tasks) & (df.metric == key)]
            per = sub.groupby(["model", "seed", "task"]).value.mean().reset_index().groupby(["model", "seed"]).value.mean().unstack(0)
            for new, ref, lab in PAIRS:
                if new not in per or ref not in per:
                    continue
                ok = per[[new, ref]].dropna()
                if len(ok) < 2:
                    continue
                d = ok[new] - ok[ref]
                worse = d / ok[ref] * 100 * (1 if key in cs.LOWER_BETTER else -1)
                rows.append(dict(comparison=lab, group=gname, metric=key, split=split, new=new, ref=ref, ref_mean=ok[ref].mean(), new_mean=ok[new].mean(), worse_pct_mean=worse.mean(),
                                 worse_pct_sd=worse.std(), n_seeds_new_better=int((worse < 0).sum()), n_seeds=len(worse)))
    return pd.DataFrame(rows)


def cost_table():
    rows = []
    for d in DIRS:
        p = os.path.join(d, "run_log.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["runs"]:
            if r["model"] == "gbt" or "history" not in r:
                continue
            h, b = r["history"], r["history"][r["best_epoch"] - 1]
            rows.append(dict(model=r["model"], seed=r["seed"], n_params=r["n_params"], n_trainable=r.get("n_trainable", r["n_params"]), wall_s=r["wall_s"], cpu_s=r.get("cpu_s"),
                             best_epoch=r["best_epoch"], epochs_run=r["epochs_run"], train_loss_at_best=b["train_loss"], cal_loss_at_best=b["cal_loss"],
                             cal_loss_rise_after_best=h[-1]["cal_loss"] - b["cal_loss"], hit_max_epochs=r["epochs_run"] >= 25, src=d))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    dirs = [d for d in DIRS if os.path.exists(os.path.join(d, "metrics_long.csv"))]
    df = cs.load(dirs)
    cs.counts = pd.concat([pd.read_csv(os.path.join(d, "counts.csv")) for d in dirs if os.path.exists(os.path.join(d, "counts.csv"))]).drop_duplicates(["task", "split"])
    models = [m for m in MODELS if m in set(df.model)]
    L = ["# P1-3: matched-parameter MLP controls and no-spline ablation vs the KAN models (SYNTHETIC; EXTENSION, not part of the original six stages)\n",
         "Frozen protocol, 5 seeds, no tuning; each control uses the configuration of the KAN it controls. Cells: mean ±SD over seeds. Test = final evaluation only; CAL = decision-grade. "
         "Blank = model not trained on that task set (B2/GRU main/forecast heads come from the 16-head and 12-head runs; M1 and its MLP control were trained on all 28 heads).\n"]
    for title, tasks, key in GROUPS:
        L += [f"\n## {title}\n", cs.table(df, tasks, key, models), ""]
    pr = paired(df)
    pr.to_csv(os.path.join(OUT, "paired.csv"), index=False)
    L += ["\n## Paired comparisons, same seed (mean over the group's tasks; worse% = degradation of the first model relative to the second, NEGATIVE = first is better; n = seeds where the first is better)\n"]
    for lab in dict.fromkeys(p[2] for p in PAIRS):
        x = pr[pr.comparison == lab]
        if x.empty:
            continue
        L += [f"\n### {lab}\n", "| group | CAL worse% ±SD (n better) | TEST worse% ±SD (n better) | TEST means (first / second) |", "|---|---|---|---|"]
        for g in dict.fromkeys(x.group):
            c, t = x[(x.group == g) & (x.split == "cal")], x[(x.group == g) & (x.split == "test")]
            f = lambda r: f"{r.worse_pct_mean:+.1f} ±{r.worse_pct_sd:.1f} ({r.n_seeds_new_better}/{r.n_seeds})"      # noqa: E731
            if len(t):
                t0 = t.iloc[0]
                L.append(f"| {g} | {f(c.iloc[0]) if len(c) else ''} | {f(t0)} | {t0.new_mean:.3f} / {t0.ref_mean:.3f} |")
    ct = cost_table()
    ct.to_csv(os.path.join(OUT, "cost_and_convergence.csv"), index=False)
    s = ct.groupby(["src", "model"]).agg(params=("n_params", "first"), trainable=("n_trainable", "first"), wall_s=("wall_s", "mean"), cpu_s=("cpu_s", "mean"), best_epoch=("best_epoch", "mean"),
                                        epochs_run=("epochs_run", "mean"), n_best_le2=("best_epoch", lambda x: int((x <= 2).sum())), hit25=("hit_max_epochs", "sum"),
                                        train_at_best=("train_loss_at_best", "mean"), cal_at_best=("cal_loss_at_best", "mean"), cal_rise=("cal_loss_rise_after_best", "mean"), n=("seed", "count")).reset_index()
    s = s[s.model.isin(MODELS)]
    L += ["\n## Computational cost and convergence (CPU, 4 torch threads; existing models from their original runs, so wall times may reflect different machine load)\n",
          "| run dir | model | params | trainable | wall s/seed | CPU s/seed | mean best epoch | seeds best epoch <= 2 | runs hitting 25 epochs | train loss at best | cal loss at best | mean cal-loss rise after best | n |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.src.replace('results/', '')} | {r.model} | {int(r.params)} | {int(r.trainable)} | {r.wall_s:.0f} | {r.cpu_s:.0f} | {r.best_epoch:.1f} | {int(r.n_best_le2)}/{int(r.n)} | {int(r.hit25)} | "
                 f"{r.train_at_best:.4f} | {r.cal_at_best:.4f} | {r.cal_rise:+.4f} | {int(r.n)} |")
    brows = []
    for d in ("results/stage3", "results/stage3b_kan_reg/final", "results/stage4_m1", f"{BASE}/b2_main", f"{BASE}/m1_all"):
        p = os.path.join(d, "bootstrap_ci_test.csv")
        if os.path.exists(p):
            b = pd.read_csv(p)
            brows.append(b[(b.seed == 0) & b.model.isin(MODELS)])
    if brows:
        b = pd.concat(brows)
        L += ["\n## Seed-0 participant-bootstrap 95% CIs, test (point [lo, hi]); overlapping intervals = difference not resolvable\n", "| task | metric | " + " | ".join(m for m in MODELS if m in set(b.model)) + " |",
              "|---|---|" + "---|" * len([m for m in MODELS if m in set(b.model)])]
        for (t, k), x in b.groupby(["task", "metric"], sort=False):
            c = {r.model: f"{r.point:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]" for r in x.itertuples()}
            L.append(f"| {t} | {k} | " + " | ".join(c.get(m, "") for m in MODELS if m in set(b.model)) + " |")
    open(os.path.join(OUT, "p13_comparison.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(OUT, "p13_comparison.md"))


if __name__ == "__main__":
    main()
