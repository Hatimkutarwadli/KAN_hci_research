"""Updated baseline comparison across result directories (SYNTHETIC data).
Reads metrics_long.csv files only (no model is re-run). Test numbers are FINAL EVALUATION REPORTING; the
'strongest on calibration' column is what any further decision must rely on.

Usage: python src/compare_stage3.py results/stage3 results/stage3b_kan_reg results/stage3c_forecast results/stage3d_summary
"""
import os
import sys

import numpy as np
import pandas as pd

import baselines as bl

LOWER_BETTER = {"rmse", "mae"}


def load(dirs):
    parts = []
    for d in dirs:
        df = pd.read_csv(os.path.join(d, "metrics_long.csv"))
        df["source"] = d
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # B0_reference exists in every run for its own task set; keep one copy per (task, split, metric)
    b0 = df[df.model == "B0_reference"].drop_duplicates(["task", "split", "metric"])
    return pd.concat([df[df.model != "B0_reference"], b0], ignore_index=True)


def agg(df, split, task, metric):
    v = df[(df.split == split) & (df.task == task) & (df.metric == metric)].groupby("model").value.agg(["mean", "std", "count"])
    return v


def cell(v):
    return f"{v['mean']:.3f}" + (f" ±{v['std']:.3f}" if v["count"] > 1 else "")


def strongest(v, metric):
    v = v[v.index != "B0_reference"]
    return (v["mean"].idxmin() if metric in LOWER_BETTER else v["mean"].idxmax()) if len(v) else ""


def table(df, tasks, key, models):
    out = []
    for t in tasks:
        cnt = counts[counts.task == t].set_index("split")
        te = cnt.loc["test"]
        pos = "" if pd.isna(te.positive_origins) else f" / {int(te.positive_origins):,} pos ({int(te.positive_participants)} ppl)"
        row = [t, f"{int(te.eligible_origins):,}{pos}"]
        vt, vc = agg(df, "test", t, key), agg(df, "cal", t, key)
        for m in models:
            row.append(cell(vt.loc[m]) if m in vt.index else "")
        row += [strongest(vc, key), strongest(vt, key)]
        out.append(row)
    hdr = ["task", "test eligible origins / positives"] + models + [f"best on CAL ({key})", f"best on TEST ({key})"]
    return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in out)


if __name__ == "__main__":
    *dirs, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    df = load(dirs)
    counts = pd.concat([pd.read_csv(os.path.join(d, "counts.csv")) for d in dirs]).drop_duplicates(["task", "split"])
    models = ["B0_reference", "gbt", "gru", "kan", "kan_reg", "m1_static", "m1"]
    models = [m for m in models if m in set(df.model)]
    groups = [("NOWCAST severity (RMSE)", bl.PRIMARY[:2], "rmse"), ("NOWCAST severity (R2)", bl.PRIMARY[:2], "r2"),
              ("NOWCAST status (AP)", bl.PRIMARY[2:4], "ap"), ("NOWCAST status (AUROC)", bl.PRIMARY[2:4], "auroc"),
              ("PRIMARY sustained onset (AP)", bl.PRIMARY[4:], "ap"), ("PRIMARY sustained onset (AUROC)", bl.PRIMARY[4:], "auroc"),
              ("SECONDARY first-crossing onset (AP)", bl.SECONDARY, "ap"),
              ("FORECAST severity at t+h (RMSE)", bl.FORECAST[:6], "rmse"), ("FORECAST severity at t+h (R2)", bl.FORECAST[:6], "r2"),
              ("FORECAST status at t+h (AP)", bl.FORECAST[6:], "ap"), ("FORECAST status at t+h (AUROC)", bl.FORECAST[6:], "auroc")]
    lines = ["# Updated baseline comparison (SYNTHETIC data; test = final evaluation only)\n",
             "NN cells: mean ±SD over seeds; gbt: single deterministic run. 'best on CAL' is the decision-grade ranking.\n"]
    for title, tasks, key in groups:
        if not all(t in set(df.task) for t in tasks):
            continue
        lines += [f"\n## {title}\n", table(df, tasks, key, models), ""]
    # strongest per group by mean rank on calibration and on test
    lines.append("\n## Strongest baseline per group (mean rank over the group's tasks; metric = RMSE for severity, AP for status/onset)\n")
    lines.append("| group | best on CAL | best on TEST |\n|---|---|---|")
    for gname, tasks, key in (("nowcast severity", bl.PRIMARY[:2], "rmse"), ("nowcast status", bl.PRIMARY[2:4], "ap"),
                              ("sustained onset", bl.PRIMARY[4:], "ap"), ("first-crossing onset", bl.SECONDARY, "ap"),
                              ("forecast severity", bl.FORECAST[:6], "rmse"), ("forecast status", bl.FORECAST[6:], "ap")):
        if not all(t in set(df.task) for t in tasks):
            continue
        res = []
        for split in ("cal", "test"):
            ranks = []
            for t in tasks:
                v = agg(df, split, t, key)
                v = v[v.index != "B0_reference"]["mean"]
                ranks.append(v.rank(ascending=key in LOWER_BETTER))
            res.append(pd.concat(ranks, axis=1).mean(axis=1).sort_values())
        lines.append(f"| {gname} | " + ", ".join(f"{m} ({r:.2f})" for m, r in res[0].items()) + " | " +
                     ", ".join(f"{m} ({r:.2f})" for m, r in res[1].items()) + " |")
    with open(os.path.join(out, "updated_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    df.to_csv(os.path.join(out, "metrics_all_long.csv"), index=False)
    print("\n".join(lines))
