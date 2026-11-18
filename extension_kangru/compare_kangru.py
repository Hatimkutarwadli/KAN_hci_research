"""EXTENSION comparison: existing results (read-only) + results/ext_kangru, by task. Retrains nothing. Reuses src/compare_stage3.py helpers.
Usage: python extension_kangru/compare_kangru.py results/ext_kangru/comparison"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]
import baselines as bl  # noqa: E402
import compare_stage3 as cs  # noqa: E402

DIRS = ["results/stage3", "results/stage3b_kan_reg/final", "results/stage3c_forecast", "results/stage4_m1",
        "results/stage4_ablation", "results/ext_kangru"]
MODELS = ["B0_reference", "gbt", "gru", "kan", "kan_reg", "m1_static", "m1", "kangru_fixed", "kangru"]
GROUPS = [("NOWCAST severity (RMSE)", bl.PRIMARY[:2], "rmse"), ("NOWCAST status (AP)", bl.PRIMARY[2:4], "ap"),
          ("PRIMARY sustained onset (AP)", bl.PRIMARY[4:], "ap"), ("PRIMARY sustained onset (AUROC)", bl.PRIMARY[4:], "auroc"),
          ("SECONDARY first-crossing onset (AP)", bl.SECONDARY, "ap"),
          ("FORECAST severity at t+h (RMSE)", bl.FORECAST[:6], "rmse"), ("FORECAST status at t+h (AP)", bl.FORECAST[6:], "ap")]

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    df = cs.load(DIRS)
    cs.counts = pd.concat([pd.read_csv(os.path.join(d, "counts.csv")) for d in DIRS if os.path.exists(os.path.join(d, "counts.csv"))]).drop_duplicates(["task", "split"])
    if not len(cs.counts):
        raise SystemExit("no counts")
    models = [m for m in MODELS if m in set(df.model)]
    L = ["# KAN-GRU extension: comparison on matching tasks (SYNTHETIC; EXTENSION, not part of the original six stages)\n",
         "Cells: mean ±SD over 5 seeds (gbt: one deterministic run). Test = final evaluation only. Every model below was evaluated on each listed task.\n"]
    for title, tasks, key in GROUPS:
        L += [f"\n## {title}\n", cs.table(df, tasks, key, models), ""]
    L.append("\n## Mean rank over each group's tasks (excluding B0; lower = better). Decision-grade = CAL.\n")
    L.append("| group | CAL ranking | TEST ranking |\n|---|---|---|")
    for gname, tasks, key in (("nowcast severity", bl.PRIMARY[:2], "rmse"), ("nowcast status", bl.PRIMARY[2:4], "ap"),
                              ("sustained onset", bl.PRIMARY[4:], "ap"), ("first-crossing onset", bl.SECONDARY, "ap"),
                              ("forecast severity", bl.FORECAST[:6], "rmse"), ("forecast status", bl.FORECAST[6:], "ap"),
                              ("ALL 28 tasks (rmse for severity, AP otherwise)", bl.ALL_HEADS, None)):
        res = []
        for split in ("cal", "test"):
            ranks = []
            for t in tasks:
                k = key or ("rmse" if t.startswith("sev") else "ap")
                v = cs.agg(df, split, t, k)
                v = v[v.index != "B0_reference"]["mean"]
                ranks.append(v.rank(ascending=k in cs.LOWER_BETTER))
            res.append(pd.concat(ranks, axis=1).mean(axis=1).sort_values())
        L.append(f"| {gname} | " + ", ".join(f"{m} ({r:.2f})" for m, r in res[0].items()) + " | " + ", ".join(f"{m} ({r:.2f})" for m, r in res[1].items()) + " |")
    open(os.path.join(out, "kangru_comparison.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[-12:]))
