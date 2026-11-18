"""Stage 9 step 2 comparison on CALIBRATION data (SYNTHETIC): original GRU (Stage 8 metrics, model 'gru'/part 'main') vs GRU R0.O0.T1, identical Stage 8 scenarios.
Reads CSVs only. Seed s of the baseline is paired with seed s of the new model (same init and batch order; they differ only by training augmentation).
'worse%' of the new model relative to the baseline on the SAME scenario: rmse (new-base)/base*100, ap/auroc (base-new)/base*100;  NEGATIVE = new model better.
Usage: python extension_v2/compare_v2.py results/stage9_v2 [config]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "src")]
import analyze_missing as am  # noqa: E402
import inject as ij  # noqa: E402

GROUPS = ["sev_nowcast_rmse", "status_ap", "onset_ap", "onset_auroc"]
SCEN_ORDER = [s["id"] for s in ij.registry()]


def read(files):
    return pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True)


def values(df, label):
    d = df.copy()
    d["group"] = [am.group_of(t, k) for t, k in zip(d.task, d.metric)]
    d = d[d.group.notna()]
    v = d.groupby(["scenario", "seed", "group", "task", "metric"]).value.mean().reset_index()            # mean over mask replicates
    v = v.groupby(["scenario", "seed", "group"]).value.mean().reset_index()                                  # mean over tasks in group
    v["model"] = label
    return v


def worse(group, new, base):
    return np.where(group == "sev_nowcast_rmse", (new - base) / base * 100, (base - new) / base * 100)


def main(root, cfg):
    base = read(glob.glob("results/stage8_missing/cal/metrics/*.csv"))
    base = base[(base.model == "gru") & (base.part == "main")]
    new = read(glob.glob(os.path.join(root, "cal", f"metrics_{cfg}", "*.csv")))
    assert new.scenario.nunique() == len(SCEN_ORDER), new.scenario.nunique()
    vb, vn = values(base, "base"), values(new, cfg)
    m = vb.merge(vn, on=["scenario", "seed", "group"], suffixes=("_b", "_n"))
    m["worse_pct"] = worse(m.group, m.value_n, m.value_b)
    m.to_csv(os.path.join(root, "analysis", f"step2_cal_paired_{cfg}.csv"), index=False)
    L = [f"# Stage 9 step 2 - {cfg} vs original GRU, CALIBRATION split (SYNTHETIC; origins 27,29,...,119)\n",
         "Per seed (5 paired seeds): value = mean over mask replicates and tasks in the group. `Δ%` = worse% of the new model vs the baseline on the same scenario (NEGATIVE = new model better), mean ±SD over seeds; `n` = seeds (of 5) where the new model is better.\n"]
    for g in GROUPS:
        L += [f"\n## {g}\n", "| scenario | baseline | new | Δ% (new vs base) | n better |", "|---|---|---|---|---|"]
        for sc in SCEN_ORDER:
            x = m[(m.scenario == sc) & (m.group == g)]
            if x.empty:
                continue
            L.append(f"| {sc} | {x.value_b.mean():.4f} | {x.value_n.mean():.4f} | {x.worse_pct.mean():+.1f} ±{x.worse_pct.std():.1f} | {(x.worse_pct < 0).sum()}/{len(x)} |")
    # degradation against each model's OWN control
    key = ["seed", "group"]
    c = m[m.scenario == "C0"].set_index(key)[["value_b", "value_n"]]
    d = m[m.scenario != "C0"].join(c, on=key, rsuffix="_c")
    d["deg_b"] = np.where(d.group == "sev_nowcast_rmse", (d.value_b - d.value_b_c) / d.value_b_c * 100, (d.value_b_c - d.value_b) / d.value_b_c * 100)
    d["deg_n"] = np.where(d.group == "sev_nowcast_rmse", (d.value_n - d.value_n_c) / d.value_n_c * 100, (d.value_n_c - d.value_n) / d.value_n_c * 100)
    L += ["\n## Degradation vs each model's own clean control (worse% , + = worse; mean ±SD over 5 seeds)\n", "| scenario | group | baseline GRU | new | "[:-1], "|---|---|---|---|"]
    L = L[:-2] + ["| scenario | group | baseline GRU | new |", "|---|---|---|---|"]
    for sc in ["S1_r20", "S1_r40", "S2_r20_L7", "S2_r40_L7", "S3_k3", "S3_k7", "S3_k14", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S4_circadian_absent", "S5_rand40", "S5_blk40", "Z0"]:
        for g in GROUPS:
            x = d[(d.scenario == sc) & (d.group == g)]
            L.append(f"| {sc} | {g} | {x.deg_b.mean():+.1f} ±{x.deg_b.std():.1f} | {x.deg_n.mean():+.1f} ±{x.deg_n.std():.1f} |")
    # calibration of predicted risk (weakness 2)
    def mp(df, label):
        t = df[(df.metric == "mean_pred") & df.task.str.startswith("onset_sust")]
        return t.groupby(["scenario", "seed"]).value.mean().reset_index().assign(model=label)
    pb, pn = mp(base, "base"), mp(new, cfg)
    pc = {k: p[p.scenario == "C0"].set_index("seed").value for k, p in (("base", pb), ("new", pn))}
    sev = lambda df, label: df[(df.metric == "mean_pred") & df.task.str.startswith("sev_")].groupby(["scenario", "seed"]).value.mean().reset_index().assign(model=label)
    sb, sn = sev(base, "base"), sev(new, cfg)
    L += ["\n## Mean predicted onset probability, ratio to the model's own control (1.00 = unchanged) and severity shift in points (weakness 2)\n",
          "| scenario | onset ratio baseline | onset ratio new | severity shift baseline | severity shift new |", "|---|---|---|---|---|"]
    for sc in ["S1_r10", "S1_r20", "S1_r40", "S1_r60", "S2_r20_L7", "S2_r40_L7", "S3_k7", "S3_k14", "S5_rand40", "S5_blk40", "S1_r40_O", "Z0"]:
        r = []
        for p, k in ((pb, "base"), (pn, "new")):
            x = p[p.scenario == sc].set_index("seed").value
            r.append(f"{(x / pc[k]).mean():.2f} ±{(x / pc[k]).std():.2f}")
        for s_, k in ((sb, "base"), (sn, "new")):
            x = s_[s_.scenario == sc].set_index("seed").value
            c0 = s_[s_.scenario == "C0"].set_index("seed").value
            r.append(f"{(x - c0).mean():+.2f} ±{(x - c0).std():.2f}")
        L.append(f"| {sc} | " + " | ".join(r) + " |")
    L += ["\n## Mean predicted onset probability vs observed prevalence, clean control (calibration-in-the-large)\n"]
    prev = base[(base.scenario == "C0") & (base.metric == "prevalence") & base.task.str.startswith("onset_sust")].groupby("seed").value.mean()
    L.append(f"prevalence {prev.mean():.4f}; baseline mean prediction {pc['base'].mean():.4f}; new mean prediction {pc['new'].mean():.4f}")
    open(os.path.join(root, "analysis", f"step2_cal_{cfg}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.makedirs(os.path.join(sys.argv[1], "analysis"), exist_ok=True)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "R0O0T1")
