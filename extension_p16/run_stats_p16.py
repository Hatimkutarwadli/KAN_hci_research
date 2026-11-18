"""P1-6 step 2 (SYNTHETIC; EXTENSION): participant-cluster bootstrap values from the regenerated per-origin predictions (results/p1_stage9_stats/predictions). Row 0 of every array is the pooled point
estimate (identical to the Stage 8/9 CSV metric); rows 1..B are bootstrap resamples with the SAME shared participant counts for every model, seed, scenario, replicate and task (B = 2000, fixed seed).
Group values follow the Stage 9 convention: mean over mask replicates, then mean over the tasks of the group. Also accumulates per-participant mean losses (squared error for severity,
Brier for status/onset; averaged over seeds, replicates, tasks, origins) for the descriptive participant-consistency analysis, and the event counts.
Writes results/p1_stage9_stats/boot/{boot_values.npz, participant_loss.npz, event_counts.csv, nan_resamples.csv, config.json}; refuses to overwrite. Usage: python extension_p16/run_stats_p16.py"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "extension_p16")]
import analyze_missing as am  # noqa: E402  (read-only: Stage 8/9 metric-to-group mapping)
import inject as ij  # noqa: E402
import regen_p16 as rg  # noqa: E402
import stats16 as st  # noqa: E402

OUT = "results/p1_stage9_stats"
B, BOOT_SEED = 2000, 20261120


def loss_group(part, task):
    if task.startswith("sev_"):
        return "sev_nowcast" if task.endswith("_t0") else "sev_forecast"
    if task.startswith("status_"):
        return f"status_{part}"
    return "onset" if task.startswith("onset_sust") else None


def main():
    os.chdir(ROOT)
    od = os.path.join(OUT, "boot")
    if os.path.isdir(od) and os.listdir(od):
        raise SystemExit(f"{od} exists and is not empty; refusing to overwrite.")
    os.makedirs(od, exist_ok=True)
    rep_ok = json.load(open(os.path.join(OUT, "repro", "repro_check.json")))
    assert rep_ok["n_failures"] == 0, "reproduction gate failed; do not run the analysis"
    meta = json.load(open(os.path.join(OUT, "regen_meta.json")))
    d, pi, ti, *_ = rg.setup()
    N = len(d["pid"])
    heads = sorted({h for v in meta["heads"].values() for h in v if "cross" not in h})
    Y = {h: d["y_" + h][pi, ti] for h in heads}
    C = st.boot_counts(B, N, BOOT_SEED)
    reg = {s["id"]: s for s in ij.registry()}
    acc, pl_sum, pl_n, nan_ct = {}, {}, {}, {}
    t0 = time.time()
    for sid in meta["scenarios"]:
        for rep in reg[sid]["reps"]:
            pred = np.load(os.path.join(OUT, "predictions", f"{sid}_rep{rep}.npz"))
            for c in meta["configs"]:
                for p in rg.PARTS:
                    hl = meta["heads"][f"{c}|{p}"]
                    for seed in rg.SEEDS:
                        P = pred[f"{c}|{p}|{seed}"]
                        for j, h in enumerate(hl):
                            if "cross" in h:
                                continue
                            y = Y[h]
                            ok = ~np.isnan(y)
                            g, pp, yy = pi[ok], P[ok, j].astype(float), y[ok]
                            lg = loss_group(p, h)
                            if h.startswith("sev_"):
                                e2 = (pp - yy) ** 2
                                vals = {"rmse": st.weighted_rmse(e2, g, C, N)}
                            else:
                                e2 = (pp - yy) ** 2
                                ap, auc = st.weighted_ap_auroc(yy.astype(int), pp, g, C)
                                vals = {"ap": ap}
                                if h.startswith("onset_sust"):
                                    vals["auroc"] = auc
                                    vals["mean_pred"] = st.weighted_mean(pp, g, C, N)
                            for metric, v in vals.items():
                                grp = "onset_meanpred" if metric == "mean_pred" else am.group_of(h, metric)
                                if grp is None:
                                    continue
                                k = (c, p, sid, seed, grp, h)
                                a = acc.setdefault(k, [np.zeros(B + 1), 0])
                                a[0] += v
                                a[1] += 1
                                if np.isnan(v).any():
                                    nan_ct[k] = nan_ct.get(k, 0) + int(np.isnan(v).sum())
                            k2 = (c, p, sid, lg)
                            pl_sum[k2] = pl_sum.get(k2, 0) + np.bincount(g, weights=e2, minlength=N)
                            pl_n[k2] = pl_n.get(k2, 0) + np.bincount(g, minlength=N)
            print(f"{sid} rep{rep} {time.time() - t0:.0f}s", flush=True)
    # group values: mean over reps (per task), then mean over tasks in the group
    per_task = {k: v[0] / v[1] for k, v in acc.items()}
    groups = {}
    for (c, p, sid, seed, grp, h), v in per_task.items():
        groups.setdefault((c, p, sid, seed, grp), []).append(v)
    bv = {"|".join(map(str, k)): np.mean(vs, 0) for k, vs in groups.items()}
    np.savez_compressed(os.path.join(od, "boot_values.npz"), **bv)
    np.savez_compressed(os.path.join(od, "participant_loss.npz"), **{"|".join(map(str, k)): pl_sum[k] / np.maximum(pl_n[k], 1) for k in pl_sum})
    np.save(os.path.join(od, "counts_C.npy"), C.astype(np.int16))
    ec = []
    for h in heads:
        if h.startswith("sev_"):
            continue
        y = Y[h]
        ok = ~np.isnan(y)
        pos_p = np.unique(pi[ok][y[ok] == 1])
        ec.append(dict(task=h, n_origins=int(ok.sum()), n_positive_origins=int((y[ok] == 1).sum()), prevalence=float((y[ok] == 1).mean()), n_participants=N, participants_with_event=len(pos_p)))
    pd.DataFrame(ec).to_csv(os.path.join(od, "event_counts.csv"), index=False)
    pd.DataFrame([dict(config=k[0], part=k[1], scenario=k[2], seed=k[3], group=k[4], task=k[5], n_nan_resample_values=v) for k, v in nan_ct.items()]).to_csv(os.path.join(od, "nan_resamples.csv"), index=False)
    json.dump(dict(B=B, boot_seed=BOOT_SEED, N=N, scenarios=meta["scenarios"], configs=meta["configs"], decision_sha256=meta["decision_sha256"], n_arrays=len(bv),
                   note="row 0 = pooled point estimate; rows 1..B = participant-cluster bootstrap with shared counts C (counts_C.npy)"), open(os.path.join(od, "config.json"), "w"), indent=1)
    print("done", f"{time.time() - t0:.0f}s", len(bv), "arrays")


if __name__ == "__main__":
    main()
