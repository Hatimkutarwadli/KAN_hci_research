"""Stage 8 analysis of results/stage8_missing/{cal,test}/metrics (SYNTHETIC). Reads CSV/NPZ only; trains nothing.
Degradation is always relative to the SAME model/seed/task under the control C0 (frozen windows through the identical pipeline; origins 27,29,...,119):
  rmse: (v - c)/c * 100  (positive = worse);   ap / auroc: (c - v)/c * 100  (positive = worse).
Decomposition: effect_O = degradation with values lost but ORIGINAL flag channels kept (physiological signal loss, informative flags intact);
  effect_flags = degradation(H) - degradation(O) (what telling the model about MCAR gaps / diluting the flags adds); Z0 = removing the informative flags
  with NO signal loss. Terms need not add up (interactions).
Usage: python extension_missing/analyze_missing.py results/stage8_missing
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "extension_missing"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import inject as ij  # noqa: E402
import metrics as mt  # noqa: E402

MODELS = ["gbt", "gru", "kan", "kan_reg", "m1_static", "m1", "kangru_fixed", "kangru"]
GROUP_ORDER = ["sev_nowcast_rmse", "sev_forecast_rmse", "status_ap", "onset_ap", "onset_auroc"]
DECOMP = ["S1_r20", "S1_r40", "S2_r20_L7", "S2_r40_L7", "S3_k7", "S3_k14"]
BOOT_SC = ["S1_r40", "S2_r40_L7", "S3_k7", "S3_k14", "S5_blk40", "Z0", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S4_circadian_absent"]
BOOT_TASKS = [("sev_dep_t0", "rmse"), ("onset_sust_dep_h14", "ap"), ("onset_sust_anx_h14", "ap")]
N_BOOT, BOOT_SEED = 200, 20261116


def group_of(task, metric):
    if metric == "rmse":
        return "sev_nowcast_rmse" if task.endswith("_t0") else "sev_forecast_rmse"
    if metric == "ap" and task.startswith("status_"):
        return "status_ap"
    if task.startswith("onset_sust"):
        return {"ap": "onset_ap", "auroc": "onset_auroc"}.get(metric)
    return None


def load(root, split):
    files = sorted(glob.glob(os.path.join(root, split, "metrics", "*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True), len(files)


def degradation(df):
    key = ["model", "part", "seed", "task", "metric"]
    m = df.groupby(["scenario"] + key).value.mean().reset_index()                      # mean over mask replicates
    c = m[m.scenario == "C0"].set_index(key).value.rename("ctrl")
    x = m.join(c, on=key)
    x["group"] = [group_of(t, k) for t, k in zip(x.task, x.metric)]
    d = x[x.group.notna() & (x.scenario != "C0")].copy()
    d["deg_pct"] = np.where(d.metric == "rmse", (d.value - d.ctrl) / d.ctrl * 100, (d.ctrl - d.value) / d.ctrl * 100)
    s = d.groupby(["scenario", "model", "seed", "group"]).deg_pct.mean().reset_index()         # mean over tasks in group
    return x, s


def cell(g):
    return f"{g.mean():+.1f}" + (f" ±{g.std():.1f}" if len(g) > 1 else "")


def table(s, group, scen, models=MODELS):
    L = ["| scenario | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for sc in scen:
        row = []
        for m in models:
            g = s[(s.scenario == sc) & (s.model == m) & (s.group == group)].deg_pct
            row.append(cell(g) if len(g) else "")
        L.append(f"| {sc} | " + " | ".join(row) + " |")
    return "\n".join(L)


def decomposition(s):
    L = []
    for group in ("sev_nowcast_rmse", "onset_ap"):
        L += [f"\n#### {group}: signal loss (flags original) | flag change (H - O) | total (H, frozen semantics)\n",
              "| scenario | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for base in DECOMP:
            for lab, fn in (("signal loss O", lambda m: s[(s.scenario == base + "_O") & (s.model == m) & (s.group == group)].deg_pct),
                            ("flag change H-O", lambda m: s[(s.scenario == base) & (s.model == m) & (s.group == group)].set_index("seed").deg_pct
                             - s[(s.scenario == base + "_O") & (s.model == m) & (s.group == group)].set_index("seed").deg_pct),
                            ("total H", lambda m: s[(s.scenario == base) & (s.model == m) & (s.group == group)].deg_pct)):
                L.append(f"| {base} {lab} | " + " | ".join(cell(fn(m)) for m in MODELS) + " |")
        L.append("| Z0 flags removed alone | " + " | ".join(cell(s[(s.scenario == "Z0") & (s.model == m) & (s.group == group)].deg_pct) for m in MODELS) + " |")
    return "\n".join(L)


def shift_table(x, scen):
    """mean predicted value vs control: onset heads = ratio of mean predicted probability, severity = shift in points."""
    mp = x[x.metric == "mean_pred"].copy()
    c = mp[mp.scenario == "C0"].set_index(["model", "part", "seed", "task"]).value
    mp = mp.drop(columns=["ctrl"], errors="ignore").join(c.rename("ctrl"), on=["model", "part", "seed", "task"])
    mp["kind"] = np.where(mp.task.str.startswith("sev_"), "sev", np.where(mp.task.str.startswith("onset_sust"), "onset", "other"))
    mp["delta"] = np.where(mp.kind == "sev", mp.value - mp.ctrl, mp.value / mp.ctrl)
    L = []
    for kind, lab in (("onset", "mean predicted onset probability, ratio to control (1.00 = unchanged)"), ("sev", "mean predicted severity, shift vs control (points)")):
        L += [f"\n#### {lab}\n", "| scenario | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for sc in scen:
            row = []
            for m in MODELS:
                g = mp[(mp.scenario == sc) & (mp.model == m) & (mp.kind == kind)].groupby("seed").delta.mean()
                row.append(f"{g.mean():.2f}" if kind == "onset" else f"{g.mean():+.2f}")
            L.append(f"| {sc} | " + " | ".join(row) + " |")
    return "\n".join(L)


def figures(s, split, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = dict(zip(MODELS, ["#8c564b", "#1f77b4", "#2ca02c", "#98df8a", "#ff7f0e", "#d62728", "#9467bd", "#e377c2"]))
    fam = [("S1 random whole-day gaps", ["S1_r10", "S1_r20", "S1_r40", "S1_r60"]), ("S2 block (7-day) whole-day gaps", ["S2_r10_L7", "S2_r20_L7", "S2_r40_L7", "S2_r60_L7"])]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, (title, scs) in enumerate(fam):
        for j, g in enumerate(("sev_nowcast_rmse", "onset_ap")):
            for m in MODELS:
                v = [s[(s.scenario == sc) & (s.model == m) & (s.group == g)].deg_pct.mean() for sc in scs]
                ax[i, j].plot([0, 10, 20, 40, 60][:1] + [10, 20, 40, 60], [0] + v, marker="o", ms=3, color=cols[m], label=m)
            ax[i, j].set(title=f"{title}: {g}", xlabel="added missing rate (%)", ylabel="degradation vs control (%, + = worse)")
            ax[i, j].title.set_fontsize(9)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{split}: degradation vs added missingness (SYNTHETIC; every 2nd origin day)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig1_rate_response_{split}.png"), dpi=130)
    plt.close(fig)
    scs = [f"S4_{g}_{k}" for g in ij.GROUPS for k in ("rand40", "blk40", "absent")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, g in zip(ax, ("sev_nowcast_rmse", "onset_ap")):
        M = np.array([[s[(s.scenario == sc) & (s.model == m) & (s.group == g)].deg_pct.mean() for m in MODELS] for sc in scs])
        im = a.imshow(M, aspect="auto", cmap="Reds", vmin=0)
        a.set_xticks(range(len(MODELS)), MODELS, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(len(scs)), scs, fontsize=8)
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                a.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center", fontsize=7)
        a.set_title(f"{g}: single-signal-group loss, degradation % (+ = worse)", fontsize=9)
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig2_single_group_{split}.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for a, base in zip(ax, ("S1_r40", "S2_r40_L7", "S3_k14")):
        for k, (lab, off) in enumerate((("signal loss (O)", 0), ("flag change (H-O)", 1), ("flags removed alone (Z0)", 2))):
            vals = []
            for m in MODELS:
                O = s[(s.scenario == base + "_O") & (s.model == m) & (s.group == "onset_ap")].deg_pct.mean()
                H = s[(s.scenario == base) & (s.model == m) & (s.group == "onset_ap")].deg_pct.mean()
                Z = s[(s.scenario == "Z0") & (s.model == m) & (s.group == "onset_ap")].deg_pct.mean()
                vals.append([O, H - O, Z][k])
            a.bar(np.arange(len(MODELS)) + (k - 1) * 0.27, vals, 0.27, label=lab)
        a.set_xticks(range(len(MODELS)), MODELS, rotation=45, ha="right", fontsize=8)
        a.set_title(f"{base}: onset AP degradation (%)", fontsize=9)
        a.axhline(0, color="k", lw=0.5)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig3_decomposition_{split}.png"), dpi=130)
    plt.close(fig)


def bootstrap_test(root, out):
    """Paired participant-level bootstrap on saved seed-0 / rep-0 test predictions: delta(metric) vs control, thresholds irrelevant."""
    d = np.load("data/processed_v1/test.npz")
    base = np.load(os.path.join(root, "predictions", "test_C0_rep0.npz"))
    pi, ti, pid = base["pi"], base["ti"], base["pid"]
    rows_of = [np.where(pi == i)[0] for i in range(len(pid))]
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.integers(0, len(pid), len(pid)) for _ in range(N_BOOT)]
    fns = {"rmse": lambda y, p: mt.reg_metrics(y, p)["rmse"], "ap": lambda y, p: mt.clf_metrics(y, p)["ap"]}
    res = []
    for fam in MODELS:
        part = next((k.split("__")[1] for k in base.files if k.startswith(fam + "__")), None)
        for task, met in BOOT_TASKS:
            keys = [k for k in base.files if k.startswith(fam + "__") and k.endswith("__" + task)]
            if not keys:
                continue
            key = keys[0]
            y = d["y_" + task][pi, ti]
            sc_preds = {"C0": base[key]}
            for sc in BOOT_SC:
                f = os.path.join(root, "predictions", f"test_{sc}_rep0.npz")
                if os.path.exists(f):
                    sc_preds[sc] = np.load(f)[key]
            idxs = [np.concatenate([rows_of[i] for i in dr]) for dr in draws]
            vals = {sc: [] for sc in sc_preds}
            for ix in idxs:
                yy = y[ix]
                ok = ~np.isnan(yy)
                for sc, p in sc_preds.items():
                    vals[sc].append(fns[met](yy[ok], p[ix][ok]))
            c = np.array(vals["C0"])
            for sc in sc_preds:
                if sc == "C0":
                    continue
                dl = np.array(vals[sc]) - c
                okall = ~np.isnan(y)
                point = fns[met](y[okall], sc_preds[sc][okall]) - fns[met](y[okall], sc_preds["C0"][okall])
                res.append(dict(model=fam, task=task, metric=met, scenario=sc, delta_point=point, ci95_lo=np.nanpercentile(dl, 2.5), ci95_hi=np.nanpercentile(dl, 97.5)))
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(out, "bootstrap_test_seed0.csv"), index=False)
    return r


def main(root):
    out = os.path.join(root, "analysis")
    os.makedirs(out, exist_ok=True)
    allscen = [s["id"] for s in ij.registry()]
    for split in ("cal", "test"):
        df, nfiles = load(root, split)
        x, s = degradation(df)
        s.to_csv(os.path.join(out, f"degradation_per_seed_{split}.csv"), index=False)
        scen_main = [sc for sc in allscen if sc != "C0" and not sc.endswith("_O") and sc != "Z0"]
        L = [f"# Stage 8 missing-data robustness ({split.upper()} split; SYNTHETIC; origins 27,29,...,119; metric files: {nfiles}/59)\n",
             "Cells: mean ±SD over 5 model seeds (gbt: 1 deterministic run) of the % degradation vs the same model's control (+ = worse); each seed's value is the mean over mask replicates and over the tasks in the group. Descriptive only, no ranking or superiority claim.\n"]
        for g in GROUP_ORDER:
            L += [f"\n### {g}\n", table(s, g, scen_main)]
        L += ["\n## Decomposition: physiological signal loss vs informative-flag effects\n", decomposition(s)]
        L += ["\n## Prediction shift vs control (key scenarios)\n", shift_table(x, [sc for sc in BOOT_SC + ["S1_r20", "S1_r40_O"] if sc in set(x.scenario)])]
        open(os.path.join(out, f"summary_{split}.md"), "w", encoding="utf-8").write("\n".join(L))
        figures(s, split, out)
    r = bootstrap_test(root, out)
    print("analysis done ->", out, len(r), "bootstrap rows")


if __name__ == "__main__":
    main(sys.argv[1])
