"""P2-2 Phase A analysis (SYNTHETIC; CALIBRATION split only) implementing results/p2_datasets/ANALYSIS_PLAN_PHASE_A.md.
  python extension_p22/analyze_p22.py --datasets v1 R1 R2 R3 --out results/p2_datasets/phase_a_cal"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p22 as C  # noqa: E402
import data as dd  # noqa: E402
import metrics as mt  # noqa: E402

KS = ("dep", "anx")
FAMS = {"sev_now": ("rmse", ["sev_{k}_t0"]), "sev_fc": ("rmse", ["sev_{k}_h7", "sev_{k}_h14", "sev_{k}_h21"]),
        "status_now": ("ap", ["status_{k}_t0"]), "status_fc": ("ap", ["status_{k}_h7", "status_{k}_h14", "status_{k}_h21"]),
        "onset_sust": ("ap", ["onset_sust_{k}_h7", "onset_sust_{k}_h14", "onset_sust_{k}_h21"]),
        "onset_cross": ("ap", ["onset_cross_{k}_h7", "onset_cross_{k}_h14", "onset_cross_{k}_h21"])}
FC = ("sev_fc", "status_fc")
MAIN_MODELS = ["gbt", "gru", "kan_reg", "mlp_b2_reg", "m1", "mlp_m1"]
FC_MODELS = ["gbt", "gru", "m1", "mlp_m1"]
AUG_PANEL = ["S1_r40", "S2_r40_L7", "S3_k3", "S3_k7", "S3_k14", "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S5_rand40", "S5_blk40"]


def taskset_of(model, fam):
    if model in ("m1", "mlp_m1"):
        return "all"
    return "forecast" if fam in FC else "main"


def models_for(fam):
    return FC_MODELS if fam in FC else MAIN_MODELS


def family_values(ds, split="cal", pred=None):
    """split/pred default to the calibration split and saved calibration predictions (Phase A/B behaviour unchanged)."""
    lab = dd.load_split(split, C.proc_dir(ds))
    rows, prev = [], []
    for fam, (met, tpl) in FAMS.items():
        for k in KS:
            for t in tpl:
                y = lab["y_" + t.format(k=k)]
                ok = ~np.isnan(y)
                if met == "ap":
                    prev.append(dict(dataset=ds, family=fam, disease=k, head=t.format(k=k), prevalence=float(y[ok].mean()), n_pos=int(y[ok].sum()),
                                     n_pos_participants=int((np.nansum(y, 1) > 0).sum())))
        for model in models_for(fam):
            ts = taskset_of(model, fam)
            for seed in ([0] if model == "gbt" else C.SEEDS):
                z = np.load(os.path.join(C.ROOT, (pred or C.pred_path)(ds, model, ts, seed, split)))
                heads = list(z["heads"])
                vals = {}
                for k in KS:
                    v = []
                    for t in tpl:
                        h = t.format(k=k)
                        y = lab["y_" + h]
                        ok = ~np.isnan(y)
                        p = z["pred"][..., heads.index(h)][ok]
                        v.append(mt.reg_metrics(y[ok], p)["rmse"] if met == "rmse" else mt.clf_metrics(y[ok], p)["ap"])
                    vals[k] = float(np.mean(v))
                vals["both"] = float(np.mean([vals["dep"], vals["anx"]]))
                for k, v in vals.items():
                    rows.append(dict(dataset=ds, family=fam, disease=k, metric=met, model=model, seed=seed, value=v))
    return pd.DataFrame(rows), pd.DataFrame(prev)


def verdict(d, metric):
    """d: per-seed differences A - B. Returns (A_better_n, label)."""
    d = np.asarray(d, float)
    better = int((d < 0).sum()) if metric == "rmse" else int((d > 0).sum())
    worse = len(d) - better - int((d == 0).sum())
    same = better == len(d) or worse == len(d)
    if same and abs(d.mean()) > d.std(ddof=1):
        lab = "consistent"
    elif same:
        lab = "leaning"
    else:
        lab = "not distinguishable"
    return better, lab


def paired(F, ds, fam, A, B, disease="both"):
    f = F[(F.dataset == ds) & (F.family == fam) & (F.disease == disease)]
    a = f[f.model == A].set_index("seed").value
    b = f[f.model == B].set_index("seed").value
    if A == "gbt":
        a = pd.Series(a.iloc[0], index=b.index)
    if B == "gbt":
        b = pd.Series(b.iloc[0], index=a.index)
    d = (a - b).dropna().sort_index()
    met = FAMS[fam][0]
    n_better, lab = verdict(d.values, met)
    return dict(dataset=ds, family=fam, disease=disease, A=A, B=B, metric=met, mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)), min_diff=float(d.min()),
                max_diff=float(d.max()), A_better_n=n_better, n=len(d), verdict=lab, A_consistently_better=bool(lab == "consistent" and n_better == len(d)),
                **{f"d_seed{s}": float(v) for s, v in d.items()})


def c4(ds, path=None):
    M = pd.read_csv(path or os.path.join(C.ROOT, C.RES, ds, "lag_recovery", "metrics.csv"))
    S = float(M.sign_ok.mean())
    Hl = float((M.peak_lag_level == M.planted_lag).mean())
    Hi = float((M.peak_lag_increment_1to5 == M.planted_lag).mean())
    lab = "supported" if (S >= 0.90 and max(Hl, Hi) <= 0.50) else ("not supported" if S - max(Hl, Hi) <= 0.10 else "mixed")
    by = M.assign(hit_lvl=M.peak_lag_level == M.planted_lag, hit_inc=M.peak_lag_increment_1to5 == M.planted_lag).groupby("model")[["sign_ok", "hit_lvl", "hit_inc", "prelag_mass", "r_with_truth", "magnitude_vs_truth"]].mean()
    return dict(dataset=ds, sign_acc=S, hit_level=Hl, hit_incr15=Hi, constant_guess=0.5, n_cases=len(M), label=lab), by.reset_index().assign(dataset=ds)


def c5(ds, path=None):
    G = pd.read_csv(path or os.path.join(C.ROOT, C.RES, ds, "augmentation", "cal_metrics.csv"))
    fams = {"severity": (["sev_dep_t0", "sev_anx_t0"], "rmse"), "status": (["status_dep_t0", "status_anx_t0"], "ap"),
            "onset": ([f"onset_sust_{k}_h{h}" for k in KS for h in (7, 14, 21)], "ap")}
    rows = []
    for fname, (tasks, met) in fams.items():
        g = G[G.task.isin(tasks) & (G.metric == met)].groupby(["model", "seed", "scenario"]).value.mean().unstack("model")   # mean over tasks and reps
        rel = (g["gru_aug"] - g["gru"]) / g["gru"] * 100 if met == "rmse" else (g["gru"] - g["gru_aug"]) / g["gru"] * 100       # negative = augmentation better
        rel = rel.reset_index().rename(columns={0: "rel"})
        for seed, sub in rel.groupby("seed"):
            rows.append(dict(dataset=ds, family=fname, seed=seed, panel_rel_change_pct=float(sub[sub.scenario.isin(AUG_PANEL)].rel.mean()),
                             clean_C0_rel_change_pct=float(sub[sub.scenario == "C0"].rel.mean())))
    R = pd.DataFrame(rows)
    out = {}
    for fname, sub in R.groupby("family"):
        d = sub.sort_values("seed").panel_rel_change_pct.values
        n_better, lab = verdict(d, "rmse")                       # negative = better for every family (sign-normalised)
        out[fname] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), better_n=n_better, verdict=lab, consistent_improve=bool(lab == "consistent" and n_better == len(d)))
    sev, st, on = out["severity"], out["status"], out["onset"]
    if sev["consistent_improve"] and not st["consistent_improve"] and not on["consistent_improve"]:
        lab = "supported"
    elif sev["better_n"] <= 1 or ((st["consistent_improve"] or on["consistent_improve"]) and not sev["consistent_improve"]):
        lab = "not supported"
    else:
        lab = "mixed"
    return dict(dataset=ds, label=lab, **{f"{f}_{k}": v for f, o in out.items() for k, v in o.items()}), R


def kendall_w(rank_table):
    """rank_table: rows = datasets, cols = models (ranks). Kendall's W without tie correction."""
    m, n = rank_table.shape
    Rj = rank_table.sum(0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def overall(labels):
    k = sum(x == "supported" for x in labels)
    return "supported" if k == len(labels) else ("not supported" if k <= 1 else "mixed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["v1", "R1", "R2", "R3"])
    ap.add_argument("--out", default=os.path.join(C.RES, "phase_a_cal"))
    a = ap.parse_args()
    os.chdir(C.ROOT)
    os.makedirs(a.out, exist_ok=True)
    parts = [family_values(ds) for ds in a.datasets]
    F = pd.concat([p[0] for p in parts])
    PV = pd.concat([p[1] for p in parts])
    F.to_csv(os.path.join(a.out, "family_per_seed.csv"), index=False)
    PV.to_csv(os.path.join(a.out, "cal_prevalence.csv"), index=False)
    summ = F.groupby(["dataset", "family", "disease", "metric", "model"]).value.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, min="min", max="max", n="count").reset_index()
    fprev = PV.groupby(["dataset", "family"]).prevalence.mean().rename("prevalence").reset_index()
    summ = summ.merge(fprev, on=["dataset", "family"], how="left")
    summ["lift"] = np.where(summ.metric == "ap", summ["mean"] / summ.prevalence, np.nan)
    summ.to_csv(os.path.join(a.out, "family_summary.csv"), index=False)

    P, verd, ranks = [], [], []
    for ds in a.datasets:
        for fam in FAMS:
            for A in models_for(fam):
                if A != "gru":
                    P.append(paired(F, ds, fam, A, "gru"))
            if fam not in FC:
                P.append(paired(F, ds, fam, "kan_reg", "mlp_b2_reg"))
            P.append(paired(F, ds, fam, "m1", "mlp_m1"))
            s = summ[(summ.dataset == ds) & (summ.family == fam) & (summ.disease == "both")].set_index("model")["mean"]
            r = (s.rank() if FAMS[fam][0] == "rmse" else (-s).rank())
            ranks += [dict(dataset=ds, family=fam, model=m_, rank=float(v)) for m_, v in r.items()]
    P = pd.DataFrame(P)
    P.to_csv(os.path.join(a.out, "paired.csv"), index=False)
    RK = pd.DataFrame(ranks)
    RK.to_csv(os.path.join(a.out, "ranks.csv"), index=False)

    lag_rows, lag_by, aug_rows, aug_seed = [], [], [], []
    for ds in a.datasets:
        vs = P[(P.dataset == ds) & (P.B == "gru")]
        beaten = sorted(set(vs[vs.A_consistently_better].family))
        c1 = "supported" if not beaten else ("mixed" if len(beaten) <= 2 else "not supported")
        gr = RK[(RK.dataset == ds) & (RK.model == "gru")]
        km = P[(P.dataset == ds) & (P.B.isin(["mlp_b2_reg", "mlp_m1"]))]
        nK = int(km.A_consistently_better.sum())
        nM = int(((km.verdict == "consistent") & (km.A_better_n == 0)).sum())
        c2 = "supported" if nK <= 2 else ("mixed" if nK <= 4 else "not supported")
        mg = P[(P.dataset == ds) & (P.A == "m1") & (P.B == "gru")]
        nm = int(mg.A_consistently_better.sum())
        c3 = "supported" if nm == 0 else ("mixed" if nm <= 2 else "not supported")
        r4, by = c4(ds)
        r5, rs = c5(ds)
        lag_rows.append(r4)
        lag_by.append(by)
        aug_rows.append(r5)
        aug_seed.append(rs)
        verd.append(dict(dataset=ds, C1=c1, C1_families_where_gru_beaten=";".join(beaten), C1_gru_rank1_families=int((gr["rank"] == 1).sum()),
                         C2=c2, C2_kan_consistently_better=nK, C2_mlp_consistently_better=nM, C3=c3, C3_m1_consistently_better=nm,
                         C4=r4["label"], C5=r5["label"]))
    V = pd.DataFrame(verd)
    V.to_csv(os.path.join(a.out, "conclusions_per_dataset.csv"), index=False)
    pd.DataFrame(lag_rows).to_csv(os.path.join(a.out, "c4_lag_summary.csv"), index=False)
    pd.concat(lag_by).to_csv(os.path.join(a.out, "c4_lag_by_model.csv"), index=False)
    pd.DataFrame(aug_rows).to_csv(os.path.join(a.out, "c5_aug_summary.csv"), index=False)
    pd.concat(aug_seed).to_csv(os.path.join(a.out, "c5_aug_per_seed.csv"), index=False)
    agg = {c: dict(per_dataset=V.set_index("dataset")[c].to_dict(), overall=overall(V[c].tolist())) for c in ("C1", "C2", "C3", "C4", "C5")}
    W = {}
    for fam in FAMS:
        t = RK[RK.family == fam].pivot(index="dataset", columns="model", values="rank")
        W[fam] = kendall_w(t.values)
    key = []
    for fam in FAMS:
        for A, B in [("m1", "gru"), ("m1", "mlp_m1"), ("kan_reg", "mlp_b2_reg")]:
            sub = P[(P.family == fam) & (P.A == A) & (P.B == B)]
            if len(sub):
                key.append(dict(family=fam, A=A, B=B, between_dataset_mean=float(sub.mean_diff.mean()), between_dataset_sd=float(sub.mean_diff.std(ddof=1)),
                                min=float(sub.mean_diff.min()), max=float(sub.mean_diff.max()), **{f"{d}": float(v) for d, v in sub.set_index("dataset").mean_diff.items()}))
        best = []
        for ds in a.datasets:
            s = summ[(summ.dataset == ds) & (summ.family == fam) & (summ.disease == "both") & (summ.model != "gru")].set_index("model")["mean"]
            bm = s.idxmin() if FAMS[fam][0] == "rmse" else s.idxmax()
            g = float(summ[(summ.dataset == ds) & (summ.family == fam) & (summ.disease == "both") & (summ.model == "gru")]["mean"].iloc[0])
            best.append(dict(dataset=ds, best_other=bm, diff=float(s[bm] - g)))
        b = pd.DataFrame(best)
        key.append(dict(family=fam, A="best_other", B="gru", between_dataset_mean=float(b["diff"].mean()), between_dataset_sd=float(b["diff"].std(ddof=1)), min=float(b["diff"].min()),
                        max=float(b["diff"].max()), **{r.dataset: float(r["diff"]) for _, r in b.iterrows()}, best_other_models=";".join(b.best_other)))
    pd.DataFrame(key).to_csv(os.path.join(a.out, "between_dataset_key_diffs.csv"), index=False)
    json.dump({"SYNTHETIC": True, "split": "cal", "test_sealed": True, "conclusions": agg, "kendall_w": W}, open(os.path.join(a.out, "conclusions.json"), "w"), indent=1)
    print(V.to_string(index=False))
    print(json.dumps(agg, indent=1))
    print("Kendall W", {k: round(v, 3) for k, v in W.items()})


if __name__ == "__main__":
    main()
