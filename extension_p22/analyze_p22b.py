"""P2-2 Phase B analysis (SYNTHETIC; CALIBRATION split only) implementing results/p2_datasets/ANALYSIS_PLAN_PHASE_B.md.
Reuses the Phase A functions unchanged (analyze_p22). Replicates = v1, R1, R2, R3; shifts are CRN-paired with R1.
  python extension_p22/analyze_p22b.py      -> results/p2_datasets/phase_b_cal/"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_p22 as A  # noqa: E402
import common_p22 as C  # noqa: E402
import data as dd  # noqa: E402

REPS = ["v1", "R1", "R2", "R3"]
SHIFTS = ["S-EFF", "S-AR", "S-MISS", "S-HET", "S-NOISE"]
FLAGGED = {"S-EFF"}                     # failed the prevalence gate; retained and flagged (approved policy)
OUT = os.path.join(C.RES, "phase_b_cal")


def target_sd(ds, split="cal"):
    lab = dd.load_split(split, C.proc_dir(ds))
    out = {}
    for fam, (met, tpl) in A.FAMS.items():
        if met == "rmse":
            out[fam] = float(np.mean([np.nanstd(np.concatenate([lab["y_" + t.format(k=k)].ravel() for t in tpl])) for k in A.KS]))
    return out


def labels(P, RK, ds, no_gbt=False):
    vs = P[(P.dataset == ds) & (P.B == "gru") & ((P.A != "gbt") if no_gbt else True)]
    beaten = sorted(set(vs[vs.A_consistently_better].family))
    c1 = "supported" if not beaten else ("mixed" if len(beaten) <= 2 else "not supported")
    km = P[(P.dataset == ds) & (P.B.isin(["mlp_b2_reg", "mlp_m1"]))]
    nK, nM = int(km.A_consistently_better.sum()), int(((km.verdict == "consistent") & (km.A_better_n == 0)).sum())
    c2 = "supported" if nK <= 2 else ("mixed" if nK <= 4 else "not supported")
    nm = int(P[(P.dataset == ds) & (P.A == "m1") & (P.B == "gru")].A_consistently_better.sum())
    c3 = "supported" if nm == 0 else ("mixed" if nm <= 2 else "not supported")
    gr = RK[(RK.dataset == ds) & (RK.model == "gru")]
    return dict(C1=c1, C1_beaten=";".join(beaten), C1_gru_rank1=int((gr["rank"] == 1).sum()), C2=c2, C2_nK=nK, C2_nM=nM, C3=c3, C3_nm=nm)


def range_label(e_raw, e_norm, rep_raw, rep_norm, delta, better_sign):
    """better_sign: +1 if larger value favours A (AP), -1 if smaller favours A (RMSE)."""
    def side(v, r):
        return 0 if min(r) <= v <= max(r) else (1 if v > max(r) else -1)
    s1, s2 = side(e_raw, rep_raw), side(e_norm, rep_norm)
    if s1 == 0 and s2 == 0:
        return "within replication noise", ""
    if s1 == 0 or s2 == 0 or s1 != s2:
        return "inconclusive (scale-dependent)", ""
    direction = "favours A" if s1 * better_sign > 0 else "favours B"
    d = np.asarray(delta, float)
    same = (np.sign(d) == s1).all()
    return ("shift-sensitive" if same and abs(d.mean()) > d.std(ddof=1) else "suggestive"), direction


def main():
    os.chdir(C.ROOT)
    os.makedirs(OUT, exist_ok=True)
    pa = os.path.join(C.RES, "phase_a_cal")
    F_rep = pd.read_csv(os.path.join(pa, "family_per_seed.csv"))
    PV_rep = pd.read_csv(os.path.join(pa, "cal_prevalence.csv"))
    parts = [A.family_values(ds) for ds in SHIFTS]
    F = pd.concat([F_rep] + [p[0] for p in parts])
    PV = pd.concat([PV_rep] + [p[1] for p in parts])
    F[F.dataset.isin(SHIFTS)].to_csv(os.path.join(OUT, "family_per_seed.csv"), index=False)
    PV[PV.dataset.isin(SHIFTS)].to_csv(os.path.join(OUT, "cal_prevalence.csv"), index=False)
    summ = F.groupby(["dataset", "family", "disease", "metric", "model"]).value.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n="count").reset_index()
    fprev = PV.groupby(["dataset", "family"]).prevalence.mean().rename("prevalence").reset_index()
    summ = summ.merge(fprev, on=["dataset", "family"], how="left")
    summ["lift"] = np.where(summ.metric == "ap", summ["mean"] / summ.prevalence, np.nan)
    summ[summ.dataset.isin(SHIFTS)].to_csv(os.path.join(OUT, "family_summary.csv"), index=False)
    TSD = {ds: target_sd(ds) for ds in REPS + SHIFTS}
    norm = {(ds, fam): (TSD[ds][fam] if A.FAMS[fam][0] == "rmse" else float(fprev[(fprev.dataset == ds) & (fprev.family == fam)].prevalence.iloc[0]))
            for ds in REPS + SHIFTS for fam in A.FAMS}
    json.dump({f"{k[0]}|{k[1]}": v for k, v in norm.items()}, open(os.path.join(OUT, "normalisers.json"), "w"), indent=1)

    P, ranks = [], []
    for ds in REPS + SHIFTS:
        for fam in A.FAMS:
            for m in A.models_for(fam):
                if m != "gru":
                    P.append(A.paired(F, ds, fam, m, "gru"))
            if fam not in A.FC:
                P.append(A.paired(F, ds, fam, "kan_reg", "mlp_b2_reg"))
            P.append(A.paired(F, ds, fam, "m1", "mlp_m1"))
            s = summ[(summ.dataset == ds) & (summ.family == fam) & (summ.disease == "both")].set_index("model")["mean"]
            r = s.rank() if A.FAMS[fam][0] == "rmse" else (-s).rank()
            ranks += [dict(dataset=ds, family=fam, model=m_, rank=float(v)) for m_, v in r.items()]
    P, RK = pd.DataFrame(P), pd.DataFrame(ranks)
    P[P.dataset.isin(SHIFTS)].to_csv(os.path.join(OUT, "paired.csv"), index=False)
    RK[RK.dataset.isin(SHIFTS)].to_csv(os.path.join(OUT, "ranks.csv"), index=False)

    # ---------------- per-dataset C1-C5 (shifts) + C1 no-GBT sensitivity (all datasets)
    c4r, c4by, c5r, c5s, V = {}, [], {}, [], []
    for ds in REPS + SHIFTS:
        c4r[ds], by = A.c4(ds)
        c5r[ds], rs = A.c5(ds)
        if ds in SHIFTS:
            c4by.append(by)
            c5s.append(rs)
        lab = labels(P, RK, ds)
        lab_ng = labels(P, RK, ds, no_gbt=True)
        V.append(dict(dataset=ds, family_type="replication" if ds in REPS else "shift", gate_flagged=ds in FLAGGED, **lab,
                      C1_noGBT=lab_ng["C1"], C1_noGBT_beaten=lab_ng["C1_beaten"], C4=c4r[ds]["label"], C5=c5r[ds]["label"]))
    V = pd.DataFrame(V)
    V.to_csv(os.path.join(OUT, "conclusions_per_dataset.csv"), index=False)
    pd.DataFrame([c4r[d] for d in REPS + SHIFTS]).to_csv(os.path.join(OUT, "c4_lag_summary.csv"), index=False)
    pd.concat(c4by).to_csv(os.path.join(OUT, "c4_lag_by_model.csv"), index=False)
    pd.DataFrame([c5r[d] for d in REPS + SHIFTS]).to_csv(os.path.join(OUT, "c5_aug_summary.csv"), index=False)
    pd.concat(c5s).to_csv(os.path.join(OUT, "c5_aug_per_seed.csv"), index=False)

    # ---------------- shift effects vs replicate range (primary)
    E = []
    keys = P[P.dataset == "R1"][["family", "A", "B"]].drop_duplicates().values.tolist()
    for fam, a_, b_ in keys:
        sgn = -1 if A.FAMS[fam][0] == "rmse" else 1
        pr = P[(P.family == fam) & (P.A == a_) & (P.B == b_) & (P.disease == "both")].set_index("dataset")
        rep_raw = [pr.loc[d, "mean_diff"] for d in REPS]
        rep_norm = [pr.loc[d, "mean_diff"] / norm[(d, fam)] for d in REPS]
        dR1 = pr.loc["R1", ["d_seed0", "d_seed1", "d_seed2"]].values.astype(float)
        for s in SHIFTS:
            e = pr.loc[s, "mean_diff"]
            delta = pr.loc[s, ["d_seed0", "d_seed1", "d_seed2"]].values.astype(float) - dR1
            lab, direc = range_label(e, e / norm[(s, fam)], rep_raw, rep_norm, delta, sgn)
            E.append(dict(shift=s, gate_flagged=s in FLAGGED, family=fam, A=a_, B=b_, metric=A.FAMS[fam][0], E_shift=e, E_norm_shift=e / norm[(s, fam)],
                          rep_min=min(rep_raw), rep_max=max(rep_raw), rep_norm_min=min(rep_norm), rep_norm_max=max(rep_norm), E_R1=pr.loc["R1", "mean_diff"],
                          delta_mean=float(delta.mean()), delta_sd=float(delta.std(ddof=1)), **{f"delta_seed{i}": float(v) for i, v in enumerate(delta)},
                          shift_verdict_within=pr.loc[s, "verdict"], label=lab, direction=direc))
    E = pd.DataFrame(E)
    E.to_csv(os.path.join(OUT, "shift_effects.csv"), index=False)

    # lag and augmentation vs replicate range
    LG = []
    for s in SHIFTS:
        hm = {d: max(c4r[d]["hit_level"], c4r[d]["hit_incr15"]) for d in REPS + [s]}
        LG.append(dict(shift=s, sign_acc=c4r[s]["sign_acc"], hit_level=c4r[s]["hit_level"], hit_incr15=c4r[s]["hit_incr15"], H_max=hm[s],
                       rep_H_max_min=min(hm[d] for d in REPS), rep_H_max_max=max(hm[d] for d in REPS), rep_sign_min=min(c4r[d]["sign_acc"] for d in REPS),
                       within_range=bool(min(hm[d] for d in REPS) <= hm[s] <= max(hm[d] for d in REPS))))
    LG = pd.DataFrame(LG)
    LG.to_csv(os.path.join(OUT, "shift_lag.csv"), index=False)
    AUGS = pd.concat([A.c5(d)[1] for d in REPS])
    AU = []
    for s in SHIFTS:
        sp = c5s[SHIFTS.index(s)]
        for fam in ("severity", "status", "onset"):
            rep = AUGS[AUGS.family == fam].groupby("dataset").panel_rel_change_pct.mean()
            x = sp[sp.family == fam].sort_values("seed").panel_rel_change_pct.values
            r1 = AUGS[(AUGS.dataset == "R1") & (AUGS.family == fam)].sort_values("seed").panel_rel_change_pct.values
            v = x.mean()
            outside = 0 if rep.min() <= v <= rep.max() else (1 if v > rep.max() else -1)
            d = x - r1
            lab = "within replication noise" if outside == 0 else ("shift-sensitive" if (np.sign(d) == outside).all() and abs(d.mean()) > d.std(ddof=1) else "suggestive")
            AU.append(dict(shift=s, family=fam, rel_change_pct_mean=float(v), seeds=";".join(f"{t:.2f}" for t in x), better_n=int((x < 0).sum()),
                           verdict=A.verdict(x, "rmse")[1], rep_min=float(rep.min()), rep_max=float(rep.max()), delta_vs_R1_mean=float(d.mean()), label=lab,
                           direction="" if outside == 0 else ("augmentation helps more" if outside < 0 else "augmentation helps less")))
    AU = pd.DataFrame(AU)
    AU.to_csv(os.path.join(OUT, "shift_aug.csv"), index=False)

    # rank robustness (Spearman vs replicate mean rank; LOO reference)
    RR = []
    for fam in A.FAMS:
        t = RK[RK.family == fam].pivot(index="dataset", columns="model", values="rank")
        loo = [t.loc[d].corr(t.loc[[x for x in REPS if x != d]].mean(), method="spearman") for d in REPS]
        for s in SHIFTS:
            rho = t.loc[s].corr(t.loc[REPS].mean(), method="spearman")
            RR.append(dict(shift=s, family=fam, spearman_vs_rep_mean=float(rho), rep_loo_min=float(min(loo)), rep_loo_max=float(max(loo)), rank_shifted=bool(rho < min(loo)),
                           gru_rank=float(t.loc[s, "gru"]), shift_order=";".join(t.loc[s].sort_values().index)))
    RR = pd.DataFrame(RR)
    RR.to_csv(os.path.join(OUT, "rank_robustness.csv"), index=False)

    # ---------------- research-question labels (pre-registered)
    def count(sub, fav):
        return int(((sub.label == "shift-sensitive") & (sub.direction == fav)).sum())
    rq = {}
    k = E[(E["shift"] == "S-EFF") & E.B.isin(["mlp_b2_reg", "mlp_m1"])]
    npos, nneg, ninc = count(k, "favours A"), count(k, "favours B"), int((k.label == "inconclusive (scale-dependent)").sum())
    rq["RQ3_S-EFF_KAN_gain"] = dict(label="inconclusive" if ninc >= 3 else ("supported" if npos >= 3 and npos > nneg else ("not supported" if npos == 0 else "mixed")),
                                    n_plus=npos, n_minus=nneg, n_inconclusive=ninc, n_suggestive=int((k.label == "suggestive").sum()), n=len(k), gate_flagged=True)
    for q, s in (("RQ4a_S-AR_lag", "S-AR"), ("RQ4b_S-EFF_lag", "S-EFF")):
        r = LG.set_index("shift").loc[s]
        lab = ("supported" if r.H_max > 0.5 else "mixed") if r.H_max > r.rep_H_max_max else "not supported"
        rq[q] = dict(label=lab, H_max=float(r.H_max), rep_H_max_max=float(r.rep_H_max_max), sign_acc=float(r.sign_acc), gate_flagged=s in FLAGGED)
    sev = AU[(AU["shift"] == "S-MISS") & (AU.family == "severity")].iloc[0]
    rq["RQ5_S-MISS_aug_severity"] = dict(label="supported" if (sev.verdict == "consistent" and sev.better_n == 3) else ("mixed" if sev.better_n == 2 or sev.verdict == "leaning" and sev.better_n == 3 else "not supported"),
                                         rel_change_pct=float(sev.rel_change_pct_mean), better_n=int(sev.better_n), magnitude_label=sev.label, magnitude_direction=sev.direction,
                                         C5_label_S_MISS=c5r["S-MISS"]["label"])
    k = E[(E["shift"] == "S-HET") & (E.A == "m1") & (E.B == "gru")]
    npos, nneg, ninc = count(k, "favours A"), count(k, "favours B"), int((k.label == "inconclusive (scale-dependent)").sum())
    rq["RQ6a_S-HET_M1"] = dict(label="inconclusive" if ninc >= 3 else ("supported" if npos >= 3 and npos > nneg else ("not supported" if npos == 0 else "mixed")),
                               n_plus=npos, n_minus=nneg, n_inconclusive=ninc, n_suggestive=int((k.label == "suggestive").sum()), n=len(k))
    rob = {}
    Vs = V[V.dataset.isin(SHIFTS)]
    for c in ("C1", "C1_noGBT", "C2", "C3", "C4", "C5"):
        def lab_(sub):
            n = int((sub[c] == "supported").sum())
            return n, ("supported" if n == len(sub) else ("mixed" if n >= len(sub) - 2 else "not supported"))
        n_all, l_all = lab_(Vs)
        n_ex, l_ex = lab_(Vs[~Vs.gate_flagged])
        rep = V[V.dataset.isin(REPS)][c]
        rob[c] = dict(per_shift=Vs.set_index("dataset")[c].to_dict(), n_supported=n_all, of=len(Vs), label=l_all,
                      sensitivity_excl_gate_flagged=dict(n_supported=n_ex, of=int((~Vs.gate_flagged).sum()), label=l_ex),
                      replication=dict(per_dataset=V[V.dataset.isin(REPS)].set_index("dataset")[c].to_dict(), overall=A.overall(rep.tolist())))
    rq["robustness_of_replication_conclusions"] = rob
    rq["rank_shifted_families"] = RR[RR.rank_shifted][["shift", "family"]].values.tolist()
    json.dump({"SYNTHETIC": True, "split": "cal", "test_sealed": True, "plan_sha256": open(os.path.join(C.RES, "plan_phase_b_sha256.txt")).read().split()[0], "results": rq},
              open(os.path.join(OUT, "conclusions_phase_b.json"), "w"), indent=1, default=str)
    print(V.to_string(index=False))
    print(E.groupby(["shift", "label"]).size().unstack(fill_value=0).to_string())
    print(json.dumps(rq, indent=1, default=str))


if __name__ == "__main__":
    main()
