"""P2-2 locked TEST analysis (SYNTHETIC) implementing results/p2_datasets/TEST_EVALUATION_PLAN.md sections 3-10.
Reuses the Phase A/B rule functions unchanged (analyze_p22, analyze_p22b). Primary replicates = R1-R3 (fresh test); v1 = historical reference (sensitivity only).
  python extension_p22/analyze_test_p22.py    -> results/p2_datasets/test/analysis/"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_p22 as A  # noqa: E402
import analyze_p22b as B  # noqa: E402
import common_p22 as C  # noqa: E402
import eval_test_p22 as E  # noqa: E402

FRESH, SHIFTS, REF = E.FRESH, E.SHIFTS, ["v1"]
ALL = REF + FRESH + SHIFTS
FLAGGED = {"S-EFF"}
YARD = {"primary_R1-R3": FRESH, "sensitivity_v1+R1-R3": REF + FRESH}
OUT = os.path.join(E.TEST_ROOT, "analysis")
LAB = ["C1", "C1_noGBT", "C2", "C3", "C4", "C5"]


def rep_rule3(labels):
    k = sum(x == "supported" for x in labels)
    return "supported" if k == len(labels) else ("mixed" if k == len(labels) - 1 else "not supported")


def shift_effects(P, norm, reps):
    out = []
    for fam, a_, b_ in P[P.dataset == "R1"][["family", "A", "B"]].drop_duplicates().values.tolist():
        sgn = -1 if A.FAMS[fam][0] == "rmse" else 1
        pr = P[(P.family == fam) & (P.A == a_) & (P.B == b_) & (P.disease == "both")].set_index("dataset")
        rr, rn = [pr.loc[d, "mean_diff"] for d in reps], [pr.loc[d, "mean_diff"] / norm[(d, fam)] for d in reps]
        dR1 = pr.loc["R1", ["d_seed0", "d_seed1", "d_seed2"]].values.astype(float)
        for s in SHIFTS:
            e = pr.loc[s, "mean_diff"]
            delta = pr.loc[s, ["d_seed0", "d_seed1", "d_seed2"]].values.astype(float) - dR1
            lab, direc = B.range_label(e, e / norm[(s, fam)], rr, rn, delta, sgn)
            out.append(dict(shift=s, qualified_exploratory=s in FLAGGED, family=fam, A=a_, B=b_, metric=A.FAMS[fam][0], E_shift=e, E_norm_shift=e / norm[(s, fam)],
                            rep_min=min(rr), rep_max=max(rr), rep_norm_min=min(rn), rep_norm_max=max(rn), E_R1=pr.loc["R1", "mean_diff"],
                            delta_mean=float(delta.mean()), delta_sd=float(delta.std(ddof=1)), **{f"delta_seed{i}": float(v) for i, v in enumerate(delta)},
                            shift_verdict_within=pr.loc[s, "verdict"], label=lab, direction=direc))
    return pd.DataFrame(out)


def lag_aug_rank(c4r, c5s, RK, reps):
    LG = []
    for s in SHIFTS:
        hm = {d: max(c4r[d]["hit_level"], c4r[d]["hit_incr15"]) for d in reps + [s]}
        LG.append(dict(shift=s, sign_acc=c4r[s]["sign_acc"], hit_level=c4r[s]["hit_level"], hit_incr15=c4r[s]["hit_incr15"], H_max=hm[s],
                       rep_H_max_min=min(hm[d] for d in reps), rep_H_max_max=max(hm[d] for d in reps), within_range=bool(min(hm[d] for d in reps) <= hm[s] <= max(hm[d] for d in reps))))
    AU = []
    for s in SHIFTS:
        for fam in ("severity", "status", "onset"):
            rep = pd.Series({d: c5s[d][c5s[d].family == fam].panel_rel_change_pct.mean() for d in reps})
            x = c5s[s][c5s[s].family == fam].sort_values("seed").panel_rel_change_pct.values
            r1 = c5s["R1"][c5s["R1"].family == fam].sort_values("seed").panel_rel_change_pct.values
            v = x.mean()
            outside = 0 if rep.min() <= v <= rep.max() else (1 if v > rep.max() else -1)
            d = x - r1
            lab = "within replication noise" if outside == 0 else ("shift-sensitive" if (np.sign(d) == outside).all() and abs(d.mean()) > d.std(ddof=1) else "suggestive")
            AU.append(dict(shift=s, family=fam, rel_change_pct_mean=float(v), seeds=";".join(f"{t:.2f}" for t in x), better_n=int((x < 0).sum()),
                           verdict=A.verdict(x, "rmse")[1], rep_min=float(rep.min()), rep_max=float(rep.max()), delta_vs_R1_mean=float(d.mean()), label=lab,
                           direction="" if outside == 0 else ("augmentation helps more" if outside < 0 else "augmentation helps less")))
    RR = []
    for fam in A.FAMS:
        t = RK[RK.family == fam].pivot(index="dataset", columns="model", values="rank")
        loo = [t.loc[d].corr(t.loc[[x for x in reps if x != d]].mean(), method="spearman") for d in reps]
        for s in SHIFTS:
            rho = t.loc[s].corr(t.loc[reps].mean(), method="spearman")
            RR.append(dict(shift=s, family=fam, spearman_vs_rep_mean=float(rho), rep_loo_min=float(min(loo)), rank_shifted=bool(rho < min(loo)), gru_rank=float(t.loc[s, "gru"])))
    return pd.DataFrame(LG), pd.DataFrame(AU), pd.DataFrame(RR)


def rq_labels(Ef, LG, AU, V, c5r):
    def count(sub, fav):
        return int(((sub.label == "shift-sensitive") & (sub.direction == fav)).sum())

    def kan_rule(k):
        npos, nneg, ninc = count(k, "favours A"), count(k, "favours B"), int((k.label == "inconclusive (scale-dependent)").sum())
        lab = "inconclusive" if ninc >= 3 else ("supported" if npos >= 3 and npos > nneg else ("not supported" if npos == 0 else "mixed"))
        return dict(label=lab, n_plus=npos, n_minus=nneg, n_inconclusive=ninc, n_suggestive=int((k.label == "suggestive").sum()), n=len(k))
    rq = {"RQ3_S-EFF_KAN_gain": dict(**kan_rule(Ef[(Ef["shift"] == "S-EFF") & Ef.B.isin(["mlp_b2_reg", "mlp_m1"])]), qualified_exploratory=True)}
    for q, s in (("RQ4a_S-AR_lag", "S-AR"), ("RQ4b_S-EFF_lag", "S-EFF")):
        r = LG.set_index("shift").loc[s]
        rq[q] = dict(label=("supported" if r.H_max > 0.5 else "mixed") if r.H_max > r.rep_H_max_max else "not supported", H_max=float(r.H_max),
                     rep_H_max_max=float(r.rep_H_max_max), sign_acc=float(r.sign_acc), qualified_exploratory=s in FLAGGED)
    sev = AU[(AU["shift"] == "S-MISS") & (AU.family == "severity")].iloc[0]
    rq["RQ5_S-MISS_aug_severity"] = dict(label="supported" if (sev.verdict == "consistent" and sev.better_n == 3) else ("mixed" if sev.better_n == 2 or sev.verdict == "leaning" and sev.better_n == 3 else "not supported"),
                                         rel_change_pct=float(sev.rel_change_pct_mean), better_n=int(sev.better_n), magnitude_label=sev.label, magnitude_direction=sev.direction,
                                         C5_label_S_MISS=c5r["S-MISS"]["label"])
    rq["RQ6a_S-HET_M1"] = kan_rule(Ef[(Ef["shift"] == "S-HET") & (Ef.A == "m1") & (Ef.B == "gru")])
    return rq


def main():
    os.chdir(C.ROOT)
    E.verify_lock()
    assert os.path.exists(os.path.join(E.TEST_ROOT, "test_access_log.json")), "test evaluation not completed"
    if os.path.isdir(OUT) and os.listdir(OUT):
        raise SystemExit(f"{OUT} not empty; refusing to overwrite")
    os.makedirs(OUT, exist_ok=True)

    # ---- families, prevalence, event counts
    parts = {ds: A.family_values(ds, "test", pred=E.test_pred_path) for ds in ALL}
    F = pd.concat([parts[d][0] for d in ALL])
    PV = pd.concat([parts[d][1] for d in ALL])
    F.to_csv(os.path.join(OUT, "family_per_seed.csv"), index=False)
    PV.to_csv(os.path.join(OUT, "test_event_counts.csv"), index=False)
    ev = []
    for ds in ALL:
        p = PV[PV.dataset == ds]
        ms, mc = int(p[p.family == "onset_sust"].n_pos_participants.min()), int(p[p.family == "onset_cross"].n_pos_participants.min())
        ev.append(dict(dataset=ds, min_sust_onset_participants=ms, min_cross_onset_participants=mc, below_minimum_test=ms < 25,
                       status_now_prevalence=float(p[p.family == "status_now"].prevalence.mean()), onset_sust_prevalence=float(p[p.family == "onset_sust"].prevalence.mean())))
    EV = pd.DataFrame(ev)
    EV.to_csv(os.path.join(OUT, "test_event_flags.csv"), index=False)
    summ = F.groupby(["dataset", "family", "disease", "metric", "model"]).value.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, min="min", max="max", n="count").reset_index()
    fprev = PV.groupby(["dataset", "family"]).prevalence.mean().rename("prevalence").reset_index()
    summ = summ.merge(fprev, on=["dataset", "family"], how="left")
    summ["lift"] = np.where(summ.metric == "ap", summ["mean"] / summ.prevalence, np.nan)
    summ.to_csv(os.path.join(OUT, "family_summary.csv"), index=False)
    norm = {}
    for ds in ALL:
        tsd = B.target_sd(ds, "test")
        for fam in A.FAMS:
            norm[(ds, fam)] = tsd[fam] if A.FAMS[fam][0] == "rmse" else float(fprev[(fprev.dataset == ds) & (fprev.family == fam)].prevalence.iloc[0])
    json.dump({f"{k[0]}|{k[1]}": v for k, v in norm.items()}, open(os.path.join(OUT, "normalisers.json"), "w"), indent=1)

    # ---- paired comparisons and ranks
    P, ranks = [], []
    for ds in ALL:
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
    P.to_csv(os.path.join(OUT, "paired.csv"), index=False)
    RK.to_csv(os.path.join(OUT, "ranks.csv"), index=False)

    # ---- per-dataset conclusions
    c4r, c5r, c5s, V, by4 = {}, {}, {}, [], []
    for ds in ALL:
        c4r[ds], by = A.c4(ds, path=os.path.join(E.TEST_ROOT, ds, "lag_recovery", "metrics.csv"))
        c5r[ds], c5s[ds] = A.c5(ds, path=os.path.join(E.TEST_ROOT, ds, "augmentation", "test_metrics.csv"))
        by4.append(by)
        la, ln = B.labels(P, RK, ds), B.labels(P, RK, ds, no_gbt=True)
        role = "historical reference (test previously inspected)" if ds in REF else ("fresh replicate" if ds in FRESH else "shift")
        V.append(dict(dataset=ds, role=role, qualified_exploratory=ds in FLAGGED, below_minimum_test=bool(EV.set_index("dataset").loc[ds, "below_minimum_test"]), **la,
                      C1_noGBT=ln["C1"], C1_noGBT_beaten=ln["C1_beaten"], C4=c4r[ds]["label"], C5=c5r[ds]["label"]))
    V = pd.DataFrame(V)
    V.to_csv(os.path.join(OUT, "conclusions_per_dataset.csv"), index=False)
    pd.DataFrame([c4r[d] for d in ALL]).to_csv(os.path.join(OUT, "c4_lag_summary.csv"), index=False)
    pd.concat(by4).to_csv(os.path.join(OUT, "c4_lag_by_model.csv"), index=False)
    pd.DataFrame([c5r[d] for d in ALL]).to_csv(os.path.join(OUT, "c5_aug_summary.csv"), index=False)
    pd.concat(c5s.values()).to_csv(os.path.join(OUT, "c5_aug_per_seed.csv"), index=False)

    # ---- replication conclusions
    Vi = V.set_index("dataset")
    repl = {c: dict(per_dataset=Vi[c].to_dict(), primary_R1_R3=rep_rule3([Vi.loc[d, c] for d in FRESH]), sensitivity_v1_R1_R3=A.overall([Vi.loc[d, c] for d in REF + FRESH])) for c in LAB}
    W = {name: {fam: A.kendall_w(RK[(RK.family == fam) & RK.dataset.isin(ds_)].pivot(index="dataset", columns="model", values="rank").values) for fam in A.FAMS}
         for name, ds_ in YARD.items()}
    key = []
    for fam in A.FAMS:
        for a_, b_ in [("m1", "gru"), ("m1", "mlp_m1"), ("kan_reg", "mlp_b2_reg")]:
            sub = P[(P.family == fam) & (P.A == a_) & (P.B == b_) & (P.disease == "both")].set_index("dataset").mean_diff
            if len(sub):
                key.append(dict(family=fam, A=a_, B=b_, fresh_mean=float(sub[FRESH].mean()), fresh_sd=float(sub[FRESH].std(ddof=1)), fresh_min=float(sub[FRESH].min()),
                                fresh_max=float(sub[FRESH].max()), **{d: float(sub[d]) for d in ALL}))
    pd.DataFrame(key).to_csv(os.path.join(OUT, "between_dataset_key_diffs.csv"), index=False)

    # ---- shift effects, lag/aug/rank vs replicate range, RQs (primary and sensitivity yardsticks)
    shift_out = {}
    for name, reps in YARD.items():
        Ef = shift_effects(P, norm, reps)
        LG, AU, RR = lag_aug_rank(c4r, c5s, RK, reps)
        for t, df in (("shift_effects", Ef), ("shift_lag", LG), ("shift_aug", AU), ("rank_robustness", RR)):
            df.to_csv(os.path.join(OUT, f"{t}__{name}.csv"), index=False)
        shift_out[name] = dict(rq=rq_labels(Ef, LG, AU, V, c5r), label_counts=Ef.groupby(["shift", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
                               rank_shifted=RR[RR.rank_shifted][["shift", "family"]].values.tolist())
    Vs = V[V.dataset.isin(SHIFTS)]
    rob = {}
    for c in LAB:
        def lab_(sub):
            n = int((sub[c] == "supported").sum())
            return n, ("supported" if n == len(sub) else ("mixed" if n >= len(sub) - 2 else "not supported"))
        n1, l1 = lab_(Vs)
        n2, l2 = lab_(Vs[~Vs.qualified_exploratory])
        rob[c] = dict(per_shift=Vs.set_index("dataset")[c].to_dict(), n_supported=n1, of=len(Vs), label=l1, excl_S_EFF=dict(n_supported=n2, of=int((~Vs.qualified_exploratory).sum()), label=l2))

    # ---- calibration versus test
    pa, pb = os.path.join(C.RES, "phase_a_cal"), os.path.join(C.RES, "phase_b_cal")
    fs_cal = pd.concat([pd.read_csv(os.path.join(pa, "family_summary.csv")), pd.read_csv(os.path.join(pb, "family_summary.csv"))])
    cm = fs_cal[["dataset", "family", "disease", "model", "mean"]].merge(summ[["dataset", "family", "disease", "model", "mean", "prevalence", "lift"]],
                                                                        on=["dataset", "family", "disease", "model"], suffixes=("_cal", "_test"))
    cm["test_minus_cal"] = cm.mean_test - cm.mean_cal
    cm.to_csv(os.path.join(OUT, "cal_vs_test_family_means.csv"), index=False)
    pc = pd.concat([pd.read_csv(os.path.join(pa, "paired.csv")), pd.read_csv(os.path.join(pb, "paired.csv"))])
    pp = pc[["dataset", "family", "disease", "A", "B", "mean_diff", "verdict", "A_better_n"]].merge(P[["dataset", "family", "disease", "A", "B", "mean_diff", "verdict", "A_better_n"]],
                                                                                                 on=["dataset", "family", "disease", "A", "B"], suffixes=("_cal", "_test"))
    pp["sign_agree"] = np.sign(pp.mean_diff_cal) == np.sign(pp.mean_diff_test)
    cons = (pp.verdict_cal == "consistent")
    pp["cal_consistent_replicated_on_test"] = np.where(cons, (pp.verdict_test == "consistent") & pp.sign_agree, np.nan)
    pp.to_csv(os.path.join(OUT, "cal_vs_test_paired.csv"), index=False)
    b_ = pp[pp.disease == "both"]
    ag = {g: dict(n=int(len(s)), sign_agreement=float(s.sign_agree.mean()), n_cal_consistent=int((s.verdict_cal == "consistent").sum()),
                  cal_consistent_also_consistent_on_test=float(s[s.verdict_cal == "consistent"].sign_agree.where(s[s.verdict_cal == "consistent"].verdict_test == "consistent", False).mean())
                  if (s.verdict_cal == "consistent").any() else None)
          for g, s in list(b_.groupby(b_.dataset.isin(SHIFTS).map({True: "shifts", False: "v1+R1-R3"}))) + [("all", b_)]}
    lc = pd.read_csv(os.path.join(pb, "conclusions_per_dataset.csv")).set_index("dataset")
    lab_cmp = pd.DataFrame([dict(dataset=d, **{f"{c}_cal": lc.loc[d, c] for c in LAB}, **{f"{c}_test": Vi.loc[d, c] for c in LAB}) for d in ALL])
    lab_cmp.to_csv(os.path.join(OUT, "cal_vs_test_labels.csv"), index=False)
    rq_cal = json.load(open(os.path.join(pb, "conclusions_phase_b.json")))["results"]

    res = {"SYNTHETIC": True, "split": "test", "lock_sha256": E.sha(E.LOCK), "v1_role": "historical reference; test previously inspected; sensitivity only",
           "S-EFF": "gate-failing; qualified/exploratory; retained", "stage5_observational_regression": "not performed (not added post hoc)",
           "event_flags": EV.to_dict(orient="records"), "replication": repl, "kendall_w": W, "shift": shift_out, "robustness_across_shifts": rob,
           "cal_vs_test_paired_agreement": ag,
           "cal_rq_labels_v1_included_yardstick": {k: v["label"] for k, v in rq_cal.items() if isinstance(v, dict) and "label" in v}}
    json.dump(res, open(os.path.join(OUT, "conclusions_test.json"), "w"), indent=1, default=str)
    print(V.to_string(index=False))
    print(json.dumps({k: res[k] for k in ("replication", "robustness_across_shifts", "cal_vs_test_paired_agreement")}, indent=1, default=str))
    for n, so in shift_out.items():
        print(n, json.dumps({k: v["label"] for k, v in so["rq"].items()}), so["rank_shifted"])
        print(pd.DataFrame(so["label_counts"]).T.to_string())


if __name__ == "__main__":
    main()
