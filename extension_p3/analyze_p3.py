"""P3 calibration analysis (SYNTHETIC; CALIBRATION split only) implementing P3_PLAN_LOCKED.md s5-s8 + P3_PLAN_AMENDMENT_1.md.
  python extension_p3/analyze_p3.py [--root runs] [--out analysis_cal]
Reads per-job metrics.json (calibration, full 200-participant set), builds family values, paired verdicts across draws (n<=400) or seeds (n=600),
reproducibility across R4 and R5, RQ1a / RQ1b labels at n=50 and n=100 independently, secondary descriptives, sensitivities S1-S3 (R4)."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p3 as P  # noqa: E402
import analyze_p22 as A  # noqa: E402

PRIMARY = ("sev_now", "sev_fc", "status_now", "status_fc")
RQ1A = [("kan_reg", "mlp_b2_reg", f) for f in ("sev_now", "status_now", "onset_sust", "onset_cross")] + \
       [("m1", "mlp_m1", f) for f in ("sev_now", "sev_fc", "status_now", "status_fc", "onset_sust", "onset_cross")]
RQ1B = {"kan_reg": ("sev_now", "status_now"), "m1": ("sev_now", "sev_fc", "status_now", "status_fc")}
GAP = "unclassified (rule gap)"


def families(metrics):
    out = {}
    for fam, (met, tpl) in A.FAMS.items():
        hs = [t.format(k=k) for k in A.KS for t in tpl]
        if all(h in metrics for h in hs):
            out[fam] = (met, float(np.mean([metrics[h]["value"] for h in hs])))
    return out


def collect(root, variant):
    rows, jobs = [], []
    for d in glob.glob(os.path.join(P.RES, root, "*", "*")):
        if d.endswith(".tmp") or not os.path.isdir(d):
            continue
        r = json.load(open(os.path.join(d, "run.json")))
        sc = json.load(open(os.path.join(d, "scaler.json")))
        jobs.append(dict(ds=r["ds"], id=r["id"], model=r["model"], ts=r["ts"], n=r["n"], draw=r["draw"], seed=r["seed"], finite=r["finite"],
                         best_epoch=r.get("best_epoch"), cpu_s=r["cpu_s"], wall_s=r["wall_s"], partial_rerun=r.get("partial_rerun", False),
                         subset_hash=hash(tuple(sc["train_pids"]))))
        for fam, (met, v) in families(json.load(open(os.path.join(d, "metrics.json")))).items():
            rows.append(dict(ds=r["ds"], variant=variant, model=r["model"], n=r["n"], draw=r["draw"], seed=r["seed"], family=fam, metric=met, value=v))
    return pd.DataFrame(rows), pd.DataFrame(jobs)


def units(F, ds, n, model, fam, seed=None):
    """Per-unit values: per draw (seed mean) for n<=400; per seed for n=600 (single draw). seed: restrict to one seed (sensitivity like-for-like)."""
    f = F[(F.ds == ds) & (F.n == n) & (F.model == model) & (F.family == fam)]
    if seed is not None and model != "gbt":
        f = f[f.seed == seed]
    if n == 600:
        return f.groupby("seed").value.mean()
    return f.groupby("draw").value.mean()


def compare(F, ds, n, a, b, fam, seed=None, Fb=None):
    ua, ub = units(F, ds, n, a, fam, seed), units(F if Fb is None else Fb, ds, n, b, fam, seed)
    if len(ua) == 0 or len(ub) == 0:
        return None
    if len(ua) == 1 and len(ub) > 1:                     # GBT single value at n=600 vs seeds
        ua = pd.Series(ua.iloc[0], index=ub.index)
    if len(ub) == 1 and len(ua) > 1:
        ub = pd.Series(ub.iloc[0], index=ua.index)
    d = (ua - ub).dropna()
    if len(d) < 2:
        return None
    met = A.FAMS[fam][0]
    nb, lab = A.verdict(d.values, met)
    better = "A" if (lab == "consistent" and nb == len(d)) else ("B" if (lab == "consistent" and nb == 0) else "")
    return dict(ds=ds, n=n, A=a, B=b, family=fam, metric=met, mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)), n_units=len(d), A_better_n=nb,
                verdict=lab, consistent_better=better, **{f"u{i}": float(v) for i, v in enumerate(d.values)})


def rq1a(Pr, n):
    s = Pr[(Pr.n == n)]
    rows = []
    for a, b, fam in RQ1A:
        c = {ds: s[(s.ds == ds) & (s.A == a) & (s.B == b) & (s.family == fam)] for ds in P.DATASETS}
        cb = {ds: (c[ds].consistent_better.iloc[0] if len(c[ds]) else "") for ds in P.DATASETS}
        rows.append(dict(n=n, A=a, B=b, family=fam, primary=fam in PRIMARY, R4=cb["R4"], R5=cb["R5"],
                         reproducible_KAN=cb["R4"] == cb["R5"] == "A", reproducible_MLP=cb["R4"] == cb["R5"] == "B",
                         KAN_one_dataset_only=(cb["R4"] == "A") != (cb["R5"] == "A")))
    T = pd.DataFrame(rows)
    nK, nM, one = int(T.reproducible_KAN.sum()), int(T.reproducible_MLP.sum()), int(T.KAN_one_dataset_only.sum())
    perds = {ds: int((T[ds] == "A").sum()) for ds in P.DATASETS}
    if nK >= 3 and nK > nM:                                                          # P3_PLAN_AMENDMENT_2: complete, ordered rule
        lab = "supported"
    elif 1 <= nK <= 2 or one >= 3 or (nK >= 3 and nM >= nK) or (nK == 0 and max(perds.values()) >= 2):
        lab = "mixed"
    elif nK == 0 and all(v <= 1 for v in perds.values()):
        lab = "not supported"
    else:
        lab = GAP                                                                    # unreachable after amendment 2 (kept as a guard)
    prim = T[T.primary]
    return lab, dict(n=n, n_reproducible_KAN=nK, n_reproducible_MLP=nM, n_KAN_one_dataset_only=one, KAN_consistent_per_dataset=perds,
                     primary_only=dict(reproducible_KAN=int(prim.reproducible_KAN.sum()), reproducible_MLP=int(prim.reproducible_MLP.sum()))), T


def rq1b(Pr, n):
    s = Pr[(Pr.n == n) & (Pr.B == "gru")]
    rows, per = [], {}
    for m, fams in RQ1B.items():
        for fam in fams:
            cb = {ds: (s[(s.ds == ds) & (s.A == m) & (s.family == fam)].consistent_better.iloc[0] if len(s[(s.ds == ds) & (s.A == m) & (s.family == fam)]) else "") for ds in P.DATASETS}
            rows.append(dict(n=n, model=m, family=fam, R4=cb["R4"], R5=cb["R5"], reproducible_KAN=cb["R4"] == cb["R5"] == "A",
                             reproducible_GRU=cb["R4"] == cb["R5"] == "B", KAN_one_dataset_only=(cb["R4"] == "A") != (cb["R5"] == "A")))
        T = pd.DataFrame([r for r in rows if r["model"] == m])
        per[m] = dict(rep_adv=int(T.reproducible_KAN.sum()), rep_loss=int(T.reproducible_GRU.sum()), one_ds_adv=int(T.KAN_one_dataset_only.sum()))
    sup = [m for m, v in per.items() if v["rep_adv"] >= 2 and v["rep_loss"] == 0]
    if sup:
        lab = "supported"
    elif any(v["rep_adv"] == 1 or v["one_ds_adv"] >= 2 or (v["rep_adv"] >= 1 and v["rep_loss"] >= 1) or v["rep_adv"] >= 2 for v in per.values()):
        lab = "mixed"
    elif all(v["rep_adv"] == 0 and v["one_ds_adv"] <= 1 for v in per.values()):
        lab = "not supported"
    else:
        lab = GAP
    return lab, dict(n=n, per_model=per, supported_by=sup), pd.DataFrame(rows)


def all_pairs(F, dss, sizes, seed=None, Fb=None):
    out = []
    for ds in dss:
        for n in sizes:
            for fam in A.FAMS:
                for a in ("gbt", "kan_reg", "mlp_b2_reg", "m1", "mlp_m1"):
                    r = compare(F, ds, n, a, "gru", fam, seed, Fb)
                    if r:
                        out.append(r)
                for a, b in (("kan_reg", "mlp_b2_reg"), ("m1", "mlp_m1")):
                    r = compare(F, ds, n, a, b, fam, seed)
                    if r:
                        out.append(r)
    return pd.DataFrame(out, columns=None if out else ["ds", "n", "A", "B", "family", "metric", "mean_diff", "sd_diff", "n_units", "A_better_n", "verdict", "consistent_better"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="analysis_cal")
    a = ap.parse_args()
    os.chdir(P.ROOT)
    out = os.path.join(P.RES, a.out)
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} not empty; refusing to overwrite")
    os.makedirs(out, exist_ok=True)
    F, J = collect(a.root, "primary")
    expected = {j["id"] + "|" + j["ds"] for ds in P.DATASETS for j in P.jobs_main(ds)}
    have = set(J.id + "|" + J.ds)
    integ = dict(expected=len(expected), present=len(have & expected), missing=sorted(expected - have)[:20], unexpected=sorted(have - expected)[:20],
                 all_finite=bool(J.finite.all()), partial_reruns=J[J.partial_rerun].id.tolist(),
                 subsets_identical_across_models=bool(J.groupby(["ds", "n", "draw"]).subset_hash.nunique().max() == 1))
    F.to_csv(os.path.join(out, "family_values.csv"), index=False)
    J.to_csv(os.path.join(out, "jobs.csv"), index=False)
    # B3 collapse indicator
    nn = J[J.model != "gbt"]
    b3 = nn.assign(le1=nn.best_epoch <= 1).groupby(["ds", "model", "ts", "n"]).le1.mean().reset_index()
    b3.to_csv(os.path.join(out, "B3_best_epoch_le1.csv"), index=False)
    # learning curves (mean and SD over units)
    lc = []
    for (ds, model, fam, n), g in F.groupby(["ds", "model", "family", "n"]):
        u = units(F, ds, n, model, fam)
        lc.append(dict(ds=ds, model=model, family=fam, n=n, mean=float(u.mean()), sd=float(u.std(ddof=1)) if len(u) > 1 else np.nan, n_units=len(u)))
    LC = pd.DataFrame(lc)
    LC.to_csv(os.path.join(out, "learning_curves.csv"), index=False)
    Pr = all_pairs(F, P.DATASETS, P.SIZES)
    Pr.to_csv(os.path.join(out, "paired.csv"), index=False)
    res = {"SYNTHETIC": True, "split": "cal", "test_sealed": True, "integrity": integ, "B3_flags": b3[b3.le1 > 0.5].to_dict(orient="records")}
    tabs = []
    for n in (50, 100):
        la, ia, Ta = rq1a(Pr, n)
        lb, ib, Tb = rq1b(Pr, n)
        res[f"RQ1a_n{n}"] = dict(label=la, **ia)
        res[f"RQ1b_n{n}"] = dict(label=lb, **ib)
        Ta.to_csv(os.path.join(out, f"RQ1a_n{n}.csv"), index=False)
        Tb.to_csv(os.path.join(out, f"RQ1b_n{n}.csv"), index=False)
    # secondary: same counts at n = 200, 400, 600 (descriptive, no labels)
    sec = {}
    for n in (200, 400, 600):
        _, ia, _ = rq1a(Pr, n)
        _, ib, _ = rq1b(Pr, n)
        sec[n] = dict(RQ1a_counts=ia, RQ1b_counts=ib)
    res["secondary_counts"] = sec
    # crossover: sign of the mean difference at each n, per dataset and comparison
    cr = []
    for (ds, A_, B_, fam), g in Pr.groupby(["ds", "A", "B", "family"]):
        g = g.set_index("n").mean_diff.reindex(P.SIZES)
        sg = np.sign(g.values)
        flip = [int(P.SIZES[i]) for i in range(1, len(sg)) if np.isfinite(sg[i]) and np.isfinite(sg[0]) and sg[i] != sg[0]]
        cr.append(dict(ds=ds, A=A_, B=B_, family=fam, **{f"diff_n{n}": float(v) for n, v in g.items()}, first_flip_n=flip[0] if flip else None))
    pd.DataFrame(cr).to_csv(os.path.join(out, "crossover.csv"), index=False)
    # sensitivities (R4; seed 0 like-for-like with the primary seed-0 runs; descriptive)
    sens = {}
    for v, root in (("S2", "sens_S2"), ("S3", "sens_S3"), ("S1", "sens_S1")):
        if not glob.glob(os.path.join(P.RES, root, "R4", "*")):
            sens[v] = "not run"
            continue
        Fs, Js = collect(root, v)
        Fs.to_csv(os.path.join(out, f"{v}_family_values.csv"), index=False)
        rows = []
        for n in sorted(Fs.n.unique()):
            for fam in A.FAMS:
                for a_, b_ in (("kan_reg", "gru"), ("m1", "gru"), ("kan_reg", "mlp_b2_reg"), ("m1", "mlp_m1"), ("mlp_b2_reg", "gru"), ("gbt", "gru")):
                    prim = compare(F, "R4", n, a_, b_, fam, seed=0)
                    if v == "S2":                        # regularised GRU replaces the GRU; other models from the primary run (seed 0)
                        sv = compare(F, "R4", n, a_, b_, fam, seed=0, Fb=Fs) if b_ == "gru" else None
                    else:
                        sv = compare(Fs, "R4", n, a_, b_, fam, seed=0)
                    if prim and sv:
                        rows.append(dict(variant=v, n=n, A=a_, B=b_, family=fam, primary_mean_diff=prim["mean_diff"], primary_verdict=prim["verdict"],
                                         primary_better=prim["consistent_better"], sens_mean_diff=sv["mean_diff"], sens_verdict=sv["verdict"], sens_better=sv["consistent_better"]))
        S = pd.DataFrame(rows)
        S.to_csv(os.path.join(out, f"{v}_comparison.csv"), index=False)
        sens[v] = dict(jobs=len(Js), all_finite=bool(Js.finite.all()), n_comparisons=len(S),
                       better_label_changed=int((S.primary_better != S.sens_better).sum()) if len(S) else 0)
    res["sensitivities"] = sens
    json.dump(res, open(os.path.join(out, "conclusions_cal.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("RQ1") or k in ("integrity", "sensitivities")}, indent=1, default=str))


if __name__ == "__main__":
    main()
