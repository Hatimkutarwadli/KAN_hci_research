"""Stage 8 runner: evaluate FROZEN checkpoints on artificially degraded windows. Inference only; nothing is trained, nothing existing is written.
Each (scenario, replicate) writes one CSV under <out>/<split>/metrics/ (never overwritten; finished files are skipped, so the run is resumable).
Usage (project root):  python extension_missing/run_missing.py --split cal --scenarios C0 S1_r20 --out results/stage8_missing
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_kangru"), os.path.join(ROOT, "extension_missing")]
import data as dd  # noqa: E402
import features as ft  # noqa: E402
import inject as ij  # noqa: E402
import metrics as mt  # noqa: E402
import models as md  # noqa: E402
import models_kangru as mk  # noqa: E402

mk.register()
KEYH = ["sev_dep_t0", "sev_anx_t0", "onset_sust_dep_h14", "onset_sust_anx_h14"]
KEY_SCEN = {"C0", "Z0", "S1_r40", "S1_r40_O", "S2_r40_L7", "S2_r40_L7_O", "S3_k7", "S3_k7_O", "S3_k14", "S3_k14_O", "S5_blk40",
            "S4_activity_absent", "S4_sleep_absent", "S4_cardiac_absent", "S4_circadian_absent"}
TOL = 1e-3


def unit_specs():
    s3, s3c = "results/stage3", "results/stage3c_forecast"
    sp = []
    for part, d in (("main", s3), ("forecast", s3c)):
        sp.append(("gbt", part, d, "gbt", [0]))
        sp += [(f, part, d, f, range(5)) for f in ("gru", "kan")]
    sp.append(("kan_reg", "main", "results/stage3b_kan_reg/final", "kan_reg", range(5)))
    sp.append(("kan_reg", "forecast", s3c, "kan_reg", range(5)))
    for f, d in (("m1", "results/stage4_m1"), ("m1_static", "results/stage4_ablation"), ("kangru", "results/ext_kangru"), ("kangru_fixed", "results/ext_kangru")):
        sp.append((f, "all", d, f, range(5)))
    return sp


def build(fam, heads):
    kw = {}
    name = fam
    if fam == "kan_reg":
        name, kw = "kan", {"dropout": 0.3}
    elif fam in ("m1", "m1_static"):
        kw = {"dropout": 0.3, "heads": heads}
    elif fam in ("kangru", "kangru_fixed"):
        kw = {"heads": heads}
    return md.MODELS[name](n_out=len(heads), **kw)


def load_units():
    units = []
    for fam, part, d, name, seeds in unit_specs():
        for seed in seeds:
            if fam == "gbt":
                models = joblib.load(os.path.join(d, "checkpoints", f"gbt_seed{seed}.joblib"))
                heads = [t for t in models if "cross" not in t]
                units.append(dict(fam=fam, part=part, seed=seed, dir=d, name=name, gbt=models, heads=heads))
            else:
                ck = torch.load(os.path.join(d, "checkpoints", f"{name}_seed{seed}.pt"), weights_only=False)
                m = build(fam, list(ck["heads"]))
                m.load_state_dict(ck["state_dict"])                          # strict: a wrong rebuild fails here
                m.eval()
                units.append(dict(fam=fam, part=part, seed=seed, dir=d, name=name, model=m, heads=list(ck["heads"]), ymean=ck["ymean"], ystd=ck["ystd"],
                                  n_reg=sum(h.startswith("sev_") for h in ck["heads"])))
    return units


@torch.no_grad()
def predict(u, W, S, F=None, bs=4096):
    if u["fam"] == "gbt":
        out = []
        for h in u["heads"]:
            m = u["gbt"][h]
            out.append(m.predict(F) if h.startswith("sev_") else m.predict_proba(F)[:, 1])
        return np.stack(out, 1).astype(np.float32)
    outs = []
    for a in range(0, len(W), bs):
        o = u["model"](torch.from_numpy(W[a:a + bs]), torch.from_numpy(S[a:a + bs]))
        k = u["n_reg"]
        outs.append(torch.cat([o[:, :k] * u["ystd"] + u["ymean"], torch.sigmoid(o[:, k:])], 1))
    return torch.cat(outs).numpy()


def gbt_features(W, S, chunk=8192):
    return np.vstack([ft.window_features(W[a:a + chunk], S[a:a + chunk]) for a in range(0, len(W), chunk)])


def evaluate(P, u, Y, base):
    rows = []
    for j, h in enumerate(u["heads"]):
        if "cross" in h:
            continue
        y = Y[h]
        ok = ~np.isnan(y)
        yy, pp = y[ok], P[ok, j].astype(float)
        if h.startswith("sev_"):
            m = mt.reg_metrics(yy, pp)
            vals = {"rmse": m["rmse"], "r2": m["r2"], "mean_pred": float(pp.mean()), "mean_true": float(yy.mean())}
        else:
            m = mt.clf_metrics(yy, pp)
            vals = {"ap": m["ap"], "auroc": m["auroc"], "brier": m["brier"], "mean_pred": float(pp.mean()), "prevalence": m["prevalence"]}
        rows += [dict(base, model=u["fam"], part=u["part"], seed=u["seed"], task=h, metric=k, value=v) for k, v in vals.items()]
    return rows


def verify_control(P, u, split, pi, ti):
    z = np.load(os.path.join(u["dir"], "predictions", f"{u['name']}_seed{u['seed']}_{split}.npz"))
    sj = list(z["heads"])
    return max(float(np.nanmax(np.abs(P[:, j].astype(float) - z["pred"][pi, ti, sj.index(h)]))) for j, h in enumerate(u["heads"]) if h in sj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["cal", "test"])
    ap.add_argument("--scenarios", nargs="+", default=["all"])
    ap.add_argument("--out", default="results/stage8_missing")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2, help="origin spacing in days (origins 27, 27+stride, ...); approved protocol: 2")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.chdir(ROOT)
    reg = ij.registry()
    scen = reg if a.scenarios == ["all"] else [s for s in reg if s["id"] in a.scenarios]
    d = dd.load_split(a.split)
    X0, pid, static = d["X"], d["pid"], d["static"]
    N = len(pid)
    pi, ti = np.meshgrid(np.arange(N), np.arange(27, 120, a.stride), indexing="ij")
    pi, ti = pi.ravel(), ti.ravel()
    assert ti.min() == 27 and ((ti - 27) % a.stride == 0).all()
    S = static[pi].astype(np.float32)
    R = ij.reconstruct_raw(X0)
    units = load_units()
    heads_all = sorted({h for u in units for h in u["heads"]})
    Y = {h: d["y_" + h][pi, ti] for h in heads_all}
    mdir = os.path.join(a.out, a.split, "metrics")
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(os.path.join(a.out, "predictions"), exist_ok=True)
    cfgp = os.path.join(a.out, f"config_{a.split}.json")
    if not os.path.exists(cfgp):
        json.dump({"SYNTHETIC": True, "split": a.split, "origin_stride_days": a.stride, "origins": f"27, 27+{a.stride}, ... <=119", "n_origins_per_participant": len(range(27, 120, a.stride)),
                   "threads": a.threads, "scenario_registry": [{k: v for k, v in x.items()} for x in reg]}, open(cfgp, "w"), indent=1, default=str)
    else:
        assert json.load(open(cfgp))["origin_stride_days"] == a.stride, "mixing origin sets is not allowed"
    for sc in scen:
        for rep in sc["reps"]:
            fn = os.path.join(mdir, f"{sc['id']}_rep{rep}.csv")
            if os.path.exists(fn):
                print("skip (exists)", fn)
                continue
            t0 = time.perf_counter()
            W = ij.make_windows(sc, rep, a.split, R, X0, pi, ti)
            t_win = time.perf_counter() - t0
            if sc["id"] == "C0":
                assert np.array_equal(W, ij.windows(X0, pi, ti)), "C0 windows differ from frozen windows"
            with threadpool_limits(a.threads):
                t1 = time.perf_counter()
                F = gbt_features(W, S)
                t_feat = time.perf_counter() - t1
                rows, tfam, ver, keep = [], {}, {}, {}
                for u in units:
                    t2 = time.perf_counter()
                    P = predict(u, W, S, F)
                    tfam[u["fam"]] = tfam.get(u["fam"], 0.0) + time.perf_counter() - t2
                    rows += evaluate(P, u, Y, dict(split=a.split, scenario=sc["id"], family=sc["family"], rep=rep))
                    if sc["id"] == "C0":
                        ver[f"{u['fam']}/{u['part']}/seed{u['seed']}"] = verify_control(P, u, a.split, pi, ti)
                    if sc["id"] in KEY_SCEN and rep == 0 and u["seed"] == 0:
                        for j, h in enumerate(u["heads"]):
                            if h in KEYH:
                                keep[f"{u['fam']}__{u['part']}__{h}"] = P[:, j].astype(np.float32)
            if ver:
                json.dump(ver, open(os.path.join(a.out, a.split, "control_verification.json"), "w"), indent=1)
                bad = {k: v for k, v in ver.items() if v > TOL}
                print(f"control verification: max |pred - saved| = {max(ver.values()):.2e}; units over tol {TOL}: {len(bad)}")
                if bad:
                    raise SystemExit(f"STOP: control predictions do not reproduce saved predictions: {bad}")
            if keep:
                np.savez_compressed(os.path.join(a.out, "predictions", f"{a.split}_{sc['id']}_rep{rep}.npz"), pi=pi, ti=ti, pid=pid, **keep)
            pd.DataFrame(rows).to_csv(fn, index=False)
            tot = time.perf_counter() - t0
            with open(os.path.join(a.out, a.split, "timing.jsonl"), "a") as f:
                f.write(json.dumps(dict(scenario=sc["id"], rep=rep, total_s=tot, windows_s=t_win, gbt_features_s=t_feat, per_family_s=tfam)) + "\n")
            print(f"{a.split} {sc['id']} rep{rep}: {tot:.0f}s (windows {t_win:.1f}, gbt feats {t_feat:.1f}, " + ", ".join(f"{k} {v:.0f}" for k, v in tfam.items()) + ")", flush=True)


if __name__ == "__main__":
    main()
