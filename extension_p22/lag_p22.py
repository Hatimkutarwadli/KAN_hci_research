"""P2-2 C4: P1-4 single-day perturbation lag profiles on one dataset (SYNTHETIC; calibration split only; inference from saved checkpoints).
Analytic truth parameterised from the dataset's ACTUAL simulator config (lags, effect signs, mean reversion a, benchmark slope from train+cal truth).
  python extension_p22/lag_p22.py --ds R1     -> results/p2_datasets/<ds>/lag_recovery/{metrics.csv, profiles.npz, truth.json}"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p22 as C  # noqa: E402
import baselines as bl  # noqa: E402
import causal_analysis as ca  # noqa: E402
import common_p14 as P14  # noqa: E402  (pure helpers only: shifted-free metrics, truth formula, predict2)
import models as md  # noqa: E402
import models_p13  # noqa: E402
from simulate import UNITS  # noqa: E402

models_p13.register()
MODELS = {"gru": ("gru", "main", {}), "m1": ("m1", "all", {"dropout": 0.3}), "mlp_m1": ("mlp_m1", "all", {"dropout": 0.3})}
NEAR = list(range(11))


def planted(cfg):
    """(feature, outcome) -> (lag, sign) from the actual config."""
    return {("sleep_variability", "dep"): (cfg["lag_sv"], int(np.sign(cfg["eff_sv_dep"]))), ("steps", "dep"): (cfg["lag_steps"], int(np.sign(cfg["eff_steps_dep"]))),
            ("steps", "anx"): (cfg["lag_steps"], int(np.sign(cfg["eff_steps_anx"]))), ("hrv", "anx"): (cfg["lag_hrv"], int(np.sign(cfg["eff_hrv_anx"])))}


def cal_windows(ds, n=1000, seed=0):
    """Identical selection rule to P1-4 load_cal_windows, for any dataset."""
    bl.configure("all")
    d = bl.load_data(C.proc_dir(ds), None, splits=("cal",))["cal"]
    sel = np.sort(np.random.RandomState(seed).choice(len(d["pi"]), n, replace=False))
    off = np.arange(-27, 1)
    x = torch.from_numpy(d["X"][d["pi"][sel][:, None], d["ti"][sel][:, None] + off])
    s = torch.from_numpy(d["static"][d["pi"][sel]])
    obs = (x[..., 8] == 0)[..., None].expand(-1, -1, 10).clone()
    obs[..., 6] &= (x[..., 9] == 0)
    return x, s, obs


def benchmarks_traincal(ds):
    raw = os.path.join(C.ROOT, C.raw_dir(ds))
    T, tp, cfg = ca.load_truth(raw)
    assert cfg["sev_scale"] == ca.SCALE
    sp = pd.read_csv(os.path.join(raw, "splits.csv")).sort_values("pid")
    keep = sp.split.isin(["train", "cal"]).values
    return ca.benchmarks({k: v[keep] for k, v in T.items()}, tp[keep].reset_index(drop=True), cfg), cfg


def load(ds, model, seed):
    arch, ts, kw = MODELS[model]
    ck = torch.load(os.path.join(C.ROOT, C.ckpt_path(ds, model, ts, seed)), map_location="cpu", weights_only=False)
    heads = list(ck["heads"])
    kw = dict(kw)
    if arch.endswith("m1"):
        kw["heads"] = heads
    m = md.MODELS[arch](n_out=len(heads), **kw)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["ymean"], ck["ystd"], heads


@torch.no_grad()
def profiles(model, ym, ys, heads, x, s, obs):
    idx = torch.tensor([heads.index("sev_dep_t0"), heads.index("sev_anx_t0")])
    base = P14.predict2(model, x, s, ym, ys, idx)
    P = np.zeros((8, len(NEAR), 2), np.float32)
    for c in (0, 4, 6):                                   # steps, sleep_variability, hrv (the planted channels)
        for j in NEAR:
            xx = x.clone()
            xx[:, 27 - j, c] += obs[:, 27 - j, c].float()
            P[c, j] = (P14.predict2(model, xx, s, ym, ys, idx) - base).mean(0).numpy()
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--n-origins", type=int, default=1000)
    a = ap.parse_args()
    os.chdir(C.ROOT)
    torch.set_num_threads(4)
    out = os.path.join(C.RES, a.ds, "lag_recovery")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} not empty; refusing to overwrite")
    os.makedirs(out, exist_ok=True)
    bench, cfg = benchmarks_traincal(a.ds)
    pm = json.load(open(os.path.join(C.proc_dir(a.ds), "preprocessing_meta.json")))
    spu = {f: pm["scaler_std"][f] / UNITS[f][1] for f in ("steps", "sleep_variability", "hrv")}
    PL = planted(cfg)
    x, s, obs = cal_windows(a.ds, a.n_origins)
    rows, prof = [], {}
    truth = {f"{f}->{k}": dict(lag=L, sign=sg, beta_per_std_unit=bench[(f, k)] * spu[f], a=cfg["a"]) for (f, k), (L, sg) in PL.items()}
    for model in MODELS:
        for seed in C.SEEDS:
            m, ym, ys, heads = load(a.ds, model, seed)
            P = profiles(m, ym, ys, heads, x, s, obs)
            prof[f"{model}_seed{seed}"] = P
            for (f, k), (L, sg) in PL.items():
                p = P[P14.CH.index(f), :, {"dep": 0, "anx": 1}[k]].astype(float)
                T = P14.truth_level(truth[f"{f}->{k}"]["beta_per_std_unit"], L, NEAR, cfg["a"])
                rows.append(dict(dataset=a.ds, model=model, seed=seed, feature=f, outcome=k, planted_lag=L, **P14.lag_metrics(p, L, sg), r_with_truth=P14.corr(p, T),
                                 magnitude_vs_truth=float(p.sum() / T.sum())))
            print(a.ds, model, seed, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False)
    np.savez_compressed(os.path.join(out, "profiles.npz"), **prof)
    json.dump({"SYNTHETIC": True, "split": "cal", "n_origins": a.n_origins, "truth": truth, "benchmark_source": "train+cal truth of this dataset"}, open(os.path.join(out, "truth.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
