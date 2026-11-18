"""Diagnostics that the M1 latent KAN transition learns temporal state evolution (SYNTHETIC data).
Uses TRAIN and CALIBRATION participants only (the test split is never loaded). Ground-truth latent states from
truth.csv are used ONLY here, post hoc, for a linear-probe diagnostic - never for training or model selection.

Usage: python src/m1_diagnostics.py results/stage4_m1 results/stage4_ablation results/stage4_diagnostics
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import baselines as bl
import metrics as mt
import models as md

torch.set_num_threads(4)
SEEDS = (0, 1, 2, 3, 4)


def load_model(path, kind):
    ck = torch.load(path, weights_only=False)
    m = md.MODELS[kind](heads=ck["heads"], n_out=len(ck["heads"]), dropout=0.3)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["ymean"], ck["ystd"]


@torch.no_grad()
def forward_all(m, T, idx, bs=2048):
    outs, zs, trs = [], [], []
    for a in range(0, len(idx), bs):
        x, s, _ = bl.batch(T, idx[a:a + bs])
        o, z, tr = m(x, s, return_latent=True)
        outs.append(o)
        zs.append(z)
        if tr is not None:
            trs.append(tr)
    return torch.cat(outs), torch.cat(zs), (torch.cat(trs) if trs else None)


@torch.no_grad()
def cal_loss(m, T, ym, ys, perturb=None, bs=2048):
    tot, cnt = 0.0, 0.0
    for a in range(0, len(T["pi"]), bs):
        x, s, y = bl.batch(T, torch.arange(a, min(a + bs, len(T["pi"]))))
        if perturb is not None:
            x = perturb(x)
        l, v = bl.head_loss_terms(m(x, s), y, ym, ys)
        tot, cnt = tot + l.sum(0), cnt + v.sum(0)
    return float((tot / cnt.clamp(min=1)).mean())


def ridge_r2(Xtr, ytr, Xev, yev, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xev - mu) / sd
    A1, B1 = np.c_[A, np.ones(len(A))], np.c_[B, np.ones(len(B))]
    w = np.linalg.solve(A1.T @ A1 + lam * np.eye(A1.shape[1]), A1.T @ ytr)
    p = B1 @ w
    return float(1 - ((yev - p) ** 2).sum() / ((yev - yev.mean()) ** 2).sum())


def main(m1_dir, abl_dir, out):
    os.makedirs(out, exist_ok=True)
    bl.configure("all")
    data = bl.load_data(splits=("train", "cal"))          # test deliberately not loaded
    assert set(data) == {"train", "cal"}
    Tc, Tt = bl.to_t(data["cal"]), bl.to_t(data["train"])
    truth = pd.read_csv("data/synthetic_v1/truth.csv", usecols=["pid", "day", "z_dep", "z_anx"]).sort_values(["pid", "day"])
    zt = {k: truth[f"z_{k}"].values.reshape(1000, 120) for k in ("dep", "anx")}
    rng = np.random.default_rng(0)
    sub = {s: np.sort(rng.choice(len(data[s]["pi"]), 4000 if s == "cal" else 8000, replace=False)) for s in data}
    R = {"models": {}}
    for kind, d_ in (("m1", m1_dir), ("m1_static", abl_dir)):
        rows = []
        for seed in SEEDS:
            m, ym, ys = load_model(os.path.join(d_, "checkpoints", f"{kind}_seed{seed}.pt"), kind)
            r = {"seed": seed}
            # (A) recency sensitivity: noise on the oldest 7 days vs the most recent 7 days (same noise draws), change in outputs
            g = torch.Generator().manual_seed(seed)
            idx = torch.from_numpy(sub["cal"])
            x0, s0, _ = bl.batch(Tc, idx)
            base, _, _ = forward_all(m, Tc, idx)
            noise = torch.randn(x0.shape, generator=g)
            for name, sl in (("old7", slice(0, 7)), ("recent7", slice(21, 28))):
                xp = x0.clone()
                xp[:, sl, :8] += noise[:, sl, :8]                       # perturb value channels only
                with torch.no_grad():
                    o = m(xp, s0)
                r[f"sens_{name}"] = float((o - base).abs().mean())
            r["sens_ratio_recent_over_old"] = r["sens_recent7"] / max(r["sens_old7"], 1e-9)
            # (B) window-order shuffle on calibration: loss and sustained-onset AP
            L0 = cal_loss(m, Tc, ym, ys)
            perm_g = torch.Generator().manual_seed(100 + seed)
            perm = torch.randperm(28, generator=perm_g)
            L1 = cal_loss(m, Tc, ym, ys, perturb=lambda x: x[:, perm])
            r.update(cal_loss=L0, cal_loss_shuffled_days=L1, cal_loss_increase_when_shuffled=L1 - L0)
            # (C) latent trajectory statistics on calibration origins
            _o, z, tr = forward_all(m, Tc, idx)
            if tr is not None:
                r["mean_abs_state_step"] = float((tr[:, 1:] - tr[:, :-1]).abs().mean())
                r["temporal_std_of_state_within_window"] = float(tr.std(1).mean())
                r["corr_z_T_with_z_T_minus_7"] = float(np.mean([np.corrcoef(tr[:, -1, j].numpy(), tr[:, -8, j].numpy())[0, 1] for j in range(tr.shape[2])]))
            # (D) linear probe: true latent at the origin day from z_T; fit on TRAIN participants, evaluate on CALIBRATION participants
            Z = {}
            for s_, T_ in (("train", Tt), ("cal", Tc)):
                ii = torch.from_numpy(sub[s_])
                _, zz, _ = forward_all(m, T_, ii)
                pids = data[s_]["pid"][data[s_]["pi"][sub[s_]]]
                days = data[s_]["ti"][sub[s_]]
                Z[s_] = (zz.numpy(), pids, days)
            for k in ("dep", "anx"):
                ytr = zt[k][Z["train"][1], Z["train"][2]]
                yev = zt[k][Z["cal"][1], Z["cal"][2]]
                r[f"probe_R2_true_z_{k}"] = ridge_r2(Z["train"][0], ytr, Z["cal"][0], yev)
            rows.append(r)
        df = pd.DataFrame(rows)
        R["models"][kind] = {"per_seed": rows, "mean": df.drop(columns="seed").mean().to_dict(), "sd": df.drop(columns="seed").std().to_dict()}
    # reference probes without any model: last-day features and window-mean features (10 channels each), same fit/eval split
    ref = {}
    for name, fn in (("last_day_channels", lambda W: W[:, -1, :]), ("window_mean_channels", lambda W: W.mean(1))):
        feats = {}
        for s_, T_ in (("train", Tt), ("cal", Tc)):
            ii = torch.from_numpy(sub[s_])
            x, _, _ = bl.batch(T_, ii)
            feats[s_] = (fn(x).numpy(), data[s_]["pid"][data[s_]["pi"][sub[s_]]], data[s_]["ti"][sub[s_]])
        for k in ("dep", "anx"):
            ref[f"{name}_probe_R2_true_z_{k}"] = ridge_r2(feats["train"][0], zt[k][feats["train"][1], feats["train"][2]],
                                                          feats["cal"][0], zt[k][feats["cal"][1], feats["cal"][2]])
    R["reference_probes_no_model"] = ref
    with open(os.path.join(out, "diagnostics.json"), "w") as f:
        json.dump(R, f, indent=1)
    keys = ["sens_old7", "sens_recent7", "sens_ratio_recent_over_old", "cal_loss", "cal_loss_shuffled_days", "cal_loss_increase_when_shuffled",
            "mean_abs_state_step", "temporal_std_of_state_within_window", "corr_z_T_with_z_T_minus_7", "probe_R2_true_z_dep", "probe_R2_true_z_anx"]
    for kind in R["models"]:
        print(kind)
        for k in keys:
            if k in R["models"][kind]["mean"]:
                print(f"  {k}: {R['models'][kind]['mean'][k]:.4f} +-{R['models'][kind]['sd'][k]:.4f}")
    print("reference probes:", {k: round(v, 3) for k, v in ref.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
