"""P1-3 interpretability analysis (SYNTHETIC; EXTENSION). Loads EXISTING and new checkpoints read-only (nothing is retrained or overwritten) and asks whether learned shapes and feature
sensitivities recover the PLANTED structure of the simulator (sleep variability -> depression (+), steps -> depression (-), steps -> anxiety (-), HRV -> anxiety (-)).
Uses the CALIBRATION split only (no test data), a fixed random subset of origins, and the model-agnostic window-shift partial dependence that applies identically to KAN, MLP and GRU.
Recovered shape/sign/rank is a statement about what the PREDICTOR learned from observational synthetic data (which also contains the planted state->feature reverse paths and
collinear channels); it is not a causal estimate.
Usage: python extension_p13/interpret_p13.py   -> results/p1_kan_mlp_control/interpretability/*.csv|npz"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p13")]
os.chdir(ROOT)
import baselines as bl  # noqa: E402
import models as md  # noqa: E402
import models_p13 as m13  # noqa: E402

OUT = "results/p1_kan_mlp_control/interpretability"
BASE = "results/p1_kan_mlp_control"
CH = ["steps", "calories", "sleep_duration", "sleep_efficiency", "sleep_variability", "resting_hr", "hrv", "circadian_regularity"]
# planted (outcome, channel) -> expected sign of the association with severity; shape of the planted bounded transform g (src/simulate.py: g_sv=tanh(x), g_st=tanh(0.7x), g_hr=tanh(0.8x))
PLANTED = {("dep", "sleep_variability"): (+1, lambda d: np.tanh(d)), ("dep", "steps"): (-1, lambda d: np.tanh(0.7 * d)),
           ("anx", "steps"): (-1, lambda d: np.tanh(0.7 * d)), ("anx", "hrv"): (-1, lambda d: np.tanh(0.8 * d))}
HEADS_OF_INTEREST = {"sev_dep_t0": "dep", "sev_anx_t0": "anx", "status_dep_t0": "dep", "status_anx_t0": "anx"}
DELTAS = np.linspace(-2, 2, 9)
LAGS = [0, 1, 2, 3, 5, 7, 10, 14, 20, 27]
N_ORIG, SEEDS = 2000, range(5)

# (label, architecture key, checkpoint pattern, constructor kwargs)
SPECS = [("gru", "gru", "results/stage3/checkpoints/gru_seed{s}.pt", {}),
         ("kan", "kan", "results/stage3/checkpoints/kan_seed{s}.pt", {}),
         ("kan_reg", "kan", "results/stage3b_kan_reg/final/checkpoints/kan_reg_seed{s}.pt", {"dropout": 0.3}),
         ("kan_nospline", "kan_nospline", BASE + "/b2_main/checkpoints/kan_nospline_seed{s}.pt", {}),
         ("mlp_b2", "mlp_b2", BASE + "/b2_main/checkpoints/mlp_b2_seed{s}.pt", {}),
         ("mlp_b2_reg", "mlp_b2", BASE + "/b2_main/checkpoints/mlp_b2_reg_seed{s}.pt", {"dropout": 0.3}),
         ("m1", "m1", "results/stage4_m1/checkpoints/m1_seed{s}.pt", {"dropout": 0.3}),
         ("mlp_m1", "mlp_m1", BASE + "/m1_all/checkpoints/mlp_m1_seed{s}.pt", {"dropout": 0.3})]
NULL_OF = {"gru": "gru", "kan": "kan", "kan_nospline": "kan_nospline", "mlp_b2": "mlp_b2", "m1": "m1", "mlp_m1": "mlp_m1"}     # untrained (random-initialisation) nulls


def build(arch, n_out, heads, kw):
    kw = dict(kw)
    if arch.endswith("m1"):
        kw["heads"] = list(heads)
    return md.MODELS[arch](n_out=n_out, **kw)


@torch.no_grad()
def predict(model, x, s, ym, ys, idx, reg, bs=1000):
    model.eval()
    outs = []
    for a in range(0, len(x), bs):
        o = model(x[a:a + bs], s[a:a + bs])[:, idx]
        outs.append(o * ys[idx] + ym[idx] if reg else torch.sigmoid(o))
    return torch.cat(outs)


def analyse(label, model, ym, ys, heads, x, s, y_true, obs, rng_seed):
    """Window-shift PDP, shift sensitivity, loss-based permutation importance and lag profile for the 4 heads of interest."""
    rows_pdp, rows_sens, rows_lag = [], [], []
    for h, k in HEADS_OF_INTEREST.items():
        if h not in heads:
            continue
        j, reg = heads.index(h), h.startswith("sev")
        idx = torch.tensor([j])
        f = lambda xx: predict(model, xx, s, ym, ys, idx, reg)[:, 0]      # noqa: E731
        base = f(x)
        sd = float(base.std()) + 1e-9
        g = torch.Generator().manual_seed(rng_seed)
        perm = torch.randperm(len(x), generator=g)
        yt = torch.from_numpy(y_true[h])
        base_err = float(((base - yt) ** 2).mean().sqrt()) if reg else None
        for c, cname in enumerate(CH):
            m = obs[..., c].float()
            vals = []
            for d in DELTAS:
                xx = x.clone()
                xx[..., c] += float(d) * m
                vals.append(float(f(xx).mean()))
                rows_pdp.append(dict(head=h, channel=cname, delta=float(d), value=vals[-1]))
            vals = np.array(vals)
            xp = x.clone()
            xp[..., c] = x[perm][..., c]
            p = f(xp)
            rows_sens.append(dict(head=h, channel=cname, sens=float((vals.max() - vals.min()) / sd), slope_central=float(np.polyfit(DELTAS[2:7], vals[2:7], 1)[0]),
                                  mean_abs_change_perm=float((p - base).abs().mean() / sd), perm_rmse_increase=(float(((p - yt) ** 2).mean().sqrt()) - base_err) if reg else np.nan))
        for cname in ("sleep_variability", "steps", "hrv"):
            c = CH.index(cname)
            for lag in LAGS:
                xx = x.clone()
                xx[:, -1 - lag, c] += obs[:, -1 - lag, c].float()                          # +1 SD on the single day t-lag only
                rows_lag.append(dict(head=h, channel=cname, lag=lag, effect=float((f(xx) - base).mean())))
    return rows_pdp, rows_sens, rows_lag


def recovery(pdp, sens):
    """Per (model, seed): sign accuracy, shape correlation with planted g, mean rank of planted channels among the 8 channels (1 = most sensitive), calories-above-steps confound flag."""
    out = []
    for h in HEADS_OF_INTEREST:
        p, s_ = pdp[pdp["head"] == h], sens[sens["head"] == h]
        if p.empty:
            continue
        k = HEADS_OF_INTEREST[h]
        rank = s_.set_index("channel").sens.rank(ascending=False)
        planted = [c for (kk, c) in PLANTED if kk == k]
        r_sign, r_shape = [], []
        for c in planted:
            sg, g = PLANTED[(k, c)]
            v = p[p.channel == c].sort_values("delta").value.values
            r_sign.append(float(np.sign(s_.set_index("channel").slope_central[c]) == sg and s_.set_index("channel").sens[c] > 1e-4))      # a numerically flat response counts as no recovery
            vc = v - v[len(v) // 2]
            ref = sg * g(DELTAS)
            r_shape.append(float(np.corrcoef(vc, ref)[0, 1]) if vc.std() > 1e-12 else 0.0)
        out.append(dict(head=h, sign_acc=np.mean(r_sign), shape_r=np.mean(r_shape), mean_rank_planted=float(np.mean([rank[c] for c in planted])),
                        recall_top3=float(np.mean([rank[c] <= 3 for c in planted])), calories_above_steps=float(rank["calories"] < rank["steps"]),
                        top_channel=rank.idxmin()))
    return out


def kan_edges(model, grid=np.linspace(-3, 3, 121)):
    """First-layer KAN edge functions phi_{o,(c,j)}(x) of the 3/5/7-day encoders: returns {scale: (phi (n_grid, d, C, k))}. KANConv/CausalKANConv patch input index = c*k + j."""
    res = {}
    xg = torch.tensor(grid, dtype=torch.float32)
    for conv in model.convs:
        L = conv.kan
        n_in = L.n_in
        with torch.no_grad():
            B = L.bases(xg[:, None].expand(-1, n_in))                                   # (n_grid, n_in, G+k)
            phi = torch.nn.functional.silu(xg)[:, None, None] * L.base_weight.T[None] + torch.einsum("gik,oik->gio", B, L.spline_weight.view(L.n_out, n_in, -1))
        res[conv.k] = phi.reshape(len(grid), conv.kan.n_in // conv.k, conv.k, L.n_out).permute(0, 3, 1, 2).numpy()      # (grid, o, C, k)
    return res, grid


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "recovery_per_seed.csv")):
        raise SystemExit("interpretability outputs exist; refusing to overwrite")
    bl.configure("all")
    m13.register()
    data = bl.load_data("data/processed_v1", None, splits=("cal",))["cal"]
    rs = np.random.RandomState(0)
    sel = np.sort(rs.choice(len(data["pi"]), N_ORIG, replace=False))
    off = np.arange(-27, 1)
    x = torch.from_numpy(data["X"][data["pi"][sel][:, None], data["ti"][sel][:, None] + off])
    s = torch.from_numpy(data["static"][data["pi"][sel]])
    obs = (x[..., 8] == 0)[..., None].expand(-1, -1, 10).clone()
    obs[..., 6] &= (x[..., 9] == 0)
    y_true = {h: data["Y"][sel, bl.ALL_HEADS.index(h)].astype(np.float32) for h in HEADS_OF_INTEREST}
    for h in y_true:                                                                       # status/sev labels exist at every eligible origin; guard anyway
        y_true[h] = np.nan_to_num(y_true[h], nan=0.0)
    allp, alls, alll, allr, alle, edge_fn = [], [], [], [], [], {}
    # association-only reference: Pearson r between the window mean of each channel (observed days) and the severity target
    assoc = []
    wm = np.array([[np.nanmean(np.where(obs[i, :, c].numpy(), x[i, :, c].numpy(), np.nan)) for c in range(8)] for i in range(len(x))])
    wm = np.nan_to_num(wm)
    for h in ("sev_dep_t0", "sev_anx_t0"):
        for c, cn in enumerate(CH):
            assoc.append(dict(head=h, channel=cn, pearson_r=float(np.corrcoef(wm[:, c], y_true[h])[0, 1])))
    pd.DataFrame(assoc).to_csv(os.path.join(OUT, "association_baseline.csv"), index=False)

    jobs = [(lab, arch, pat.format(s=sd), kw, sd, False) for lab, arch, pat, kw in SPECS for sd in SEEDS] + \
           [(lab + "_untrained", arch, None, kw, sd, True) for lab, arch, pat, kw in SPECS if lab in NULL_OF for sd in SEEDS]
    for lab, arch, path, kw, sd, null in jobs:
        if null:
            torch.manual_seed(sd)
            heads = bl.ALL_HEADS if arch.endswith("m1") else bl.MAIN_HEADS
            ym, ys = torch.zeros(8 if arch.endswith("m1") else 2), torch.ones(8 if arch.endswith("m1") else 2)
            model = build(arch, len(heads), heads, kw)
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            heads, ym, ys = list(ck["heads"]), ck["ymean"], ck["ystd"]
            model = build(arch, len(heads), heads, kw)
            model.load_state_dict(ck["state_dict"])
        pdp, sens, lag = analyse(lab, model, ym, ys, heads, x, s, y_true, obs, 100 + sd)
        for rows, store in ((pdp, allp), (sens, alls), (lag, alll)):
            for r in rows:
                r.update(model=lab, seed=sd)
            store += rows
        pdf, sdf = pd.DataFrame(pdp), pd.DataFrame(sens)
        for r in recovery(pdf, sdf):
            allr.append(dict(model=lab, seed=sd, **r))
        if arch in ("kan", "kan_nospline", "m1") and not null:
            e, grid = kan_edges(model)
            for k, phi in e.items():
                edge_fn[f"{lab}|{sd}|{k}"] = phi.astype(np.float32)
                imp = np.abs(phi).mean(0)                                                    # (o, C, k): mean |phi| over the grid
                for c, cn in enumerate(CH + ["missing_day", "hrv_missing"]):
                    alle.append(dict(model=lab, seed=sd, scale=k, channel=cn, edge_importance=float(imp[:, c].sum())))
        print(lab, sd, "done", flush=True)
    pd.DataFrame(allp).to_csv(os.path.join(OUT, "pdp_curves.csv"), index=False)
    pd.DataFrame(alls).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    pd.DataFrame(alll).to_csv(os.path.join(OUT, "lag_profile.csv"), index=False)
    pd.DataFrame(allr).to_csv(os.path.join(OUT, "recovery_per_seed.csv"), index=False)
    pd.DataFrame(alle).to_csv(os.path.join(OUT, "kan_edge_importance.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "kan_edge_functions.npz"), grid=grid, **edge_fn)
    print("done")


if __name__ == "__main__":
    main()
