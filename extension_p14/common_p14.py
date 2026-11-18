"""P1-4 EXTENSION (SYNTHETIC; not part of the original six stages): shared helpers for model-based lag recovery. Existing checkpoints, data and results are only READ.
Conventions: window = 28 days ending at the prediction origin; 'lag j' = the day tau-j (j = 0 is the origin day, index 27-j in the window). Planted lag L (src/simulate.py): the feature on
day tau-L enters the INCREMENT of the latent state z on day tau (z[t+1] uses u[t+1-L]); z mean-reverts with rate a, so a one-day unit impulse on day tau-j changes z(tau) by
c_j = beta * (1-a)^(j-L) for j >= L and by 0 for j < L (analytic level response; beta = the Stage 5 benchmark in severity points per unit of the true feature). The increment response is
c_j - c_{j-1} = beta at j = L (then a small negative tail), which is the quantity the Stage 5 distributed-lag regression estimates. This analytic truth ignores the (weak) state -> feature feedback."""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p13"), os.path.join(ROOT, "extension_kangru")]

import baselines as bl  # noqa: E402
import models as md  # noqa: E402

CH = ["steps", "calories", "sleep_duration", "sleep_efficiency", "sleep_variability", "resting_hr", "hrv", "circadian_regularity"]
PLANTED = {("sleep_variability", "dep"): 3, ("steps", "dep"): 2, ("steps", "anx"): 2, ("hrv", "anx"): 1}
SIGN = {("sleep_variability", "dep"): +1, ("steps", "dep"): -1, ("steps", "anx"): -1, ("hrv", "anx"): -1}
HEADS = {"dep": "sev_dep_t0", "anx": "sev_anx_t0"}
LAGS = list(range(0, 11)) + [12, 14, 17, 20, 24, 27]
NEAR = list(range(0, 11))                                   # consecutive lags used for profile metrics
A_REV = 0.08                                                # mean-reversion rate of the simulator (Config.a)

# (label, architecture key, checkpoint pattern, constructor kwargs); all trained on the frozen processed data
SPECS = {"gru": ("gru", "results/stage3/checkpoints/gru_seed{s}.pt", {}),
         "m1": ("m1", "results/stage4_m1/checkpoints/m1_seed{s}.pt", {"dropout": 0.3}),
         "mlp_m1": ("mlp_m1", "results/p1_kan_mlp_control/m1_all/checkpoints/mlp_m1_seed{s}.pt", {"dropout": 0.3}),
         "kangru": ("kangru", "results/ext_kangru/checkpoints/kangru_seed{s}.pt", {})}


def register():
    import models_kangru
    import models_p13
    models_p13.register()
    models_kangru.register()


def build(arch, n_out, heads, kw):
    kw = dict(kw)
    if arch.endswith("m1") or arch.startswith("kangru"):
        kw["heads"] = list(heads)
    return md.MODELS[arch](n_out=n_out, **kw)


def load_model(label, seed, untrained=False):
    """Returns (model in eval mode, ymean, ystd, head list). untrained=True: same architecture, random initialisation (null model)."""
    arch, pat, kw = SPECS[label]
    if untrained:
        torch.manual_seed(seed)
        heads = list(bl.ALL_HEADS if (arch.endswith("m1") or arch.startswith("kangru")) else bl.MAIN_HEADS)
        n_reg = 8 if len(heads) == 28 else 2
        model, ym, ys = build(arch, len(heads), heads, kw), torch.zeros(n_reg), torch.ones(n_reg)
    else:
        ck = torch.load(os.path.join(ROOT, pat.format(s=seed)), map_location="cpu", weights_only=False)
        heads, ym, ys = list(ck["heads"]), ck["ymean"], ck["ystd"]
        model = build(arch, len(heads), heads, kw)
        model.load_state_dict(ck["state_dict"])
    return model.eval(), ym, ys, heads


def head_index(heads):
    return torch.tensor([heads.index(HEADS["dep"]), heads.index(HEADS["anx"])])


@torch.no_grad()
def predict2(model, x, s, ym, ys, idx, bs=1000):
    """Native-scale severity of the two nowcast heads (dep, anx): (N, 2)."""
    outs = []
    for a in range(0, len(x), bs):
        o = model(x[a:a + bs], s[a:a + bs])[:, idx]
        outs.append(o * ys[idx] + ym[idx])
    return torch.cat(outs)


def load_cal_windows(n=2000, seed=0):
    """Fixed random subset of calibration origins (no test data): windows x (n,28,10), static s, observed-day mask obs (n,28,10), participant ids."""
    bl.configure("all")
    d = bl.load_data("data/processed_v1", None, splits=("cal",))["cal"]
    sel = np.sort(np.random.RandomState(seed).choice(len(d["pi"]), n, replace=False))
    off = np.arange(-27, 1)
    x = torch.from_numpy(d["X"][d["pi"][sel][:, None], d["ti"][sel][:, None] + off])
    s = torch.from_numpy(d["static"][d["pi"][sel]])
    obs = (x[..., 8] == 0)[..., None].expand(-1, -1, 10).clone()
    obs[..., 6] &= (x[..., 9] == 0)
    return x, s, obs, d["pid"][d["pi"][sel]], d


def estimate_ar1(d):
    """Within-person lag-1 autocorrelation of each standardised channel on consecutive OBSERVED days of the calibration split (data-visible, no ground truth)."""
    X, miss = d["X"], d["X"][..., 8] == 1
    rho = []
    for c in range(8):
        v = np.where(miss | ((X[..., 9] == 1) if c == 6 else False), np.nan, X[..., c])
        v = v - np.nanmean(v, axis=1, keepdims=True)
        a, b = v[:, 1:].ravel(), v[:, :-1].ravel()
        ok = ~np.isnan(a) & ~np.isnan(b)
        rho.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    return np.array(rho)


def std_per_unit(ch):
    """Standardised-input units per simulator unit u of channel ch inverse: +1 standardised unit == k u-units (k = scaler_std / UNITS_sd)."""
    from simulate import UNITS
    meta = json.load(open(os.path.join(ROOT, "data/processed_v1/preprocessing_meta.json")))
    return meta["scaler_std"][ch] / UNITS[ch][1]


def benchmarks():
    c = json.load(open(os.path.join(ROOT, "results/stage5_causal/analysis_config.json")))["benchmarks_severity_points_per_u_per_day"]
    return {("sleep_variability", "dep"): c["sleep_variability->dep"], ("steps", "dep"): c["steps->dep"], ("steps", "anx"): c["steps->anx"], ("hrv", "anx"): c["hrv->anx"]}


def truth_level(beta, L, lags=NEAR, a=A_REV):
    """Analytic level response (severity points) of the origin-day severity to a one-day +1 unit impulse j days earlier."""
    return np.array([beta * (1 - a) ** (j - L) if j >= L else 0.0 for j in lags])


def truth_ar(beta, L, rho, lags=NEAR, a=A_REV):
    """Same, when the impulse persists as the feature's own AR(1) (day tau-j+k is shifted by rho^k): sum_k rho^k c_{j-k}."""
    c = {m: (beta * (1 - a) ** (m - L) if m >= L else 0.0) for m in range(0, 28)}
    return np.array([sum(rho ** k * c[j - k] for k in range(j + 1)) for j in lags])


def increments(P):
    """First difference over consecutive lags (Stage 5 compatible): D_0 = P_0, D_j = P_j - P_{j-1}. P indexed by NEAR (0..10)."""
    P = np.asarray(P, float)
    return np.concatenate([[P[0]], np.diff(P)])


def lag_metrics(P, L, sign):
    """Quantitative lag-recovery metrics of a level profile P over lags 0..10 against planted lag L / expected sign. Largest value is NOT treated as the true lag: peak lag, increment lag and
    pre-lag mass are reported side by side."""
    P = np.asarray(P, float)
    D = increments(P)
    tot = np.abs(P).sum() + 1e-12
    other = np.delete(np.abs(D), L)
    onset = int(np.argmax(np.abs(P) >= 0.5 * np.abs(P).max()))                  # rise-time estimator: first lag reaching 50% of the largest |response|
    return dict(peak_lag_level=int(np.argmax(np.abs(P))), peak_lag_increment=int(np.argmax(np.abs(D))), peak_lag_increment_1to5=int(1 + np.argmax(np.abs(D[1:6]))), onset_lag50=onset, prelag_mass=float(np.abs(P[:L]).sum() / tot),
                spike_ratio=float(abs(D[L]) / (other.mean() + 1e-12)), sign_ok=float(np.sign(P.sum()) == sign), total_abs=float(np.abs(P).sum()))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0
