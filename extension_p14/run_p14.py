"""P1-4 compute step (SYNTHETIC; EXTENSION): input-perturbation lag profiles and Integrated-Gradients day x channel attributions for the saved GRU, M1, M1-MLP and KAN-GRU checkpoints
(and random-initialisation null models of the same architectures). Calibration split only; nothing is retrained; checkpoints/data/results are only read. Writes raw arrays to
results/p1_lag_recovery/raw/{label}[_untrained]_seed{s}.npz. Refuses to overwrite.
Usage: python extension_p14/run_p14.py [--quick]
Perturbation profile P[ch, lag, head]: change of the origin-day SEVERITY prediction (points) for +1 standardised unit on day tau-lag of one channel (observed days only), averaged over origins
(also over two participant halves for stability). 'ar': the impulse persists as the channel's own AR(1) (estimated from the data). 'coupled': primary channel and its generator-coupled
correlate are moved together (oracle-informed control). IG: baseline = all-zero value window (the train-mean day), 16 midpoint steps, 600 origins."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p14 as C  # noqa: E402

OUT = "results/p1_lag_recovery/raw"
COUPLED = {"steps": ("calories", 0.7), "sleep_variability": ("circadian_regularity", -0.3)}       # generator couplings (ground-truth-informed control only)


def shifted(x, obs, spec):
    """spec: list of (channel index, weights over the 28 window days, amount)."""
    xx = x.clone()
    for c, w, amt in spec:
        xx[..., c] += (torch.as_tensor(w, dtype=torch.float32)[None] * amt) * obs[..., c].float()
    return xx


def halves_mean(eff, pid):
    """eff (N, 2) -> (2 heads, 3): overall, half A (even participant ids), half B"""
    even = torch.from_numpy(pid % 2 == 0)
    return torch.stack([eff.mean(0), eff[even].mean(0), eff[~even].mean(0)], dim=1)      # (2 heads, 3)


def integrated_gradients(model, x, s, ym, ys, idx, steps=16):
    """IG for the two severity heads; returns signed and absolute mean attribution (2, 28 lags, 10 channels) and the completeness error (mean |sum IG - (f(x)-f(0))|)."""
    n = len(x)
    grads = torch.zeros(2, n, 28, 10)
    for k in range(steps):
        xa = ((k + 0.5) / steps * x).clone().requires_grad_(True)
        out = model(xa, s)[:, idx] * ys[idx] + ym[idx]
        for h in range(2):
            g, = torch.autograd.grad(out[:, h].sum(), xa, retain_graph=h == 0)
            grads[h] += g / steps
    attr = grads * x[None]
    with torch.no_grad():
        f1, f0 = model(x, s)[:, idx] * ys[idx] + ym[idx], model(torch.zeros_like(x), s)[:, idx] * ys[idx] + ym[idx]
    comp = (attr.sum((2, 3)).T - (f1 - f0)).abs().mean().item()
    flip = lambda t: t.flip(1)                                      # day index 27 (origin) -> lag 0          # noqa: E731
    return flip(attr.mean(1)).numpy(), flip(attr.abs().mean(1)).numpy(), comp


def run_one(label, seed, untrained, x, s, obs, pid, rho, quick):
    model, ym, ys, heads = C.load_model(label, seed, untrained)
    idx = C.head_index(heads)
    base = C.predict2(model, x, s, ym, ys, idx)
    lags = C.LAGS if not quick else [0, 1, 2, 3]
    pert = np.zeros((8, len(lags), 2, 3), np.float32)
    for c in range(8):
        for li, j in enumerate(lags):
            w = np.zeros(28)
            w[27 - j] = 1.0
            pert[c, li] = halves_mean(C.predict2(model, shifted(x, obs, [(c, w, 1.0)]), s, ym, ys, idx) - base, pid).numpy()
    planted_ch = ["steps", "sleep_variability", "hrv"]
    ar = np.zeros((3, len(lags), 2, 3), np.float32)
    coup = np.zeros((2, len(lags), 2, 3), np.float32)
    for ci, name in enumerate([] if untrained else planted_ch):       # AR-propagated / coupled controls: trained models only (null models: single-day profile only)
        c = C.CH.index(name)
        for li, j in enumerate(lags):
            w = np.zeros(28)
            w[27 - j:] = rho[c] ** np.arange(j + 1)
            ar[ci, li] = halves_mean(C.predict2(model, shifted(x, obs, [(c, w, 1.0)]), s, ym, ys, idx) - base, pid).numpy()
    for ci, name in enumerate([] if untrained else ["steps", "sleep_variability"]):
        co, mult = COUPLED[name]
        amt2 = mult * C.std_per_unit(name) / C.std_per_unit(co)
        for li, j in enumerate(lags):
            w = np.zeros(28)
            w[27 - j] = 1.0
            coup[ci, li] = halves_mean(C.predict2(model, shifted(x, obs, [(C.CH.index(name), w, 1.0), (C.CH.index(co), w, amt2)]), s, ym, ys, idx) - base, pid).numpy()
    ig_s = ig_a = np.zeros((2, 28, 10), np.float32)
    comp = np.nan
    if not untrained:
        n_ig = 60 if quick else 600
        ig_s, ig_a, comp = integrated_gradients(model, x[:n_ig], s[:n_ig], ym, ys, idx)
    return dict(lags=np.array(lags), pert=pert, ar=ar, coupled=coup, ig_signed=ig_s, ig_abs=ig_a, ig_completeness_err=np.array(comp), base_sd=base.std(0).numpy(), ar_rho=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--models", nargs="+", default=["gru", "mlp_m1", "kangru", "m1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--null-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-origins", type=int, default=1000)
    a = ap.parse_args()
    os.chdir(C.ROOT)
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty; refusing to overwrite.")
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(4)
    C.register()
    x, s, obs, pid, d = C.load_cal_windows(200 if a.quick else a.n_origins)
    rho = C.estimate_ar1(d)
    print("estimated within-person AR(1):", dict(zip(C.CH, np.round(rho, 2))), flush=True)
    for untrained in (False, True):
        for label in a.models:
            for seed in (a.null_seeds if untrained else a.seeds):
                t0 = time.time()
                r = run_one(label, seed, untrained, x, s, obs, pid, rho, a.quick)
                np.savez_compressed(os.path.join(a.out, f"{label}{'_untrained' if untrained else ''}_seed{seed}.npz"), **r)
                print(label, "untrained" if untrained else "trained", seed, f"{time.time() - t0:.0f}s", "IG completeness err", float(r["ig_completeness_err"]), flush=True)
    print("done")


if __name__ == "__main__":
    main()
