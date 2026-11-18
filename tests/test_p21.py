"""Tests for P2-1 (hybrid KAN-ODE, no-flow ablation, GRU-D-style GRU). SYNTHETIC data. Run: python tests/test_p21.py
The test split is never loaded here."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p21")]
import baselines as bl  # noqa: E402
import models as md  # noqa: E402
import models_p21 as mp  # noqa: E402

H = list(bl.ALL_HEADS)


def _x(B=4, L=28, seed=0, p_miss=0.3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, 10, generator=g)
    miss = (torch.rand(B, L, generator=g) < p_miss).float()
    hm = torch.maximum(miss, (torch.rand(B, L, generator=g) < 0.1).float())
    x[..., 8], x[..., 9] = miss, hm
    return x, torch.randn(B, 4, generator=g)


def test_rk4_accuracy_and_order():
    k, T = torch.tensor([0.3, 1.0, 2.0], dtype=torch.float64), 4.0
    errs = []
    for S in (2, 4, 8):
        z = torch.ones(3, dtype=torch.float64)
        for _ in range(int(T * S)):
            z = mp.rk4_step(lambda u: -k * u, z, 1.0 / S)
        errs.append(float((z - torch.exp(-k * T)).abs().max()))
    order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    assert errs[-1] < 1e-6 and min(order) > 3.7, (errs, order)
    # h = 1 (pilot setting). Stability: |R(-kappa h)| < 1 over the whole allowed range kappa h in [0.05, 2] (RK4 limit ~2.785).
    kh = torch.linspace(mp.KMIN, mp.KMAX, 200, dtype=torch.float64)
    R = mp.rk4_step(lambda u: -kh * u, torch.ones_like(kh), 1.0)
    assert float(R.abs().max()) < 1.0
    # Accuracy is only good for kappa h <= 1 (at kappa h = 2, R = 1/3 vs exp(-2) = 0.135): measured during training via S=1 vs S=4.
    k1 = torch.tensor([0.3, 1.0], dtype=torch.float64)
    z = torch.ones(2, dtype=torch.float64)
    for _ in range(4):
        z = mp.rk4_step(lambda u: -k1 * u, z, 1.0)
    assert float((z - torch.exp(-k1 * T)).abs().max()) < 2e-3


def test_reduction_to_m1():
    torch.manual_seed(0)
    m1 = md.MODELS["m1"](n_out=28, heads=H, dropout=0.3).eval()
    ko = mp.KANODENet(H, n_out=28, flow=False, dropout=0.3).eval()
    ko.load_state_dict(m1.state_dict(), strict=True)              # identical parameter set
    ko.gate = False
    x, s = _x()
    assert torch.equal(m1(x, s), ko(x, s))
    assert md.n_params(ko) == md.n_params(m1) == 20932
    assert md.n_params(mp.KANODENet(H, n_out=28, flow=True, dropout=0.3)) == 20932 + 640 + 8


def test_missing_day_gating():
    x, s = _x(p_miss=0.4)
    miss = x[..., 8] > 0
    for flow in (False, True):
        torch.manual_seed(1)
        m = mp.KANODENet(H, n_out=28, flow=flow).eval()
        _, _, traj = m(x, s, return_latent=True)
        zbar = torch.tanh(m.z0(s))
        for b in range(x.shape[0]):
            for t in range(1, 28):
                if miss[b, t]:
                    prev = traj[b:b + 1, t - 1]
                    exp = mp.rk4_step(lambda u: m.vf(u, zbar[b:b + 1]), prev, 1.0) if flow else prev
                    assert torch.allclose(traj[b:b + 1, t], exp, atol=1e-6), (flow, b, t)
                else:
                    assert not torch.allclose(traj[b, t], traj[b, t - 1])


def test_causality_of_latent_trajectory():
    torch.manual_seed(2)
    m = mp.KANODENet(H, n_out=28, flow=True).eval()
    x, s = _x()
    _, _, tr1 = m(x, s, return_latent=True)
    x2 = x.clone()
    x2[:, 15:, :8] += 5.0
    _, _, tr2 = m(x2, s, return_latent=True)
    assert torch.equal(tr1[:, :15], tr2[:, :15]) and not torch.allclose(tr1[:, 20:], tr2[:, 20:])
    g = mp.GRUDNet(n_out=16).eval()                               # GRU-D: a window truncated at day 15 is unaffected by later days
    assert torch.equal(g(x[:, :15], s), g(x2[:, :15], s))


def test_boundedness_inside_spline_grid():
    torch.manual_seed(3)
    m = mp.KANODENet(H, n_out=28, flow=True).eval()
    with torch.no_grad():                                         # adversarially large forcing and kappa at its maximum
        for p in m.kan_f.parameters():
            p.mul_(20.0)
        m.rho.fill_(20.0)
    x, s = _x(B=16, p_miss=0.8)
    x[..., :8] *= 5
    for S, tol in ((16, 1e-4), (1, None)):
        m.S = S
        m.reset_stats()
        m(x, s)
        mx = m.stats["eval"]["max_abs_z"]
        if tol is not None:
            assert mx < mp.GRID + tol, mx                         # near-exact flow respects the analytic bound |z| < 3
        print(f"  boundedness: S={S} max|z|={mx:.4f} (grid edge {mp.GRID})")
    assert float(m.kappa().max()) <= mp.KMAX and float(m.kappa().min()) >= mp.KMIN


def test_grud_masks_deltas_and_imputation():
    x = torch.zeros(1, 6, 10)
    x[0, :, 0] = torch.tensor([1.0, 9.0, 9.0, 4.0, 9.0, 9.0])     # value channel 0 (9 = forward-filled garbage on missing days)
    x[0, :, 8] = torch.tensor([0, 1, 1, 0, 1, 0.])                # missing_day
    x[0, :, 9] = torch.tensor([0, 1, 1, 1, 1, 0.])                # hrv_missing
    m, d = mp.GRUDNet.masks_deltas(x)
    assert d[0, :, 0].tolist() == [0, 1, 2, 3, 1, 2] and d[0, :, 1].tolist() == [0, 1, 2, 3, 4, 5]
    assert m[0, :, 0].tolist() == [1, 0, 0, 1, 0, 1] and m[0, :, mp.HRV].tolist() == [1, 0, 0, 0, 0, 1]
    g = mp.GRUDNet(n_out=16)
    with torch.no_grad():
        g.wx.fill_(0.5)
        g.bx.zero_()
    seen = []
    orig = g.cell.forward
    g.cell.forward = lambda inp, h: (seen.append(inp[0, 0].item()), orig(inp, h))[1]
    g(x, torch.zeros(1, 4))
    exp = [1.0, np.exp(-0.5) * 1.0, np.exp(-1.0) * 1.0, 4.0, np.exp(-0.5) * 4.0, 9.0]   # decay of the last observed value towards 0
    assert np.allclose(seen, exp, atol=1e-6), (seen, exp)
    for n_out, ref in ((16, 10864), (12, 10732)):
        assert abs(md.n_params(mp.GRUDNet(n_out=n_out)) - ref) / ref < 0.02


def test_grud_training_determinism():
    mp.register()
    bl.configure("main")
    d = bl.load_data("data/processed_v1", {"train": 30, "cal": 10}, splits=("train", "cal"))
    sd = []
    for _ in range(2):
        m, *_ = bl.fit_nn("grud", 0, d["train"], d["cal"], epochs=1, log=lambda *a: None)
        sd.append(m.state_dict())
    assert all(torch.equal(sd[0][k], sd[1][k]) for k in sd[0])


def test_quick_pipeline_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "k")
        r = subprocess.run([sys.executable, "extension_p21/run_p21.py", "train", "--model", "kanode", "--taskset", "all", "--out", out, "--quick"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        rl = json.load(open(os.path.join(out, "run_log.json")))
        assert rl["test_split_loaded"] is False and rl["n_params"] == 21580 and rl["final_diagnostics"]["kappa_h_max"] <= 2.0
        assert len(rl["epoch_diagnostics"]) == rl["epochs_run"] and "kappa" in rl["epoch_diagnostics"][0]


def test_existing_files_unchanged():
    man = "results/p2_kanode/manifest_pre.json"
    if not os.path.exists(man):
        print("  (no manifest yet; skipped)")
        return
    r = subprocess.run([sys.executable, "extension_p21/run_p21.py", "verify", "--manifest", man], capture_output=True, text=True)
    print("  " + r.stdout.strip())
    assert r.returncode == 0


def test_pilot_seed0_outputs_unchanged():
    man = "results/p2_kanode/manifest_pilot_seed0.json"
    if not os.path.exists(man):
        return
    import hashlib
    pre = json.load(open(man))
    bad = [p for p, h in pre.items() if hashlib.sha256(open(p, "rb").read()).hexdigest() != h]
    assert not bad, bad


def test_seed0_reproduction_with_seed_argument():
    """GRU-D seed 0 retrained via the --seed code path must reproduce the pilot checkpoint exactly."""
    rep = "results/p2_kanode/full/repro_grud_main_seed0/checkpoints/grud_seed0.pt"
    if not os.path.exists(rep):
        print("  (reproduction run not present; skipped)")
        return
    a = torch.load(rep, weights_only=False)["state_dict"]
    b = torch.load("results/p2_kanode/pilot/grud_main/checkpoints/grud_seed0.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in b)


def test_reference_metrics_recomputed_exactly():
    """The analysis recomputes reference metrics from saved predictions; they must equal the saved Stage 4 / Stage 3 calibration metrics."""
    import pandas as pd
    import analyze_p21 as an
    lab = __import__("data").load_split("cal")
    for ref, task, met in (("m1", "onset_sust_dep_h7", "ap"), ("m1", "sev_anx_h14", "rmse"), ("gru_main", "status_anx_t0", "ap")):
        z = np.load(an.pred_file(ref, 2, "cal"))
        y, p = lab["y_" + task], z["pred"][..., list(z["heads"]).index(task)]
        ok = ~np.isnan(y)
        mine = an.metric(met, y[ok], p[ok])
        d, name = an.REF[ref]
        ml = pd.read_csv(os.path.join(d, "metrics_long.csv"))
        saved = ml[(ml.model == name) & (ml.seed == 2) & (ml.task == task) & (ml.split == "cal") & (ml.metric == met)].value.iloc[0]
        assert abs(mine - saved) < 1e-6, (ref, task, mine, saved)


def test_paired_verdict_rules():
    import pandas as pd
    import analyze_p21 as an
    rows = []
    for s, (a, b) in enumerate([(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]):
        rows += [dict(family="f", disease="dep", metric="ap", model="kanode", seed=s, value=a), dict(family="f", disease="dep", metric="ap", model="kanode_noflow", seed=s, value=b + 0.01 * s)]
    P = an.paired(pd.DataFrame(rows), ["family", "disease", "metric"])
    r = P[(P.A == "kanode") & (P.B == "kanode_noflow")].iloc[0]
    assert r.A_better_n == 0 and r.verdict == "consistent"


if __name__ == "__main__":
    torch.set_num_threads(4)
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"{len(tests)} tests pass")
