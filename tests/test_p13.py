"""Tests for P1-3 (matched-MLP control for KAN, interpretability). Run: python tests/test_p13.py. Uses train/calibration data only (the test split is never loaded here)."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p13")]
import baselines as bl  # noqa: E402
import models as md  # noqa: E402
import models_p13 as m13  # noqa: E402
import interpret_p13 as ip  # noqa: E402

m13.register()
SUB = {"train": 40, "cal": 20, "test": 10}


def _small(ts="main"):
    bl.configure(ts)
    return bl.load_data("data/processed_v1", SUB, splits=("train", "cal"))


def test_layer_parameter_matching():
    for n_in, n_out in [(30, 8), (50, 8), (70, 8), (52, 32), (24, 12), (16, 8), (12, 16), (8, 6)]:
        kan, mlp = md.n_params(md.KANLayer(n_in, n_out)), md.n_params(m13.MLPLayer(n_in, n_out))
        assert kan == 10 * n_in * n_out and abs(mlp - kan) / kan < 0.03, (n_in, n_out, kan, mlp)


def test_model_parameter_matching_and_wiring():
    b_k, b_m = md.MTLNetLite(n_out=16), m13.MTLNetMLP(n_out=16)
    assert abs(md.n_params(b_m) - md.n_params(b_k)) / md.n_params(b_k) < 0.005
    H = list(bl.ALL_HEADS)
    m_k, m_m = md.MODELS["m1"](n_out=28, heads=H), md.MODELS["mlp_m1"](n_out=28, heads=H)
    assert abs(md.n_params(m_m) - md.n_params(m_k)) / md.n_params(m_k) < 0.01
    assert torch.equal(m_k.gather_idx, m_m.gather_idx)                                  # same head routing
    x, s = torch.randn(3, 28, 10), torch.randn(3, 4)
    assert b_m(x, s).shape == b_k(x, s).shape == (3, 16) and m_m(x, s).shape == m_k(x, s).shape == (3, 28)
    assert b_m.head.weight.shape == b_k.head.weight.shape and b_m.norm.weight.shape == b_k.norm.weight.shape


def test_mlp_encoder_is_causal():
    torch.manual_seed(0)
    c = m13.CausalMLPConv(10, 5, 8).eval()
    x = torch.randn(2, 28, 10)
    x2 = x.clone()
    x2[:, 20:] += 5.0                                                                  # perturb the future only
    a, b = c(x), c(x2)
    assert torch.allclose(a[:, :20], b[:, :20]) and not torch.allclose(a[:, 20:], b[:, 20:])


def test_nospline_ablation_freezes_splines():
    data = _small()
    torch.manual_seed(3)
    a = md.MTLNetLite(n_out=16)
    torch.manual_seed(3)
    b = m13.MTLNetNoSpline(n_out=16)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and (torch.equal(va, vb) or "spline_weight" in ka)              # same init stream, only splines zeroed
    assert m13.n_trainable(b) < md.n_params(b) / 5
    model, *_ = bl.fit_nn("kan_nospline", 0, data["train"], data["cal"], epochs=1, log=lambda *_: None)
    assert all(float(m.spline_weight.abs().max()) == 0.0 for m in model.modules() if isinstance(m, md.KANLayer))


def test_mlp_training_deterministic_and_seed_dependent():
    data = _small()
    f = lambda sd: bl.fit_nn("mlp_b2", sd, data["train"], data["cal"], epochs=2, log=lambda *_: None)[0]      # noqa: E731
    a, b, c = f(1), f(1), f(2)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))


def test_m1_mlp_trains_and_evaluates_all_heads():
    data = _small("all")
    model, ym, ys, info = bl.fit_nn("mlp_m1", 0, data["train"], data["cal"], epochs=1, model_kwargs={"heads": list(bl.HEADS), "dropout": 0.3}, log=lambda *_: None)
    P = bl.predict_nn(model, bl.to_t(data["cal"]), ym, ys)
    assert P.shape[1] == 28 and np.isfinite(P).all() and ((P[:, 8:] >= 0) & (P[:, 8:] <= 1)).all()
    bl.configure("main")


def test_kan_edge_functions_reconstruct_layer_output():
    torch.manual_seed(0)
    layer = md.KANLayer(6, 4)
    conv = type("C", (), {})()
    conv.kan, conv.k = layer, 3
    model = type("M", (), {"convs": [conv]})()
    phi_all, grid = ip.kan_edges(model, grid=np.linspace(-3, 3, 61))
    phi = phi_all[3]                                                                    # (grid, o, C, k)
    x = torch.rand(5, 6) * 4 - 2
    xi = np.interp(x.numpy(), grid, np.arange(len(grid))).round().astype(int)           # nearest grid point approximation
    rec = np.zeros((5, 4))
    for i in range(6):
        c, j = divmod(i, 3)
        rec += phi[xi[:, i], :, c, j]
    g_step = grid[1] - grid[0]
    xs = torch.tensor(grid[xi], dtype=torch.float32)
    assert torch.allclose(layer(xs), torch.tensor(rec, dtype=torch.float32), atol=1e-4)    # sum of univariate edge functions == layer output (exact at grid points)
    assert g_step > 0


class _Toy(torch.nn.Module):
    """Known ground truth: sev_dep rises with tanh(sleep_variability), sev_anx falls with tanh(0.8*hrv); other outputs constant."""

    def forward(self, x, s):
        o = torch.zeros(len(x), 28)
        o[:, 0] = torch.tanh(x[..., 4]).mean(1)
        o[:, 4] = -torch.tanh(0.8 * x[..., 6]).mean(1)
        return o


def test_recovery_metrics_find_planted_structure_in_toy_model():
    import pandas as pd
    torch.manual_seed(0)
    x, s = torch.randn(300, 28, 10), torch.zeros(300, 4)
    obs = torch.ones(300, 28, 10, dtype=torch.bool)
    y = {h: np.zeros(300, np.float32) for h in ip.HEADS_OF_INTEREST}
    pdp, sens, lag = ip.analyse("toy", _Toy(), torch.zeros(8), torch.ones(8), list(bl.ALL_HEADS), x, s, y, obs, 0)
    rec = {r["head"]: r for r in ip.recovery(pd.DataFrame(pdp), pd.DataFrame(sens))}
    d = rec["sev_dep_t0"]
    assert d["top_channel"] == "sleep_variability"
    assert rec["sev_anx_t0"]["top_channel"] == "hrv"
    assert rec["sev_dep_t0"]["sign_acc"] == 0.5 and rec["sev_anx_t0"]["sign_acc"] == 0.5           # one of the two planted channels per head is learned with the right sign
    sd = pd.DataFrame(sens)
    r_sv = sd[(sd["head"] == "sev_dep_t0") & (sd.channel == "sleep_variability")].iloc[0]
    assert r_sv.slope_central > 0
    lagdf = pd.DataFrame(lag)
    e = lagdf[(lagdf["head"] == "sev_dep_t0") & (lagdf.channel == "sleep_variability")].set_index("lag").effect
    assert e.min() > 0 and np.allclose(e.values, e.values[0], rtol=0.3)                           # toy averages the window -> every lag contributes (similar sizes)


def test_runner_refuses_nonempty_output_directory():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "x"), "w").close()
        r = subprocess.run([sys.executable, "extension_p13/run_p13.py", "--models", "mlp_b2", "--out", d, "--quick"], capture_output=True, text=True)
        assert r.returncode != 0 and "refusing to overwrite" in (r.stdout + r.stderr)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
