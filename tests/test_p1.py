"""Tests for the P1-1 baselines (extension_p1/). Run: python tests/test_p1.py. Small subsets only; trains for 1-2 epochs on 80 participants; the test split is never used for any decision."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p1")]
import baselines as bl  # noqa: E402
import models as md  # noqa: E402
import models_p1 as mp  # noqa: E402

FROZEN = set(md.MODELS)          # registry before our (runtime-only) registration
mp.register()


def test_registry_extends_without_changing_frozen_models():
    assert FROZEN == {"kan", "gru", "m1", "m1_static"}
    assert {"transformer", "cnnlstm"} <= set(md.MODELS) and set(md.MODELS) - {"transformer", "cnnlstm"} == FROZEN


def test_parameter_budget_close_to_gru():
    for n_out in (16, 12):
        g = md.n_params(md.GRUNet(n_out=n_out))
        for cls in (mp.TransformerNet, mp.CNNLSTMNet):
            n = md.n_params(cls(n_out=n_out))
            assert abs(n - g) / g < 0.10, (cls.__name__, n, g)


def test_shapes_finite_and_deterministic_forward():
    x, s = torch.randn(5, 28, 10), torch.randn(5, 4)
    for name in ("transformer", "cnnlstm"):
        torch.manual_seed(0)
        m = md.MODELS[name](n_out=16).eval()
        y1, y2 = m(x, s), m(x, s)
        assert y1.shape == (5, 16) and torch.isfinite(y1).all() and torch.equal(y1, y2)


def test_cnn_is_causal():
    torch.manual_seed(0)
    m = mp.CNNLSTMNet().eval()
    x = torch.randn(3, 28, 10)
    x2 = x.clone()
    x2[:, 20:] += torch.randn(3, 8, 10)
    c = lambda z: F.relu(m.conv(F.pad(z.transpose(1, 2), (m.k - 1, 0))))  # noqa: E731
    assert torch.allclose(c(x)[:, :, :20], c(x2)[:, :, :20])                  # conv features at days < 20 unaffected by changes at days >= 20


def test_transformer_uses_day_order_and_last_day():
    torch.manual_seed(0)
    m = mp.TransformerNet().eval()
    x, s = torch.randn(4, 28, 10), torch.randn(4, 4)
    perm = torch.randperm(28)
    assert not torch.allclose(m(x, s), m(x[:, perm], s), atol=1e-5)             # positional information is used
    assert m.pe.shape[0] >= 28 and torch.allclose(m.pe[0, 0::2], torch.zeros(12))   # sin(0) = 0


def test_training_reproducible_and_seed_dependent():
    data = bl.load_data("data/processed_v1", {"train": 40, "cal": 20, "test": 10}, splits=("train", "cal"))
    bl.configure("main")
    for name in ("transformer", "cnnlstm"):
        m1, y1, s1, i1 = bl.fit_nn(name, 3, data["train"], data["cal"], epochs=2, log=lambda *_: None)
        m2, y2, s2, i2 = bl.fit_nn(name, 3, data["train"], data["cal"], epochs=2, log=lambda *_: None)
        m3, *_ = bl.fit_nn(name, 4, data["train"], data["cal"], epochs=2, log=lambda *_: None)
        assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
        assert i1["history"] == i2["history"]
        assert not all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m3.state_dict().values()))
        T = bl.to_t(data["cal"])
        assert np.array_equal(bl.predict_nn(m1, T, y1, s1), bl.predict_nn(m2, T, y2, s2))


def test_fit_signature_has_no_test_input_and_windows_end_at_origin():
    import inspect
    assert "test" not in inspect.signature(bl.fit_nn).parameters
    data = bl.load_data("data/processed_v1", {"train": 10, "cal": 10, "test": 10}, splits=("cal",))
    T = bl.to_t(data["cal"])
    idx = torch.arange(20)
    x, _, _ = bl.batch(T, idx)
    pi, ti = T["pi"][idx], T["ti"][idx]
    assert torch.equal(x[:, -1], T["X"][pi, ti]) and x.shape[1] == 28            # last window day = origin day


def test_runner_refuses_nonempty_output_directory():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "x"), "w").close()
        r = subprocess.run([sys.executable, "extension_p1/run_p1.py", "--out", d, "--quick"], capture_output=True, text=True)
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
