"""Tests for P1-4 (model-based lag recovery). Run: python tests/test_p14.py. Calibration split only; checkpoints are only read (hash-checked)."""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "extension_p14")]
import common_p14 as C  # noqa: E402
import run_p14 as R  # noqa: E402

C.register()
bl = C.bl


def test_analytic_truth_has_zero_prelag_and_increment_at_planted_lag():
    for (f, k), L in C.PLANTED.items():
        P = C.truth_level(2.0, L)
        assert np.all(P[:L] == 0) and P[L] == 2.0 and np.all(np.diff(P[L:]) < 0)              # impulse enters at lag L, then decays at (1-a)
        D = C.increments(P)
        assert int(np.argmax(np.abs(D))) == L and abs(D[L] - 2.0) < 1e-12 and np.all(D[L + 1:] < 0)
        m = C.lag_metrics(P, L, 1)
        assert m["peak_lag_increment"] == L and m["peak_lag_level"] == L and m["prelag_mass"] == 0 and m["spike_ratio"] > 3
        assert m["onset_lag50"] == L and m["peak_lag_increment_1to5"] == L
    assert np.allclose(C.truth_ar(1.0, 2, 0.0), C.truth_level(1.0, 2))                       # no persistence -> single-day truth
    assert C.truth_ar(1.0, 2, 0.8)[3] > C.truth_level(1.0, 2)[3]                              # persistence spreads the effect


def test_lag_metrics_do_not_equate_largest_response_with_planted_lag():
    P = 1.0 * 0.8 ** np.arange(11)                                                            # reverse-path / autocorrelation-like profile: largest at lag 0, decaying
    m = C.lag_metrics(P, 3, +1)
    assert m["peak_lag_level"] == 0 and m["peak_lag_increment"] == 0 and m["prelag_mass"] > 0.3 and m["spike_ratio"] < 3 and m["onset_lag50"] == 0
    flat = C.lag_metrics(np.ones(11), 3, +1)
    assert flat["prelag_mass"] > 0.2 and flat["spike_ratio"] < 1.5


def test_shift_touches_only_requested_day_channel_and_observed_days():
    x = torch.zeros(4, 28, 10)
    obs = torch.ones(4, 28, 10, dtype=torch.bool)
    obs[1, 25, 6] = False
    w = np.zeros(28)
    w[25] = 1.0
    xx = R.shifted(x, obs, [(6, w, 2.0)])
    assert xx[0, 25, 6] == 2.0 and xx[1, 25, 6] == 0.0                                        # missing day not perturbed
    assert float(xx.abs().sum()) == 2.0 * 3                                                   # only 3 origins x that one day x that one channel changed
    w2 = np.zeros(28)
    w2[25:] = 0.5 ** np.arange(3)
    assert torch.allclose(R.shifted(x, obs, [(0, w2, 1.0)])[0, 25:, 0], torch.tensor([1.0, 0.5, 0.25]))


class _DayModel(torch.nn.Module):
    """Toy predictor that depends ONLY on steps two days before the origin (lag 2), with weight -3; other outputs 0."""

    def forward(self, x, s):
        o = torch.zeros(len(x), 28)
        o[:, 0] = -3.0 * x[:, 27 - 2, 0]
        o[:, 4] = 1.0 * x[:, 27 - 1, 6]
        return o


def test_perturbation_profile_recovers_toy_lag_and_halves_average_to_overall():
    x = torch.randn(40, 28, 10)
    s, pid = torch.zeros(40, 4), np.arange(40)
    obs = torch.ones_like(x, dtype=torch.bool)
    m, ym, ys = _DayModel(), torch.zeros(8), torch.ones(8)
    idx = torch.tensor([0, 4])
    base = C.predict2(m, x, s, ym, ys, idx)
    prof = []
    for j in range(6):
        w = np.zeros(28)
        w[27 - j] = 1.0
        h = R.halves_mean(C.predict2(m, R.shifted(x, obs, [(0, w, 1.0)]), s, ym, ys, idx) - base, pid)
        assert h.shape == (2, 3) and torch.allclose((h[:, 1] + h[:, 2]) / 2, h[:, 0], atol=1e-6)
        prof.append(float(h[0, 0]))
    assert int(np.argmax(np.abs(prof))) == 2 and abs(prof[2] + 3.0) < 1e-5 and sum(abs(p) for j, p in enumerate(prof) if j != 2) < 1e-6


def test_integrated_gradients_completeness_and_lag_orientation():
    x = torch.randn(30, 28, 10)
    s = torch.zeros(30, 4)
    ig_s, ig_a, comp = R.integrated_gradients(_DayModel(), x, s, torch.zeros(8), torch.ones(8), torch.tensor([0, 4]))
    assert comp < 1e-4                                                                        # sum of attributions == f(x) - f(baseline)
    assert int(np.argmax(ig_a[0, :, 0])) == 2 and ig_a[0, :, 0].argmax() == 2 and ig_a[0, :, 1:].sum() < 1e-6      # dep head: only steps at lag 2
    assert int(np.argmax(ig_a[1, :, 6])) == 1                                                  # anx head: only HRV at lag 1


def test_checkpoints_load_read_only_with_expected_heads_and_are_deterministic():
    path = "results/stage3/checkpoints/gru_seed0.pt"
    h0 = hashlib.sha256(open(path, "rb").read()).hexdigest()
    model, ym, ys, heads = C.load_model("gru", 0)
    assert len(heads) == 16 and C.head_index(heads).tolist() == [0, 1]
    m1, _, _, h1 = C.load_model("m1", 0)
    assert len(h1) == 28 and C.head_index(h1).tolist() == [0, 4]
    x = torch.randn(5, 28, 10)
    s = torch.zeros(5, 4)
    a = C.predict2(model, x, s, ym, ys, C.head_index(heads))
    assert torch.equal(a, C.predict2(model, x, s, ym, ys, C.head_index(heads)))
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == h0                           # nothing written back
    for label in C.SPECS:
        C.load_model(label, 0)                                                                 # all four architectures load (kangru compatible)


def test_data_are_calibration_only_and_stage5_benchmarks_consistent():
    bl.configure("all")
    d = bl.load_data("data/processed_v1", None, splits=("cal",))
    assert list(d) == ["cal"]
    x, s, obs, pid, dd = C.load_cal_windows(50)
    assert x.shape == (50, 28, 10) and s.shape == (50, 4) and obs.shape == x.shape
    assert not torch.isnan(x).any() and (x[..., 8].mean() < 0.5)
    for key, b in C.benchmarks().items():
        assert np.sign(b) == C.SIGN[key]
    rho = C.estimate_ar1(dd)
    assert rho.shape == (8,) and np.all((rho > 0) & (rho < 1))
    assert all(C.std_per_unit(c) > 0.5 for c in ("steps", "sleep_variability", "hrv"))


def test_saved_results_reproduce_for_one_model_seed():
    p = "results/p1_lag_recovery/raw/gru_seed0.npz"
    if not os.path.exists(p):
        return
    saved = np.load(p)
    x, s, obs, pid, d = C.load_cal_windows(1000)
    torch.set_num_threads(4)
    r = R.run_one("gru", 0, False, x, s, obs, pid, C.estimate_ar1(d), False)
    assert np.allclose(r["pert"], saved["pert"], atol=1e-5) and np.allclose(r["ig_abs"], saved["ig_abs"], atol=1e-5)


def test_runner_refuses_nonempty_output_directory():
    with tempfile.TemporaryDirectory() as dname:
        open(os.path.join(dname, "x"), "w").close()
        r = subprocess.run([sys.executable, "extension_p14/run_p14.py", "--out", dname, "--quick"], capture_output=True, text=True)
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
