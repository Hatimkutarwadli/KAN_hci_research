"""Tests for the Stage 9 v2 pipeline (extension_v2/). Run: python tests/test_v2.py   (train and calibration data only; the test split is never loaded)."""
import inspect
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing"), os.path.join(ROOT, "extension_v2")]
os.chdir(ROOT)
import baselines as bl  # noqa: E402
import data as dd  # noqa: E402
import inject as ij  # noqa: E402
import train_v2 as tv  # noqa: E402
import v2  # noqa: E402

OFF = np.arange(-27, 1)
_cache = {}


def split(name):
    if name not in _cache:
        d = dd.load_split(name)
        R = ij.reconstruct_raw(d["X"])
        _cache[name] = (d, R, v2.pad_raw(R))
    return _cache[name]


def origins(N, n, seed):
    r = np.random.default_rng(seed)
    return r.integers(0, N, n), r.integers(27, 120, n)


def test_identity_clean_windows():
    for name in ("train", "cal"):
        d, R, Rp = split(name)
        pi, ti = origins(len(d["pid"]), 4000, 1)
        frozen = d["X"][pi[:, None], ti[:, None] + OFF]
        mine = v2.represent(v2.slice31(Rp, pi, ti))
        assert mine.dtype == frozen.dtype and np.array_equal(mine, frozen), name


def test_identity_stage8_scenarios():
    d, R, Rp = split("cal")
    pi, ti = origins(len(d["pid"]), 3000, 2)
    reg = {s["id"]: s for s in ij.registry()}
    for sid in ("S1_r20", "S2_r20_L7", "S4_sleep_rand40", "S4_cardiac_absent", "S5_blk40"):
        sc = reg[sid]
        W8 = ij.make_windows(sc, 0, "cal", R, d["X"], pi, ti)
        Rg = ij.inject_series(R, sc["spec"], ij.rng_for(sc["base"], 0, "cal"))
        assert np.array_equal(W8, v2.represent(v2.slice31(v2.pad_raw(Rg), pi, ti))), sid
    for k in (3, 7, 14):
        S = v2.slice31(Rp, pi, ti)
        S[:, v2.L31 - k:, :] = np.nan
        assert np.array_equal(ij.make_windows(reg[f"S3_k{k}"], 0, "cal", R, d["X"], pi, ti), v2.represent(S)), k


def test_causality():
    d, R, Rp = split("cal")
    pi, ti = origins(len(d["pid"]), 500, 3)
    S = v2.slice31(Rp, pi, ti)
    S = v2.augment(S, np.random.default_rng(0), 1.0)
    base = v2.represent(S)
    rr = np.random.default_rng(4)
    for tau in (0, 5, 13, 20, 26):
        S2 = S.copy()
        S2[:, 3 + tau + 1:] = np.where(rr.random(S2[:, 3 + tau + 1:].shape) < 0.5, np.nan, rr.normal(size=S2[:, 3 + tau + 1:].shape))
        assert np.array_equal(v2.represent(S2)[:, :tau + 1], base[:, :tau + 1]), tau


def test_augmentation_adds_only_and_flags_are_honest():
    d, R, Rp = split("train")
    pi, ti = origins(len(d["pid"]), 3000, 5)
    S = v2.slice31(Rp, pi, ti)
    A = v2.augment(S, np.random.default_rng(1))
    assert (np.isnan(A)[np.isnan(S)]).all()                                # original gaps never filled
    assert np.array_equal(A[~np.isnan(A)], S[~np.isnan(A)])                  # observed values never altered
    assert np.isnan(A).sum() > np.isnan(S).sum()
    W = v2.represent(A)
    assert np.array_equal(W[:, :, 8:10], ij.honest_masks(A[:, 3:]).astype(np.float32))
    assert np.array_equal(W[:, :, 8] == 1, np.isnan(A[:, 3:]).all(-1))       # missing_day <=> all 8 signals missing
    assert np.array_equal(W[:, :, 9] == 1, np.isnan(A[:, 3:, 6]))            # hrv_missing <=> HRV missing
    assert not np.isnan(W).any()


def test_augmentation_independent_of_data_labels_and_ids():
    params = list(inspect.signature(v2.gap_mask).parameters)
    assert params == ["B", "rng", "p_aug", "rmax", "L"], params                # no label / latent / pid argument exists
    assert list(inspect.signature(v2.augment).parameters) == ["S", "rng", "p_aug", "rmax"]
    a = v2.gap_mask(500, np.random.default_rng(9))
    b = v2.gap_mask(500, np.random.default_rng(9))
    assert np.array_equal(a, b)
    S1 = np.random.default_rng(0).normal(size=(500, 31, 8))
    S2 = np.random.default_rng(1).normal(size=(500, 31, 8)) * 5
    n1 = np.isnan(v2.augment(S1, np.random.default_rng(7)))
    n2 = np.isnan(v2.augment(S2, np.random.default_rng(7)))
    assert np.array_equal(n1, n2)                                             # identical gaps whatever the data


def test_augmentation_distribution():
    m = v2.gap_mask(40000, np.random.default_rng(11))
    on = m.any((1, 2))
    assert 0.3 < on.mean() <= 0.5 + 0.01                                      # at most P_AUG of windows get any gap
    assert m.mean() < 0.15
    covered_frac = m.all(2).mean(1)
    assert (covered_frac == 1.0).mean() < 0.001                               # a fully empty window is a rare tail event of heavy block gaps (12/40000 observed)
    assert m[:, -14:, :].all((1, 2)).mean() < 0.03                            # no explicit trailing-outage generator
    whole = m.all(2).any(1)
    sg =[(m[:, :, SIG].any((1, 2)) & ~m[:, :, [j for j in range(8) if j not in SIG]].any((1, 2))).sum() for SIG in ([0, 1], [2, 3, 4], [5, 6], [7])]
    assert min(sg) > 100                                                      # single-group gaps occur for every group
    assert whole.sum() > 1000                                                 # whole-day gaps occur
    assert np.array_equal(v2.gap_mask(100, np.random.default_rng(0), p_aug=0.0), np.zeros((100, 31, 8), bool))


def test_train_only_no_test_access():
    pids = {s: set(np.load(f"data/processed_v1/{s}.npz")["pid"].tolist()) for s in ("train", "cal", "test")}
    assert not (pids["train"] & pids["cal"]) and not (pids["train"] & pids["test"]) and not (pids["cal"] & pids["test"])
    bl.configure("main")
    data = bl.load_data("data/processed_v1", {"train": 20, "cal": 10, "test": 10}, splits=("train", "cal"))
    assert set(data) == {"train", "cal"} and set(data["train"]["pid"].tolist()) <= pids["train"]
    for fn in ("train_v2.py",):
        assert '"test"' not in open(os.path.join("extension_v2", fn)).read(), fn
    r = subprocess.run([sys.executable, "extension_v2/eval_v2.py", "--split", "test", "--config", "R0O0T1"], capture_output=True, text=True)
    assert r.returncode != 0 and "locked" in (r.stdout + r.stderr)


def test_p0_reproduces_frozen_fit_nn():
    bl.configure("main")
    torch.set_num_threads(bl.THREADS)
    data = bl.load_data("data/processed_v1", {"train": 60, "cal": 20, "test": 20}, splits=("train", "cal"))
    m0, ym0, ys0, _ = bl.fit_nn("gru", 0, data["train"], data["cal"], epochs=2, log=lambda *_: None)
    raw = ij.reconstruct_raw(data["train"]["X"])
    m1, ym1, ys1, _ = tv.fit_aug(0, data["train"], data["cal"], raw, p_aug=0.0, epochs=2, log=lambda *_: None)
    for k, v in m0.state_dict().items():
        assert torch.allclose(v, m1.state_dict()[k], atol=1e-6), k
    assert torch.allclose(ym0, ym1) and torch.allclose(ys0, ys1)
    m2, *_ = tv.fit_aug(0, data["train"], data["cal"], raw, p_aug=0.5, epochs=2, log=lambda *_: None)
    assert any(not torch.allclose(v, m2.state_dict()[k], atol=1e-6) for k, v in m0.state_dict().items())   # augmentation actually changes training


def test_p0_matches_saved_stage3_epoch1():
    """full train data, seed 0, 1 epoch: train/cal loss equal the logged Stage 3 GRU epoch 1 (train 0.2931, cal 0.2220)."""
    bl.configure("main")
    torch.set_num_threads(bl.THREADS)
    data = bl.load_data("data/processed_v1", splits=("train", "cal"))
    raw = ij.reconstruct_raw(data["train"]["X"])
    _, _, _, info = tv.fit_aug(0, data["train"], data["cal"], raw, p_aug=0.0, epochs=1, log=lambda *_: None)
    h = info["history"][0]
    assert abs(h["train_loss"] - 0.2931) < 6e-4 and abs(h["cal_loss"] - 0.2220) < 6e-4, h


CONFIGS = [(r, o) for r in ("R0", "R1") for o in ("O0", "O1", "O2", "O3")]


def test_all_registry_scenarios_match_stage8_for_R0O0():
    d, R, Rp = split("cal")
    pi, ti = origins(len(d["pid"]), 2000, 6)
    for sc in ij.registry():
        for rep in sc["reps"]:
            S, So = v2.scenario_slices(sc, rep, "cal", R, Rp, pi, ti)
            assert np.array_equal(v2.represent(S, "R0", "O0", sc["flags"], So), ij.make_windows(sc, rep, "cal", R, d["X"], pi, ti)), (sc["id"], rep)


def test_fill_rules_hand_example():
    S = np.full((1, 31, 8), np.nan)
    S[0, 5], S[0, 6] = 2.0, 4.0
    idx = [4, 5, 6, 7, 9, 10, 12, 30]
    exp = {"O0": [0, 2, 4, 4, 4, 0, 0, 0], "O1": [0, 2, 4, 4, 4, 4, 4, 4], "O3": [0, 2, 4, 4, 4, 3, 3, 3],
           "O2": [0, 2, 4, 4 * np.exp(-1 / 7), 4 * np.exp(-3 / 7), 4 * np.exp(-4 / 7), 4 * np.exp(-6 / 7), 4 * np.exp(-24 / 7)]}
    for o, e in exp.items():
        assert np.allclose(v2.fill(S, o)[0, idx, 0], e, atol=1e-9), o


def test_fills_keep_observed_values_and_agree_with_frozen_inside_3_days():
    d, R, Rp = split("train")
    pi, ti = origins(len(d["pid"]), 3000, 7)
    S = v2.augment(v2.slice31(Rp, pi, ti), np.random.default_rng(3), 1.0)
    f0 = v2.fill(S, "O0")
    _, Ag = v2._last_age(S)
    obs = ~np.isnan(S)
    for o in ("O1", "O2", "O3"):
        f = v2.fill(S, o)
        assert np.array_equal(f[obs], S[obs]), o
        assert not np.isnan(f).any()
        if o in ("O1", "O3"):
            assert np.array_equal(f[Ag <= 3], f0[Ag <= 3]), o                 # identical to the frozen fill while the forward-fill is still active
    f1, f3 = v2.fill(S, "O1"), v2.fill(S, "O3")
    assert np.array_equal(f1[(Ag > 3) & np.isfinite(Ag)], v2._last_age(S)[0][(Ag > 3) & np.isfinite(Ag)])
    assert (f3[Ag > 3] != 0).any() and (f0[Ag > 3] == 0).all()


def test_causality_all_configs():
    d, R, Rp = split("cal")
    pi, ti = origins(len(d["pid"]), 400, 8)
    S = v2.augment(v2.slice31(Rp, pi, ti), np.random.default_rng(1), 1.0)
    rr = np.random.default_rng(5)
    for r, o in CONFIGS:
        base = v2.represent(S, r, o)
        for tau in (2, 10, 22):
            S2 = S.copy()
            S2[:, 3 + tau + 1:] = np.where(rr.random(S2[:, 3 + tau + 1:].shape) < 0.5, np.nan, rr.normal(size=S2[:, 3 + tau + 1:].shape))
            assert np.array_equal(v2.represent(S2, r, o)[:, :tau + 1], base[:, :tau + 1]), (r, o, tau)


def test_R1_per_signal_flags():
    d, R, Rp = split("train")
    pi, ti = origins(len(d["pid"]), 3000, 9)
    W0 = d["X"][pi[:, None], ti[:, None] + OFF]
    S = v2.slice31(Rp, pi, ti)
    W1 = v2.represent(S, "R1", "O0")
    assert W1.shape[2] == 16 == v2.n_channels("R1") and np.array_equal(W1[:, :, :8], W0[:, :, :8])
    assert np.array_equal(W1[:, :, 8:], np.isnan(S[:, 3:]).astype(np.float32))
    for j in range(8):                                                         # on the frozen (clean) data per-signal flags are redundant copies of the 2 frozen flags
        assert np.array_equal(W1[:, :, 8 + j], W0[:, :, 9] if j == 6 else W0[:, :, 8]), j
    A = v2.augment(S, np.random.default_rng(2), 1.0)
    assert np.array_equal(v2.represent(A, "R1", "O2")[:, :, 8:], np.isnan(A[:, 3:]).astype(np.float32))   # honest per-signal flags under augmentation
    So = v2.represent(A, "R1", "O0", "O", S)                                    # 'original flags kept': flags come from the clean slice, values from the gapped one
    assert np.array_equal(So[:, :, 8:], W1[:, :, 8:]) and not np.array_equal(So[:, :, :8], W1[:, :, :8])
    assert not v2.represent(A, "R1", "O1", "Z", S)[:, :, 8:].any()


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n, flush=True)
