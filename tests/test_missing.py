"""Tests for Stage 8 masking pipeline (extension_missing/inject.py). Run: python tests/test_missing.py"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "extension_missing"))
os.chdir(ROOT)
import inject as ij  # noqa: E402
import preprocess as pp  # noqa: E402


def load(split):
    return np.load(f"data/processed_v1/{split}.npz")["X"]


def origins(N):
    pi, ti = np.meshgrid(np.arange(N), np.arange(27, 120), indexing="ij")
    return pi.ravel(), ti.ravel()


def test_identity_reconstruction_all_splits():
    for s in ("train", "cal", "test"):
        X = load(s)
        R = ij.reconstruct_raw(X)
        assert np.array_equal(ij.rebuild(R, ij.honest_masks(R)), X), s            # bit-identical to the frozen processed array
        assert np.array_equal(ij.honest_masks(R), X[:, :, 8:10].astype(float)), s    # mask semantics reproduce frozen masks


def test_control_and_zero_windows():
    X = load("cal")
    pi, ti = origins(len(X))
    R = ij.reconstruct_raw(X)
    W0 = ij.windows(X, pi, ti)
    assert np.array_equal(ij.series_windows(R, X, pi, ti, None, "H", None), W0)
    assert np.array_equal(ij.trailing_windows(R, X, pi, ti, 0, "H"), W0)           # k=0 trailing path == frozen windows (exact ffill carry-in)
    assert np.array_equal(ij.trailing_windows(R, X, pi, ti, 0, "O"), W0)
    Z = ij.series_windows(R, X, pi, ti, None, "Z", None)
    assert (Z[:, :, 8:] == 0).all() and np.array_equal(Z[:, :, :8], W0[:, :, :8])


def test_injection_only_removes_and_respects_original_gaps():
    X = load("cal")
    R = ij.reconstruct_raw(X)
    spec = dict(scope="all", rate=0.4, block=1)
    Rg = ij.inject_series(R, spec, np.random.default_rng(0))
    assert np.isnan(Rg)[np.isnan(R)].all()                                          # original gaps stay gaps
    assert np.array_equal(Rg[~np.isnan(Rg)], R[~np.isnan(Rg)])                      # surviving values untouched
    obs = ~np.isnan(R[:, :, 0])
    assert abs((np.isnan(Rg[:, :, 0]) & obs).sum() / obs.sum() - 0.4) < 0.02        # added rate on observed days ~ target


def test_block_structure_and_rate():
    d = ij.drop_mask((2000, 120), 0.4, 7, np.random.default_rng(1))
    assert abs(d.mean() - 0.4) < 0.02
    runs = []
    for row in d[:300]:
        r = 0
        for v in row:
            if v: r += 1
            elif r: runs.append(r); r = 0
    assert np.mean(runs) > 7                                                       # blocks merge but are at least 7 long on average
    r = ij.drop_mask((500, 120), 0.4, 1, np.random.default_rng(2))
    assert abs(r.mean() - 0.4) < 0.02 and ij.drop_mask((5, 10), 1.0, 1, None).all()


def test_group_scopes():
    X = load("cal")
    R = ij.reconstruct_raw(X)
    Rg = ij.inject_series(R, dict(scope="group", group="sleep", rate=1.0, block=1), np.random.default_rng(0))
    assert np.isnan(Rg[:, :, 2:5]).all() and np.array_equal(np.isnan(Rg[:, :, [0, 1, 5, 7]]), np.isnan(R[:, :, [0, 1, 5, 7]]))
    m = ij.honest_masks(Rg)
    assert (m[:, :, 0] == X[:, :, 8]).all()                                         # missing_day unchanged: other signals still observed
    Rc = ij.inject_series(R, dict(scope="group", group="cardiac", rate=1.0, block=1), np.random.default_rng(0))
    assert (ij.honest_masks(Rc)[:, :, 1] == 1).all()                                # HRV gap is flagged by the frozen semantics
    Rm = ij.inject_series(R, dict(scope="multi", rate=0.4, block=1), np.random.default_rng(0))
    assert len({round(float(np.isnan(Rm[:, :, j]).mean()), 3) for j in (0, 2, 5, 7)}) > 1     # groups drawn independently


def test_trailing_outage_causal_and_exact():
    X = load("cal")
    pi, ti = origins(len(X))
    R = ij.reconstruct_raw(X)
    k = 7
    W = ij.trailing_windows(R, X, pi, ti, k, "H")
    W0 = ij.windows(X, pi, ti)
    assert np.array_equal(W[:, :28 - k, :], W0[:, :28 - k, :])                       # days before the outage are bit-identical (causal)
    assert (W[:, -k:, 8] == 1).all()                                                # outage flagged under 'H'
    Wo = ij.trailing_windows(R, X, pi, ti, k, "O")
    assert np.array_equal(Wo[:, :, 8:], W0[:, :, 8:])                               # 'O' keeps original flags
    # values on outage days: forward-filled for <=3 days from the last observation, then 0
    i = np.where((W0[:, -k - 1, 8] == 0) & (W0[:, -k - 1, 9] == 0))[0][:50]
    for j in i:
        assert np.allclose(W[j, -k:-k + 3, :6], W0[j, -k - 1, :6]) and (W[j, -k + 3:, :6] == 0).all()


def test_no_future_leak_series_scenarios():
    X = load("cal")
    pi, ti = origins(len(X))
    R = ij.reconstruct_raw(X)
    spec = dict(scope="all", rate=0.4, block=7)
    A = ij.series_windows(R, X, pi, ti, spec, "H", np.random.default_rng(5))
    R2 = R.copy()
    R2[:, 100:] = R2[:, 100:] * 0 + 7.0                                            # change the future (days >= 100)
    B = ij.series_windows(R2, X, pi, ti, spec, "H", np.random.default_rng(5))
    sel = ti < 100
    assert np.array_equal(A[sel], B[sel])                                          # windows ending before day 100 cannot see it


def test_registry_and_determinism():
    sc = ij.registry()
    assert len([s for s in sc if not s["id"].endswith("_O") and s["id"] != "Z0"]) == 28 and len(sc) == 35
    ids = [s["id"] for s in sc]
    assert len(set(ids)) == len(ids)
    a = ij.rng_for("S1_r20", 0, "cal").random(3)
    assert np.array_equal(a, ij.rng_for("S1_r20", 0, "cal").random(3)) and not np.array_equal(a, ij.rng_for("S1_r20", 1, "cal").random(3))
    assert not np.array_equal(a, ij.rng_for("S1_r20", 0, "test").random(3))
    s = {x["id"]: x for x in sc}
    assert s["S1_r20_O"]["base"] == "S1_r20" and s["S1_r20_O"]["spec"] == s["S1_r20"]["spec"]       # O shares draw with its base
    X = load("cal")
    pi, ti = origins(len(X))
    R = ij.reconstruct_raw(X)
    WH = ij.make_windows(s["S1_r20"], 0, "cal", R, X, pi, ti)
    WO = ij.make_windows(s["S1_r20_O"], 0, "cal", R, X, pi, ti)
    assert np.array_equal(WH[:, :, :8], WO[:, :, :8]) and np.array_equal(WO[:, :, 8:], ij.windows(X, pi, ti)[:, :, 8:])


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
