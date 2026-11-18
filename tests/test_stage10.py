"""Tests for the Stage 10 (P0) analyses in extension_stage10/. Run: python tests/test_stage10.py   (reads saved predictions / processed data; trains nothing; test labels are only used by the one reproduction check)."""
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "extension_stage10"), os.path.join(ROOT, "src")]
import common as C  # noqa: E402
import conformal as cf  # noqa: E402
import p02_calibration as p2  # noqa: E402
import p03_ensemble as p3  # noqa: E402
import p04_oracle as p4  # noqa: E402


# ---------------------------------------------------------------- shared metrics
def test_ece_and_calibration():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    assert C.ece_equal_count(y, p) < 0.02                              # calibrated
    assert C.ece_equal_count(y, np.clip(p * 0.2, 0, 1)) > 0.2           # badly under-confident
    a, b = C.calib_slope_intercept(y, p)
    assert abs(a) < 0.1 and abs(b - 1) < 0.1
    assert all(np.isnan(C.calib_slope_intercept(np.zeros(10), np.full(10, 0.2))))


def test_group_masks_partition_participants():
    d = C.load_split("test")
    m = C.group_masks(d)
    assert m["sex=F"].sum() + m["sex=M"].sum() == len(d["pid"])
    assert sum(m[f"age_group={g}"].sum() for g in range(3)) == len(d["pid"])
    assert m["age_group=2"].sum() < C.MIN_N                           # the small group must be flagged UNRELIABLE


# ---------------------------------------------------------------- P0-2 conformal sets
def test_quantile_matches_stage6_rule_and_inf_when_too_few():
    rng = np.random.default_rng(1)
    S = rng.random((3, 50))
    for alpha in (0.1, 0.05):
        q = p3.quantile_rows(S, alpha)
        assert np.allclose(q, [cf.conformal_q(r, alpha) for r in S])
    assert np.isinf(p3.quantile_rows(rng.random((2, 5)), 0.05)).all()   # ceil(6*0.95)=6 > 5
    P = rng.random((2, 30))
    Y = np.array([[1] + [0] * 29] * 2)                              # one positive: Mondrian positive-class quantile must be +inf at 90%
    q1, q0 = p2.class_quantiles(P, Y, 0.10, True)
    assert np.isinf(q1).all() and np.isfinite(q0).all()


def test_set_membership_and_eval_consistency():
    p = np.array([[0.9, 0.5, 0.05, 0.3]])
    y = np.array([[1, 0, 0, 1]])
    q1, q0 = np.array([0.4]), np.array([0.6])
    i1, i0 = p2.set_membership(p, q1[:, None], q0[:, None])
    assert i1.tolist() == [[True, False, False, False]]                  # 1 in set iff 1-p <= 0.4
    assert i0.tolist() == [[False, True, True, True]]                    # 0 in set iff p <= 0.6
    e = p2.eval_sets(p, y, q1, q0)
    assert abs(e["coverage"][0] - 0.75) < 1e-12 and abs(e["set_size"][0] - 1.0) < 1e-12 and abs(e["cov_pos"][0] - 0.5) < 1e-12


def test_conformal_coverage_guarantee_on_exchangeable_data():
    rng = np.random.default_rng(2)
    covs_m, covs_pos = [], []
    for _ in range(300):
        p = rng.beta(0.6, 8, 400)
        y = (rng.random(400) < p).astype(float)
        pc, yc, pt, yt = p[:200][None], y[:200][None], p[200:], y[200:]
        q1, q0 = p2.class_quantiles(pc, yc, 0.10, False)
        covs_m.append(p2.eval_sets(pt[None], yt[None], q1, q0)["coverage"][0])
        q1, q0 = p2.class_quantiles(pc, yc, 0.10, True)
        e = p2.eval_sets(pt[None], yt[None], q1, q0)
        if not np.isnan(e["cov_pos"][0]):
            covs_pos.append(e["cov_pos"][0])
    assert np.mean(covs_m) >= 0.89                                      # marginal guarantee (finite-sample, on average)
    assert np.mean(covs_pos) >= 0.88                                    # class-conditional guarantee for positives


def test_quantiles_never_use_test_labels():
    rng = np.random.default_rng(3)
    pc, yc = rng.random((4, 80)), (rng.random((4, 80)) < 0.1).astype(float)
    a = p2.class_quantiles(pc, yc, 0.1, True)
    b = p2.class_quantiles(pc.copy(), yc.copy(), 0.1, True)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))              # function has no test input at all (signature check below)
    import inspect
    assert list(inspect.signature(p2.class_quantiles).parameters) == ["P", "Y", "alpha", "mondrian"]


def test_origin_draws_use_only_eligible_origins():
    Y = np.full((5, 120), np.nan)
    Y[0, 30:40] = 1
    Y[1, 100:] = 0                                                      # outside 27..98 -> participant excluded
    Y[2, 27:99] = 0
    ids, T = p2.draw_origins(Y, np.random.default_rng(0), 50)
    assert ids.tolist() == [0, 2] and T.shape == (50, 2)
    assert ((T[:, 0] >= 30) & (T[:, 0] < 40)).all() and ((T[:, 1] >= 27) & (T[:, 1] <= 98)).all()


# ---------------------------------------------------------------- P0-3 ensemble
def test_winkler_and_ensemble_std():
    assert abs(p3.winkler(5.0, 4.0, 6.0, 0.1) - 2.0) < 1e-12              # covered: width only
    assert abs(p3.winkler(8.0, 4.0, 6.0, 0.1) - (2 + 20 * 2)) < 1e-12     # missed above by 2
    P = np.random.default_rng(0).normal(size=(5, 3, 4))
    m, s = p3.ensemble(P)
    assert np.allclose(m, P.mean(0)) and np.allclose(s, P.std(0, ddof=1))


def test_normalised_interval_reduces_to_constant_when_sigma_constant():
    rng = np.random.default_rng(4)
    mu, y = rng.normal(size=(3, 200)), rng.normal(size=(3, 200))
    sd = np.ones((3, 200))
    q0, lo0, hi0 = p3.one_method(mu, sd, y, mu, sd, y, 0.1, lambda s: np.ones_like(s))
    q1, lo1, hi1 = p3.one_method(mu, sd, y, mu, sd, y, 0.1, lambda s: s + 0.0)
    assert np.allclose(lo0, lo1) and np.allclose(hi0, hi1)


def test_single_seed_reproduces_stage6_quantile():
    s6 = pd.read_csv("results/stage6_conformal/coverage_width_by_seed.csv")
    r = s6[(s6.model == "gru") & (s6.seed == 0) & (s6.task == "sev_dep_t0") & (s6.level == "90%")].iloc[0]
    cal, tst = C.load_split("cal"), C.load_split("test")
    rng = np.random.default_rng(cf.ORIGIN_SEED)
    t_cal = cf.draw_origins(len(cal["pid"]), rng)
    P, _ = C.load_pred("gru", "sev_dep_t0", 0, "cal")
    a = np.arange(len(cal["pid"]))
    q = p3.quantile_rows(np.abs(cal["y_sev_dep_t0"][a, t_cal] - P[a, t_cal])[None, :], 0.10)[0]
    assert abs(q - r.q) < 1e-9


# ---------------------------------------------------------------- P0-4 oracle
def _fake_truth(rng, N=4, T=40):
    return dict(z_dep=rng.normal(size=(N, T)), z_anx=rng.normal(size=(N, T)), stress=rng.normal(size=(N, T)), u=rng.normal(size=(N, T, 8)), traits=rng.normal(size=(N, 6)))


def test_oracle_features_use_only_past_days():
    rng = np.random.default_rng(5)
    tr, static = _fake_truth(rng), rng.random((4, 4))
    for tier in ("A", "B"):
        base = p4.build_features(tr, static, tier, (1.0, 1.0))
        t0 = 25
        pert = {k: (v.copy() if v.ndim > 1 else v) for k, v in tr.items()}
        for k in ("z_dep", "z_anx", "stress"):
            pert[k][:, t0 + 1:] = rng.normal(size=pert[k][:, t0 + 1:].shape) * 50
        pert["u"][:, t0 + 1:] = rng.normal(size=pert["u"][:, t0 + 1:].shape) * 50
        alt = p4.build_features(pert, static, tier, (1.0, 1.0))
        assert np.allclose(base[:, :t0 + 1], alt[:, :t0 + 1], equal_nan=True), tier          # features at days <= t0 unchanged by anything after t0
        assert not np.allclose(base[:, t0 + 1:], alt[:, t0 + 1:], equal_nan=True)            # sanity: later days do change


def test_oracle_excludes_outcome_derived_columns():
    assert not any(("onset" in c) or ("prevalent" in c) for c in p4.TRAITS)
    src = open(os.path.join(ROOT, "extension_stage10", "p04_oracle.py"), encoding="utf-8").read()
    assert "onset_day" not in src.split("def load_truth")[1].split("def fit_predict")[0]       # loader never selects outcome-derived columns


def test_run_length_and_shift():
    f = np.array([[0, 1, 1, 0, 1, 1, 1]], float)
    assert p4.run_length(f)[0].tolist() == [0, 1, 2, 0, 1, 2, 3]
    a = np.arange(5, dtype=float)[None, :, None]
    s = p4.shift(a, 2)[0, :, 0]
    assert np.isnan(s[:2]).all() and s[2:].tolist() == [0, 1, 2]


def test_oracle_model_selection_ignores_test_inputs():
    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(600, 5)), None
    y = X[:, 0] * 2 + rng.normal(size=600)
    Xc, yc = rng.normal(size=(300, 5)), None
    yc = Xc[:, 0] * 2 + rng.normal(size=300)
    (_, it1) = p4.fit_predict("sev_dep_t0", X, y, Xc, yc, [rng.normal(size=(10, 5))])
    (_, it2) = p4.fit_predict("sev_dep_t0", X, y, Xc, yc, [rng.normal(size=(10, 5)) * 100])
    assert it1 == it2 and it1 in p4.GRID


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
