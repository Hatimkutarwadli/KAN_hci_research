"""Tests for P1-6 (statistical analysis of Stage 9 robustness). Run: python tests/test_p16.py
FIRST test: regenerated metrics reproduce the existing Stage 8/9 CSVs. Existing results are only read; the manifest test checks nothing existing was touched."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "extension_p16"), os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing")]
import stats16 as st  # noqa: E402

OUT = "results/p1_stage9_stats"


def test_00_regenerated_metrics_reproduce_saved_stage8_and_stage9_csvs():
    import repro_check_p16 as rc
    rep = json.load(open(f"{OUT}/repro/repro_check.json"))
    assert rep["n_failures"] == 0 and rep["n_values_compared"] == 30600 and rep["max_abs_diff"] < rc.TOL
    # live re-check on two scenario files (full comparison of every task/metric/seed/config/part)
    import regen_p16 as rg
    import inject as ij
    meta = json.load(open(f"{OUT}/regen_meta.json"))
    d, pi, ti, *_ = rg.setup()
    Y = {h: d["y_" + h][pi, ti] for h in sorted({h for v in meta["heads"].values() for h in v})}
    reg = {s["id"]: s for s in ij.registry()}
    for sid, rp in (("S3_k7", 0), ("S5_blk40", 1)):
        pred = np.load(f"{OUT}/predictions/{sid}_rep{rp}.npz")
        for c in meta["configs"]:
            sv = rc.saved_metrics(c, sid, rp)
            for p in rg.PARTS:
                new = rc.recompute(pred, meta, Y, c, p, 2, reg[sid], rp)
                old = sv[p][sv[p].seed == 2]
                m = new.merge(old, on=["task", "metric"], suffixes=("_n", "_o"))
                assert len(m) == len(new) == len(old) and (m.value_n - m.value_o).abs().max() < rc.TOL


def test_regeneration_is_deterministic_forward_only():
    import regen_p16 as rg
    import eval_v2 as ev
    import inject as ij
    import run_missing as rm
    import v2
    import torch
    d, pi, ti, R, Rp, S_static = rg.setup()
    sc = {s["id"]: s for s in ij.registry()}["S2_r40_L7"]
    S, S_orig = v2.scenario_slices(sc, 1, "test", R, Rp, pi, ti)
    u = ev.load_units("R0O2T1", seeds=[3])[0]
    W = v2.represent(S, "R0", "O2", sc["flags"], S_orig)
    torch.set_num_threads(6)                                                                # regeneration used 6 threads; other thread counts differ only by float rounding (<= 2e-6)
    P = rm.predict(u, W, S_static).astype(np.float32)
    saved = np.load(f"{OUT}/predictions/S2_r40_L7_rep1.npz")["R0O2T1|main|3"]
    assert np.array_equal(P, saved)
    torch.set_num_threads(2)
    assert np.abs(rm.predict(u, W, S_static).astype(np.float32) - saved).max() < 1e-5
    #                                                      # bit-identical regeneration; no training involved (checkpoint only loaded)


def test_weighted_ap_auroc_equal_sklearn_on_expanded_samples_including_ties():
    rng = np.random.default_rng(0)
    N, n = 30, 600
    g = rng.integers(0, N, n)
    y = (rng.random(n) < 0.1 + 0.2 * (g % 3 == 0)).astype(int)
    s = np.round(rng.random(n) * (1 + y), 1)                                                 # coarse scores => many ties
    C = np.vstack([np.ones(N), rng.multinomial(N, np.ones(N) / N, 5)]).astype(float)
    ap, auc = st.weighted_ap_auroc(y, s, g, C, chunk=2)
    for k in range(len(C)):
        rows = np.repeat(np.arange(n), C[k][g].astype(int))
        assert abs(ap[k] - average_precision_score(y[rows], s[rows])) < 1e-12
        assert abs(auc[k] - roc_auc_score(y[rows], s[rows])) < 1e-12


def test_weighted_rmse_mean_and_degenerate_resamples():
    rng = np.random.default_rng(1)
    N, n = 20, 200
    g = rng.integers(0, N, n)
    e2 = rng.random(n)
    C = np.vstack([np.ones(N), rng.multinomial(N, np.ones(N) / N, 3)]).astype(float)
    r = st.weighted_rmse(e2, g, C, N)
    for k in range(len(C)):
        rows = np.repeat(np.arange(n), C[k][g].astype(int))
        assert abs(r[k] - np.sqrt(e2[rows].mean())) < 1e-12 and abs(st.weighted_mean(e2, g, C, N)[k] - e2[rows].mean()) < 1e-12
    y = np.zeros(n, int)                                                                      # no positives in any resample -> NaN, not an error
    ap, auc = st.weighted_ap_auroc(y, rng.random(n), g, C)
    assert np.isnan(ap).all() and np.isnan(auc).all()


def test_identity_row_reproduces_saved_stage9_metric_on_real_predictions():
    import metrics as mt
    import regen_p16 as rg
    d, pi, ti, *_ = rg.setup()
    pred = np.load(f"{OUT}/predictions/S1_r40_rep0.npz")
    heads = json.load(open(f"{OUT}/regen_meta.json"))["heads"]["R0O3T1|main"]
    j = heads.index("status_dep_t0")
    y, p = d["y_status_dep_t0"][pi, ti], pred["R0O3T1|main|0"][:, j].astype(float)
    ok = ~np.isnan(y)
    C = st.boot_counts(20, 200, 5)
    ap, auc = st.weighted_ap_auroc(y[ok].astype(int), p[ok], pi[ok], C)
    m = mt.clf_metrics(y[ok], p[ok])
    assert abs(ap[0] - m["ap"]) < 1e-10 and abs(auc[0] - m["auroc"]) < 1e-10
    saved = pd.read_csv("results/stage9_v2/test/metrics_R0O3T1/S1_r40_rep0.csv")
    v = saved[(saved.seed == 0) & (saved.task == "status_dep_t0") & (saved.metric == "ap")].value.iloc[0]
    assert abs(ap[0] - v) < 1e-5


def test_bootstrap_counts_shared_fixed_and_paired_self_contrast_is_zero():
    C1, C2 = st.boot_counts(50, 200, 7), st.boot_counts(50, 200, 7)
    assert np.array_equal(C1, C2) and (C1[0] == 1).all() and np.allclose(C1.sum(1), 200)
    assert not np.array_equal(C1, st.boot_counts(50, 200, 8))
    rng = np.random.default_rng(0)
    g = rng.integers(0, 200, 3000)
    e = rng.random(3000)
    a = st.weighted_rmse(e, g, C1, 200)
    assert np.all(st.worse_pct("sev_nowcast_rmse", a, a) == 0)                               # paired: a model against itself is exactly 0 in every resample
    assert st.worse_pct("onset_ap", 0.18, 0.2) == 10.0 and st.worse_pct("sev_nowcast_rmse", 1.1, 1.0) == 10.000000000000009 or abs(st.worse_pct("sev_nowcast_rmse", 1.1, 1.0) - 10) < 1e-9


def test_holm_and_benjamini_hochberg_against_hand_computed_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(st.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(st.benjamini_hochberg([0.01, 0.04, 0.03, 0.005]), [0.02, 0.04, 0.04, 0.02])
    assert np.all(st.holm([0.2]) == 0.2) and np.all(st.benjamini_hochberg([1.0, 1.0]) == 1.0)
    pp = np.random.default_rng(2).random(50)
    assert (st.holm(pp) >= pp - 1e-12).all() and (st.benjamini_hochberg(pp) >= pp - 1e-12).all() and (st.holm(pp) >= st.benjamini_hochberg(pp) - 1e-12).all()


def test_centred_bootstrap_test_has_nominal_type_i_error_and_power():
    rng = np.random.default_rng(3)
    N = 120
    rej0 = rej1 = 0
    sims = 300
    for _ in range(sims):
        shared = rng.normal(0, 1, N)                                                            # participant-level difficulty shared by both models (paired)
        a = shared + rng.normal(0, 0.5, N)
        b0 = shared + rng.normal(0, 0.5, N)
        b1 = b0 - 0.45                                                                          # true improvement
        C = st.boot_counts(300, N, int(rng.integers(1 << 30)))
        for b, which in ((b0, 0), (b1, 1)):
            T = (C @ (b - a)) / C.sum(1)
            p = st.two_sided_centred_p(T)
            if which == 0:
                rej0 += p < 0.05
            else:
                rej1 += p < 0.05
    assert 0.02 <= rej0 / sims <= 0.10 and rej1 / sims > 0.95                              # about 5% false positives under the null; high power for a large effect


def test_tertile_groups_depend_only_on_labels_and_top_share_example():
    sev = np.random.default_rng(4).random(200)
    g = st.tertile_groups(sev)
    assert np.bincount(g).tolist() in ([67, 67, 66], [66, 67, 67], [67, 66, 67]) and g[np.argmin(sev)] == 0 and g[np.argmax(sev)] == 2
    assert np.array_equal(g, st.tertile_groups(np.exp(sev * 3)))                             # invariant to monotone transforms; no model output involved
    imp = np.array([5.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(st.top_share(imp, 0.1) - 0.5) < 1e-12 and np.isnan(st.top_share(-np.ones(10)))


def test_existing_stage8_stage9_files_are_unchanged():
    m = json.load(open(f"{OUT}/pre_manifest.json"))
    for f in ("results/stage9_v2/decision/decision_cal.json", "results/stage9_v2/analysis/final_test_report.md", "results/stage9_v2/analysis/final_cal_decision.md"):
        assert hashlib.sha256(open(f, "rb").read()).hexdigest() == m[f]

    def tree(root):
        items = []
        for dp, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                s = os.stat(p)
                items.append((p.replace("\\", "/"), s.st_size, int(s.st_mtime)))
        items.sort()
        return hashlib.sha256(json.dumps(items).encode()).hexdigest()

    for r in ("results/stage9_v2", "results/stage8_missing", "results/stage3", "results/stage3c_forecast", "data/processed_v1"):
        assert tree(r) == m[r][0], r


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
                if name.startswith("test_00"):
                    print("REPRODUCTION FAILED - stop and report.")
                    break
    sys.exit(1 if fails else 0)
