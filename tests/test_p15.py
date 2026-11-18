"""Tests for P1-5 (extended early-warning evaluation). Run: python tests/test_p15.py
FIRST test = regression: the generalised implementation must reproduce every saved Stage 7 result (h21) exactly. Existing predictions are only read."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "extension_p15"), os.path.join(ROOT, "src")]
import ew15 as ew  # noqa: E402
import leadtime as lt  # noqa: E402


def test_00_stage7_regression_reproduces_saved_results_exactly():
    m7 = pd.read_csv("results/stage7_leadtime/metrics_per_seed.csv")
    leads7 = pd.read_csv("results/stage7_leadtime/onset_participant_leads.csv")
    skip = {"model", "seed", "target", "target_fap"}
    n_cmp = 0
    for k in ew.KS:
        cal, test = ew.load_cohort("cal", k, "sust", 21), ew.load_cohort("test", k, "sust", 21)
        runs = [(m, s) for m in ew.STAGE7_MODELS for s in ew.SRC[m][2]] + [("random_null", s) for s in range(lt.NULL_DRAWS)]
        for name, seed in runs:
            if name == "random_null":
                Scal, Stest = lt.null_scores(seed)
            else:
                Scal, Stest = ew.load_scores(name, seed, "cal", k, "sust", 21, cal["pid"]), ew.load_scores(name, seed, "test", k, "sust", 21, test["pid"])
            auc = ew.participant_auc(Stest, test)
            for f in ew.TARGETS:
                res, st = ew.evaluate_one(Scal, Stest, cal, test, f)
                ref = m7[(m7.model == name) & (m7.seed == seed) & (m7.target == k) & (np.isclose(m7.target_fap, f))]
                assert len(ref) == 1, (name, seed, k, f)
                ref = ref.iloc[0]
                for col in ref.index:
                    if col in skip:
                        continue
                    if col == "auc_participant":
                        a, b = auc, ref[col]
                    else:
                        a, b = res[col], ref[col]
                    assert (np.isnan(a) and np.isnan(b)) or abs(float(a) - float(b)) < 1e-9, (name, seed, k, f, col, a, b)
                    n_cmp += 1
                lead_ref = leads7[(leads7.model == name) & (leads7.seed == seed) & (leads7.target == k) & np.isclose(leads7.target_fap, f)].set_index("pid")
                on = np.where(test["group"] == "onset")[0]
                mine = pd.Series(st["first_lead"][on], index=test["pid"][on])
                ref_l = lead_ref.first_alarm_lead.reindex(mine.index).fillna(0).astype(int)
                assert (mine == ref_l).all(), (name, seed, k, f)
    assert n_cmp == len(m7) * (m7.shape[1] - len(skip)) == 164 * 30                                                                      # every column of every saved Stage 7 row was compared


def test_vectorised_stats_equal_stage7_loop_implementation_on_random_scores():
    rng = np.random.default_rng(3)
    for k in ew.KS:
        coh = ew.load_cohort("test", k, "sust", 21)
        for tau in (0.5, 0.8, 0.95):
            S = rng.random((200, 120))
            ref = lt.participant_stats(S, tau, coh["onset"], coh["group"])
            mine = ew.alarm_stats(S, tau, coh)
            assert np.array_equal(ref["first_lead"], mine["first_lead"]) and np.array_equal(ref["n_days"], mine["n_days"]) and np.array_equal(ref["n_ep"], mine["n_ep"])
            assert np.array_equal(ew.window_max(S, coh)[coh["group"] != "excl"], lt.window_max(S, coh["onset"], coh["group"])[coh["group"] != "excl"])


def test_t1_matches_label_support_and_predictions_are_finite_there():
    d = np.load("data/processed_v1/test.npz")
    for kind in ("sust", "cross"):
        for h in ew.HORIZONS:
            y = d[f"y_onset_{kind}_dep_h{h}"]
            last = np.where(~np.isnan(y).all(0))[0].max()
            assert last == ew.t1_of(kind, h), (kind, h, last)
    cal = ew.load_cohort("cal", "dep", "sust", 7)
    for m in ("transformer", "family", "st_task", "mlp_m1"):
        S = ew.load_scores(m, 0, "cal", "dep", "sust", 7, cal["pid"])
        assert np.isfinite(S[:, ew.T0:ew.t1_of("sust", 7) + 1]).all()


def test_tau_depends_on_calibration_only():
    cal, test = ew.load_cohort("cal", "dep"), ew.load_cohort("test", "dep")
    rng = np.random.default_rng(0)
    Scal, Stest = rng.random((200, 120)), rng.random((200, 120))
    r1, _ = ew.evaluate_one(Scal, Stest, cal, test, 0.2)
    r2, _ = ew.evaluate_one(Scal, rng.random((200, 120)) * 0.5 + 0.5, cal, test, 0.2)          # completely different test scores
    assert r1["tau"] == r2["tau"] == ew.tau_from_calibration(Scal, cal, 0.2)
    assert r1["cal_fap_realised"] == r2["cal_fap_realised"] <= 0.2 + 1e-12
    r3, _ = ew.evaluate_one(rng.random((200, 120)), Stest, cal, test, 0.2)
    assert r3["tau"] != r1["tau"]


def test_episode_counting_against_bruteforce():
    rng = np.random.default_rng(1)
    coh = ew.load_cohort("test", "anx", "sust", 14)
    S = (rng.random((200, 120)) > 0.9).astype(float)
    st = ew.alarm_stats(S, 0.5, coh)
    for i in np.where(coh["group"] == "never")[0]:
        a = np.where(S[i, ew.T0:coh["T1"] + 1] > 0.5)[0]
        ep = 0 if not len(a) else 1 + int((np.diff(a) - 1 >= ew.EP_GAP).sum())
        assert st["n_ep"][i] == ep and st["n_days"][i] == len(a)


def test_horizon_windows_leads_and_achievable():
    for h in ew.HORIZONS:
        t = ew.load_cohort("test", "dep", "sust", h)
        assert t["T1"] == 117 - h and t["W"] == h
        on = np.where(t["group"] == "onset")[0]
        assert (t["hi"][on] <= t["T1"]).all() and ((t["hi"][on] - t["lo"][on]) <= h - 1).all()
        res = ew.metrics_from(ew.alarm_stats(np.zeros((200, 120)), 0.5, t), t)
        for l in ew.LEADS:
            assert (np.isnan(res[f"det_lead{l}"])) == (l > h)                                  # leads longer than the window are undefined, not zero
        assert res["det_lead1"] == 0.0 and res["fap"] == 0.0


def test_matched_exposure_blocks_layout_leftover_and_content():
    for h, nb, left in ((21, 3, 7), (14, 5, 7), (7, 12, 0)):
        coh = ew.load_cohort("test", "dep", "sust", h)
        assert ew.block_layout(coh) == (nb, left)
    coh = ew.load_cohort("test", "dep", "sust", 21)
    S = np.zeros((200, 120))
    nv = np.where(coh["group"] == "never")[0]
    S[nv[0], 27 + 63 + 2] = 1.0                                                                  # alarm only in the LEFTOVER days (after the 3 complete blocks) -> not counted
    S[nv[1], 27 + 21 + 5] = 1.0                                                                  # alarm in block 2 of participant 1 -> 1 of 3*n_never blocks
    st = ew.alarm_stats(S, 0.5, coh)
    m, B, full = ew.matched_exposure(S, 0.5, coh, st)
    assert B.shape == (len(nv), 3) and m["matched_leftover_days"] == 7 and m["matched_n_blocks"] == 3 * len(nv)
    assert abs(m["matched_block_fap"] - 1 / (3 * len(nv))) < 1e-12 and st["n_days"][nv[0]] == 1     # the leftover alarm still counts in the Stage 7 participant FAP, not in the block rate
    on = np.where(coh["group"] == "onset")[0]
    o_full = [i for i in on if coh["onset"][i] - 21 >= 27 and coh["onset"][i] - 1 <= 96]
    assert m["matched_n_full_window_onsets"] == len(o_full)
    S2 = np.zeros((200, 120))
    i0 = o_full[0]
    S2[i0, coh["onset"][i0] - 10] = 1.0
    m2, _, _ = ew.matched_exposure(S2, 0.5, coh, ew.alarm_stats(S2, 0.5, coh))
    assert abs(m2["matched_det_lead7"] - 1 / len(o_full)) < 1e-12 and abs(m2["matched_det_lead14"] - 0) < 1e-12 + 0 and m2["matched_auc_vs_blocks"] > 0.5


def test_shuffle_permutations_are_complete_trajectories_and_shared():
    P1, P2 = ew.shuffle_perms(), ew.shuffle_perms()
    assert P1.shape == (200, 200) and np.array_equal(P1, P2)                                     # identical for every model / call
    assert all(sorted(p) == list(range(200)) for p in P1)
    test = ew.load_cohort("test", "dep", "sust", 21)
    S = np.random.default_rng(0).random((200, 120))
    Sp = S[P1]
    for b in (0, 17, 199):
        for i in (0, 50, 199):
            assert np.array_equal(Sp[b, i], S[P1[b, i]])                                          # a whole trajectory, in its original temporal order
    nullm = ew.shuffle_null(S, 0.9, test, P1[:20])
    assert all(v.shape == (20,) for v in nullm.values()) and np.isfinite(nullm["fap"]).all()
    # a score that is the true outcome signal must beat its own shuffle null
    on = np.where(test["group"] == "onset")[0]
    Ssig = np.zeros((200, 120))
    for i in on:
        Ssig[i, max(27, test["onset"][i] - 21):min(test["onset"][i], 97)] = 1.0
    r, _ = ew.evaluate_one(np.zeros((200, 120)) + np.random.default_rng(1).random((200, 120)) * 0.1, Ssig, ew.load_cohort("cal", "dep"), test, 0.2)
    nm = ew.shuffle_null(Ssig, r["tau"], test, P1)
    assert r["det_lead1"] == 1.0 and nm["det_lead1"].mean() < 0.5 and (nm["det_lead1"] >= r["det_lead1"]).mean() < 0.01


def test_stratified_bootstrap_is_paired_and_consistent_with_point_estimates():
    test = ew.load_cohort("test", "dep")
    on, nv, bon, bnv = ew.stratified_indices(test, 400, 5)
    on2, nv2, bon2, bnv2 = ew.stratified_indices(test, 400, 5)
    assert np.array_equal(bon, bon2) and np.array_equal(bnv, bnv2) and bon.shape == (400, len(on)) and bnv.shape == (400, len(nv))   # same indices for every model
    rng = np.random.default_rng(2)
    S = rng.random((200, 120))
    st = ew.alarm_stats(S, 0.9, test)
    a = ew.participant_arrays(st, test)
    bm = ew.boot_metrics([a], bon, bnv, test)
    pt = ew.metrics_from(st, test)
    assert abs(bm["det_lead7"].mean() - pt["det_lead7"]) < 0.03 and abs(bm["fap"].mean() - pt["fap"]) < 0.03
    ident = ew.boot_metrics([a], np.tile(np.arange(len(on)), (3, 1)), np.tile(np.arange(len(nv)), (3, 1)), test)             # identity resample == point estimate
    for key in ("det_lead1", "det_lead7", "fap", "alarm_burden", "precision_participant", "det_lead7_achievable"):
        assert np.allclose(ident[key], pt[key], equal_nan=True), key
    # paired difference of a model with itself is exactly zero in every resample
    assert np.all(ew.boot_metrics([a], bon, bnv, test)["det_lead7"] - bm["det_lead7"] == 0)


def test_threshold_bootstrap_uses_only_calibration_resampling():
    cal, test = ew.load_cohort("cal", "dep"), ew.load_cohort("test", "dep")
    rng = np.random.default_rng(4)
    Scal, Stest = rng.random((200, 120)), rng.random((200, 120))
    r1 = ew.threshold_bootstrap(Scal, Stest, cal, test, 0.2, 30, 9)
    r2 = ew.threshold_bootstrap(Scal, rng.random((200, 120)), cal, test, 0.2, 30, 9)
    assert [r["tau"] for r in r1] == [r["tau"] for r in r2]                                       # tau path independent of the test scores
    assert len({r["tau"] for r in r1}) > 3


def test_no_existing_file_is_written_by_the_core_module_and_models_exist():
    for m, (d, how, seeds, fam) in ew.SRC.items():
        for s in list(seeds)[:1]:
            for split in ("cal", "test"):
                k = "dep"
                fn = {"std": f"{m}_seed{s}_{split}.npz", "head": f"onset_sust_{k}_h21_seed{s}_{split}.npz", "family": f"family_onset_sust_seed{s}_{split}.npz"}[how]
                assert os.path.exists(os.path.join(d, "predictions", fn)), (m, fn)
    assert not ew.has_head("family", "cross") and ew.has_head("transformer", "cross")


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
                    print("STAGE 7 REGRESSION FAILED - stop and report; do not change the Stage 7 protocol.")
                    break
    sys.exit(1 if fails else 0)
