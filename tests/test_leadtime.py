"""Tests for Stage 7 lead-time analysis (src/leadtime.py). Run: python tests/test_leadtime.py"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import leadtime as lt  # noqa: E402


def at_risk_from_onsets(onsets, N=None):
    """onsets: list of onset day / -1 never / 'prev' prevalent. Builds at_risk (NaN for t<27)."""
    A = np.ones((len(onsets), 120))
    for i, o in enumerate(onsets):
        if o == "prev":
            A[i] = 0
        elif o >= 0:
            A[i, o:] = 0
    A[:, :27] = np.nan
    return A


def test_cohort_groups():
    on, g = lt.cohort_from_at_risk(at_risk_from_onsets([40, -1, "prev", 27, 28, 117]))
    assert list(g) == ["onset", "never", "excl", "excl", "onset", "onset"]
    assert on[0] == 40 and on[4] == 28 and on[5] == 117 and on[1] == -1


def test_tau_convention():
    m = np.arange(1, 11, dtype=float)                       # 10 never-onset maxima 1..10
    assert lt.tau_from_calibration(m, 0.2) == 8.0            # k=2 -> 3rd largest = 8 ; alarm iff >8 -> {9,10} = 20%
    assert (m > lt.tau_from_calibration(m, 0.2)).mean() <= 0.2
    assert lt.tau_from_calibration(m, 0.0) == 10.0 and (m > 10).sum() == 0
    t = np.array([5, 5, 5, 5, 1, 1, 1, 1, 1, 1], float)      # ties: conservative (<= target)
    assert (t > lt.tau_from_calibration(t, 0.2)).mean() <= 0.2
    for f in (0.1, 0.2):
        r = np.random.default_rng(0).random(135)
        assert (r > lt.tau_from_calibration(r, f)).mean() <= f + 1e-12


def test_lead_arithmetic_and_window():
    on, g = lt.cohort_from_at_risk(at_risk_from_onsets([60, 30, 110]))
    S = np.zeros((3, 120))
    S[0, 45] = 1                                             # onset 60, alarm day 45 -> lead 15
    S[0, 30] = 1                                             # alarm 30 days early (outside window) -> must be ignored
    S[1, 27] = 1                                             # onset 30, alarm at day 27 -> lead 3 (window truncated at 27)
    S[2, 97] = 1                                             # onset 110, alarm day 97 > 96 -> not an origin, ignored
    st = lt.participant_stats(S, 0.5, on, g)
    assert list(st["first_lead"]) == [15, 3, 0]
    S2 = np.zeros((1, 120))
    S2[0, 38] = 1                                            # onset 60: window starts day 39 -> day 38 ignored
    on1, g1 = lt.cohort_from_at_risk(at_risk_from_onsets([60]))
    assert lt.participant_stats(S2, 0.5, on1, g1)["first_lead"][0] == 0
    S2[0, 39] = 1
    assert lt.participant_stats(S2, 0.5, on1, g1)["first_lead"][0] == 21
    S2[0, 60] = 1                                            # alarm on/after onset day never counts
    S3 = np.zeros((1, 120))
    S3[0, 60:] = 1
    assert lt.participant_stats(S3, 0.5, on1, g1)["first_lead"][0] == 0
    S4 = np.zeros((1, 120)); S4[0, 45] = 0.5               # strict '>'
    assert lt.participant_stats(S4, 0.5, on1, g1)["first_lead"][0] == 0


def test_never_onset_and_exclusions():
    on, g = lt.cohort_from_at_risk(at_risk_from_onsets([-1, -1, 40, 20, "prev"]))
    S = np.zeros((5, 120))
    S[0, 80] = 1                                             # never-onset false alarm
    S[3, 50] = 1                                             # excluded participant alarm: must not count anywhere
    S[4, 50] = 1
    S[1, 97:] = 1                                            # alarms after day 96 are outside the window
    S[0, 90:93] = 1
    S[0, 30] = 1
    st = lt.participant_stats(S, 0.5, on, g)
    m = lt.stat_metrics(st)
    assert m["n_never"] == 2 and m["n_onset"] == 1 and m["fap"] == 0.5 and m["det_lead1"] == 0.0
    assert st["n_ep"][0] == 3 and st["n_days"][0] == 5       # episodes: day 30 | day 80 | days 90-92 (gaps >= 7)
    assert abs(m["episodes_per_100_participant_days"] - 3 / (2 * 70) * 100) < 1e-9
    assert m["precision_participant"] == 0.0


def test_precision_recall_and_achievable():
    on, g = lt.cohort_from_at_risk(at_risk_from_onsets([40, 80, -1, -1]))
    S = np.zeros((4, 120))
    S[0, 35] = 1
    S[1, 70] = 1
    S[2, 60] = 1
    st = lt.participant_stats(S, 0.5, on, g)
    m = lt.stat_metrics(st)
    assert m["det_lead1"] == 1.0 and m["det_lead7"] == 0.5 and m["fap"] == 0.5 and abs(m["precision_participant"] - 2 / 3) < 1e-12
    assert not lt.achievable(30, 7) and lt.achievable(60, 21) and lt.achievable(110, 5) and lt.achievable(117, 21) and not lt.achievable(28, 2) and lt.achievable(28, 1)


def test_auc_matches_brute_force():
    r = np.random.default_rng(1)
    on, g = lt.cohort_from_at_risk(at_risk_from_onsets([50, 60, 70, -1, -1, -1, -1]))
    S = r.random((7, 120))
    auc, M = lt.participant_auc(S, on, g)
    a, b = M[g == "onset"], M[g == "never"]
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert abs(auc - brute) < 1e-12


def test_test_data_cannot_influence_tau():
    on, g = lt.cohort_from_at_risk(at_risk_from_onsets([50, 60, -1, -1, -1, -1, -1, -1, -1, -1]))
    cal = dict(onset=on, group=g, Y=np.full((10, 120), np.nan))
    test = dict(onset=on, group=g, Y=np.full((10, 120), np.nan))
    r = np.random.default_rng(2)
    Sc, S1, S2 = r.random((10, 120)), r.random((10, 120)), r.random((10, 120)) * 5
    a, _ = lt.run_one(Sc, S1, cal, test, 0.2)
    b, _ = lt.run_one(Sc, S2, cal, test, 0.2)
    assert a["tau"] == b["tau"] and a["cal_fap_realised"] <= 0.2


def test_null_same_protocol_and_deterministic():
    c1, t1 = lt.null_scores(0)
    c2, t2 = lt.null_scores(0)
    assert np.array_equal(c1, c2) and np.array_equal(t1, t2) and not np.array_equal(c1, t1)
    assert not np.array_equal(lt.null_scores(1)[0], c1)
    cal, test = lt.load_cohort("cal", "dep"), lt.load_cohort("test", "dep")
    res, _ = lt.run_one(c1, t1, cal, test, 0.1)
    assert res["cal_fap_realised"] <= 0.1 and 0 <= res["fap"] <= 1 and res["n_onset"] == 32       # runs through the same run_one as every model


def test_real_labels_match_truth_and_saved_preds():
    for k in ("dep", "anx"):
        for split in ("cal", "test"):
            c = lt.load_cohort(split, k)
            tp = pd.read_csv("data/synthetic_v1/truth_participants.csv").set_index("pid")
            td = tp.loc[c["pid"], f"onset_day_sust_{k}"].to_numpy()
            ev = c["group"] == "onset"
            assert np.array_equal(c["onset"][ev], td[ev].astype(int))              # onset day agrees with the simulator's recorded onset
            assert (td[c["group"] == "never"] != td[c["group"] == "never"]).all()  # never-onset have no onset day (NaN)
            assert c["onset"][ev].min() >= 28
        assert (lt.load_cohort("test", k)["group"] == "onset").sum() == 32
    S = lt.load_scores("results/stage3", "gru", 0, "test", "dep", lt.load_cohort("test", "dep")["pid"])
    assert S.shape == (200, 120)


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
