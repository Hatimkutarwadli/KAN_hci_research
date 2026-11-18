"""Stage 5 tests (causal-analysis machinery). Run: python -m unittest tests/test_causal.py"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import causal_analysis as ca  # noqa: E402

RAW = os.path.join(ROOT, "data", "synthetic_v1")
RES = os.path.join(ROOT, "results", "stage5_causal")


class TestMechanics(unittest.TestCase):
    def test_design_alignment_and_no_future_regressors(self):
        N, D = 3, ca.D
        Y = np.tile(np.arange(D, dtype=float) ** 2, (N, 1))              # Y[t] = t^2
        A = np.tile(np.arange(D, dtype=float), (N, 1))                    # A[t] = t
        names, X, y, pid = ca.design(Y, [("x|L1", A, 1), ("x|L3", A, 3), ("same", A, 0)], diff=True, tau_lo=5)
        tau = np.tile(np.arange(5, D), N)
        np.testing.assert_array_equal(y, tau ** 2 - (tau - 1) ** 2)       # dependent = Y[tau] - Y[tau-1]
        np.testing.assert_array_equal(X[:, 0], tau - 1)                   # lag L=1 -> day tau-1
        np.testing.assert_array_equal(X[:, 1], tau - 3)
        np.testing.assert_array_equal(X[:, 2], tau)                       # same-day only where explicitly requested
        self.assertTrue((X <= tau[:, None]).all())                        # never a future day

    def test_within_demeaning_and_cluster_ols_recover_known_slope(self):
        rng = np.random.default_rng(1)
        N, T = 300, 60
        person = rng.normal(0, 3, N)[:, None]
        x = rng.normal(size=(N, T)) + 0.8 * person                       # x correlated with the person effect (confounding across people)
        y = 2.0 * x + 5 * person + rng.normal(size=(N, T)) * (1 + (np.arange(N)[:, None] % 3))
        pid = np.repeat(np.arange(N), T)
        Xd, yd, pidd = ca.within(x.reshape(-1, 1), y.reshape(-1), pid)
        np.testing.assert_allclose(np.bincount(pidd, weights=Xd[:, 0]), 0, atol=1e-8)
        b, se, g = ca.ols_cr(Xd, yd, pidd)
        self.assertLess(abs(b[0] - 2.0), 3 * se[0])                      # fixed effects remove the between-person confounding
        self.assertEqual(g, N)
        lo, hi = ca.boot_ci(Xd, yd, pidd, B=200)
        self.assertTrue(lo[0] <= b[0] <= hi[0])
        self.assertTrue(0.7 < (hi[0] - lo[0]) / (2 * 1.96 * se[0]) < 1.4)  # bootstrap and clustered SE agree

    def test_complete_case_and_masking_keep_same_rows(self):
        A = np.random.default_rng(0).normal(size=(5, ca.D))
        T = np.random.default_rng(1).normal(size=(5, ca.D))
        A[2, 10] = np.nan
        M = ca.mask_like(T, A)
        self.assertTrue(np.isnan(M[2, 10]) and np.isfinite(M[2, 11]))


@unittest.skipUnless(os.path.isdir(RAW), "dataset not present")
class TestDataSeparation(unittest.TestCase):
    def test_observational_loader_never_needs_truth_files(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            for f in ("observations.csv", "targets.csv"):
                shutil.copy(os.path.join(RAW, f), os.path.join(tmp, f))
            A = ca.load_observational(tmp)                                  # works with NO truth.csv / truth_participants.csv / meta.json
        self.assertEqual(set(A), set(ca.FEATS) | {"stress_proxy", "sev_dep", "sev_anx"})
        self.assertFalse({"z_dep", "z_anx", "stress", "u_steps"} & set(A))
        self.assertEqual(A["steps"].shape, (1000, ca.D))

    def test_raw_data_untouched(self):
        import preprocess as pp
        pp.verify_raw(RAW)


@unittest.skipUnless(os.path.isfile(os.path.join(RES, "recovery_summary.csv")), "Stage 5 results not present")
class TestSavedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.rec = pd.read_csv(os.path.join(RES, "recovery_summary.csv"))
        cls.eff = pd.read_csv(os.path.join(RES, "effect_lag_table.csv"))

    def test_benchmark_signs_and_planted_lags(self):
        sub = self.rec.drop_duplicates("relationship").set_index("relationship")
        self.assertGreater(sub.loc["sleep_variability -> dep", "benchmark"], 0)
        for r in ("steps -> dep", "steps -> anx", "hrv -> anx"):
            self.assertLess(sub.loc[r, "benchmark"], 0)
        self.assertEqual(sub.loc["sleep_variability -> dep", "planted_lag"], 3)
        self.assertEqual(sub.loc["hrv -> anx", "planted_lag"], 1)

    def test_oracle_estimator_recovers_planted_effects(self):
        """The estimator itself is sound: with true variables (E3) the planted lag is the peak and the effect is within 25%."""
        o = self.rec[self.rec.variant == "E3_oracle_true_stress"]
        self.assertTrue(o.peak_is_planted_lag.all())
        self.assertTrue(((o.ratio_to_benchmark - 1).abs() < 0.25).all())
        self.assertTrue(o.sign_correct.all())

    def test_every_variant_uses_same_rows(self):
        n = self.eff.groupby("variant").n_rows.first()
        self.assertEqual(n.nunique(), 1)

    def test_ground_truth_variants_are_flagged(self):
        flagged = self.eff.groupby("variant").uses_ground_truth.first()
        self.assertFalse(flagged["V1_adjusted_for_other_features"] or flagged["V2_plus_stress_proxy"])
        for v in flagged.index:
            if v.startswith("E"):
                self.assertTrue(flagged[v], v)


if __name__ == "__main__":
    unittest.main()
