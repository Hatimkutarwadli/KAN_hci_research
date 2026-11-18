"""Stage 6 tests (split conformal + subgroup reliability). Run: python -m unittest tests/test_conformal.py"""
import hashlib
import os
import sys
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import conformal as cf  # noqa: E402


class TestCore(unittest.TestCase):
    def test_quantile_rank_rule(self):
        s = np.arange(1, 101, dtype=float)                                   # n = 100
        self.assertEqual(cf.conformal_q(s, 0.10), 91.0)                      # ceil(101*0.9) = 91
        self.assertEqual(cf.conformal_q(s, 0.05), 96.0)                      # ceil(101*0.95) = 96
        self.assertEqual(cf.conformal_q(np.arange(1, 10, dtype=float), 0.05), float("inf"))   # n too small for the level
        self.assertEqual(cf.conformal_q(np.array([3.0, 1.0, 2.0]), 0.5), 2.0)  # unsorted input; ceil(4*0.5)=2

    def test_interval_construction_and_coverage_arithmetic(self):
        pred = np.array([5.0, 10.0, 20.0])
        lo, hi = cf.make_interval(pred, 2.0)
        np.testing.assert_array_equal(lo, [3, 8, 18])
        np.testing.assert_array_equal(hi, [7, 12, 22])
        y = np.array([7.0, 12.1, 18.0])                                      # boundary values count as covered
        self.assertAlmostEqual(cf.coverage(y, lo, hi), 2 / 3)
        k, n = 30, 40
        lo_w, hi_w = cf.wilson(k, n)
        self.assertTrue(lo_w < k / n < hi_w and 0 <= lo_w and hi_w <= 1)
        self.assertGreater(cf.wilson(3, 4)[1] - cf.wilson(3, 4)[0], cf.wilson(300, 400)[1] - cf.wilson(300, 400)[0])   # small n -> wide

    def test_calibration_separation(self):
        rng = np.random.default_rng(0)
        yc, pc = rng.normal(10, 3, 200), rng.normal(10, 3, 200)
        yt, pt = rng.normal(10, 3, 200), rng.normal(10, 3, 200)
        q1, cov1, w1, _, _ = cf.calibrate_and_evaluate(pc, yc, pt, yt, 0.05)
        q2, cov2, w2, _, _ = cf.calibrate_and_evaluate(pc, yc, pt + 5, rng.normal(0, 50, 200), 0.05)   # change test data completely
        self.assertEqual(q1, q2)                                             # q depends on calibration pairs only
        self.assertEqual(w1, w2)
        q3, *_ = cf.calibrate_and_evaluate(pc, yc + 1.0, pt, yt, 0.05)       # change calibration labels
        self.assertNotEqual(q1, q3)
        self.assertAlmostEqual(w1, 2 * q1)                                   # constant width 2q

    def test_origin_draws_one_per_participant_and_deterministic(self):
        a = cf.draw_origins(200, np.random.default_rng(cf.ORIGIN_SEED))
        b = cf.draw_origins(200, np.random.default_rng(cf.ORIGIN_SEED))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (200,))
        self.assertTrue(a.min() >= cf.ORIG_LO and a.max() <= cf.ORIG_HI)
        self.assertEqual(cf.ORIG_LO, 27)
        self.assertLessEqual(cf.ORIG_HI + 21, 119)                           # every horizon eligible at every origin

    def test_coverage_near_nominal_under_exchangeability(self):
        """Sanity check of the implementation: iid residuals, n_cal = 200 -> average coverage ~ nominal."""
        rng = np.random.default_rng(1)
        for alpha in (0.10, 0.05):
            covs = []
            for _ in range(1500):
                sc = np.abs(rng.normal(size=200))
                q = cf.conformal_q(sc, alpha)
                covs.append(np.mean(np.abs(rng.normal(size=400)) <= q))
            self.assertGreaterEqual(np.mean(covs), 1 - alpha - 0.003)        # guarantee is >= 1-alpha on average
            self.assertLess(np.mean(covs), 1 - alpha + 0.01)


RES = os.path.join(ROOT, "results", "stage6_conformal")


@unittest.skipUnless(os.path.isfile(os.path.join(RES, "config.json")), "Stage 6 results not present")
class TestSavedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.iv = pd.read_csv(os.path.join(RES, "intervals_primary_draw.csv"))
        cls.by_seed = pd.read_csv(os.path.join(RES, "coverage_width_by_seed.csv"))
        cls.sub = pd.read_csv(os.path.join(RES, "subgroup_by_seed.csv"))
        cls.sub_sum = pd.read_csv(os.path.join(RES, "subgroup_summary.csv"))
        cls.splits = pd.read_csv(os.path.join(ROOT, "data", "synthetic_v1", "splits.csv")).set_index("pid").split

    def test_participant_level_calibration(self):
        g = self.iv.groupby(["model", "seed", "task", "level", "split"])
        self.assertTrue((g.pid.nunique() == g.size()).all())                 # exactly one origin per participant
        n = g.size().unstack("split")
        self.assertTrue((n["cal"] == 200).all() and (n["test"] == 200).all())   # 200 calibration points, not 18,600 correlated days
        self.assertTrue((self.iv.origin_day.between(cf.ORIG_LO, cf.ORIG_HI)).all())
        for s in ("cal", "test"):
            ids = self.iv[self.iv.split == s].pid.unique()
            self.assertTrue((self.splits.loc[ids] == s).all())               # calibration rows are calibration participants only
        origins = self.iv.drop_duplicates(["split", "pid"])[["split", "pid", "origin_day"]]
        self.assertEqual(len(origins), len(self.iv[["split", "pid", "origin_day"]].drop_duplicates()))   # same origin for every model/task

    def test_intervals_and_coverage_consistent(self):
        t = self.iv[self.iv.split == "test"]
        np.testing.assert_allclose(t.hi - t.lo, 2 * t.q, atol=1e-9)
        np.testing.assert_allclose((t.lo + t.hi) / 2, t.pred, atol=1e-9)
        self.assertTrue((t.covered == ((t.y >= t.lo) & (t.y <= t.hi))).all())
        cov = t.groupby(["model", "seed", "task", "level"]).covered.mean().rename("c").reset_index()
        m = self.by_seed.merge(cov, on=["model", "seed", "task", "level"])
        np.testing.assert_allclose(m.coverage_primary, m.c, atol=1e-9)
        np.testing.assert_allclose(self.by_seed.width, 2 * self.by_seed.q, atol=1e-9)
        self.assertTrue(self.by_seed.coverage_primary.between(0, 1).all())

    def test_q_uses_calibration_rows_only(self):
        for (model, seed, task, level), g in list(self.iv.groupby(["model", "seed", "task", "level"]))[:40]:
            c = g[g.split == "cal"]
            alpha = {"90%": 0.10, "95%": 0.05}[level]
            self.assertAlmostEqual(cf.conformal_q(np.abs(c.y - c.pred), alpha), c.q.iloc[0], places=9)

    def test_subgroup_counts_and_flags(self):
        s = self.sub_sum[(self.sub_sum.model == "gru") & (self.sub_sum.task == "sev_dep_t0") & (self.sub_sum.level == "95%")].set_index(["group_type", "group"])
        self.assertEqual(int(s.loc[("sex", "F"), "n_test"] + s.loc[("sex", "M"), "n_test"]), 200)
        self.assertEqual(int(sum(s.loc[("age_group", g), "n_test"] for g in ("0", "1", "2"))), 200)
        self.assertTrue(str(s.loc[("age_group", "2"), "reliability_flag"]).startswith("UNRELIABLE"))
        self.assertEqual(s.loc[("sex", "F"), "reliability_flag"], "ok")
        self.assertTrue((s.wilson_lo <= s.coverage_primary_mean).all() and (s.coverage_primary_mean <= s.wilson_hi).all())

    def test_previous_results_unchanged(self):
        for chk in ("stage3_checksums.sha256", "stage3bcd_checksums.sha256", "stage45_checksums.sha256"):
            with open(os.path.join(ROOT, "results", chk)) as f:
                for line in f:
                    h, p = line.split()
                    with open(os.path.join(ROOT, p), "rb") as g:
                        self.assertEqual(hashlib.sha256(g.read()).hexdigest(), h, p)


if __name__ == "__main__":
    unittest.main()
