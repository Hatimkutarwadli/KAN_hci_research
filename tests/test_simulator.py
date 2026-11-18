"""Simulator tests (synthetic data). Run: python -m unittest tests/test_simulator.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simulate import Config, simulate, MODEL_INPUT_COLUMNS  # noqa: E402


class TestSimulator(unittest.TestCase):
    cfg = Config(n_participants=60, seed=7)

    def test_reproducible(self):
        a, b = simulate(self.cfg), simulate(self.cfg)
        for k in a:
            self.assertTrue(a[k].equals(b[k]), k)

    def test_no_future_leakage(self):
        """Perturbing ALL randomness from day 60 on must not change inputs/targets before day 60."""
        d = 60
        a, b = simulate(self.cfg), simulate(self.cfg, perturb_after=d)
        for name in ("observations", "targets_sev"):
            pass
        oa, ob = a["observations"], b["observations"]
        ma, mb = oa[oa.day < d], ob[ob.day < d]
        np.testing.assert_allclose(ma.drop(columns=[]).values, mb.values, equal_nan=True)
        ta, tb = a["targets"], b["targets"]
        cols = ["severity_dep", "severity_anx", "status_dep", "status_anx"]
        np.testing.assert_allclose(ta.loc[ta.day < d, cols].values, tb.loc[tb.day < d, cols].values)
        # sanity: the perturbation does change later latent states
        za, zb = a["truth"], b["truth"]
        self.assertFalse(np.allclose(za.loc[za.day > d + 5, "z_dep"], zb.loc[zb.day > d + 5, "z_dep"]))

    def test_inputs_exclude_truth_and_targets(self):
        o = simulate(self.cfg)
        inputs = set(o["observations"].columns) - {"pid", "day"}
        forbidden = (set(o["truth"].columns) | set(o["targets"].columns)) - {"pid", "day"}
        self.assertFalse(inputs & forbidden)

    def test_onset_censoring_and_risk_set(self):
        o = simulate(self.cfg)
        tg, D = o["targets"], self.cfg.n_days
        for k in ("dep", "anx"):
            for h in self.cfg.horizons:
                col = f"onset_{k}_h{h}"
                self.assertTrue(tg.loc[tg.day > D - 1 - h, col].isna().all())
                self.assertTrue(tg.loc[tg[f"at_risk_{k}"] == 0, col].isna().all())
                # a positive label means the latent state really crosses within (t, t+h]
                pos = tg[tg[col] == 1]
                z = o["truth"].set_index(["pid", "day"])[f"z_{k}"]
                thr = getattr(self.cfg, f"z_thr_{k}")
                for pid, day in list(zip(pos.pid, pos.day))[:50]:
                    fut = z.loc[pid].loc[day + 1: day + h].values
                    self.assertTrue((fut >= thr).any())
                    self.assertTrue((z.loc[pid].loc[: day].values < thr).all())

    def test_hrv_missing_flag(self):
        ob = simulate(self.cfg)["observations"]
        np.testing.assert_array_equal(ob.hrv_missing.values, ob.hrv.isna().astype(int).values)
        self.assertTrue((ob.loc[ob.missing_day == 1, "hrv_missing"] == 1).all())
        self.assertGreater(((ob.hrv_missing == 1) & (ob.missing_day == 0)).sum(), 0)  # HRV-only gaps exist

    def test_sustained_onset_definition_and_censoring(self):
        o = simulate(self.cfg)
        tg, tp, D, P = o["targets"], o["truth_participants"], self.cfg.n_days, self.cfg.persistence_days
        for k in ("dep", "anx"):
            z = o["truth"].pivot(index="pid", columns="day", values=f"z_{k}").values
            st = z >= getattr(self.cfg, f"z_thr_{k}")
            for pid in range(self.cfg.n_participants):
                exp = np.nan
                if not st[pid, 0]:
                    for d in range(D - P + 1):
                        if st[pid, d:d + P].all():
                            exp = d
                            break
                got = tp.loc[pid, f"onset_day_sust_{k}"]
                self.assertTrue((np.isnan(exp) and np.isnan(got)) or exp == got, (k, pid, exp, got))
                cross = tp.loc[pid, f"onset_day_{k}"]
                if not np.isnan(got):
                    self.assertGreaterEqual(got, cross)      # sustained onset never precedes first crossing
            for h in self.cfg.horizons:
                col = f"onset_sust_{k}_h{h}"
                self.assertTrue(tg.loc[tg.day + h + P - 1 > D - 1, col].isna().all())    # needs confirmation days
                self.assertTrue(tg.loc[tg[f"at_risk_sust_{k}"] == 0, col].isna().all())
                m = tg.merge(tp[["pid", f"onset_day_sust_{k}"]], on="pid")
                sub = m[m[col].notna()]
                exp = ((sub[f"onset_day_sust_{k}"] > sub.day) & (sub[f"onset_day_sust_{k}"] <= sub.day + h)).astype(float)
                self.assertTrue((exp.values == sub[col].values).all())
                self.assertTrue(tg.loc[tg.pid.isin(tp.pid[tp[f"prevalent_{k}"] == 1]), col].isna().all())

    def test_participant_split_disjoint(self):
        o = simulate(self.cfg)
        self.assertEqual(o["splits"].pid.nunique(), self.cfg.n_participants)


if __name__ == "__main__":
    unittest.main()
