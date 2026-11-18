"""Preprocessing tests. Run: python -m unittest tests/test_preprocess.py
Unit tests use toy arrays; artifact tests (skipped if absent) check data/processed_v1 against data/synthetic_v1."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import preprocess as pp  # noqa: E402
import data as dd  # noqa: E402
from simulate import Config, simulate  # noqa: E402

RAW = os.path.join(ROOT, "data", "synthetic_v1")
PROC = os.path.join(ROOT, "data", "processed_v1")
HAVE = os.path.isdir(PROC) and os.path.isdir(RAW)
nan = np.nan


class TestUnits(unittest.TestCase):
    def test_ffill_limit_and_causality(self):
        x = np.array([[1, nan, nan, nan, nan, 2, nan]], dtype=float)
        out = pp.causal_ffill(x, 3)
        np.testing.assert_array_equal(out[0], [1, 1, 1, 1, np.nan, 2, 2])  # 4th missing day not filled

    def test_leading_missing_not_filled_then_zero(self):
        v = np.array([[[nan], [nan], [5.0], [nan]]])
        sc = {"mean": np.array([5.0]), "std": np.array([2.0])}
        X = pp.transform(v, np.zeros((1, 4, 2)), sc)
        np.testing.assert_allclose(X[0, :, 0], [0, 0, 0, 0])          # leading NaN -> 0; later fill = (5-5)/2 = 0
        v2 = np.array([[[nan], [nan], [9.0], [nan]]])
        X2 = pp.transform(v2, np.zeros((1, 4, 2)), sc)
        np.testing.assert_allclose(X2[0, :, 0], [0, 0, 2.0, 2.0])      # no backward fill of leading gap

    def test_ffill_never_uses_future(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5, 40, 3))
        x[rng.random(x.shape) < 0.4] = nan
        y = x.copy()
        y[:, 20:] = rng.normal(size=(5, 20, 3))                          # change the future only
        np.testing.assert_array_equal(pp.causal_ffill(x, 3)[:, :20], pp.causal_ffill(y, 3)[:, :20])

    def test_scaler_train_only_observed_only(self):
        rng = np.random.default_rng(1)
        tr = rng.normal(10, 2, size=(50, 30, 2)); tr[rng.random(tr.shape) < 0.3] = nan
        sc = pp.fit_scaler(tr)
        np.testing.assert_allclose(sc["mean"], np.nanmean(tr.reshape(-1, 2), axis=0))
        np.testing.assert_allclose(sc["std"], np.nanstd(tr.reshape(-1, 2), axis=0))
        te = rng.normal(100, 50, size=(10, 30, 2))                       # very different test data
        X = pp.transform(te, np.zeros((10, 30, 2)), sc)
        self.assertGreater(np.abs(X[:, :, :2].mean()), 5)                # test is NOT re-centred

    def test_transform_shape_masks_no_nan(self):
        v = np.random.default_rng(2).normal(size=(3, 12, 8)); v[0, 3] = nan
        m = np.zeros((3, 12, 2)); m[0, 3] = 1
        X = pp.transform(v, m, pp.fit_scaler(v))
        self.assertEqual(X.shape, (3, 12, 10))
        self.assertFalse(np.isnan(X).any())
        np.testing.assert_array_equal(X[:, :, 8:], m)

    def test_shift_future(self):
        a = np.arange(10, dtype=float)[None, :]
        s = pp.shift_future(a, 3)
        np.testing.assert_array_equal(s[0], [3, 4, 5, 6, 7, 8, 9, nan, nan, nan])

    def test_onset_labels_toy(self):
        D = 20
        st = np.zeros((4, D), dtype=int)
        st[0, 8:12] = 1            # sustained onset at day 8
        st[1, 5] = 1               # 1-day blip, never sustained
        st[2, 0:5] = 1             # prevalent
        st[3, 6:8] = 1; st[3, 15:19] = 1   # 2-day blip, then sustained at 15
        L = pp.build_labels(st, lookback=3, horizons=(4,), P=3)
        t = np.arange(D)
        y0 = L["onset_sust_h4"][0]
        for ti in t:
            if ti < 2 or ti + 4 + 2 > D - 1:
                self.assertTrue(np.isnan(y0[ti]), ti)                    # origin or censoring
            elif ti >= 8:
                self.assertTrue(np.isnan(y0[ti]), ti)                    # no longer at risk
            else:
                self.assertEqual(y0[ti], 1.0 if 8 - 4 <= ti < 8 else 0.0, ti)
        self.assertTrue(np.nanmax(L["onset_sust_h4"][1]) == 0)           # blip never an onset
        self.assertTrue(np.isnan(L["onset_sust_h4"][2]).all())           # prevalent
        self.assertEqual(L["onset_sust_h4"][3][11], 1.0)                 # onset 15 in (11,15]
        self.assertEqual(L["onset_sust_h4"][3][10], 0.0)
        # first-crossing (secondary) fires on the blip; censoring differs: cross needs t+h<=19, sust needs t+h+2<=19
        self.assertEqual(L["onset_cross_h4"][1][3], 1.0)
        self.assertTrue(np.isnan(L["onset_sust_h4"][3][14]))             # t+h+2=20>19 (also at risk until 15)

    def test_window_boundaries(self):
        N, D, C = 3, 60, 10
        X = np.zeros((N, D, C), dtype=np.float32)
        X[:, :, 0] = np.arange(D)[None, :]
        y = np.full((N, D), np.nan, dtype=np.float32)
        y[:, 27:50] = 1
        d = {"X": X, "static": np.zeros((N, 4), np.float32), "pid": np.arange(N), "y_toy": y}
        Xw, st, yy, pid, t = dd.make_windows(d, "toy")
        self.assertEqual(Xw.shape, (N * 23, 28, C))
        self.assertEqual(t.min(), 27)
        for i in (0, 10, 22):
            self.assertEqual(Xw[i, 0, 0], t[i] - 27)                      # first window day = t-27
            self.assertEqual(Xw[i, -1, 0], t[i])                          # last window day = t (no future)
        y[:, 20] = 1
        with self.assertRaises(AssertionError):                           # origin before full lookback rejected
            dd.make_windows(d, "toy")


@unittest.skipUnless(HAVE, "processed/raw data not present")
class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = dd.load_meta(PROC)
        cls.raw_meta = json.load(open(os.path.join(RAW, "meta.json")))
        cls.S = {s: dd.load_split(s, PROC) for s in ("train", "cal", "test")}
        cls.sp = pd.read_csv(os.path.join(RAW, "splits.csv"))
        cls.obs = pd.read_csv(os.path.join(RAW, "observations.csv")).sort_values(["pid", "day"])
        cls.tg = pd.read_csv(os.path.join(RAW, "targets.csv")).sort_values(["pid", "day"])

    def test_split_integrity(self):
        ids = {s: set(self.S[s]["pid"].tolist()) for s in self.S}
        for s in ids:
            self.assertEqual(ids[s], set(self.sp.pid[self.sp.split == s]))
        self.assertFalse(ids["train"] & ids["cal"] or ids["train"] & ids["test"] or ids["cal"] & ids["test"])
        self.assertEqual(sum(len(v) for v in ids.values()), 1000)
        self.assertEqual({s: len(v) for s, v in ids.items()}, {"train": 600, "cal": 200, "test": 200})

    def test_input_columns_exclude_forbidden(self):
        used = self.meta["input_columns_used"]
        self.assertNotIn("stress_proxy", used)
        self.assertNotIn("stress_proxy", self.meta["channel_names"])
        forbidden = set(pd.read_csv(os.path.join(RAW, "truth.csv"), nrows=2).columns) | set(self.tg.columns)
        self.assertFalse((set(self.meta["channel_names"]) | set(used)) & (forbidden - {"pid", "day"}))
        self.assertEqual(sorted(used), sorted(self.raw_meta["features"] + ["missing_day", "hrv_missing"]))
        for s in self.S:
            self.assertEqual(self.S[s]["X"].shape[1:], (120, 10))
            self.assertEqual(self.S[s]["static"].shape[1], 4)
            self.assertFalse(np.isnan(self.S[s]["X"]).any())

    def test_masks_match_raw_missingness(self):
        full = {}
        for s, d in self.S.items():
            for i, p in enumerate(d["pid"]):
                full[p] = d["X"][i]
        o = self.obs
        md = o.missing_day.values.reshape(1000, 120)
        hm = o.hrv_missing.values.reshape(1000, 120)
        for p in (0, 17, 400, 999):
            np.testing.assert_array_equal(full[p][:, 8], md[p])
            np.testing.assert_array_equal(full[p][:, 9], hm[p])
        # the mask definitions themselves
        self.assertTrue((o.hrv_missing == o.hrv.isna().astype(int)).all())

    def test_scaling_train_only(self):
        tr = self.S["train"]["X"]
        feats = self.meta["channel_names"][:8]
        for i, f in enumerate(feats):
            maskcol = 9 if f == "hrv" else 8
            obs = tr[:, :, maskcol] == 0
            v = tr[:, :, i][obs]
            self.assertAlmostEqual(float(v.mean()), 0.0, places=3, msg=f)
            self.assertAlmostEqual(float(v.std()), 1.0, places=3, msg=f)
        # same transformation on train/cal/test: recompute scaler from raw TRAIN participants and compare to saved meta
        tr_ids = set(self.S["train"]["pid"].tolist())
        sub = self.obs[self.obs.pid.isin(tr_ids)]
        for f in feats:
            self.assertAlmostEqual(float(sub[f].mean()), self.meta["scaler_mean"][f], places=6)
            self.assertAlmostEqual(float(sub[f].std(ddof=0)), self.meta["scaler_std"][f], places=6)
        # cal/test are not re-centred on their own statistics
        te = self.S["test"]["X"][:, :, 0][self.S["test"]["X"][:, :, 8] == 0]
        self.assertNotEqual(round(float(te.mean()), 6), 0.0)

    def test_values_match_independent_pandas_ffill(self):
        feats = self.meta["channel_names"][:8]
        o = self.obs.copy()
        for f in feats:
            z = (o[f] - self.meta["scaler_mean"][f]) / self.meta["scaler_std"][f]
            o[f + "_z"] = z.groupby(o.pid).ffill(limit=3).fillna(0.0)
        full = {}
        for s, d in self.S.items():
            for i, p in enumerate(d["pid"]):
                full[p] = d["X"][i]
        for p in (3, 250, 777):
            g = o[o.pid == p]
            for i, f in enumerate(feats):
                np.testing.assert_allclose(full[p][:, i], g[f + "_z"].values, atol=1e-5)

    def test_future_perturbation_does_not_change_past_inputs(self):
        cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in self.raw_meta["config"].items()})
        out = simulate(cfg, perturb_after=60)["observations"].sort_values(["pid", "day"])
        feats = self.meta["channel_names"][:8]
        values = out[feats].values.reshape(1000, 120, 8)
        masks = out[["missing_day", "hrv_missing"]].values.reshape(1000, 120, 2).astype(float)
        sc = {"mean": np.array([self.meta["scaler_mean"][f] for f in feats]), "std": np.array([self.meta["scaler_std"][f] for f in feats])}
        Xp = pp.transform(values, masks, sc)
        full = np.zeros((1000, 120, 10), np.float32)
        for s, d in self.S.items():
            full[d["pid"]] = d["X"]
        np.testing.assert_allclose(Xp[:, :60], full[:, :60], atol=1e-5)    # past identical under the same train stats
        self.assertFalse(np.allclose(Xp[:, 70:], full[:, 70:]))             # future really differs

    def test_windows_from_processed(self):
        d = self.S["test"]
        Xw, st, y, pid, t = dd.make_windows(d, "onset_sust_dep_h14")
        self.assertEqual(Xw.shape[1:], (28, 10))
        self.assertGreaterEqual(t.min(), 27)
        rng = np.random.default_rng(0)
        pos = {p: i for i, p in enumerate(d["pid"])}
        for j in rng.choice(len(t), 200, replace=False):
            np.testing.assert_array_equal(Xw[j], d["X"][pos[pid[j]], t[j] - 27:t[j] + 1])
        self.assertTrue(set(np.unique(y)) <= {0.0, 1.0})

    def test_onset_labels_independent_recomputation(self):
        """Recompute primary and secondary labels by plain loops from daily status (not build_labels)."""
        N, D, P, LB = 1000, 120, 3, 27
        full = {k: np.full((N, D), np.nan) for k in []}
        S = {}
        for s, d in self.S.items():
            for key in d:
                if key.startswith("y_"):
                    S.setdefault(key, np.full((N, D), np.nan, dtype=np.float32))[d["pid"]] = d[key]
        for k in ("dep", "anx"):
            st = self.tg[f"status_{k}"].values.reshape(N, D)
            for p in range(0, N):
                s = st[p]
                prevalent = s[0] == 1
                on_s = next((d_ for d_ in range(D - P + 1) if s[d_:d_ + P].all()), None)
                on_c = next((d_ for d_ in range(D) if s[d_]), None)
                for h in (7, 14, 21):
                    for t in range(D):
                        a, b = S[f"y_onset_sust_{k}_h{h}"][p, t], S[f"y_onset_cross_{k}_h{h}"][p, t]
                        e_s = e_c = np.nan
                        if t >= LB and not prevalent:
                            if (on_s is None or on_s > t) and t + h + P - 1 <= D - 1:
                                e_s = float(on_s is not None and t < on_s <= t + h)
                            if (on_c is None or on_c > t) and t + h <= D - 1:
                                e_c = float(on_c is not None and t < on_c <= t + h)
                        self.assertTrue((np.isnan(a) and np.isnan(e_s)) or a == e_s, (k, p, t, h, "sust", a, e_s))
                        self.assertTrue((np.isnan(b) and np.isnan(e_c)) or b == e_c, (k, p, t, h, "cross", b, e_c))
        # also agrees with the simulator's stored onset days
        tp = pd.read_csv(os.path.join(RAW, "truth_participants.csv"))
        for k in ("dep", "anx"):
            st = self.tg[f"status_{k}"].values.reshape(N, D)
            for p in range(N):
                s = st[p]
                on_s = next((d_ for d_ in range(D - P + 1) if s[d_:d_ + P].all()), None)
                exp = np.nan if (s[0] == 1 or on_s is None) else on_s
                got = tp.loc[p, f"onset_day_sust_{k}"]
                self.assertTrue((np.isnan(exp) and np.isnan(got)) or exp == got)

    def test_forecast_targets_boundaries(self):
        d = self.S["test"]
        sev0, sev14 = d["y_sev_dep_t0"], d["y_sev_dep_h14"]
        self.assertTrue(np.isnan(sev0[:, :27]).all())
        self.assertTrue(np.isnan(sev14[:, 106:]).all())            # t+14 > 119
        i, t = 5, 60
        p = d["pid"][i]
        raw = self.tg[self.tg.pid == p].severity_dep.values
        self.assertAlmostEqual(float(sev14[i, t]), float(raw[t + 14]), places=4)
        self.assertAlmostEqual(float(sev0[i, t]), float(raw[t]), places=4)

    def test_raw_unchanged_and_reproducible(self):
        pp.verify_raw(RAW)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            pp.run(RAW, os.path.join(tmp, "again"), readonly=False)
            for s in ("train", "cal", "test"):
                a, b = dd.load_split(s, PROC), dd.load_split(s, os.path.join(tmp, "again"))
                self.assertEqual(set(a), set(b))
                for k in a:
                    np.testing.assert_array_equal(a[k], b[k])


if __name__ == "__main__":
    unittest.main()
