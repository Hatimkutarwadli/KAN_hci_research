"""Stage 3 baseline tests. Run: python -m unittest tests/test_baselines.py"""
import hashlib
import inspect
import json
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import baselines as bl  # noqa: E402
import features as ft  # noqa: E402
import metrics as mt  # noqa: E402
import models as md  # noqa: E402

torch.set_num_threads(2)
H = len(bl.HEADS)


def toy_data(seed=0, n=(10, 5, 5)):
    rng = np.random.default_rng(seed)
    out = {}
    for s, N in zip(("train", "cal", "test"), n):
        X = rng.normal(size=(N, 120, 10)).astype(np.float32)
        X[:, :, 8:] = (rng.random((N, 120, 2)) < 0.15)
        static = rng.integers(0, 2, size=(N, 4)).astype(np.float32)
        pi, ti = np.meshgrid(np.arange(N), np.arange(27, 120), indexing="ij")
        pi, ti = pi.ravel(), ti.ravel()
        Y = np.full((len(pi), H), np.nan, np.float32)
        Y[:, :2] = rng.normal(8, 3, size=(len(pi), 2))
        for j in range(2, H):
            Y[:, j] = (rng.random(len(pi)) < 0.1).astype(np.float32)
            Y[rng.random(len(pi)) < 0.3, j] = np.nan if j >= 4 else Y[0, j]   # onset heads have undefined origins
        out[s] = dict(X=X, static=static, pid=np.arange(N), pi=pi, ti=ti, Y=Y, pid_o=np.arange(N)[pi])
    return out


class TestModels(unittest.TestCase):
    def test_spline_bases_partition_of_unity(self):
        layer = md.KANLayer(3, 2)
        x = torch.linspace(-2.99, 2.99, 50).unsqueeze(1).repeat(1, 3)
        b = layer.bases(x)
        self.assertEqual(b.shape, (50, 3, layer.G + layer.k))
        torch.testing.assert_close(b.sum(-1), torch.ones(50, 3), atol=1e-5, rtol=0)

    def test_fused_kan_conv_equals_direct_kan_layer(self):
        torch.manual_seed(0)
        conv = md.KANConv(10, 5, 8)
        x = torch.randn(4, 28, 10)
        p = x.unfold(1, 5, 1).reshape(4, 24, -1)
        h = conv.kan(p)
        ref = torch.cat([h.mean(1), h.amax(1)], -1)
        conv2 = md.KANConv(10, 5, 8)
        conv2.kan.base_weight.data = conv.kan.base_weight.data.clone()
        Gk = conv.kan.G + conv.kan.k
        conv2.kan.spline_weight.data = conv.kan.spline_weight.data.view(8, 10, 5, Gk).permute(0, 1, 3, 2).reshape(8, -1)
        torch.testing.assert_close(conv2(x), ref, atol=1e-5, rtol=1e-5)

    def test_output_shapes_and_budget(self):
        x, s = torch.randn(6, 28, 10), torch.randn(6, 4)
        for name in ("kan", "gru"):                                               # baselines; M1 is tested in test_m1.py
            cls = md.MODELS[name]
            m = cls(n_out=H)
            self.assertEqual(m(x, s).shape, (6, H), name)
            self.assertLess(md.n_params(m), 50_000, name)

    def test_masked_loss_ignores_nan_labels(self):
        out = torch.randn(8, H)
        y = torch.rand(8, H).round()
        y[:, :2] = torch.randn(8, 2)
        y[0:3, 5] = float("nan")
        ym, ys = torch.zeros(2), torch.ones(2)
        l1 = bl.masked_loss(out, y, ym, ys)
        out2 = out.clone()
        out2[0:3, 5] = 1e3                      # change predictions only where the label is undefined
        self.assertAlmostEqual(float(l1), float(bl.masked_loss(out2, y, ym, ys)), places=6)
        y_all_nan = y.clone()
        y_all_nan[:, 6] = float("nan")
        self.assertTrue(torch.isfinite(bl.masked_loss(out, y_all_nan, ym, ys)))


class TestFeatures(unittest.TestCase):
    def test_shapes_and_toy_values(self):
        W = np.zeros((1, 28, 10), np.float32)
        W[0, :, 0] = np.arange(28)
        W[0, 20:, 8] = 1                        # last 8 days missing (all features except HRV pattern)
        W[0, :, 9] = 1                          # HRV never observed
        f = ft.window_features(W, np.array([[1, 1, 0, 0]], np.float32))
        self.assertEqual(f.shape, (1, len(ft.NAMES)))
        n = {k: i for i, k in enumerate(ft.NAMES)}
        self.assertAlmostEqual(float(f[0, n["steps_mean28"]]), 9.5, places=4)       # mean of days 0..19
        self.assertAlmostEqual(float(f[0, n["steps_last_obs"]]), 19.0, places=4)
        self.assertAlmostEqual(float(f[0, n["steps_slope28"]]), 1.0, places=4)
        self.assertEqual(float(f[0, n["trailing_missing"]]), 8.0)
        self.assertTrue(np.isnan(f[0, n["hrv_mean28"]]))                             # no observed HRV
        self.assertEqual(float(f[0, n["trailing_hrv_missing"]]), 28.0)
        self.assertAlmostEqual(float(f[0, n["miss_frac28"]]), 8 / 28, places=5)
        self.assertTrue(np.isnan(f[0, n["steps_mean3"]]))                            # last 3 days all missing
        self.assertTrue(np.isnan(f[0, n["steps_mean7"]]))                            # days 21..27 all missing
        W[0, 24:, 8] = 1
        W[0, 20:24, 8] = 0                      # days 20..23 observed again -> last-7 window has 3 observed days
        f2 = ft.window_features(W, np.array([[1, 1, 0, 0]], np.float32))
        self.assertAlmostEqual(float(f2[0, n["steps_mean7"]]), np.mean([21, 22, 23]), places=4)

    def test_features_use_only_window_days(self):
        d = toy_data()["train"]
        a = bl.gbt_features(d)
        d2 = dict(d)
        X2 = d["X"].copy()
        for p, t in zip(d["pi"][:1], d["ti"][:1]):
            pass
        # perturb every day AFTER the origin of a chosen (pid, t); features of that origin must be unchanged
        pid, t = 3, 60
        X2[pid, t + 1:] += 5.0
        d2["X"] = X2
        b = bl.gbt_features(d2)
        row = int(np.where((d["pi"] == pid) & (d["ti"] == t))[0][0])
        np.testing.assert_array_equal(a[row], b[row])
        row_after = int(np.where((d["pi"] == pid) & (d["ti"] == t + 30))[0][0])
        self.assertFalse(np.array_equal(a[row_after], b[row_after]))


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        y, p = np.array([0., 1, 2, 3]), np.array([0., 1, 2, 5])
        self.assertAlmostEqual(mt.reg_metrics(y, p)["rmse"], 1.0, places=6)
        yy = np.array([0, 0, 1, 1])
        m = mt.clf_metrics(yy, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
        self.assertEqual((m["ap"], m["auroc"], m["precision"], m["recall"], m["f1"]), (1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(m["n_pos"], 2)
        thr = mt.best_f1_threshold(yy, np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertTrue(0.2 < thr <= 0.8)

    def test_participant_bootstrap_ci_contains_point(self):
        rng = np.random.default_rng(0)
        pid = np.repeat(np.arange(40), 20)
        y = (rng.random(800) < 0.2).astype(float)
        p = np.clip(y * 0.3 + rng.random(800) * 0.7, 0, 1)
        lo, hi = mt.participant_bootstrap(y, p, pid, lambda a, b: mt.clf_metrics(a, b)["auroc"], 100)
        pt = mt.clf_metrics(y, p)["auroc"]
        self.assertTrue(lo <= pt <= hi)


class TestReproducibilityAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = toy_data()

    def _train(self, name, seed):
        m, ym, ys, info = bl.fit_nn(name, seed, self.data["train"], self.data["cal"], epochs=2, bs=128, log=lambda *_: None)
        return bl.predict_nn(m, bl.to_t(self.data["cal"]), ym, ys), info

    def test_nn_reproducible_same_seed(self):
        for name in ("gru", "kan"):
            p1, i1 = self._train(name, 3)
            p2, i2 = self._train(name, 3)
            np.testing.assert_array_equal(p1, p2)
            self.assertEqual(i1["best_epoch"], i2["best_epoch"])
            p3, _ = self._train(name, 4)
            self.assertFalse(np.array_equal(p1, p3))
            self.assertEqual(p1.shape, (len(self.data["cal"]["pi"]), H))
            self.assertTrue(np.isfinite(p1).all())

    def test_gbt_deterministic(self):
        d = self.data
        Xtr, Xca = bl.gbt_features(d["train"]), bl.gbt_features(d["cal"])
        j = 4
        tr, ca = d["train"]["Y"][:, j], d["cal"]["Y"][:, j]
        a, _ = bl.fit_gbt_task(bl.HEADS[j], Xtr[~np.isnan(tr)], tr[~np.isnan(tr)], Xca[~np.isnan(ca)], ca[~np.isnan(ca)], 0, (3,), 20, 10)
        b, _ = bl.fit_gbt_task(bl.HEADS[j], Xtr[~np.isnan(tr)], tr[~np.isnan(tr)], Xca[~np.isnan(ca)], ca[~np.isnan(ca)], 0, (3,), 20, 10)
        np.testing.assert_array_equal(a.predict_proba(Xca)[:, 1], b.predict_proba(Xca)[:, 1])

    def test_training_functions_never_receive_test_data(self):
        self.assertEqual(list(inspect.signature(bl.fit_nn).parameters)[:4], ["name", "seed", "tr", "ca"])
        self.assertNotIn("test", " ".join(inspect.signature(bl.fit_nn).parameters))
        self.assertNotIn("test", " ".join(inspect.signature(bl.fit_gbt_task).parameters))

    def test_selection_and_thresholds_independent_of_test_labels(self):
        d = self.data
        P = {s: np.random.default_rng(1).random((len(d[s]["pi"]), H)).astype(np.float32) for s in d}
        P["train"][:, :2] *= 20
        P["cal"][:, :2] *= 20
        P["test"][:, :2] *= 20
        rows1, thr1 = bl.evaluate("m", 0, P, d)
        d2 = {s: dict(v) for s, v in d.items()}
        d2["test"]["Y"] = np.random.default_rng(9).permuted(d["test"]["Y"], axis=0)    # scramble TEST labels only
        rows2, thr2 = bl.evaluate("m", 0, P, d2)
        self.assertEqual(thr1, thr2)                                                    # thresholds come from CAL only
        key = lambda r: (r["task"], r["split"], r["metric"])
        r1, r2 = {key(r): r["value"] for r in rows1}, {key(r): r["value"] for r in rows2}
        same_cal = all(np.isclose(r1[k], r2[k], equal_nan=True) for k in r1 if k[1] in ("train", "cal"))
        self.assertTrue(same_cal)
        self.assertFalse(all(np.isclose(r1[k], r2[k], equal_nan=True) for k in r1 if k[1] == "test"))
        d3 = {s: dict(v) for s, v in d.items()}
        d3["cal"]["Y"] = np.random.default_rng(5).permuted(d["cal"]["Y"], axis=0)       # scramble CAL labels -> thresholds move
        _, thr3 = bl.evaluate("m", 0, P, d3)
        self.assertNotEqual(thr1, thr3)

    def test_processed_data_unchanged(self):
        proc = os.path.join(ROOT, "data", "processed_v1")
        if not os.path.isdir(proc):
            self.skipTest("no processed data")
        meta = json.load(open(os.path.join(proc, "preprocessing_meta.json")))
        for fn, h in meta["output_file_sha256"].items():
            with open(os.path.join(proc, fn), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), h, fn)


class TestRegularizationAndForecast(unittest.TestCase):
    def tearDown(self):
        bl.configure("main")

    def test_dropout_option_keeps_parameter_count_and_default_is_unchanged(self):
        m0, m1 = md.MTLNetLite(n_out=H), md.MTLNetLite(n_out=H, dropout=0.3)
        self.assertEqual(md.n_params(m0), md.n_params(m1))
        self.assertFalse(any(isinstance(x, torch.nn.Dropout) for x in m0.modules()))
        self.assertEqual(sum(isinstance(x, torch.nn.Dropout) for x in m1.modules()), 2)
        self.assertEqual(list(m0.state_dict()), list(m1.state_dict()))
        m1.eval()
        m0.load_state_dict(m1.state_dict())
        x, s = torch.randn(3, 28, 10), torch.randn(3, 4)
        torch.testing.assert_close(m0.eval()(x, s), m1(x, s))                     # dropout is inactive in eval mode

    def test_selection_loader_can_exclude_test(self):
        proc = os.path.join(ROOT, "data", "processed_v1")
        if not os.path.isdir(proc):
            self.skipTest("no processed data")
        d = bl.load_data(proc, {"train": 4, "cal": 3, "test": 3}, splits=("train", "cal"))
        self.assertEqual(set(d), {"train", "cal"})

    def test_forecast_taskset_and_toy_training(self):
        bl.configure("forecast")
        self.assertEqual(len(bl.HEADS), 12)
        self.assertEqual(bl.N_REG, 6)
        self.assertTrue(all(h.startswith("sev_") for h in bl.HEADS[:6]) and all(h.startswith("status_") for h in bl.HEADS[6:]))
        self.assertTrue(set(bl.HEADS).isdisjoint(bl.MAIN_HEADS))                    # kept separate from onset/nowcast tasks
        rng = np.random.default_rng(0)
        n = 200
        tr = {"X": rng.normal(size=(5, 120, 10)).astype(np.float32), "static": np.zeros((5, 4), np.float32),
              "pi": np.repeat(np.arange(5), 40), "ti": np.tile(np.arange(27, 67), 5)}
        Y = np.full((n, 12), np.nan, np.float32)
        Y[:, :6] = rng.normal(8, 3, (n, 6))
        Y[:, 6:] = (rng.random((n, 6)) < 0.1)
        tr["Y"] = Y
        m, ym, ys, _ = bl.fit_nn("gru", 0, tr, tr, epochs=1, bs=64, log=lambda *_: None)
        p = bl.predict_nn(m, bl.to_t(tr), ym, ys)
        self.assertEqual(p.shape, (n, 12))
        self.assertTrue(((p[:, 6:] >= 0) & (p[:, 6:] <= 1)).all())

    def test_forecast_labels_are_future_values_with_censoring(self):
        proc = os.path.join(ROOT, "data", "processed_v1")
        raw = os.path.join(ROOT, "data", "synthetic_v1")
        if not (os.path.isdir(proc) and os.path.isdir(raw)):
            self.skipTest("no data")
        import pandas as pd
        d = bl.dd.load_split("test", proc)
        tg = pd.read_csv(os.path.join(raw, "targets.csv")).sort_values(["pid", "day"])
        for k in ("dep", "anx"):
            st = tg[f"status_{k}"].values.reshape(1000, 120)
            sv = tg[f"severity_{k}"].values.reshape(1000, 120)
            for h in (7, 14, 21):
                ys, yv = d[f"y_status_{k}_h{h}"], d[f"y_sev_{k}_h{h}"]
                self.assertTrue(np.isnan(ys[:, :27]).all() and np.isnan(ys[:, 120 - h:]).all())    # t<27 and t+h>119 undefined
                self.assertTrue(np.isfinite(ys[:, 27:120 - h]).all())
                for i in (0, 7, 150):
                    p = d["pid"][i]
                    for t in (27, 60, 119 - h):
                        self.assertEqual(ys[i, t], st[p, t + h])
                        self.assertAlmostEqual(float(yv[i, t]), float(sv[p, t + h]), places=4)


RES = os.path.join(ROOT, "results", "stage3")


@unittest.skipUnless(os.path.isfile(os.path.join(RES, "run_log.json")), "Stage 3 results not present")
class TestSavedResults(unittest.TestCase):
    def test_outputs(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(RES, "metrics_long.csv"))
        counts = pd.read_csv(os.path.join(RES, "counts.csv"))
        self.assertEqual(set(df.task), set(bl.HEADS))
        self.assertEqual(set(df.split), {"train", "cal", "test"})
        self.assertEqual(set(df.model) - {"B0_reference"}, {"gbt", "kan", "gru"})
        # counts agree with the frozen Stage 2 metadata
        meta = json.load(open(os.path.join(ROOT, "data", "processed_v1", "preprocessing_meta.json")))
        c = counts.set_index(["task", "split"])
        for t in ("onset_sust_dep_h14", "onset_cross_anx_h7", "onset_sust_anx_h21"):
            for s in ("train", "cal", "test"):
                e = meta["split_counts"][s][t]
                self.assertEqual(int(c.loc[(t, s), "eligible_origins"]), e["eligible_origins"])
                self.assertEqual(int(c.loc[(t, s), "positive_origins"]), e["positive_origins"])
        for f in os.listdir(os.path.join(RES, "predictions")):
            z = np.load(os.path.join(RES, "predictions", f))
            self.assertEqual(z["pred"].shape, (200, 120, H))
            self.assertTrue(np.isnan(z["pred"][:, :27]).all() and np.isfinite(z["pred"][:, 27:]).all())
        # B0 reference: AP equals prevalence (sanity of the metric code)
        r = df[(df.model == "B0_reference") & (df.split == "test") & (df.task == "onset_sust_dep_h14")].set_index("metric").value
        self.assertAlmostEqual(r["ap"], r["prevalence"], places=6)


RES_F = os.path.join(ROOT, "results", "stage3c_forecast")
RES_R = os.path.join(ROOT, "results", "stage3b_kan_reg")


@unittest.skipUnless(os.path.isfile(os.path.join(RES_F, "run_log.json")), "forecast results not present")
class TestForecastAndRegularizationResults(unittest.TestCase):
    def test_forecast_outputs_shapes_and_counts(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(RES_F, "metrics_long.csv"))
        counts = pd.read_csv(os.path.join(RES_F, "counts.csv")).set_index(["task", "split"])
        self.assertEqual(set(df.task), set(bl.FORECAST))
        self.assertTrue(set(df.task).isdisjoint(bl.MAIN_HEADS))                      # separate from onset/nowcast
        self.assertEqual(set(df.model) - {"B0_reference"}, {"gbt", "gru", "kan", "kan_reg"})
        for t in bl.FORECAST:
            h = int(t.split("_h")[1])
            for s, n in (("train", 600), ("cal", 200), ("test", 200)):
                self.assertEqual(int(counts.loc[(t, s), "eligible_origins"]), n * (93 - h))   # t in 27..119-h, all participants
        for f in os.listdir(os.path.join(RES_F, "predictions")):
            z = np.load(os.path.join(RES_F, "predictions", f))
            self.assertEqual(z["pred"].shape, (200, 120, 12))
            self.assertTrue(np.isnan(z["pred"][:, :27]).all() and np.isfinite(z["pred"][:, 27:]).all())
        r = df[(df.model == "B0_reference") & (df.split == "test") & (df.task == "status_dep_h14")].set_index("metric").value
        self.assertAlmostEqual(r["ap"], r["prevalence"], places=6)

    def test_selection_rule_and_test_not_used(self):
        import json as js
        sel = js.load(open(os.path.join(RES_R, "selection", "selection_table.json")))
        best = min(sel["table"], key=lambda r: r["mean_best_cal_loss"])
        self.assertEqual((best["wd"], best["dropout"]), (sel["chosen"]["wd"], sel["chosen"]["dropout"]))
        src = open(os.path.join(ROOT, "src", "kan_regularization.py"), encoding="utf-8").read()
        self.assertIn('splits=("train", "cal")', src)
        self.assertNotIn("test", src.replace("test split", "").replace("test file", "").replace("TEST", "").replace("test is", "").replace("Test", "").replace("tests", ""))

    def test_stage3_results_unchanged(self):
        import hashlib
        chk = os.path.join(ROOT, "results", "stage3_checksums.sha256")
        for line in open(chk):
            h, p = line.split()
            with open(os.path.join(ROOT, p), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), h, p)


if __name__ == "__main__":
    unittest.main()
