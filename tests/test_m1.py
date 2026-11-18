"""Stage 4 tests for the proposed latent KAN model (M1) and its no-transition ablation. Run: python -m unittest tests/test_m1.py"""
import hashlib
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import baselines as bl  # noqa: E402
import models as md  # noqa: E402

torch.set_num_threads(2)


def make(mode="m1", **kw):
    bl.configure("all")
    return md.MODELS[mode](heads=list(bl.HEADS), n_out=len(bl.HEADS), **kw)


class TestArchitecture(unittest.TestCase):
    def tearDown(self):
        bl.configure("main")

    def test_shapes_and_parameter_budget(self):
        for mode in ("m1", "m1_static"):
            m = make(mode)
            x, s = torch.randn(6, 28, 10), torch.randn(6, 4)
            out, z, traj = m(x, s, return_latent=True)
            self.assertEqual(out.shape, (6, 28))
            self.assertEqual(z.shape, (6, 8))
            if mode == "m1":
                self.assertEqual(traj.shape, (6, 28, 8))
            else:
                self.assertIsNone(traj)                                             # no trajectory without a transition
            self.assertLess(md.n_params(m), 30_000)

    def test_head_routing_separate_dep_anx_heads(self):
        m = make("m1")
        for lin in m.head_lin:
            torch.nn.init.zeros_(lin.weight)
        m.head_lin[0].bias.data = torch.arange(14, dtype=torch.float32) + 1          # depression head
        m.head_lin[1].bias.data = torch.arange(14, dtype=torch.float32) + 101        # anxiety head
        out = m.eval()(torch.randn(2, 28, 10), torch.randn(2, 4))
        for j, name in enumerate(bl.HEADS):
            k = "dep" if "_dep_" in name else "anx"
            pos = [t.format(k=k) for t in md.COND_TASKS].index(name)
            self.assertAlmostEqual(float(out[0, j]), pos + 1 + (0 if k == "dep" else 100), places=5, msg=name)
        self.assertEqual(len(set(bl.HEADS)), 28)

    def test_encoder_and_latent_transition_are_causal(self):
        m = make("m1").eval()
        x, s = torch.randn(4, 28, 10), torch.randn(4, 4)
        _, _, tr1 = m(x, s, return_latent=True)
        x2 = x.clone()
        x2[:, 15:] += torch.randn(4, 13, 10) * 3                                    # change days 15.. only
        _, _, tr2 = m(x2, s, return_latent=True)
        torch.testing.assert_close(tr1[:, :15], tr2[:, :15], atol=1e-6, rtol=0)    # z_0..z_14 unchanged (no future info)
        self.assertFalse(torch.allclose(tr1[:, 20:], tr2[:, 20:]))                  # later states do change

    def test_prediction_at_origin_ignores_days_after_origin(self):
        m = make("m1").eval()
        rng = np.random.default_rng(0)
        X = torch.from_numpy(rng.normal(size=(3, 120, 10)).astype(np.float32))
        S = torch.zeros(3, 4)
        T = {"X": X, "S": S, "pi": torch.tensor([1]), "ti": torch.tensor([60]), "Y": torch.zeros(1, 28)}
        x1, s1, _ = bl.batch(T, torch.tensor([0]))
        X2 = X.clone()
        X2[:, 61:] += 9.0                                                           # everything after the origin day
        T2 = dict(T, X=X2)
        x2, s2, _ = bl.batch(T2, torch.tensor([0]))
        torch.testing.assert_close(m(x1, s1), m(x2, s2))

    def test_state_depends_on_temporal_order_and_early_days(self):
        m = make("m1").eval()
        x, s = torch.randn(8, 28, 10), torch.zeros(8, 4)
        _, z, _ = m(x, s, return_latent=True)
        perm = torch.randperm(28, generator=torch.Generator().manual_seed(1))
        _, z_shuf, _ = m(x[:, perm], s, return_latent=True)
        self.assertFalse(torch.allclose(z, z_shuf, atol=1e-4))                      # order matters (evolution, not a bag)
        x2 = x.clone()
        x2[:, :5] += 2.0
        _, z2, _ = m(x2, s, return_latent=True)
        self.assertFalse(torch.allclose(z, z2, atol=1e-5))                          # early days still influence the state
        _, _, tr = m(x, s, return_latent=True)
        self.assertGreater(float((tr[:, 1:] - tr[:, :-1]).abs().mean()), 1e-4)      # the latent actually moves over the window

    def test_dropout_inactive_in_eval_and_default_off(self):
        m = make("m1", dropout=0.3)
        self.assertEqual(sum(isinstance(x, torch.nn.Dropout) for x in m.modules()), 2)
        x, s = torch.randn(3, 28, 10), torch.randn(3, 4)
        m.eval()
        torch.testing.assert_close(m(x, s), m(x, s))
        self.assertFalse(any(isinstance(x, torch.nn.Dropout) for x in make("m1").modules()))

    def test_training_reproducible_and_uses_only_train_cal(self):
        bl.configure("all")
        rng = np.random.default_rng(0)
        d = {}
        for sname, N in (("train", 8), ("cal", 4)):
            pi, ti = np.meshgrid(np.arange(N), np.arange(27, 60), indexing="ij")
            Y = np.full((N * 33, 28), np.nan, np.float32)
            Y[:, :8] = rng.normal(8, 3, (N * 33, 8))
            Y[:, 8:] = (rng.random((N * 33, 20)) < 0.1)
            d[sname] = {"X": rng.normal(size=(N, 120, 10)).astype(np.float32), "static": np.zeros((N, 4), np.float32),
                        "pi": pi.ravel(), "ti": ti.ravel(), "Y": Y}
        outs = []
        for _ in range(2):
            m, ym, ys, info = bl.fit_nn("m1", 3, d["train"], d["cal"], epochs=2, bs=64, model_kwargs={"heads": list(bl.HEADS)}, log=lambda *_: None)
            outs.append(bl.predict_nn(m, bl.to_t(d["cal"]), ym, ys))
        np.testing.assert_array_equal(outs[0], outs[1])
        self.assertEqual(outs[0].shape, (4 * 33, 28))


RES = os.path.join(ROOT, "results", "stage4_m1")
RES_ABL = os.path.join(ROOT, "results", "stage4_ablation")


@unittest.skipUnless(os.path.isfile(os.path.join(RES, "run_log.json")), "Stage 4 results not present")
class TestStage4Results(unittest.TestCase):
    def test_saved_outputs(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(RES, "metrics_long.csv"))
        counts = pd.read_csv(os.path.join(RES, "counts.csv")).set_index(["task", "split"])
        self.assertEqual(set(df.task), set(bl.ALL_HEADS))
        self.assertIn("m1", set(df.model))
        if os.path.isfile(os.path.join(RES_ABL, "run_log.json")):
            self.assertIn("m1_static", set(pd.read_csv(os.path.join(RES_ABL, "metrics_long.csv")).model))
        s3 = pd.read_csv(os.path.join(ROOT, "results", "stage3", "counts.csv")).set_index(["task", "split"])
        for t in ("onset_sust_dep_h14", "status_anx_t0", "onset_cross_dep_h21"):
            for s in ("train", "cal", "test"):
                self.assertEqual(int(counts.loc[(t, s), "eligible_origins"]), int(s3.loc[(t, s), "eligible_origins"]))
        for f in os.listdir(os.path.join(RES, "predictions")):
            z = np.load(os.path.join(RES, "predictions", f))
            self.assertEqual(z["pred"].shape, (200, 120, 28))
            self.assertTrue(np.isnan(z["pred"][:, :27]).all() and np.isfinite(z["pred"][:, 27:]).all())

    def test_previous_results_unchanged(self):
        for chk in ("stage3_checksums.sha256", "stage3bcd_checksums.sha256"):
            with open(os.path.join(ROOT, "results", chk)) as f:
                for line in f:
                    h, p = line.split()
                    with open(os.path.join(ROOT, p), "rb") as g:
                        self.assertEqual(hashlib.sha256(g.read()).hexdigest(), h, p)


if __name__ == "__main__":
    unittest.main()
