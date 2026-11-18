"""Tests for the KAN-GRU extension (synthetic data not needed). Run: python -I -m pytest tests/test_kangru.py  (or python tests/test_kangru.py)"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_kangru")]
import baselines as bl  # noqa: E402
import models as md  # noqa: E402
import models_kangru as mk  # noqa: E402

bl.configure("all")
HEADS = list(bl.HEADS)
FROZEN = ("kan", "gru", "m1", "m1_static")


def build(fusion):
    torch.manual_seed(0)
    return mk.KANGRUNet(heads=HEADS, fusion=fusion)


def test_shapes_and_gate():
    for f in ("adaptive", "fixed"):
        m = build(f).eval()
        x, s = torch.randn(5, 28, 10), torch.randn(5, 4)
        assert m(x, s).shape == (5, 28)
        br, gate, fused, out = m.sequence(x)
        assert br.shape == (5, 28, 3, 8) and fused.shape == (5, 28, 8) and out.shape == (5, 28, 32)
        if f == "adaptive":
            assert gate.shape == (5, 28, 3) and torch.allclose(gate.sum(-1), torch.ones(5, 28), atol=1e-5)
        else:
            assert gate is None


def test_causality():
    for f in ("adaptive", "fixed"):
        m = build(f).eval()
        x = torch.randn(3, 28, 10)
        x2 = x.clone()
        x2[:, 20:] += torch.randn(3, 8, 10)          # change days 20..27 only
        a, b = m.sequence(x), m.sequence(x2)
        for u, v in zip(a, b):
            if u is not None:
                assert torch.allclose(u[:, :20], v[:, :20], atol=1e-6)      # nothing before day 20 changes
        assert not torch.allclose(a[3][:, 27], b[3][:, 27])               # but the future does respond


def test_registry_untouched():
    before = {k: md.MODELS[k] for k in FROZEN}
    mk.register()
    assert all(md.MODELS[k] is before[k] for k in FROZEN) and set(md.MODELS) == set(FROZEN) | {"kangru", "kangru_fixed"}


def test_param_counts_reported():
    for f in ("adaptive", "fixed"):
        print(f, md.n_params(build(f)))


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
