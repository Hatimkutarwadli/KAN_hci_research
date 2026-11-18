"""Tests for P1-2 (multitask vs single-task GRU). Run: python tests/test_p12.py. Uses train/calibration data only (the test split is never loaded here)."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p12")]
import common_p12 as C  # noqa: E402

bl, md = C.bl, C.md
SUB = {"train": 40, "cal": 20, "test": 10}


def _small():
    bl.configure("all")
    return bl.load_data("data/processed_v1", SUB, splits=("train", "cal"))


def test_unit_definitions():
    for part, P in C.PARTS.items():
        assert len(set(P["st"])) == len(P["st"])
        assert sorted(sum(P["family"].values(), [])) == sorted(P["st"])          # families partition the single-task heads
        assert not any("cross" in h for h in P["st"])
    assert len(C.PARTS["main"]["st"]) == 10 and len(C.PARTS["forecast"]["st"]) == 12
    assert set(C.PARTS["main"]["st"]) <= set(bl.MAIN_HEADS) and set(C.PARTS["forecast"]["st"]) == set(bl.FORECAST)


def test_unit_labels_identical_to_multitask_labels_and_regression_first():
    data = _small()
    heads = C.configure_unit(["status_dep_t0", "sev_anx_t0"])
    assert heads == ["sev_anx_t0", "status_dep_t0"] and bl.N_REG == 1
    u = C.unit_data(data, heads)
    for j, h in enumerate(heads):
        assert np.array_equal(u["train"]["Y"][:, j], data["train"]["Y"][:, bl.ALL_HEADS.index(h)], equal_nan=True)
    assert u["train"]["X"] is data["train"]["X"] and np.array_equal(u["train"]["pi"], data["train"]["pi"])      # same rows / inputs


def test_parameter_budget_single_head_vs_multitask():
    g16, g1, g12 = md.n_params(md.GRUNet(n_out=16)), md.n_params(md.GRUNet(n_out=1)), md.n_params(md.GRUNet(n_out=12))
    assert g16 - g1 == 15 * 33 and abs(g1 - g16) / g16 < 0.05                    # only the last linear layer differs
    assert g12 - g1 == 11 * 33


def test_single_classification_head_trains_and_evaluates():
    data = _small()
    heads = C.configure_unit(["onset_sust_dep_h14"])
    u = C.unit_data(data, heads)
    assert bl.N_REG == 0
    m, ym, ys, info = bl.fit_nn("gru", 0, u["train"], u["cal"], epochs=1, log=lambda *_: None)
    u["test"] = u["cal"]                                                      # evaluate() expects a 'test' key; alias calibration data (test split stays unloaded)
    P = {s: bl.predict_nn(m, bl.to_t(u[s]), ym, ys) for s in u}
    assert P["cal"].shape[1] == 1 and ((P["cal"] >= 0) & (P["cal"] <= 1)).all()
    rows, thr = bl.evaluate("st_task", 0, P, u)
    assert {r["metric"] for r in rows if r["split"] == "cal"} >= {"ap", "auroc"}


def test_fit_fixed_reproduces_fit_nn_trajectory():
    data = _small()
    heads = C.configure_unit(["sev_dep_t0"])
    u = C.unit_data(data, heads)
    m1, _, _, i1 = bl.fit_nn("gru", 2, u["train"], u["cal"], epochs=2, patience=99, log=lambda *_: None)
    m2, _, _, i2 = C.fit_fixed(2, u["train"], u["cal"], 2, log=lambda *_: None)
    assert [h["train_loss"] for h in i1["history"]] == [h["train_loss"] for h in i2["history"]]        # identical training trajectory (same RNG, same steps)
    assert [h["cal_loss"] for h in i1["history"]] == [h["cal_loss"] for h in i2["history"]]
    m3, _, _, i3 = bl.fit_nn("gru", 2, u["train"], u["cal"], epochs=1, log=lambda *_: None)
    m4, _, _, _ = C.fit_fixed(2, u["train"], u["cal"], 1, log=lambda *_: None)
    assert all(torch.equal(a, b) for a, b in zip(m3.state_dict().values(), m4.state_dict().values()))


def test_training_deterministic_and_seed_dependent():
    data = _small()
    heads = C.configure_unit(["sev_dep_t0", "sev_anx_t0"])
    u = C.unit_data(data, heads)
    a, *_ = bl.fit_nn("gru", 1, u["train"], u["cal"], epochs=2, log=lambda *_: None)
    b, *_ = bl.fit_nn("gru", 1, u["train"], u["cal"], epochs=2, log=lambda *_: None)
    c, *_ = bl.fit_nn("gru", 2, u["train"], u["cal"], epochs=2, log=lambda *_: None)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))


def test_harness_with_full_head_set_reproduces_frozen_stage3_epoch1():
    """Routing the 16 main heads through configure_unit/unit_data must give the frozen multitask GRU's first epoch exactly (Stage 3 log: train 0.2931 cal 0.2220)."""
    bl.configure("all")
    data = bl.load_data("data/processed_v1", None, splits=("train", "cal"))
    heads = C.configure_unit(list(bl.MAIN_HEADS))
    assert heads == list(bl.MAIN_HEADS)
    u = C.unit_data(data, heads)
    _, _, _, info = bl.fit_nn("gru", 0, u["train"], u["cal"], epochs=1, log=lambda *_: None)
    assert round(info["history"][0]["train_loss"], 4) == 0.2931 and round(info["history"][0]["cal_loss"], 4) == 0.2220


def test_mt_best_epochs_from_existing_logs():
    assert C.mt_best_epochs("main") == {0: 4, 1: 3, 2: 4, 3: 3, 4: 3}
    assert C.mt_best_epochs("forecast") == {0: 4, 1: 3, 2: 2, 3: 2, 4: 2}


def test_runner_refuses_nonempty_output_directory():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "x"), "w").close()
        r = subprocess.run([sys.executable, "extension_p12/run_p12.py", "--design", "st_task", "--part", "main", "--out", d, "--quick"], capture_output=True, text=True)
        assert r.returncode != 0 and "refusing to overwrite" in (r.stdout + r.stderr)


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
    sys.exit(1 if fails else 0)
