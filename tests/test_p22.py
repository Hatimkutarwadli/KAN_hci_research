"""Tests for P2-2 (versioned SYNTHETIC datasets). Run: python tests/test_p22.py. No test split is loaded by any test."""
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "extension_p22")]
import common_p22 as C  # noqa: E402
import gen_p22 as G  # noqa: E402
import simulate as sim  # noqa: E402

NEW = [d for d in ("R1", "R2", "R3", "S-EFF", "S-AR", "S-MISS", "S-HET", "S-NOISE") if os.path.isdir(os.path.join(ROOT, C.raw_dir(d)))]


def _rmtree_ro(p):
    import shutil
    def onerr(f, path, _):
        os.chmod(path, stat.S_IWRITE)
        f(path)
    shutil.rmtree(p, onerror=onerr)


def test_v1_regenerates_identically_through_wrapper():
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "v1")
        cfg = G.build_config({"seed": 2027, "overrides": {}})
        out, rho = G.simulate_with(cfg, 1.0)
        sim.save(out, d, cfg, final=True)
        assert open(os.path.join(d, "checksums.sha256")).read() == open("data/synthetic_v1/checksums.sha256").read()
        assert rho == sim.RHO.tolist()
    finally:
        _rmtree_ro(tmp)


def test_configs_seeds_and_forbidden_overrides():
    streams = []
    for ds in ("R1", "R2", "R3"):
        c = C.ds_config(ds)
        assert c["id"] == ds and c["family"] == "replication" and c["overrides"] == {} and c["rho_scale"] == 1.0
        s = G.seed_streams(c["seed"])
        assert not (s & C.RESERVED_SEED_STREAMS)
        streams.append(s)
    assert not (streams[0] & streams[1]) and not (streams[0] & streams[2]) and not (streams[1] & streams[2])
    for bad in ({"z_thr_dep": 2.0}, {"seed": 5}, {"n_days": 90}, {"not_a_field": 1}):
        try:
            G.build_config({"seed": 3101, "overrides": bad})
        except AssertionError:
            continue
        raise AssertionError(f"override {bad} accepted")


def test_rho_override_is_temporary_and_crn_holds():
    base = dict(json.load(open(C.FROZEN_CFG)), n_participants=40, seed=3101)
    cfg = sim.Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in base.items()})
    rho0 = sim.RHO.copy()
    a, _ = G.simulate_with(cfg, 1.0)
    b, rb = G.simulate_with(cfg, 0.5)
    assert np.array_equal(sim.RHO, rho0) and np.allclose(rb, rho0 * 0.5)
    assert not a["truth"].equals(b["truth"])
    assert a["splits"].equals(b["splits"]) and a["participants"].equals(b["participants"])          # same static/split streams
    cfg_n = sim.Config(**{**cfg.__dict__, "sensor_sd": 0.7})                                       # sensor-noise shift: latent/targets untouched (CRN)
    c, _ = G.simulate_with(cfg_n, 1.0)
    assert a["truth"].equals(c["truth"]) and a["targets"].equals(c["targets"]) and not a["observations"].equals(c["observations"])


def test_new_datasets_immutable_and_checksummed():
    for ds in NEW:
        raw = C.raw_dir(ds)
        for line in open(os.path.join(raw, "checksums.sha256")).read().strip().splitlines():
            h, fn = line.split()
            assert hashlib.sha256(open(os.path.join(raw, fn), "rb").read()).hexdigest() == h, (ds, fn)
            assert not os.access(os.path.join(raw, fn), os.W_OK), (ds, fn)
        pm = json.load(open(os.path.join(C.proc_dir(ds), "preprocessing_meta.json")))
        for fn, h in pm["output_file_sha256"].items():
            assert hashlib.sha256(open(os.path.join(C.proc_dir(ds), fn), "rb").read()).hexdigest() == h
        prov = json.load(open(f"data/synthetic_p22/_provenance/{ds}.json"))
        assert prov["seed_streams"]["static"] == C.ds_config(ds)["seed"] and np.allclose(prov["RHO_used"], sim.RHO * C.ds_config(ds)["rho_scale"])
        assert json.load(open(os.path.join(raw, "meta.json")))["config"]["z_thr_dep"] == 2.45


def test_validation_is_sealed():
    for ds in NEW:
        v = json.load(open(os.path.join(C.RES, ds, "validation.json")))
        assert v["test_split_inspected"] is False and set(v["train_cal_onset_counts"]) == {"train", "cal"}
        assert v["gates"]["all_pass"] or ds in ("S-EFF", "S-AR", "S-MISS", "S-HET", "S-NOISE")     # replicates must pass; shifts may fail (flagged)


def test_no_p22_code_loads_test_split():
    for fn in ("run_p22.py", "lag_p22.py", "aug_p22.py", "analyze_p22.py", "analyze_p22b.py", "gen_p22.py"):
        src = open(os.path.join("extension_p22", fn)).read()
        assert not re.search(r"""load_split\(\s*["']test""", src) and '"test")' not in src.replace('("train", "cal", "test")', ""), fn
        assert not re.search(r"""splits=\([^)]*test""", src), fn


def test_lag_and_aug_reproduce_v1_references():
    A = pd.read_csv(os.path.join(C.RES, "v1/lag_recovery/metrics.csv"))
    B = pd.read_csv("results/p1_lag_recovery/analysis/metrics_per_seed.csv")
    B = B[(B.kind == "pert") & (B.population == "trained") & B.model.isin(["gru", "m1", "mlp_m1"]) & (B.seed < 3)]
    M = A.merge(B, on=["model", "seed", "feature", "outcome"], suffixes=("", "_ref"))
    assert len(M) == 36
    for c in ("peak_lag_level", "peak_lag_increment_1to5", "sign_ok"):
        assert (M[c] == M[c + "_ref"]).all()
    assert (M.prelag_mass - M.prelag_mass_ref).abs().max() < 1e-9
    G_ = pd.read_csv(os.path.join(C.RES, "v1/augmentation/cal_metrics.csv"))
    ref = pd.read_csv("results/stage9_v2/cal/metrics_R0O0T1/S3_k7_rep0.csv")
    m = G_[(G_.model == "gru_aug") & (G_.scenario == "S3_k7")].merge(ref[ref.seed < 3], on=["seed", "task", "metric", "rep"], suffixes=("", "_ref"))
    assert len(m) > 0 and (m.value - m.value_ref).abs().max() < 1e-6


def test_verdict_rules():
    import analyze_p22 as A
    assert A.verdict([-1, -1.1, -0.9], "rmse") == (3, "consistent")
    assert A.verdict([-1, -0.01, -3], "rmse")[1] == "leaning"
    assert A.verdict([-1, 1, -1], "ap")[1] == "not distinguishable"
    assert A.overall(["supported"] * 4) == "supported" and A.overall(["supported", "mixed", "mixed", "mixed"]) == "not supported"
    assert A.overall(["supported", "supported", "mixed", "supported"]) == "mixed"


def test_runner_reproduces_stage3_gru_seed0():
    tmp = tempfile.mkdtemp()
    try:
        r = subprocess.run([sys.executable, "extension_p22/run_p22.py", "--ds", "v1", "--models", "gru", "--seeds", "0", "--out-root", tmp], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        for ts, ref in (("main", "results/stage3"), ("forecast", "results/stage3c_forecast")):
            a = np.load(os.path.join(tmp, "v1", f"gru_{ts}", "predictions", "gru_seed0_cal.npz"))["pred"]
            b = np.load(os.path.join(ref, "predictions", "gru_seed0_cal.npz"))["pred"]
            assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) == 0.0
    finally:
        _rmtree_ro(tmp)


SHIFTS = [d for d in ("S-EFF", "S-AR", "S-MISS", "S-HET", "S-NOISE") if os.path.isdir(os.path.join(ROOT, C.raw_dir(d)))]
SHIFT_FIELDS = {"S-EFF": {"eff_sv_dep", "eff_steps_dep", "eff_steps_anx", "eff_hrv_anx"}, "S-AR": set(), "S-MISS": {"miss_mean", "miss_persist"},
                "S-HET": {"sens_sd"}, "S-NOISE": {"sensor_sd"}}


def test_shift_configs_differ_only_in_declared_fields():
    frozen = json.load(open(C.FROZEN_CFG))
    r1 = C.sim_config("R1")
    for ds in ("S-EFF", "S-AR", "S-MISS", "S-HET", "S-NOISE"):
        c = C.ds_config(ds)
        assert c["family"] == "shift" and c["seed"] == C.ds_config("R1")["seed"] and c["crn_partner"] == "R1"
        assert set(c["overrides"]) == SHIFT_FIELDS[ds] and (c["rho_scale"] == 0.5) == (ds == "S-AR") and c["rho_scale"] in (0.5, 1.0)
        if ds in SHIFTS:
            m = C.sim_config(ds)
            diff = {k for k in m if m[k] != r1[k]}
            assert diff == SHIFT_FIELDS[ds], (ds, diff)
            assert m["z_thr_dep"] == frozen["z_thr_dep"] and m["z_thr_anx"] == frozen["z_thr_anx"]
            prov = json.load(open(f"data/synthetic_p22/_provenance/{ds}.json"))
            assert np.allclose(prov["RHO_used"], sim.RHO * c["rho_scale"])
    assert C.ds_config("S-EFF")["overrides"] == {k: 2 * frozen[k] for k in SHIFT_FIELDS["S-EFF"]}


def test_shift_common_random_numbers_vs_r1():
    """Approved CRN design: same participants/splits as R1 everywhere; truth/targets identical to R1 where the shift acts only on observation."""
    def rd(ds, f):
        return open(os.path.join(C.raw_dir(ds), f), "rb").read()
    for ds in SHIFTS:
        assert rd(ds, "splits.csv") == rd("R1", "splits.csv"), ds
        a, b = pd.read_csv(os.path.join(C.raw_dir(ds), "truth_participants.csv")), pd.read_csv(os.path.join(C.raw_dir("R1"), "truth_participants.csv"))
        for col in ("pid", "sex", "age_group"):
            if col in a:
                assert a[col].equals(b[col]), (ds, col)
        same_truth = rd(ds, "truth.csv") == rd("R1", "truth.csv") and rd(ds, "targets.csv") == rd("R1", "targets.csv")
        assert same_truth == (ds in ("S-MISS", "S-NOISE")), ds
        assert rd(ds, "observations.csv") != rd("R1", "observations.csv"), ds


def test_shift_gate_policy_recorded():
    for ds in SHIFTS:
        v = json.load(open(os.path.join(C.RES, ds, "validation.json")))
        assert v["test_split_inspected"] is False
        flag = os.path.join(C.RES, ds, "GATE_FAIL_FLAG.json")
        if not v["gates"]["all_pass"] and os.path.isdir(os.path.join(C.RES, ds, "gru_main")):
            assert os.path.exists(flag), ds                          # failing dataset retained and flagged, not silently used


def test_phase_a_outputs_unchanged():
    man = json.load(open(os.path.join(C.RES, "manifest_phase_a.json")))
    bad = [p for p, h in man.items() if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != h]
    print(f"  {len(man)} Phase A files; changed or missing: {len(bad)}")
    assert not bad, bad[:10]


def test_existing_files_unchanged():
    pre = json.load(open(os.path.join(C.RES, "manifest_pre.json")))
    now = C.file_hashes()
    changed = [k for k in pre if now.get(k) != pre[k]]
    print(f"  {len(pre)} pre-existing files; changed or missing: {len(changed)}")
    assert not changed, changed[:10]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and (len(sys.argv) == 1 or any(s in k for s in sys.argv[1:]))]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"{len(tests)} tests pass")
