"""P2-2 single locked TEST evaluation (SYNTHETIC; inference only) implementing results/p2_datasets/TEST_EVALUATION_PLAN.md.
  python extension_p22/eval_test_p22.py verify-cal   pre-lock: regenerate every saved CALIBRATION prediction with the test-generation code (no test access)
  python extension_p22/eval_test_p22.py lock         write results/p2_datasets/test_lock.json (SHA-256 of plan, code, configs, checkpoints, data, settings)
  python extension_p22/eval_test_p22.py run          verify lock, then evaluate every test split EXACTLY ONCE (refuses if results/p2_datasets/test exists)
No training, recalibration, tuning or selection happens here."""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p22 as C  # noqa: E402
import aug_p22 as AG  # noqa: E402
import baselines as bl  # noqa: E402
import data as dd  # noqa: E402
import inject as ij  # noqa: E402
import lag_p22 as L  # noqa: E402
import models as md  # noqa: E402
import run_missing as rm  # noqa: E402
import v2  # noqa: E402

FRESH = ["R1", "R2", "R3"]
SHIFTS = ["S-EFF", "S-AR", "S-MISS", "S-HET", "S-NOISE"]
NEW = FRESH + SHIFTS
TEST_ROOT = os.path.join(C.RES, "test")
LOCK = os.path.join(C.RES, "test_lock.json")
SETTINGS = {"torch_threads": 4, "threadpool_limit": 4, "n_lag_origins": 1000, "lag_origin_rng": "RandomState(0)", "aug_scenarios": AG.SCEN,
            "aug_origins": "27,29,...,119", "models": {m: list(C.CORE[m]["tasksets"]) for m in C.CORE} | {C.AUG: ["main"]}, "seeds": list(C.SEEDS),
            "gbt_seed": 0, "v1_role": "historical reference (test previously inspected); existing v1 test predictions reused"}
LOCK_FILES = ["results/p2_datasets/TEST_EVALUATION_PLAN.md", "results/p2_datasets/ANALYSIS_PLAN_PHASE_A.md", "results/p2_datasets/ANALYSIS_PLAN_PHASE_B.md",
              "configs/frozen_config_v1.json", "tests/test_p22.py",
              "src/baselines.py", "src/models.py", "src/data.py", "src/metrics.py", "src/simulate.py", "src/preprocess.py", "src/causal_analysis.py",
              "extension_p13/models_p13.py", "extension_missing/inject.py", "extension_missing/run_missing.py", "extension_v2/v2.py", "extension_v2/train_v2.py",
              "extension_p14/common_p14.py"]
LOCK_GLOBS = ["extension_p22/*.py", "configs/p22/*.json", "results/p2_datasets/*/*/checkpoints/*", "results/p2_datasets/phase_a_cal/*", "results/p2_datasets/phase_b_cal/*",
              "results/p2_datasets/*/*/predictions/*_cal.npz", "results/p2_datasets/*/validation.json", "results/p2_datasets/S-EFF/GATE_FAIL_FLAG.json",
              "data/processed_p22/*/*", "data/synthetic_p22/*/*", "data/synthetic_p22/_provenance/*", "data/processed_v1/test.npz", "data/processed_v1/preprocessing_meta.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def locked_paths():
    ps = set(LOCK_FILES)
    for g in LOCK_GLOBS:
        ps |= {p.replace("\\", "/") for p in glob.glob(g) if os.path.isfile(p)}
    for (m, ts), tpl in C.V1_CKPT.items():
        ps |= {tpl.format(s=s) for s in C.SEEDS}
    for (m, ts), tpl in C.V1_PRED.items():
        ps |= {tpl.format(s=s, split=sp) for s in ([0] if m == "gbt" else C.SEEDS) for sp in ("cal", "test")}
    return sorted(ps)


def verify_lock():
    lk = json.load(open(LOCK))
    bad = [p for p, h in lk["files"].items() if not os.path.exists(p) or sha(p) != h]
    assert not bad, f"LOCK VIOLATION: {len(bad)} files changed, e.g. {bad[:5]}"
    assert lk["settings"] == json.loads(json.dumps(SETTINGS)), "LOCK VIOLATION: settings changed"
    return lk


# ------------------------------------------------------------------ inference (identical code path for cal verification and test)
def build_nn(model, ck):
    spec = C.CORE[model] if model != C.AUG else None
    heads = list(ck["heads"])
    if model == C.AUG:
        return md.MODELS["gru"](n_out=len(heads), n_ch=ck.get("n_ch", 10))
    mk = {"dropout": spec["dropout"]} if spec["dropout"] > 0 else {}
    if spec["arch"].endswith("m1"):
        mk["heads"] = heads
    return md.MODELS[spec["arch"]](n_out=len(heads), **mk)


def predict(ds, model, ts, seed, split):
    bl.configure(ts)
    d = bl.load_data(C.proc_dir(ds), None, splits=(split,))[split]
    if model == "gbt":
        fitted = joblib.load(os.path.join(C.ROOT, C.ckpt_path(ds, "gbt", ts, seed).replace(".pt", ".joblib")))
        with threadpool_limits(bl.THREADS):
            F_ = bl.gbt_features(d)
            P = np.stack([fitted[t].predict(F_) if t in bl.REG else fitted[t].predict_proba(F_)[:, 1] for t in bl.HEADS], 1).astype(np.float32)
    else:
        ck = torch.load(os.path.join(C.ROOT, C.ckpt_path(ds, model, ts, seed)), map_location="cpu", weights_only=False)
        assert list(ck["heads"]) == list(bl.HEADS)
        m = build_nn(model, ck)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        P = bl.predict_nn(m, bl.to_t(d), ck["ymean"], ck["ystd"])
    full = np.full((len(d["pid"]), 120, len(bl.HEADS)), np.nan, np.float32)
    full[d["pi"], d["ti"]] = P
    return full, d["pid"], list(bl.HEADS)


def units():
    for ds in NEW:
        for model, spec in C.CORE.items():
            for ts in spec["tasksets"]:
                for seed in ([0] if model == "gbt" else C.SEEDS):
                    yield ds, model, ts, seed


def test_pred_path(ds, model, ts, seed, split="test"):
    assert split == "test"
    if ds == "v1":
        return C.pred_path(ds, model, ts, seed, "test")
    return os.path.join(TEST_ROOT, ds, f"{model}_{ts}", f"{model}_seed{seed}_test.npz")


def verify_cal():
    rows = []
    for ds, model, ts, seed in units():
        a, _, _ = predict(ds, model, ts, seed, "cal")
        b = np.load(os.path.join(C.ROOT, C.pred_path(ds, model, ts, seed, "cal")))["pred"]
        same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        rows.append(dict(dataset=ds, model=model, taskset=ts, seed=seed, same_nan=same_nan, max_abs_diff=float(np.nanmax(np.abs(a - b)))))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.RES, "test_prelock_cal_regeneration.csv"), index=False)
    ok = bool(R.same_nan.all() and (R.max_abs_diff <= 1e-6).all())
    print("ALL MATCH" if ok else "MISMATCH", len(R), "units; max diff", R.max_abs_diff.max())
    assert ok


def lock():
    if os.path.exists(LOCK):
        raise SystemExit("lock exists; refusing to overwrite")
    assert os.path.exists(os.path.join(C.RES, "test_prelock_cal_regeneration.csv")), "run verify-cal first"
    assert not os.path.exists(TEST_ROOT), "test outputs already exist"
    ps = locked_paths() + ["results/p2_datasets/test_prelock_cal_regeneration.csv"]
    files = {p: sha(p) for p in ps}
    json.dump({"SYNTHETIC": True, "created": datetime.datetime.now().isoformat(), "test_accessed_before_lock": False, "settings": SETTINGS,
               "n_files": len(files), "files": files}, open(LOCK, "w"), indent=0)
    print("locked", len(files), "files; lock sha256", sha(LOCK))


# ------------------------------------------------------------------ test-split lag (C4) and augmentation (C5): same code as lag_p22 / aug_p22 with split='test'
def test_windows(ds, n=1000, seed=0):
    bl.configure("all")
    d = bl.load_data(C.proc_dir(ds), None, splits=("test",))["test"]
    sel = np.sort(np.random.RandomState(seed).choice(len(d["pi"]), n, replace=False))
    off = np.arange(-27, 1)
    x = torch.from_numpy(d["X"][d["pi"][sel][:, None], d["ti"][sel][:, None] + off])
    s = torch.from_numpy(d["static"][d["pi"][sel]])
    obs = (x[..., 8] == 0)[..., None].expand(-1, -1, 10).clone()
    obs[..., 6] &= (x[..., 9] == 0)
    return x, s, obs


def lag_test(ds, out):
    bench, cfg = L.benchmarks_traincal(ds)
    pm = json.load(open(os.path.join(C.ROOT, C.proc_dir(ds), "preprocessing_meta.json")))
    spu = {f: pm["scaler_std"][f] / L.UNITS[f][1] for f in ("steps", "sleep_variability", "hrv")}
    PL = L.planted(cfg)
    x, s, obs = test_windows(ds)
    truth = {f"{f}->{k}": dict(lag=Lg, sign=sg, beta_per_std_unit=bench[(f, k)] * spu[f], a=cfg["a"]) for (f, k), (Lg, sg) in PL.items()}
    rows = []
    for model in L.MODELS:
        for seed in C.SEEDS:
            m, ym, ys, heads = L.load(ds, model, seed)
            P = L.profiles(m, ym, ys, heads, x, s, obs)
            for (f, k), (Lg, sg) in PL.items():
                p = P[L.P14.CH.index(f), :, {"dep": 0, "anx": 1}[k]].astype(float)
                T = L.P14.truth_level(truth[f"{f}->{k}"]["beta_per_std_unit"], Lg, L.NEAR, cfg["a"])
                rows.append(dict(dataset=ds, model=model, seed=seed, feature=f, outcome=k, planted_lag=Lg, **L.P14.lag_metrics(p, Lg, sg),
                                 r_with_truth=L.P14.corr(p, T), magnitude_vs_truth=float(p.sum() / T.sum())))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False)


def aug_test(ds, out):
    d = dd.load_split("test", C.proc_dir(ds))
    X0, pid, static = d["X"], d["pid"], d["static"]
    pi, ti = np.meshgrid(np.arange(len(pid)), np.arange(27, 120, 2), indexing="ij")
    pi, ti = pi.ravel(), ti.ravel()
    S_static = static[pi].astype(np.float32)
    R = ij.reconstruct_raw(X0)
    Rp = v2.pad_raw(R)
    us = AG.units(ds)
    Y = {h: d["y_" + h][pi, ti] for h in us[0]["heads"]}
    reg = {s["id"]: s for s in ij.registry()}
    rows = []
    with torch.no_grad():
        for sid in AG.SCEN:
            sc = reg[sid]
            for rep in sc["reps"]:
                S, S_orig = v2.scenario_slices(sc, rep, "test", R, Rp, pi, ti)
                W = v2.represent(S, "R0", "O0", sc["flags"], S_orig)
                for u in us:
                    rows += rm.evaluate(rm.predict(u, W, S_static), u, Y, dict(dataset=ds, split="test", scenario=sid, family=sc["family"], rep=rep))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "test_metrics.csv"), index=False)


def run():
    verify_lock()
    if os.path.exists(TEST_ROOT):
        raise SystemExit("results/p2_datasets/test exists: the test evaluation has already been run (exactly-once rule)")
    os.makedirs(TEST_ROOT)
    log = {"SYNTHETIC": True, "lock_sha256": sha(LOCK), "start": datetime.datetime.now().isoformat(), "datasets_fresh": FRESH, "datasets_shift": SHIFTS,
           "v1": "historical reference; existing test predictions reused; lag/aug inference on v1 test from existing checkpoints", "events": []}
    for ds, model, ts, seed in units():
        P, pid, heads = predict(ds, model, ts, seed, "test")
        p = test_pred_path(ds, model, ts, seed)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, pred=P, pid=pid, heads=np.array(heads))
        print("pred", ds, model, ts, seed, flush=True)
    log["events"].append(("predictions done", datetime.datetime.now().isoformat()))
    for ds in NEW + ["v1"]:
        lag_test(ds, os.path.join(TEST_ROOT, ds, "lag_recovery"))
        aug_test(ds, os.path.join(TEST_ROOT, ds, "augmentation"))
        print("lag/aug", ds, flush=True)
    log["end"] = datetime.datetime.now().isoformat()
    json.dump(log, open(os.path.join(TEST_ROOT, "test_access_log.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify-cal", "lock", "run"])
    a = ap.parse_args()
    os.chdir(C.ROOT)
    torch.set_num_threads(4)
    {"verify-cal": verify_cal, "lock": lock, "run": run}[a.cmd]()
