"""P3 single locked TEST evaluation (SYNTHETIC; inference only) implementing results/p3_small_data/P3_TEST_EVALUATION_PLAN.md.
  python extension_p3/eval_test_p3.py verify-cal   pre-lock: rebuild every main-run model from its checkpoint + scaler.json and regenerate its
                                                    CALIBRATION metrics with the test-inference code path (no test access); must match metrics.json
  python extension_p3/eval_test_p3.py lock         write results/p3_small_data/test_lock.json (SHA-256 of plans, code, configs, checkpoints, scalers, data, settings)
  python extension_p3/eval_test_p3.py run          verify lock, then evaluate the R4 and R5 test splits EXACTLY ONCE (refuses if results/p3_small_data/test exists)
No training, recalibration, tuning, threshold change or selection happens here. Only the 416 primary (main) jobs are evaluated; S1-S3 stay calibration-only."""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p3 as P  # noqa: E402

RUNS = os.path.join(P.RES, "runs")
TEST_ROOT = os.path.join(P.RES, "test")
LOCK = os.path.join(P.RES, "test_lock.json")
PRELOCK = os.path.join(P.RES, "test_prelock_cal_regeneration.csv")
TOL = 1e-6
SETTINGS = {"torch_threads": 4, "threadpool_limit": 4, "workers": 3, "jobs": "main (416; P3_PLAN_AMENDMENT_1 design)", "datasets": list(P.DATASETS),
            "test_participants": "all test-split participants of each dataset", "scaler": "per-job scaler.json (subset-fitted, primary)",
            "analysis": "extension_p3/analyze_test_p3.py = analyze_p3 functions (families, units, compare, rq1a [amendment 2], rq1b) on test metrics",
            "cal_reproduction_tolerance": TOL}
LOCK_FILES = ["results/p3_small_data/P3_TEST_EVALUATION_PLAN.md", "results/p3_small_data/P3_PLAN_LOCKED.md", "results/p3_small_data/P3_PLAN_LOCKED.sha256",
              "results/p3_small_data/P3_PLAN_AMENDMENT_1.md", "results/p3_small_data/P3_PLAN_AMENDMENT_1.sha256", "results/p3_small_data/P3_PLAN_AMENDMENT_2.md",
              "results/p3_small_data/P3_PLAN_AMENDMENT_2.sha256", "results/p3_small_data/manifest_p3_pre.json", "tests/test_p3.py",
              "configs/p22/R4.json", "configs/p22/R5.json", "extension_p22/common_p22.py", "extension_p22/analyze_p22.py", "extension_p13/models_p13.py",
              "src/baselines.py", "src/models.py", "src/data.py", "src/metrics.py", "src/preprocess.py"]
LOCK_GLOBS = ["extension_p3/*.py", "results/p3_small_data/subsets/*", "results/p3_small_data/analysis_cal/*",
              "results/p3_small_data/runs/*/*/checkpoints/*", "results/p3_small_data/runs/*/*/scaler.json", "results/p3_small_data/runs/*/*/metrics.json",
              "results/p3_small_data/runs/*/*/run.json", "data/processed_p22/R4/*", "data/processed_p22/R5/*", "data/synthetic_p22/R4/*", "data/synthetic_p22/R5/*"]


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
    return sorted(ps)


def verify_lock():
    lk = json.load(open(LOCK))
    bad = [p for p, h in lk["files"].items() if not os.path.exists(p) or sha(p) != h]
    assert not bad, f"LOCK VIOLATION: {len(bad)} files changed, e.g. {bad[:5]}"
    assert lk["settings"] == json.loads(json.dumps(SETTINGS)), "LOCK VIOLATION: settings changed"
    return lk


def main_jobs():
    return [j for ds in P.DATASETS for j in P.jobs_main(ds)]


# ------------------------------------------------------------------ inference (identical code path for calibration verification and test)
def infer(jb, split):
    """Rebuild the job's model from its checkpoint and scaler.json; return predictions, the evaluated data and the train subset data."""
    import joblib
    import torch
    from threadpoolctl import threadpool_limits
    import baselines as bl
    import common_p22 as C
    import models as md
    import models_p13
    import run_p3 as R
    models_p13.register()
    torch.set_num_threads(4)
    bl.configure(jb["ts"])
    d = os.path.join(RUNS, jb["ds"], jb["id"])
    sj = json.load(open(os.path.join(d, "scaler.json")))
    assert sj["scaler_mode"] == "subset"
    scaler = {"mean": np.asarray(sj["mean"]), "std": np.asarray(sj["std"])}
    tr = P.split_data(jb["ds"], "train", sj["train_pids"], scaler)
    ev = P.split_data(jb["ds"], split, P.all_pids(jb["ds"], split), scaler)
    if jb["model"] == "gbt":
        fitted = joblib.load(os.path.join(d, "checkpoints", f"gbt_seed{jb['seed']}.joblib"))
        with threadpool_limits(4):
            F_ = bl.gbt_features(ev)
            Pr = np.stack([fitted[t].predict(F_) if t in bl.REG else fitted[t].predict_proba(F_)[:, 1] for t in bl.HEADS], 1).astype(np.float32)
    else:
        ck = torch.load(os.path.join(d, "checkpoints", f"{jb['model']}_seed{jb['seed']}.pt"), map_location="cpu", weights_only=False)
        assert list(ck["heads"]) == list(bl.HEADS)
        spec = C.CORE[jb["model"]]
        mk = {"dropout": spec["dropout"]} if spec["dropout"] > 0 else {}
        if spec["arch"].endswith("m1"):
            mk["heads"] = list(ck["heads"])
        m = md.MODELS[spec["arch"]](n_out=len(ck["heads"]), **mk)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        Pr = bl.predict_nn(m, bl.to_t(ev), ck["ymean"], ck["ystd"])
    out = dict(P=Pr, ca_eval=ev, tr=tr, heads=list(bl.HEADS))
    return out, R.head_metrics(out)


def _verify_one(jb):
    _, met = infer(jb, "cal")
    ref = json.load(open(os.path.join(RUNS, jb["ds"], jb["id"], "metrics.json")))
    diffs = [abs(met[h][k] - ref[h][k]) for h in ref for k in ref[h] if isinstance(ref[h][k], float)]
    return dict(ds=jb["ds"], id=jb["id"], same_heads=sorted(met) == sorted(ref), n_values=len(diffs), max_abs_diff=float(max(diffs)),
                finite=bool(all(np.isfinite(v["value"]) for v in met.values())))


def verify_cal():
    with ProcessPoolExecutor(SETTINGS["workers"]) as ex:
        rows = list(ex.map(_verify_one, main_jobs()))
    R = pd.DataFrame(rows)
    R.to_csv(PRELOCK, index=False)
    ok = bool(R.same_heads.all() and R.finite.all() and (R.max_abs_diff <= TOL).all() and len(R) == 416)
    print("ALL MATCH" if ok else "MISMATCH", len(R), "jobs; max diff", R.max_abs_diff.max())
    assert ok


def lock():
    if os.path.exists(LOCK):
        raise SystemExit("lock exists; refusing to overwrite")
    assert os.path.exists(PRELOCK), "run verify-cal first"
    assert not os.path.exists(TEST_ROOT), "test outputs already exist"
    R = pd.read_csv(PRELOCK)
    assert len(R) == 416 and (R.max_abs_diff <= TOL).all() and R.same_heads.all()
    ps = locked_paths() + [PRELOCK.replace(P.ROOT + os.sep, "").replace("\\", "/")]
    files = {p: sha(p) for p in ps}
    json.dump({"SYNTHETIC": True, "created": datetime.datetime.now().isoformat(), "test_accessed_before_lock": False, "settings": SETTINGS,
               "n_files": len(files), "files": files}, open(LOCK, "w"), indent=0)
    os.chmod(LOCK, 0o444)
    print("locked", len(files), "files; lock sha256", sha(LOCK))


def _test_one(jb):
    P.TEST_UNLOCKED = True                       # set only inside the lock-verified run step (worker process)
    out, met = infer(jb, "test")
    ev = out["ca_eval"]
    pd_ = os.path.join(TEST_ROOT, "predictions", jb["ds"])
    md_ = os.path.join(TEST_ROOT, "metrics", jb["ds"])
    np.savez_compressed(os.path.join(pd_, jb["id"] + ".npz"), pred=out["P"], pid=ev["pid_o"], t=ev["ti"], heads=np.array(out["heads"]))
    json.dump(met, open(os.path.join(md_, jb["id"] + ".json"), "w"), indent=1)
    return dict(ds=jb["ds"], id=jb["id"], finite=bool(np.isfinite(out["P"]).all()), n_rows=int(len(out["P"])), time=datetime.datetime.now().isoformat())


def run():
    lk = verify_lock()
    if os.path.exists(TEST_ROOT):
        raise SystemExit("results/p3_small_data/test exists: the test evaluation has already been run (exactly-once rule)")
    for ds in P.DATASETS:
        os.makedirs(os.path.join(TEST_ROOT, "predictions", ds))
        os.makedirs(os.path.join(TEST_ROOT, "metrics", ds))
    log = {"SYNTHETIC": True, "lock_sha256": sha(LOCK), "lock_n_files": lk["n_files"], "start": datetime.datetime.now().isoformat(),
           "scope": "416 primary jobs; R4 and R5 test splits; inference only"}
    json.dump(log, open(os.path.join(TEST_ROOT, "test_access_log.json"), "w"), indent=1)
    with ProcessPoolExecutor(SETTINGS["workers"]) as ex:
        rows = list(ex.map(_test_one, main_jobs()))
    pd.DataFrame(rows).to_csv(os.path.join(TEST_ROOT, "test_jobs.csv"), index=False)
    log.update(end=datetime.datetime.now().isoformat(), n_jobs=len(rows), all_finite=bool(all(r["finite"] for r in rows)))
    json.dump(log, open(os.path.join(TEST_ROOT, "test_access_log.json"), "w"), indent=1)
    print("test done", len(rows), "jobs; all finite", log["all_finite"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify-cal", "lock", "run"])
    a = ap.parse_args()
    os.chdir(P.ROOT)
    {"verify-cal": verify_cal, "lock": lock, "run": run}[a.cmd]()
