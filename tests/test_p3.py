"""Tests for P3 (small-data learning curves; SYNTHETIC). Run: python tests/test_p3.py [name-filter ...]. No test split is loaded."""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "extension_p3")]
import common_p3 as P  # noqa: E402
import common_p22 as C  # noqa: E402
import data as dd  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_plan_locked():
    h = open("results/p3_small_data/P3_PLAN_LOCKED.sha256").read().split()[0]
    assert sha("results/p3_small_data/P3_PLAN_LOCKED.md") == h


def test_r4_r5_configs_streams_integrity_validation():
    used = set()
    for f in glob.glob("configs/p22/*.json"):
        c = json.load(open(f))
        if c["id"] not in ("R4", "R5"):
            used |= {c["seed"], c["seed"] + 1, c["seed"] + 2}
    used |= C.RESERVED_SEED_STREAMS
    for ds, seed in (("R4", 3401), ("R5", 3501)):
        c = C.ds_config(ds)
        assert c["seed"] == seed and c["family"] == "replication" and c["overrides"] == {} and c["rho_scale"] == 1.0
        assert not ({seed, seed + 1, seed + 2} & used)
        raw = C.raw_dir(ds)
        for line in open(os.path.join(raw, "checksums.sha256")).read().strip().splitlines():
            h, fn = line.split()
            assert sha(os.path.join(raw, fn)) == h and not os.access(os.path.join(raw, fn), os.W_OK)
        m, f = json.load(open(os.path.join(raw, "meta.json")))["config"], {**json.load(open(C.FROZEN_CFG)), "seed": seed}
        assert {k: m[k] for k in f} == f                                              # frozen config, only the seed differs
        v = json.load(open(f"results/p3_small_data/datasets/{ds}/validation.json"))
        assert v["gates"]["all_pass"] and v["test_split_inspected"] is False


def test_subsets_nested_stratified_disjoint_deterministic():
    for ds in P.DATASETS:
        S = P.subsets(ds)
        assert S == json.loads(json.dumps(P.make_subsets(ds)))                       # deterministic, equals the stored file
        r = P.raw(ds)
        train = set(np.where(r["split"] == "train")[0].tolist())
        pids, keys = P.strata(ds, "train")
        kmap = dict(zip(pids.tolist(), keys.tolist()))
        for draw in range(5):
            prev = set()
            for n in P.SIZES:
                s = S[str(draw)][str(n)]
                assert len(s) == n == len(set(s)) and set(s) <= train and s == sorted(s)        # train only, no cal/test
                assert prev <= set(s)                                                            # nested
                prev = set(s)
                for kv in set(keys.tolist()):
                    share = np.mean(keys == kv)
                    got = sum(kmap[p] == kv for p in s)
                    assert abs(got - n * share) < 1.5, (ds, draw, n, kv, got, n * share)        # proportional stratification
            assert S[str(draw)]["600"] == sorted(train)


def test_scaler_exactness_and_subset_fit():
    import baselines as bl
    bl.configure("main")
    for ds in P.DATASETS:
        tr_all = P.all_pids(ds, "train")
        sc = P.fit_subset_scaler(ds, tr_all)
        for split in ("train", "cal"):
            d = P.split_data(ds, split, P.all_pids(ds, split), sc)
            ref = bl.load_data(C.proc_dir(ds), None, splits=(split,))[split]
            for k in ("X", "static", "pid", "pi", "ti", "pid_o"):
                assert np.array_equal(d[k], ref[k]), (ds, split, k)
            assert np.array_equal(np.isnan(d["Y"]), np.isnan(ref["Y"])) and np.array_equal(np.nan_to_num(d["Y"]), np.nan_to_num(ref["Y"]))
        sub = P.subsets(ds)["0"]["50"]
        s50 = P.fit_subset_scaler(ds, sub)
        assert np.allclose(s50["mean"], np.nanmean(P.raw(ds)["values"][sub], axis=(0, 1)))


def test_test_split_sealed():
    for fn in glob.glob("extension_p3/*.py"):
        src = open(fn).read()
        assert not re.search(r"""load_split\(\s*["']test""", src) and not re.search(r"""splits=\([^)]*test""", src), fn
    for f in (lambda: P.split_data("R4", "test", [0], None), lambda: P.all_pids("R4", "test")):
        try:
            f()
        except AssertionError:
            continue
        raise AssertionError("test split not sealed")


def test_reproduces_p22_r1_gru_seed0():
    import run_p3 as R
    out = R.train_eval("R1", "gru", "main", 0, P.all_pids("R1", "train"), P.all_pids("R1", "cal"), "subset", log=lambda m: None)
    ref = np.load("results/p2_datasets/R1/gru_main/predictions/gru_seed0_cal.npz")["pred"]
    ca = out["ca_eval"]
    full = np.full_like(ref, np.nan)
    full[np.searchsorted(dd.load_split("cal", C.proc_dir("R1"))["pid"], ca["pid"])[ca["pi"]], ca["ti"]] = out["P"]
    assert np.array_equal(np.isnan(full), np.isnan(ref)) and np.nanmax(np.abs(full - ref)) <= 1e-6, np.nanmax(np.abs(full - ref))


def test_pilot_jobs_share_subsets():
    runs = glob.glob("results/p3_small_data/pilot/R4/*/scaler.json")
    by = {}
    for f in runs:
        j = json.load(open(f))
        n = re.search(r"_n(\d+)_d(\d+)_", f.replace("\\", "/")).groups()
        by.setdefault(n, set()).add(tuple(j["train_pids"]))
    assert all(len(v) == 1 for v in by.values()), "models did not receive identical subsets"


def test_existing_files_unchanged():
    man = json.load(open("results/p3_small_data/manifest_p3_pre.json"))
    bad = [p for p, h in man.items() if not os.path.exists(p) or sha(p) != h]
    print(f"  {len(man)} pre-existing files; changed or missing: {len(bad)}")
    assert not bad, bad[:10]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and (len(sys.argv) == 1 or any(s in k for s in sys.argv[1:]))]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"{len(tests)} tests pass")
