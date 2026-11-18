"""P3 shared definitions (SYNTHETIC): nested stratified train subsets, in-memory subset-fitted scaling, job design.
Implements results/p3_small_data/P3_PLAN_LOCKED.md sections 3-4 and 8. Reuses frozen code unchanged (preprocess, baselines, common_p22).
TEST SEALING: split_data refuses the test split unless TEST_UNLOCKED is set by the (separate, G3-approved, lock-verified) test step."""
import functools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "extension_p22"))
import common_p22 as C  # noqa: E402,F401  (sets sys.path for src/, extension_p13, ...)
import baselines as bl  # noqa: E402
import data as dd  # noqa: E402
import preprocess as pp  # noqa: E402

RES = os.path.join(ROOT, "results", "p3_small_data")
DATASETS = ("R4", "R5")
DSK = {"R4": 4, "R5": 5}
SIZES = (50, 100, 200, 400, 600)
DESIGN = {50: (range(5), (0, 1)), 100: (range(5), (0, 1)), 200: (range(3), (0,)), 400: (range(3), (0,)), 600: (range(1), (0, 1, 2))}   # P3_PLAN_AMENDMENT_1 (Option F)
MODELS = [("gbt", "main"), ("gbt", "forecast"), ("gru", "main"), ("gru", "forecast"), ("kan_reg", "main"), ("mlp_b2_reg", "main"), ("m1", "all"), ("mlp_m1", "all")]
COST = {"m1": 54, "kan_reg": 10, "mlp_m1": 5, "gbt": 3, "gru": 2.2, "mlp_b2_reg": 1.4}          # CPU-min per full-data run (P2-2 logs); ordering only
TEST_UNLOCKED = False


@functools.lru_cache(maxsize=4)
def raw(ds):
    rawd = os.path.join(ROOT, C.raw_dir(ds))
    meta = json.load(open(os.path.join(rawd, "meta.json")))
    feats = meta["features"]
    N, D = meta["config"]["n_participants"], meta["config"]["n_days"]
    ob = pd.read_csv(os.path.join(rawd, "observations.csv"), usecols=["pid", "day"] + feats + ["missing_day", "hrv_missing"]).sort_values(["pid", "day"])
    return dict(values=ob[feats].values.reshape(N, D, -1).astype(float), masks=ob[["missing_day", "hrv_missing"]].values.reshape(N, D, 2).astype(float),
                split=pd.read_csv(os.path.join(rawd, "splits.csv")).sort_values("pid").split.values,
                part=pd.read_csv(os.path.join(rawd, "participants.csv")).sort_values("pid"))


def strat_order(keys, rng):
    """Nested stratified order (Phase 0): random order within each stratum; global order by (i + U) / n_stratum."""
    pos = np.zeros(len(keys))
    for kv in np.unique(keys):
        idx = np.where(keys == kv)[0]
        pos[rng.permutation(idx)] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    return np.argsort(pos, kind="stable")


def strata(ds, split):
    """sex x age_group x any sustained onset (dep or anx) from that split's OWN labels (train for subsets; cal for S3)."""
    assert split in ("train", "cal")
    r = raw(ds)
    lab = dd.load_split(split, os.path.join(ROOT, C.proc_dir(ds)))
    pids = lab["pid"]
    ons = np.zeros(len(pids), bool)
    for k in ("dep", "anx"):
        for h in (7, 14, 21):
            ons |= (np.nan_to_num(lab[f"y_onset_sust_{k}_h{h}"]) > 0).any(1)
    p = r["part"].set_index("pid").loc[pids]
    return pids, (p.sex.values == "F").astype(int) * 100 + p.age_group.values * 10 + ons.astype(int)


def make_subsets(ds):
    pids, keys = strata(ds, "train")
    out = {}
    for draw in range(5):
        order = strat_order(keys, np.random.default_rng(7000 + 100 * DSK[ds] + draw))
        out[str(draw)] = {str(n): sorted(int(x) for x in pids[order[:n]]) for n in (50, 100, 200, 400)}
        out[str(draw)]["600"] = sorted(int(x) for x in pids)
    return out


def subsets(ds):
    p = os.path.join(RES, "subsets", f"{ds}.json")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({"SYNTHETIC": True, "dataset": ds, "rule": "P3_PLAN_LOCKED.md s4", "subsets": make_subsets(ds)}, open(p, "w"))
        os.chmod(p, 0o444)
    return json.load(open(p))["subsets"]


def cal_subset_S3(ds, draw, n=33):
    pids, keys = strata(ds, "cal")
    order = strat_order(keys, np.random.default_rng(9000 + draw))
    return sorted(int(x) for x in pids[order[:n]])


def all_pids(ds, split):
    assert split in ("train", "cal") or TEST_UNLOCKED, "TEST SPLIT SEALED (P3 G3 not passed)"
    return [int(x) for x in np.where(raw(ds)["split"] == split)[0]]


def fit_subset_scaler(ds, train_pids):
    return pp.fit_scaler(raw(ds)["values"][np.asarray(train_pids)])


def split_data(ds, split, pids, scaler):
    """Exact equivalent of baselines.load_data for the given participants (current bl.HEADS). scaler=None -> frozen processed arrays (S1);
    otherwise inputs rebuilt in memory with preprocess.transform (labels are scale-independent and come from the processed file)."""
    assert split in ("train", "cal") or TEST_UNLOCKED, "TEST SPLIT SEALED (P3 G3 not passed)"
    d = dd.load_split(split, os.path.join(ROOT, C.proc_dir(ds)))
    pids = np.asarray(pids)
    idx = np.searchsorted(d["pid"], pids)
    assert np.array_equal(d["pid"][idx], pids) and np.all(np.diff(pids) > 0)
    if scaler is None:
        X = d["X"][idx]
    else:
        r = raw(ds)
        X = pp.transform(r["values"][pids], r["masks"][pids], scaler)
    N = len(idx)
    pi, ti = np.meshgrid(np.arange(N), np.arange(27, 120), indexing="ij")
    pi, ti = pi.ravel(), ti.ravel()
    Y = np.stack([d["y_" + h][idx][pi, ti] for h in bl.HEADS], 1).astype(np.float32)
    return dict(X=X, static=d["static"][idx], pid=d["pid"][idx], pi=pi, ti=ti, Y=Y, pid_o=d["pid"][idx][pi])


# ------------------------------------------------------------------ jobs
def job(ds, model, ts, n, draw, seed, variant="primary"):
    jid = f"{model}_{ts}_n{n}_d{draw}_s{seed}" + ("" if variant == "primary" else f"_{variant}")
    return dict(ds=ds, model=model, ts=ts, n=n, draw=draw, seed=seed, variant=variant, id=jid, cost=COST[model] * max(n, 50) / 600)


def jobs_main(ds):
    return [job(ds, m, ts, n, dr, s) for n, (draws, seeds) in DESIGN.items() for dr in draws for m, ts in MODELS for s in ([0] if m == "gbt" else seeds)]


def jobs_pilot():
    return [job("R4", m, ts, n, 0, 0) for n in (50, 400) for m, ts in MODELS]


def jobs_sens(variant):
    if variant == "S2":
        return [job("R4", "gru", ts, n, dr, 0, "S2") for n in (50, 100) for dr in range(5) for ts in ("main", "forecast")]
    if variant == "S3":
        return [job("R4", m, ts, 100, dr, 0, "S3") for dr in range(5) for m, ts in (("gru", "main"), ("kan_reg", "main"), ("mlp_b2_reg", "main"))]
    if variant == "S1":
        return [job("R4", m, ts, n, dr, 0, "S1") for n in (50, 100) for m, ts in MODELS for dr in (range(3) if m in ("m1", "mlp_m1") else range(5))]
    raise ValueError(variant)
