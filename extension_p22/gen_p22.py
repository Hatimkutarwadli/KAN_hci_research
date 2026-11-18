"""P2-2: generate, preprocess and validate a versioned SYNTHETIC dataset through the UNCHANGED simulator (src/simulate.py) and preprocessing (src/preprocess.py).
  python extension_p22/gen_p22.py generate --ds R1        -> data/synthetic_p22/R1 (read-only, checksums) + data/processed_p22/R1 (read-only)
  python extension_p22/gen_p22.py validate --ds R1        -> results/p2_datasets/R1/validation.json  (TRAIN/CAL and simulator-level only; test sealed)
Dataset config (configs/p22/<id>.json): {"id", "family": replication|shift, "seed", "overrides": {Config fields}, "rho_scale": 1.0, "crn_partner": id|null, "purpose"}.
Everything not in 'overrides' comes from configs/frozen_config_v1.json; z thresholds are NEVER overridden (approved decision).
rho_scale multiplies simulate.RHO (wearable AR(1) persistence, a module constant) for the duration of the call only; it is recorded in
data/synthetic_p22/_provenance/<id>.json because the simulator's own meta.json does not contain RHO."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_p22 as C  # noqa: E402
import preprocess as pp  # noqa: E402
import simulate as sim  # noqa: E402

FORBIDDEN_OVERRIDES = {"z_thr_dep", "z_thr_anx", "n_participants", "n_days", "burn_in", "horizons", "persistence_days", "split_fracs", "seed"}


def build_config(dcfg):
    base = json.load(open(os.path.join(C.ROOT, C.FROZEN_CFG)))
    bad = set(dcfg.get("overrides", {})) & FORBIDDEN_OVERRIDES
    assert not bad, f"forbidden overrides {bad}"
    unknown = set(dcfg.get("overrides", {})) - set(base)
    assert not unknown, f"unknown Config fields {unknown}"
    kw = dict(base)
    kw.update(dcfg.get("overrides", {}))
    kw["seed"] = dcfg["seed"]
    return sim.Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})


def simulate_with(cfg, rho_scale=1.0):
    old = sim.RHO.copy()
    try:
        sim.RHO = old * rho_scale
        return sim.simulate(cfg), sim.RHO.tolist()
    finally:
        sim.RHO = old


def seed_streams(seed):
    return {seed, seed + 1, seed + 2}                     # static params, daily noise, split


def generate(ds):
    dcfg = C.ds_config(ds)
    assert dcfg["id"] == ds
    assert not (seed_streams(dcfg["seed"]) & C.RESERVED_SEED_STREAMS), "seed stream overlaps pilot/v1"
    cfg = build_config(dcfg)
    out, rho = simulate_with(cfg, dcfg.get("rho_scale", 1.0))
    raw = os.path.join(C.ROOT, C.raw_dir(ds))
    sim.save(out, raw, cfg, final=True)                   # refuses non-empty dir; checksums; read-only
    prov_dir = os.path.join(C.ROOT, "data/synthetic_p22/_provenance")
    os.makedirs(prov_dir, exist_ok=True)
    prov = os.path.join(prov_dir, f"{ds}.json")
    if os.path.exists(prov):
        raise SystemExit("provenance exists; refusing to overwrite")
    json.dump({"SYNTHETIC": True, "dataset": ds, "dataset_config": dcfg, "dataset_config_sha256": hashlib.sha256(open(os.path.join(C.ROOT, "configs", "p22", f"{ds}.json"), "rb").read()).hexdigest(),
               "frozen_config_sha256": hashlib.sha256(open(os.path.join(C.ROOT, C.FROZEN_CFG), "rb").read()).hexdigest(), "RHO_used": rho,
               "seed_streams": {"static": cfg.seed, "daily_noise": cfg.seed + 1, "split": cfg.seed + 2},
               "simulator_sha256": hashlib.sha256(open(os.path.join(C.ROOT, "src/simulate.py"), "rb").read()).hexdigest()}, open(prov, "w"), indent=1)
    os.chmod(prov, 0o444)
    pp.run(raw, os.path.join(C.ROOT, C.proc_dir(ds)), readonly=True)      # unchanged Stage 2 pipeline; scaler fitted on this dataset's train split
    print("generated", ds)


def validate(ds):
    """Pre-training feasibility checks: simulator-level (pooled, rates only) and TRAIN/CAL split-level. Test-split counts are never read."""
    raw = os.path.join(C.ROOT, C.raw_dir(ds))
    cfg = json.load(open(os.path.join(raw, "meta.json")))["config"]
    tg = pd.read_csv(os.path.join(raw, "targets.csv"), usecols=["pid", "day", "status_dep", "status_anx"])
    ob = pd.read_csv(os.path.join(raw, "observations.csv"), usecols=["pid", "day", "missing_day", "hrv_missing"])
    sp = pd.read_csv(os.path.join(raw, "splits.csv"))
    tp = pd.read_csv(os.path.join(raw, "truth_participants.csv"))
    N, D = sp.shape[0], cfg["n_days"]
    miss = ob.sort_values(["pid", "day"]).missing_day.values.reshape(N, D)
    gaps = []
    for row in miss:
        r = np.concatenate([[0], row, [0]])
        d = np.diff(r)
        gaps += list(np.where(d == -1)[0] - np.where(d == 1)[0])
    pmeta = json.load(open(os.path.join(C.ROOT, C.proc_dir(ds), "preprocessing_meta.json")))
    sc = {s: {k: v for k, v in pmeta["split_counts"][s].items() if k.startswith("onset_")} for s in ("train", "cal")}     # test entry deliberately not read
    v = {"SYNTHETIC": True, "dataset": ds, "test_split_inspected": False,
         "simulator_level_pooled_rates": {
             "daily_prevalence_dep": float(tg.status_dep.mean()), "daily_prevalence_anx": float(tg.status_anx.mean()),
             "missing_day_rate": float(ob.missing_day.mean()), "hrv_missing_rate": float(ob.hrv_missing.mean()),
             "gap_length_mean": float(np.mean(gaps)), "gap_length_p90": float(np.percentile(gaps, 90)), "gap_length_max": int(np.max(gaps)),
             "participant_rate_any_sustained_onset_dep": float(tp.onset_day_sust_dep.notna().mean()),
             "participant_rate_any_sustained_onset_anx": float(tp.onset_day_sust_anx.notna().mean()),
             "prevalent_rate_dep": float(tp.prevalent_dep.mean()), "prevalent_rate_anx": float(tp.prevalent_anx.mean())},
         "split_sizes": sp.split.value_counts().to_dict(),
         "train_cal_onset_counts": sc,
         "processed_has_nan": {s: bool(np.isnan(np.load(os.path.join(C.ROOT, C.proc_dir(ds), f"{s}.npz"))["X"]).any()) for s in ("train", "cal")}}
    pr = v["simulator_level_pooled_rates"]
    cal_min = min(sc["cal"][f"onset_sust_{k}_h{h}"]["positive_participants"] for k in ("dep", "anx") for h in (7, 14, 21))
    v["gates"] = {"prevalence_5_to_10pct": bool(0.05 <= pr["daily_prevalence_dep"] <= 0.10 and 0.05 <= pr["daily_prevalence_anx"] <= 0.10),
                  "cal_min_evaluable_sustained_onset_participants": cal_min, "cal_onset_participants_ge_25": bool(cal_min >= 25),
                  "no_nan_in_processed_train_cal": not any(v["processed_has_nan"].values()),
                  "label_cross_check_vs_simulator": pmeta.get("label_cross_check_vs_simulator")}
    v["gates"]["all_pass"] = bool(v["gates"]["prevalence_5_to_10pct"] and v["gates"]["cal_onset_participants_ge_25"] and v["gates"]["no_nan_in_processed_train_cal"])
    out = os.path.join(C.ROOT, C.RES, ds)
    os.makedirs(out, exist_ok=True)
    json.dump(v, open(os.path.join(out, "validation.json"), "w"), indent=1)
    print(json.dumps(v["gates"]), json.dumps(pr))
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "validate"])
    ap.add_argument("--ds", required=True)
    a = ap.parse_args()
    os.chdir(C.ROOT)
    generate(a.ds) if a.cmd == "generate" else validate(a.ds)
