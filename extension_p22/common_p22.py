"""P2-2 EXTENSION (SYNTHETIC data): shared definitions for new versioned datasets. Nothing in src/ or existing results is modified.
Datasets: 'v1' = the frozen original (data/synthetic_v1, existing results reused read-only); new datasets are described by configs/p22/<id>.json and live in
data/synthetic_p22/<id>/ (raw, read-only + checksums) and data/processed_p22/<id>/ (read-only). Results: results/p2_datasets/<id>/.
TEST SEALING: nothing in this package loads a test split, except the (separate, hash-locked) final evaluation step."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p13"), os.path.join(ROOT, "extension_missing"),
                os.path.join(ROOT, "extension_v2"), os.path.join(ROOT, "extension_p14"), os.path.join(ROOT, "extension_p22")]

FROZEN_CFG = "configs/frozen_config_v1.json"
RESERVED_SEED_STREAMS = {2026, 2027, 2028, 2029}          # pilot (2026..2028) and v1 (2027..2029)
SEEDS = (0, 1, 2)                                        # model seeds per dataset (approved)
RES = "results/p2_datasets"


def ds_config(ds):
    return json.load(open(os.path.join(ROOT, "configs", "p22", f"{ds}.json")))


def raw_dir(ds):
    return "data/synthetic_v1" if ds == "v1" else f"data/synthetic_p22/{ds}"


def proc_dir(ds):
    return "data/processed_v1" if ds == "v1" else f"data/processed_p22/{ds}"


def sim_config(ds):
    """The simulator Config fields actually used for a dataset (from the raw meta.json written by the simulator)."""
    return json.load(open(os.path.join(ROOT, raw_dir(ds), "meta.json")))["config"]


# ------------------------------------------------------------------ core model panel (approved): arch, task sets, fit_nn settings
CORE = {"gbt": dict(arch="gbt", tasksets=("main", "forecast")),
        "gru": dict(arch="gru", tasksets=("main", "forecast"), wd=1e-5, dropout=0.0),
        "kan_reg": dict(arch="kan", tasksets=("main",), wd=1e-3, dropout=0.3),
        "mlp_b2_reg": dict(arch="mlp_b2", tasksets=("main",), wd=1e-3, dropout=0.3),
        "m1": dict(arch="m1", tasksets=("all",), wd=1e-3, dropout=0.3),
        "mlp_m1": dict(arch="mlp_m1", tasksets=("all",), wd=1e-3, dropout=0.3)}
AUG = "gru_aug"                                          # Stage 9 R0O0T1 (pre-specified; not reselected)

# existing v1 artefacts (read-only)
V1_PRED = {("gbt", "main"): "results/stage3/predictions/gbt_seed{s}_{split}.npz",
           ("gbt", "forecast"): "results/stage3c_forecast/predictions/gbt_seed{s}_{split}.npz",
           ("gru", "main"): "results/stage3/predictions/gru_seed{s}_{split}.npz",
           ("gru", "forecast"): "results/stage3c_forecast/predictions/gru_seed{s}_{split}.npz",
           ("kan_reg", "main"): "results/stage3b_kan_reg/final/predictions/kan_reg_seed{s}_{split}.npz",
           ("mlp_b2_reg", "main"): "results/p1_kan_mlp_control/b2_main/predictions/mlp_b2_reg_seed{s}_{split}.npz",
           ("m1", "all"): "results/stage4_m1/predictions/m1_seed{s}_{split}.npz",
           ("mlp_m1", "all"): "results/p1_kan_mlp_control/m1_all/predictions/mlp_m1_seed{s}_{split}.npz"}
V1_CKPT = {("gru", "main"): "results/stage3/checkpoints/gru_seed{s}.pt",
           ("m1", "all"): "results/stage4_m1/checkpoints/m1_seed{s}.pt",
           ("mlp_m1", "all"): "results/p1_kan_mlp_control/m1_all/checkpoints/mlp_m1_seed{s}.pt",
           (AUG, "main"): "results/stage9_v2/checkpoints/gru_R0O0T1_seed{s}.pt"}


def run_dir(ds, model, taskset):
    return os.path.join(RES, ds, f"{model}_{taskset}")


def pred_path(ds, model, taskset, seed, split="cal"):
    if ds == "v1":
        return V1_PRED[(model, taskset)].format(s=seed, split=split)
    return os.path.join(run_dir(ds, model, taskset), "predictions", f"{model}_seed{seed}_{split}.npz")


def ckpt_path(ds, model, taskset, seed):
    if ds == "v1":
        return V1_CKPT[(model, taskset)].format(s=seed)
    return os.path.join(run_dir(ds, model, taskset), "checkpoints", f"{model}_seed{seed}.pt")


# ------------------------------------------------------------------ integrity manifest of pre-existing files
NEW_PATHS = ("extension_p22", "configs/p22", "data/synthetic_p22", "data/processed_p22", RES, "tests/test_p22.py", "__pycache__", ".claude")


def file_hashes():
    import hashlib
    out = {}
    for top in sorted(os.listdir(ROOT)):
        p0 = os.path.join(ROOT, top)
        files = [p0] if os.path.isfile(p0) else [os.path.join(r, f) for r, _, fs in os.walk(p0) for f in fs]
        for p in files:
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            if any(rel == n or rel.startswith(n + "/") or "/__pycache__/" in rel for n in NEW_PATHS) or rel in ("PROJECT_STATUS.md", "EXPERIMENT_LOG.md") or rel.startswith(".git"):
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            out[rel] = h.hexdigest()
    return out
