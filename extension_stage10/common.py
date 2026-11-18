"""Stage 10 (P0) shared helpers. SYNTHETIC data only. Post-processing of SAVED predictions: nothing is retrained.
Test labels are used only to report final metrics; no choice (hyperparameter, threshold, epsilon, recalibration map, conformal quantile) uses them."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]
import data as dd  # noqa: E402

PROC = "data/processed_v1"
MODELS = ["gbt", "gru", "kan", "kan_reg", "m1", "m1_static"]
SEEDS = {"gbt": [0]}
DIRS = {"gbt": ("results/stage3", "results/stage3c_forecast"), "gru": ("results/stage3", "results/stage3c_forecast"),
        "kan": ("results/stage3", "results/stage3c_forecast"), "kan_reg": ("results/stage3b_kan_reg/final", "results/stage3c_forecast"),
        "m1": ("results/stage4_m1", "results/stage4_m1"), "m1_static": ("results/stage4_ablation", "results/stage4_ablation")}
KS = ("dep", "anx")
SEV = [f"sev_{k}_{g}" for k in KS for g in ("t0", "h7", "h14", "h21")]
STATUS = [f"status_{k}_{g}" for k in KS for g in ("t0", "h7", "h14", "h21")]
ONSET = [f"onset_sust_{k}_h{h}" for k in KS for h in (7, 14, 21)]
CLF = STATUS + ONSET
SUBGROUPS = [("sex", "F"), ("sex", "M"), ("age_group", "0"), ("age_group", "1"), ("age_group", "2")]
MIN_N = 30          # participants per subgroup below which results are flagged UNRELIABLE
MIN_EVENT_P = 10    # participants with >=1 positive label below which classification metrics are flagged LOW EVENTS


def seeds_of(model):
    return SEEDS.get(model, [0, 1, 2, 3, 4])


def pred_path(model, task, seed, split):
    main = task.endswith("_t0") or task.startswith("onset")
    return os.path.join(DIRS[model][0 if main else 1], "predictions", f"{model}_seed{seed}_{split}.npz")


_cache = {}


def load_pred(model, task, seed, split):
    """(N, 120) float64 predictions for one head (severity points or probability)."""
    p = pred_path(model, task, seed, split)
    if p not in _cache:
        z = np.load(p)
        _cache[p] = (z["pred"], list(z["heads"]), z["pid"])
    arr, heads, pid = _cache[p]
    return arr[:, :, heads.index(task)].astype(np.float64), pid


def load_split(split):
    d = dd.load_split(split, PROC)
    pa = pd.read_csv("data/synthetic_v1/participants.csv").set_index("pid")
    d["sex"] = pa.loc[d["pid"], "sex"].values
    d["age"] = pa.loc[d["pid"], "age_group"].values.astype(int)
    return d


def group_masks(d):
    """name -> boolean mask over participants (+ 'all')."""
    m = {"all": np.ones(len(d["pid"]), bool)}
    for t, g in SUBGROUPS:
        m[f"{t}={g}"] = (d["sex"] == g) if t == "sex" else (d["age"] == int(g))
    return m


# ------------------------------------------------------------------ metrics
def ece_equal_count(y, p, bins=10):
    """Expected calibration error with equal-count bins (rare events make equal-width bins degenerate)."""
    o = np.argsort(p, kind="stable")
    ys, ps = y[o], p[o]
    e = 0.0
    for idx in np.array_split(np.arange(len(p)), bins):
        if len(idx):
            e += len(idx) / len(p) * abs(ys[idx].mean() - ps[idx].mean())
    return float(e)


def clf_summary(y, p):
    """AP, AUROC, Brier, ECE, mean prediction / prevalence. NaN when only one class is present."""
    pos = 0 < y.sum() < len(y)
    return dict(ap=float(average_precision_score(y, p)) if pos else np.nan, auroc=float(roc_auc_score(y, p)) if pos else np.nan,
                brier=float(((p - y) ** 2).mean()), ece=ece_equal_count(y, p), prevalence=float(y.mean()), mean_pred=float(p.mean()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib_slope_intercept(y, p):
    """Logistic recalibration of y on logit(p): returns (intercept, slope). Ideal (0, 1). NaN if single class."""
    if not 0 < y.sum() < len(y):
        return np.nan, np.nan
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def eligible(Y):
    """(participant idx, day idx) of rows with a defined label."""
    return np.where(~np.isnan(Y))


def boot_ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
