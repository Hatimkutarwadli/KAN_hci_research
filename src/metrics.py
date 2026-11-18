"""Metrics for the Stage 3 baselines (SYNTHETIC data). All functions take NaN-free arrays."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def reg_metrics(y, p):
    e = p - y
    ss_res, ss_tot = float((e ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "pearson": float(np.corrcoef(y, p)[0, 1]) if p.std() > 0 and y.std() > 0 else np.nan}


def best_f1_threshold(y, p):
    """Threshold maximizing F1 (chosen on CALIBRATION data only)."""
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def clf_metrics(y, p, thr=None):
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
           "ap": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
           "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
           "brier": float(((p - y) ** 2).mean())}
    if thr is not None:
        pred = p >= thr
        tp = float((pred & (y == 1)).sum())
        prec = tp / pred.sum() if pred.sum() > 0 else 0.0
        rec = tp / y.sum() if y.sum() > 0 else np.nan
        out.update({"threshold": float(thr), "precision": prec, "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0})
    return out


def participant_bootstrap(y, p, pid, fn, n_boot=200, seed=0):
    """Resample PARTICIPANTS with replacement; returns (lo, hi) 95% percentile CI of fn(y, p)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(pid)
    groups = {i: np.where(pid == i)[0] for i in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([groups[i] for i in pick])
        try:
            v = fn(y[idx], p[idx])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)
