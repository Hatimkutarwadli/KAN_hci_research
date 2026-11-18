"""Loaders for the processed SYNTHETIC dataset (output of preprocess.py)."""
import json
import os

import numpy as np

LOOKBACK = 28


def load_meta(proc_dir="data/processed_v1"):
    with open(os.path.join(proc_dir, "preprocessing_meta.json")) as f:
        return json.load(f)


def load_split(split, proc_dir="data/processed_v1"):
    z = np.load(os.path.join(proc_dir, f"{split}.npz"))
    return {k: z[k] for k in z.files}


def make_windows(d, label, lookback=LOOKBACK):
    """All eligible origins for `label` (non-NaN) as 28-day windows ending at the origin (days t-27..t).
    Returns X (M, lookback, C), static (M, S), y (M,), pid (M,), t (M,)."""
    Y = d[f"y_{label}"]
    pi, ti = np.where(~np.isnan(Y))
    assert len(ti) == 0 or ti.min() >= lookback - 1
    idx = ti[:, None] + np.arange(-(lookback - 1), 1)[None, :]
    return d["X"][pi[:, None], idx], d["static"][pi], Y[pi, ti], d["pid"][pi], ti
