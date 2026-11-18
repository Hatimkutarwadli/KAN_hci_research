"""Stage 2 preprocessing for the SYNTHETIC longitudinal dataset. Raw data is never modified.

Usage: python src/preprocess.py --raw data/synthetic_v1 --out data/processed_v1

Pipeline (all steps causal / participant-safe):
 1. verify raw file checksums; load ONLY the model-input columns declared in raw meta.json
    (stress_proxy, latent states, targets and truth are never loaded as inputs);
 2. fit per-feature mean/std on TRAIN participants' OBSERVED values only;
 3. standardize -> causal forward fill (<= 3 days, past values only) -> remaining NaN := 0 (= train mean);
 4. channels = 8 value channels + 2 raw mask channels (missing_day, hrv_missing; 1 = missing);
 5. static inputs = sex (F=1) + one-hot age_group;
 6. labels rebuilt from daily latent status (independent of the simulator's label columns, then cross-checked);
 7. window access (28 days, t-27..t) is done at load time by data.make_windows; origins t < 27 are NaN in all labels.
Output per split: X (N,120,10) float32, static (N,4), pid, and label arrays y_<name> (N,120) with NaN = undefined.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

LOOKBACK = 28            # days t-27 .. t
FFILL_LIMIT = 3
HORIZONS = (7, 14, 21)
PERSIST = 3              # primary onset: >= 3 consecutive positive days
FORBIDDEN_INPUTS = {"stress_proxy", "z_dep", "z_anx", "stress"}


# ---------------------------------------------------------------- core functions (unit-tested)
def causal_ffill(x, limit):
    """Forward fill along axis 1 using only past/present values, at most `limit` consecutive days.
    x: (N, D, ...) with NaN = missing. Days beyond `limit` after the last observation stay NaN."""
    out = x.copy()
    last = np.full_like(x[:, 0], np.nan)
    age = np.full(x[:, 0].shape, np.inf)
    for t in range(x.shape[1]):
        cur = x[:, t]
        obs = ~np.isnan(cur)
        last = np.where(obs, cur, last)
        age = np.where(obs, 0.0, age + 1)
        out[:, t] = np.where(obs, cur, np.where(age <= limit, last, np.nan))
    return out


def fit_scaler(values_train):
    """values_train: (N, D, F) raw values with NaN for missing. Observed-only mean/std per feature."""
    mean = np.nanmean(values_train, axis=(0, 1))
    std = np.nanstd(values_train, axis=(0, 1))
    return {"mean": mean, "std": np.maximum(std, 1e-8)}


def transform(values, masks, scaler, limit=FFILL_LIMIT):
    """values (N,D,F) raw w/ NaN; masks (N,D,M) raw missing flags. Returns (N,D,F+M) float32, no NaN."""
    z = (values - scaler["mean"]) / scaler["std"]
    z = causal_ffill(z, limit)
    z = np.nan_to_num(z, nan=0.0)
    return np.concatenate([z, masks], axis=2).astype(np.float32)


def build_labels(status, lookback=LOOKBACK, horizons=HORIZONS, P=PERSIST):
    """status: (N,D) daily latent status (0/1). Returns dict of (N,D) float arrays, NaN = undefined.
    PRIMARY  sust : onset = first day starting >=P consecutive positive days; label(t,h)=1 iff onset in (t,t+h];
                    needs t+h+P-1 <= D-1 (confirmation days observed).
    SECONDARY cross: onset = first positive day; needs t+h <= D-1.
    Both: participants positive on day 0 are prevalent (excluded); at risk until onset; origins t < lookback-1 are NaN."""
    N, D = status.shape
    prevalent = status[:, 0] == 1
    run = np.ones((N, D - P + 1), dtype=bool)
    for j in range(P):
        run &= status[:, j:D - P + 1 + j] == 1
    os_ = np.where(run.any(1) & ~prevalent, run.argmax(1), np.nan)[:, None]
    oc = np.where(status.any(1) & ~prevalent, status.argmax(1), np.nan)[:, None]
    t = np.arange(D)[None, :]
    ever = np.maximum.accumulate(status, axis=1) > 0
    risk_s = ~prevalent[:, None] & ~(os_ <= t)
    risk_c = ~prevalent[:, None] & ~ever
    early = t[0] < lookback - 1
    out = {"at_risk_sust": np.where(risk_s, 1.0, 0.0), "at_risk_cross": np.where(risk_c, 1.0, 0.0)}
    for h in horizons:
        a = ((os_ > t) & (os_ <= t + h)).astype(float)
        a[~risk_s] = np.nan
        a[:, t[0] + h + P - 1 > D - 1] = np.nan
        b = ((oc > t) & (oc <= t + h)).astype(float)
        b[~risk_c] = np.nan
        b[:, t[0] + h > D - 1] = np.nan
        out[f"onset_sust_h{h}"], out[f"onset_cross_h{h}"] = a, b
    for v in out.values():
        v[:, early] = np.nan
    return out


def shift_future(arr, h):
    """value at t+h placed at index t; NaN where t+h > last day."""
    out = np.full(arr.shape, np.nan)
    if h == 0:
        return arr.astype(float)
    out[:, :-h] = arr[:, h:]
    return out


# ---------------------------------------------------------------- pipeline
def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_raw(raw):
    bad = []
    with open(os.path.join(raw, "checksums.sha256")) as f:
        lines = f.read().splitlines()
    for line in lines:
        h, fn = line.split()
        if sha(os.path.join(raw, fn)) != h:
            bad.append(fn)
    if bad:
        raise SystemExit(f"raw checksum mismatch: {bad}")
    return {line.split()[1]: line.split()[0] for line in lines}


def run(raw, out, readonly=True):
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite.")
    raw_sums = verify_raw(raw)
    with open(os.path.join(raw, "meta.json")) as f:
        meta = json.load(f)
    cfg = meta["config"]
    N, D = cfg["n_participants"], cfg["n_days"]
    inputs = [c for c in meta["model_input_columns"] if "(static)" not in c]
    assert not (set(inputs) & FORBIDDEN_INPUTS) and "stress_proxy" not in inputs
    feats = meta["features"]
    mask_cols = ["missing_day", "hrv_missing"]
    assert sorted(inputs) == sorted(feats + mask_cols), inputs
    channels = feats + mask_cols

    obs = pd.read_csv(os.path.join(raw, "observations.csv"), usecols=["pid", "day"] + inputs).sort_values(["pid", "day"])
    assert len(obs) == N * D and (obs.groupby("pid").day.apply(lambda s: (s.values == np.arange(D)).all())).all()
    values = obs[feats].values.reshape(N, D, len(feats)).astype(float)
    masks = obs[mask_cols].values.reshape(N, D, 2).astype(float)
    pa = pd.read_csv(os.path.join(raw, "participants.csv")).sort_values("pid")
    static = np.column_stack([(pa.sex == "F").astype(float).values] + [(pa.age_group == g).astype(float).values for g in (0, 1, 2)])
    static_names = ["sex_F", "age_group_0", "age_group_1", "age_group_2"]
    sp = pd.read_csv(os.path.join(raw, "splits.csv")).sort_values("pid")
    split = sp.split.values
    tg = pd.read_csv(os.path.join(raw, "targets.csv")).sort_values(["pid", "day"])

    # scaler from TRAIN participants, observed values only
    tr = split == "train"
    scaler = fit_scaler(values[tr])
    X = transform(values, masks, scaler)

    # labels
    Y = {}
    for k in ("dep", "anx"):
        status = tg[f"status_{k}"].values.reshape(N, D).astype(int)
        sev = tg[f"severity_{k}"].values.reshape(N, D)
        lab = build_labels(status)
        early = np.arange(D) < LOOKBACK - 1
        for name, arr in lab.items():
            Y[f"{name.replace('onset_sust', 'onset_sust_' + k).replace('onset_cross', 'onset_cross_' + k).replace('at_risk_sust', 'at_risk_sust_' + k).replace('at_risk_cross', 'at_risk_cross_' + k)}"] = arr
        for h in (0,) + HORIZONS:
            tag = "t0" if h == 0 else f"h{h}"
            s_, v_ = shift_future(status, h), shift_future(sev, h)
            s_[:, early] = np.nan
            v_[:, early] = np.nan
            Y[f"status_{k}_{tag}"], Y[f"sev_{k}_{tag}"] = s_, v_
        # cross-check against the simulator's own label columns (t >= lookback-1)
        for h in HORIZONS:
            for mine, theirs in ((f"onset_sust_{k}_h{h}", f"onset_sust_{k}_h{h}"), (f"onset_cross_{k}_h{h}", f"onset_{k}_h{h}")):
                a = Y[mine][:, LOOKBACK - 1:]
                b = tg[theirs].values.reshape(N, D)[:, LOOKBACK - 1:]
                assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, equal_nan=True), (mine, theirs)
        for mine, theirs in ((f"at_risk_sust_{k}", f"at_risk_sust_{k}"), (f"at_risk_cross_{k}", f"at_risk_{k}")):
            assert np.array_equal(Y[mine][:, LOOKBACK - 1:], tg[theirs].values.reshape(N, D)[:, LOOKBACK - 1:].astype(float))

    os.makedirs(out, exist_ok=True)
    counts, files = {}, {}
    for s in ("train", "cal", "test"):
        m = split == s
        arrs = {"X": X[m], "static": static[m].astype(np.float32), "pid": sp.pid.values[m]}
        arrs.update({f"y_{k}": v[m].astype(np.float32) for k, v in Y.items()})
        np.savez(os.path.join(out, f"{s}.npz"), **arrs)
        counts[s] = {"participants": int(m.sum()), "X_shape": list(X[m].shape), "static_shape": list(static[m].shape)}
        for k, v in Y.items():
            if k.startswith("onset_"):
                vv = v[m]
                counts[s][k] = {"eligible_origins": int(np.isfinite(vv).sum()), "positive_origins": int(np.nansum(vv)),
                                "positive_participants": int((np.nansum(vv, axis=1) > 0).sum())}
        files[f"{s}.npz"] = sha(os.path.join(out, f"{s}.npz"))
    filled = {}
    obs_val = ~np.isnan(values)
    zf = causal_ffill((values - scaler["mean"]) / scaler["std"], FFILL_LIMIT)
    for i, f in enumerate(feats):
        miss = ~obs_val[:, :, i]
        filled[f] = {"missing_days": int(miss.sum()), "forward_filled": int((miss & ~np.isnan(zf[:, :, i])).sum()),
                     "zero_filled": int((miss & np.isnan(zf[:, :, i])).sum())}
    pmeta = {"SYNTHETIC": True, "source_raw_dir": raw, "source_raw_checksums": raw_sums, "source_seed": cfg["seed"],
             "input_columns_used": inputs, "channel_names": channels, "static_names": static_names,
             "excluded": sorted(set(meta["analysis_only_columns"]) | {"all of targets.csv", "all of truth.csv", "truth_participants.csv"}),
             "scaler_fit_on": "train participants, observed values only", "scaler_mean": dict(zip(feats, scaler["mean"].tolist())),
             "scaler_std": dict(zip(feats, scaler["std"].tolist())), "ffill_limit_days": FFILL_LIMIT,
             "remaining_missing_filled_with": 0.0, "mask_semantics": "1 = missing (raw column semantics)",
             "lookback_days": LOOKBACK, "window": "days t-27..t inclusive; first origin t=27", "horizons": list(HORIZONS),
             "primary_onset": f"onset_sust_*: first day starting >= {PERSIST} consecutive positive latent days",
             "secondary_onset": "onset_cross_*: first positive day (legacy first-crossing)",
             "label_censoring": "sust: t+h+2 <= 119; cross: t+h <= 119; origins t<27 are NaN in every label",
             "label_names": sorted(Y.keys()), "label_nan_means": "undefined (not at risk, prevalent, censored, or t<27)",
             "split_counts": counts, "feature_fill_stats": filled, "output_file_sha256": files,
             "label_cross_check_vs_simulator": "PASS (asserted in run)"}
    with open(os.path.join(out, "preprocessing_meta.json"), "w") as f:
        json.dump(pmeta, f, indent=1)
    verify_raw(raw)  # raw unchanged
    if readonly:
        for fn in os.listdir(out):
            os.chmod(os.path.join(out, fn), 0o444)
    return pmeta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/synthetic_v1")
    ap.add_argument("--out", default="data/processed_v1")
    a = ap.parse_args()
    m = run(a.raw, a.out)
    print("saved", a.out)
    print({s: m["split_counts"][s]["X_shape"] for s in ("train", "cal", "test")})
