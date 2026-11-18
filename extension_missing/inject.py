"""Stage 8 (additional analysis): controlled artificial missingness for ROBUSTNESS testing of frozen models. SYNTHETIC data only.

Nothing here modifies the frozen data or preprocessing: the standardized observed values are reconstructed from the frozen processed arrays
(observed <=> mask channel = 0), gaps are injected, and the frozen `preprocess.transform` / `causal_ffill` (limit 3, then 0) is re-applied with the
frozen train scaler (identity here because values are already standardized). Gaps are MCAR: random numbers only, never labels or latent states.

Mask-channel variants (to separate physiological signal loss from the informative missingness indicators):
  'H' honest/frozen semantics: missing_day = all 8 signals NaN, hrv_missing = HRV NaN (injected gaps are flagged; flags are diluted by MCAR gaps)
  'O' original flags kept: values are lost but the two mask channels stay exactly as in the original data (injected gaps NOT flagged)
  'Z' flags zeroed: no injection, both mask channels set to 0 (informative indicators removed, signal values untouched)
"""
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import preprocess as pp  # noqa: E402  (frozen module, imported read-only)

FEATS = ["steps", "calories", "sleep_duration", "sleep_efficiency", "sleep_variability", "resting_hr", "hrv", "circadian_regularity"]
GROUPS = {"activity": [0, 1], "sleep": [2, 3, 4], "cardiac": [5, 6], "circadian": [7]}
IDENT = {"mean": 0.0, "std": 1.0}
OFF = np.arange(-27, 1)
LOOK = 28


# ------------------------------------------------------------------ reconstruction / masks
def reconstruct_raw(X):
    """(N,120,10) frozen processed array -> standardized observed values (N,120,8) with NaN where missing."""
    R = X[:, :, :8].astype(np.float64)
    R[X[:, :, 8] == 1] = np.nan
    h = R[:, :, 6]
    h[X[:, :, 9] == 1] = np.nan
    return R


def honest_masks(R):
    """frozen mask semantics from a NaN pattern: missing_day = all 8 NaN; hrv_missing = HRV NaN."""
    return np.stack([np.isnan(R).all(-1), np.isnan(R[..., 6])], -1).astype(np.float64)


def rebuild(R, masks):
    """frozen transform (causal ffill <=3 days, then 0) + mask channels -> (N,D,10) float32."""
    return pp.transform(R, masks, IDENT)


# ------------------------------------------------------------------ gap generators
def drop_mask(shape, rate, block, rng):
    """Boolean (N,T) covered-by-gap mask. block=1: iid daily. block=L>1: blocks of L days with per-day start probability q chosen so that
    P(day covered) = 1-(1-q)^L = rate. rate>=1 -> everything."""
    N, T = shape
    if rate >= 1.0:
        return np.ones(shape, bool)
    if block <= 1:
        return rng.random(shape) < rate
    q = 1.0 - (1.0 - rate) ** (1.0 / block)
    starts = (rng.random((N, T + block - 1)) < q).astype(np.int32)
    cs = np.cumsum(starts, axis=1)
    hi = cs[:, block - 1:block - 1 + T]
    lo = np.concatenate([np.zeros((N, 1), np.int32), cs[:, :T - 1]], axis=1)
    return (hi - lo) > 0


def inject_series(R, spec, rng):
    """spec: scope in {'all','group','multi'}, rate, block, group (for 'group')."""
    out = R.copy()
    N, T, _ = R.shape
    if spec["scope"] == "all":
        out[drop_mask((N, T), spec["rate"], spec["block"], rng)] = np.nan
    elif spec["scope"] == "group":
        d = drop_mask((N, T), spec["rate"], spec["block"], rng)
        for j in GROUPS[spec["group"]]:
            out[:, :, j][d] = np.nan
    else:                                                    # 'multi': every signal group draws its own gaps independently
        for g in GROUPS:
            d = drop_mask((N, T), spec["rate"], spec["block"], rng)
            for j in GROUPS[g]:
                out[:, :, j][d] = np.nan
    return out


# ------------------------------------------------------------------ windows
def windows(Xs, pi, ti):
    return Xs[pi[:, None], ti[:, None] + OFF]


def trailing_windows(R, X0, pi, ti, k, flags):
    """Window for each origin (pi,ti) in which the last k days (t-k+1..t) are missing for ALL signals (outage right before the prediction).
    The 3-day forward-fill carry-in is exact: the slice starts 3 days before the window (NaN-padded before day 0, as in the frozen pipeline)."""
    Rp = np.concatenate([np.full((R.shape[0], 3, 8), np.nan), R], axis=1)
    idx = (ti - 27)[:, None] + np.arange(LOOK + 3)[None, :]
    S = Rp[pi[:, None], idx]                                 # (M,31,8): days t-30..t
    if k > 0:
        S[:, S.shape[1] - k:, :] = np.nan
    z = np.nan_to_num(pp.causal_ffill(S, pp.FFILL_LIMIT), nan=0.0)[:, 3:]
    Sw = S[:, 3:]
    masks = honest_masks(Sw) if flags == "H" else X0[pi[:, None], ti[:, None] + OFF][:, :, 8:10].astype(np.float64)
    return np.concatenate([z, masks], axis=2).astype(np.float32)


def series_windows(R, X0, pi, ti, spec, flags, rng):
    Rg = inject_series(R, spec, rng) if spec is not None else R
    if flags == "H":
        m = honest_masks(Rg)
    elif flags == "O":
        m = X0[:, :, 8:10].astype(np.float64)
    elif flags == "Z":
        m = np.zeros(R.shape[:2] + (2,))
    else:
        raise ValueError(flags)
    return windows(rebuild(Rg, m), pi, ti)


# ------------------------------------------------------------------ scenario registry (28 approved + decomposition variants)
def _s(id_, family, flags="H", reps=(0, 1), **spec):
    return dict(id=id_, family=family, flags=flags, reps=tuple(reps), spec=spec or None, base=id_[:-2] if id_.endswith("_O") else id_)


def registry():
    sc = [_s("C0", "control", reps=(0,))]
    for r in (10, 20, 40, 60):
        sc.append(_s(f"S1_r{r}", "S1_random_wholeday", scope="all", rate=r / 100, block=1))
    for r in (10, 20, 40, 60):
        sc.append(_s(f"S2_r{r}_L7", "S2_block_wholeday", scope="all", rate=r / 100, block=7))
    sc.append(_s("S2_r20_L3", "S2_block_wholeday", scope="all", rate=0.2, block=3))
    sc.append(_s("S2_r20_L14", "S2_block_wholeday", scope="all", rate=0.2, block=14))
    for k in (3, 7, 14):
        sc.append(dict(id=f"S3_k{k}", family="S3_trailing_outage", flags="H", reps=(0,), spec=dict(scope="trailing", k=k), base=f"S3_k{k}"))
    for g in GROUPS:
        sc.append(_s(f"S4_{g}_rand40", "S4_single_group", scope="group", group=g, rate=0.4, block=1))
        sc.append(_s(f"S4_{g}_blk40", "S4_single_group", scope="group", group=g, rate=0.4, block=7))
        sc.append(_s(f"S4_{g}_absent", "S4_single_group", reps=(0,), scope="group", group=g, rate=1.0, block=1))
    sc.append(_s("S5_rand40", "S5_multi_group", scope="multi", rate=0.4, block=1))
    sc.append(_s("S5_blk40", "S5_multi_group", scope="multi", rate=0.4, block=7))
    n_approved = len(sc)
    sc.append(_s("Z0", "decomposition_flags_zeroed", flags="Z", reps=(0,)))
    for base in ("S1_r20", "S1_r40", "S2_r20_L7", "S2_r40_L7", "S3_k7", "S3_k14"):
        b = next(s for s in sc if s["id"] == base)
        sc.append(dict(id=base + "_O", family="decomposition_flags_original", flags="O", reps=b["reps"], spec=b["spec"], base=base))
    assert n_approved == 28, n_approved
    return sc


def rng_for(base, rep, split):
    return np.random.default_rng(810000 + zlib.crc32(base.encode()) % 100000 * 7 + rep * 1000 + {"cal": 0, "test": 500}[split])


def make_windows(sc, rep, split, R, X0, pi, ti):
    """Windows (M,28,10) for a scenario. Variant 'O' shares the SAME gap draw as its base scenario (same rng seed) - only the flag channels differ."""
    spec, flags = sc["spec"], sc["flags"]
    if spec is not None and spec["scope"] == "trailing":
        return trailing_windows(R, X0, pi, ti, spec["k"], flags)
    return series_windows(R, X0, pi, ti, spec, flags, rng_for(sc["base"], rep, split))
