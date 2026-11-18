"""Stage 9 (v2 pipeline, SYNTHETIC): window representation (factors R, O) and training-time augmentation (factor T). Nothing frozen is modified.

Factors are kept separate:  R = missingness-indicator representation, O = prolonged-outage fill rule, T = training augmentation.
R0 = frozen 2 flags, R1 = 8 per-signal flags; O0 = frozen fill (ffill <= 3 days then 0), O1/O2/O3 = alternative outage fills; T0/T1 = no/with augmentation.

Representation works on a 31-day slice (days t-30..t, NaN-padded before day 0) so that the 3-day forward-fill carry-in is exact; the output window is days t-27..t.
Flags always follow the frozen semantics computed from the NaN pattern: missing_day = all 8 signals NaN, hrv_missing = HRV NaN (honest flags).

Augmentation (T1) is random numbers only: it never sees labels, latent states or participant ids. Gap families: random_day, block (3/7/14), single_group, multi_group
(rate ~ U(0, RMAX); it never removes a whole group for all days and has no explicit 'trailing outage' generator). Applied to a window with probability P_AUG.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_missing")]
import inject as ij  # noqa: E402
import preprocess as pp  # noqa: E402

L31 = 31
GROUPS = ij.GROUPS
GROUP_LIST = list(GROUPS)
SIG2GROUP = np.zeros(8, int)
for gi, g in enumerate(GROUP_LIST):
    SIG2GROUP[GROUPS[g]] = gi
P_AUG, RMAX = 0.5, 0.5
FAMILIES = ("random_day", "block", "single_group", "multi_group")
BLOCKS = {"random_day": (1,), "block": (3, 7, 14), "single_group": (1, 7), "multi_group": (1, 7)}


def pad_raw(R):
    """(N,120,8) standardized values with NaN -> (N,123,8) with 3 NaN lead days (as the frozen pipeline's ffill start)."""
    return np.concatenate([np.full((R.shape[0], 3, 8), np.nan), R], axis=1)


def slice31(Rp, pi, ti):
    idx = (ti - 27)[:, None] + np.arange(L31)[None, :]
    return Rp[pi[:, None], idx]


TAU = 7.0                                                 # O2 decay constant (days), fixed in advance


def _last_age(S):
    """last observed value and days since it (0 = observed today; inf = nothing observed yet in the slice), causal."""
    B, L, C = S.shape
    last, age = np.full((B, C), np.nan), np.full((B, C), np.inf)
    La, Ag = np.empty_like(S), np.empty_like(S)
    for t in range(L):
        cur = S[:, t]
        obs = ~np.isnan(cur)
        last, age = np.where(obs, cur, last), np.where(obs, 0.0, age + 1)
        La[:, t], Ag[:, t] = last, age
    return La, Ag


def fill(S, O="O0"):
    """(B,31,8) NaN-pattern -> filled values (B,31,8); observed values are never changed. Standardized scale, so 0 = train mean.
    O0 frozen: forward-fill <= 3 days, then 0.   O1: unlimited last-value carry-forward (0 before the first observation).
    O2: last value decayed towards 0 with exp(-age/TAU), for every age.   O3: forward-fill <= 3 days, then the causal running mean of the observed values in the slice (0 if none)."""
    if O == "O0":
        return np.nan_to_num(pp.causal_ffill(S, pp.FFILL_LIMIT), nan=0.0)
    if O == "O3":
        f3 = pp.causal_ffill(S, pp.FFILL_LIMIT)
        obs = ~np.isnan(S)
        cnt = np.cumsum(obs, axis=1)
        mean = np.where(cnt > 0, np.cumsum(np.where(obs, S, 0.0), axis=1) / np.maximum(cnt, 1), 0.0)
        return np.where(np.isnan(f3), mean, f3)
    La, Ag = _last_age(S)
    if O == "O1":
        return np.nan_to_num(La, nan=0.0)
    if O == "O2":
        with np.errstate(invalid="ignore"):
            return np.nan_to_num(La * np.exp(-Ag / TAU), nan=0.0)
    raise NotImplementedError(f"O={O}")


def represent(S, R="R0", O="O0", flags="H", S_orig=None):
    """(B,31,8) NaN-pattern slice -> (B,28,C) float32 model input.  R0: 2 frozen flags (missing_day = all 8 missing, hrv_missing), C=10.
    R1: 8 per-signal missing flags (1 = that signal missing that day), C=16.  Flags are computed from `flags`: 'H' honest (from S), 'O' original flags kept
    (from S_orig, the clean slice, so injected gaps are NOT flagged), 'Z' all zero. Values always follow the fill rule O applied to S."""
    if R not in ("R0", "R1"):
        raise NotImplementedError(f"R={R}")
    z = fill(S, O)[:, 3:]
    src = S if flags == "H" else S_orig
    if flags == "Z":
        m = np.zeros(z.shape[:2] + (2 if R == "R0" else 8,))
    elif R == "R0":
        m = ij.honest_masks(src[:, 3:])
    else:
        m = np.isnan(src[:, 3:]).astype(np.float64)
    return np.concatenate([z, m], axis=2).astype(np.float32)


def n_channels(R):
    return 10 if R == "R0" else 16


def scenario_slices(sc, rep, split, R_raw, Rp, pi, ti):
    """(S, S_orig): 31-day NaN-pattern slices of the scenario (same gap draws as Stage 8) and of the clean data."""
    S_orig = slice31(Rp, pi, ti)
    spec = sc["spec"]
    if spec is None:
        return S_orig, S_orig
    if spec["scope"] == "trailing":
        S = S_orig.copy()
        S[:, L31 - spec["k"]:, :] = np.nan
        return S, S_orig
    Rg = ij.inject_series(R_raw, spec, ij.rng_for(sc["base"], rep, split))
    return slice31(pad_raw(Rg), pi, ti), S_orig


def parse_config(name):
    """'R1O2T1' -> ('R1', 'O2', 1)"""
    return name[:2], name[2:4], int(name[5])


# ------------------------------------------------------------------ augmentation (T1)
def _cover(r, b, L, rng):
    """(n,L) bool: day covered by a gap. r (n,) per-window P(covered), b (n,) block length (same-b rows only)."""
    n = len(r)
    if b == 1:
        return rng.random((n, L)) < r[:, None]
    q = 1.0 - (1.0 - r) ** (1.0 / b)
    starts = (rng.random((n, L + b - 1)) < q[:, None]).astype(np.int32)
    cs = np.cumsum(starts, axis=1)
    hi = cs[:, b - 1:b - 1 + L]
    lo = np.concatenate([np.zeros((n, 1), np.int32), cs[:, :L - 1]], axis=1)
    return (hi - lo) > 0


def gap_mask(B, rng, p_aug=P_AUG, rmax=RMAX, L=L31):
    """(B,L,8) bool of NEWLY removed signal-days. Independent of any data: only the generator state matters."""
    fam = rng.integers(0, len(FAMILIES), B)
    on = rng.random(B) < p_aug
    rate = rng.random(B) * rmax
    g1 = rng.integers(0, len(GROUP_LIST), B)
    bsel = rng.random(B)
    blk = np.ones(B, int)
    for fi, f in enumerate(FAMILIES):
        opts = BLOCKS[f]
        sel = fam == fi
        blk[sel] = np.array(opts)[np.minimum((bsel[sel] * len(opts)).astype(int), len(opts) - 1)]
    cov = np.zeros((B, L, len(GROUP_LIST)), bool)
    for b in np.unique(blk):
        rows = np.where(blk == b)[0]
        for gi in range(len(GROUP_LIST)):
            cov[rows, :, gi] = _cover(rate[rows], int(b), L, rng)
    allg = fam <= 1                                           # random_day / block: all signals together (use group-0 draw)
    cov[allg] = cov[allg][:, :, :1]
    one = fam == 2
    cov[one] &= (np.arange(len(GROUP_LIST))[None, None, :] == g1[one][:, None, None])
    cov[~on] = False
    return cov[:, :, SIG2GROUP]


def augment(S, rng, p_aug=P_AUG, rmax=RMAX):
    """S (B,31,8) with original NaNs kept. Injected gaps only ADD missingness (original gaps are never filled)."""
    out = S.copy()
    out[gap_mask(len(S), rng, p_aug, rmax)] = np.nan
    return out
