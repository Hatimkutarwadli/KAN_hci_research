"""Window summary features for the B1 gradient-boosting baseline. Uses ONLY days t-27..t of one participant.
Input window W: (M, 28, 10) = 8 standardized value channels + missing_day + hrv_missing (1 = missing).
Observed-only summaries (filled values are ignored via the masks); NaN where no observed day exists."""
import numpy as np

FEATS = ["steps", "calories", "sleep_duration", "sleep_efficiency", "sleep_variability", "resting_hr", "hrv",
         "circadian_regularity"]
STATS = ["mean28", "std28", "min28", "max28", "last_obs", "slope28", "mean7", "mean3"]
MISS = ["miss_frac28", "hrv_miss_frac28", "miss_frac7", "hrv_miss_frac7", "trailing_missing", "trailing_hrv_missing"]
STATIC = ["sex_F", "age_group_0", "age_group_1", "age_group_2"]
NAMES = [f"{f}_{s}" for f in FEATS for s in STATS] + MISS + STATIC


def _trailing_missing(m):
    """number of consecutive missing days ending at the last day. m: (M, L) 1 = missing."""
    obs_rev = (m[:, ::-1] == 0)
    first_obs = obs_rev.argmax(1)
    return np.where(obs_rev.any(1), first_obs, m.shape[1]).astype(np.float32)


def window_features(W, static):
    M, L, _ = W.shape
    obs = np.repeat((W[:, :, 8] == 0)[:, :, None], 8, axis=2)
    obs[:, :, 6] = W[:, :, 9] == 0                       # HRV has its own gap pattern
    V = np.where(obs, W[:, :, :8], 0.0).astype(np.float64)
    Vn = np.where(obs, W[:, :, :8], np.nan)
    tt = np.arange(L, dtype=np.float64)[None, :, None]
    n = obs.sum(1).astype(np.float64)
    nn_ = np.where(n > 0, n, np.nan)
    mean = V.sum(1) / nn_
    var = (np.where(obs, (V - mean[:, None, :]) ** 2, 0.0)).sum(1) / nn_
    with np.errstate(all="ignore"):
        mn, mx = np.nanmin(Vn, axis=1), np.nanmax(Vn, axis=1)
    last_idx = L - 1 - obs[:, ::-1, :].argmax(1)
    last = np.where(n > 0, np.take_along_axis(V, last_idx[:, None, :], axis=1)[:, 0, :], np.nan)
    mt = (tt * obs).sum(1) / nn_
    cov = (obs * (tt - mt[:, None, :]) * (V - mean[:, None, :])).sum(1) / nn_
    vt = (obs * (tt - mt[:, None, :]) ** 2).sum(1) / nn_
    with np.errstate(all="ignore"):
        slope = np.where(n >= 3, cov / vt, np.nan)

    def recent(k):
        o, v = obs[:, -k:, :], V[:, -k:, :]
        c = o.sum(1).astype(np.float64)
        return np.where(c > 0, v.sum(1) / np.where(c > 0, c, np.nan), np.nan)

    per_feat = np.stack([mean, np.sqrt(var), mn, mx, last, slope, recent(7), recent(3)], axis=2)  # (M, 8, 8)
    miss = np.column_stack([W[:, :, 8].mean(1), W[:, :, 9].mean(1), W[:, -7:, 8].mean(1), W[:, -7:, 9].mean(1),
                            _trailing_missing(W[:, :, 8]), _trailing_missing(W[:, :, 9])])
    return np.column_stack([per_feat.reshape(M, -1), miss, static]).astype(np.float32)
