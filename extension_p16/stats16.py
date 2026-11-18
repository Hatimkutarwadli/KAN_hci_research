"""P1-6 statistical core (SYNTHETIC; EXTENSION). Pure functions, unit-tested (tests/test_p16.py).

Participant-cluster bootstrap by MULTIPLICITY WEIGHTS: one shared matrix C (K x N) of participant counts (row 0 = identity = the pooled point estimate used by Stage 8/9; rows 1..B = bootstrap resamples, fixed
seed) is used for EVERY model, seed, scenario, replicate and task, so all contrasts are paired. A metric for resample k is the metric of the multiset of participants with counts C[k].
Metrics are computed from these weights without copying data: weighted RMSE from per-participant sums, weighted AP/AUROC from one sort per (model, seed, task) (ties handled at distinct scores as in sklearn).
Seed handling and hypothesis tests are in analyze_p16.py; seed variability is kept separate from participant variability."""
import numpy as np

CONTRASTS = [("R0O0T1", "baseline"), ("R0O2T1", "baseline"), ("R0O3T1", "baseline"), ("R0O2T1", "R0O0T1"), ("R0O3T1", "R0O0T1")]     # (new, reference); 'baseline' = original frozen GRU
FAMS = {"severity (nowcast)": ("main", ["sev_nowcast_rmse"]), "status (nowcast)": ("main", ["status_ap"]), "onset": ("main", ["onset_ap", "onset_auroc"]),
        "forecast severity": ("forecast", ["sev_forecast_rmse"]), "forecast status": ("forecast", ["status_ap"])}


def boot_counts(B, N, seed):
    """(B+1, N) participant multiplicity matrix. Row 0 = identity (point estimate); rows 1..B = bootstrap resamples drawn with a fixed seed."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, (B, N))
    C = np.zeros((B + 1, N))
    C[0] = 1.0
    for b in range(B):
        C[b + 1] = np.bincount(idx[b], minlength=N)
    return C


def weighted_rmse(sq_err, g, C, N):
    sse = np.bincount(g, weights=sq_err, minlength=N)
    n = np.bincount(g, minlength=N).astype(float)
    return np.sqrt((C @ sse) / (C @ n))


def weighted_mean(x, g, C, N):
    return (C @ np.bincount(g, weights=x, minlength=N)) / (C @ np.bincount(g, minlength=N).astype(float))


def weighted_ap_auroc(y, s, g, C, chunk=250):
    """y in {0,1} (n,), scores s (n,), participant index g (n,), counts C (K, N). Returns AP (K,) and AUROC (K,); NaN where a resample has no positives or no negatives.
    Ties in the score are treated as one threshold (identical to sklearn's average_precision_score / roc_auc_score for integer weights)."""
    order = np.argsort(-s, kind="stable")
    ys, ss, gs = y[order].astype(float), s[order], g[order]
    ends = np.r_[ss[1:] != ss[:-1], True]
    K = C.shape[0]
    ap, auc = np.empty(K), np.empty(K)
    for a in range(0, K, chunk):
        W = C[a:a + chunk][:, gs]
        TP, FP = np.cumsum(W * ys, 1), np.cumsum(W * (1 - ys), 1)
        P, N_ = TP[:, -1], FP[:, -1]
        TPe, FPe = TP[:, ends], FP[:, ends]
        den = TPe + FPe
        prec = np.where(den > 0, TPe / np.where(den > 0, den, 1), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rec = TPe / P[:, None]
            drec = np.diff(np.concatenate([np.zeros((len(W), 1)), rec], 1), axis=1)
            ap[a:a + chunk] = np.where(P > 0, np.nansum(drec * prec, 1), np.nan)
            dpos = np.diff(np.concatenate([np.zeros((len(W), 1)), TPe], 1), axis=1)
            dneg = np.diff(np.concatenate([np.zeros((len(W), 1)), FPe], 1), axis=1)
            auc[a:a + chunk] = np.where((P > 0) & (N_ > 0), (dpos * ((N_[:, None] - FPe) + 0.5 * dneg)).sum(1) / (P * N_), np.nan)
    return ap, auc


# ------------------------------------------------------------------ contrasts
def worse_pct(group, new, ref):
    """Stage 9 convention (NEGATIVE = new better): relative RMSE increase, or relative loss in AP/AUROC, in percent."""
    return (new - ref) / ref * 100 if "rmse" in group else (ref - new) / ref * 100


def two_sided_centred_p(T):
    """Bootstrap test of H0: effect = 0 from the paired participant bootstrap. T[0] = point estimate, T[1:] = resamples. The bootstrap distribution is re-centred at the point estimate (its mean under
    the bootstrap world) so the null distribution is T_b - T_0; p = (1 + #{|T_b - T_0| >= |T_0|}) / (B + 1). Resamples with NaN are ignored (and counted by the caller)."""
    t0, tb = T[0], T[1:]
    tb = tb[np.isfinite(tb)]
    return (1 + int((np.abs(tb - t0) >= abs(t0) - 1e-15).sum())) / (len(tb) + 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, p[i] * m / rank)
        adj[i] = run
    return adj


# ------------------------------------------------------------------ participant-level descriptive helpers
def tertile_groups(severity):
    """Subgroups fixed from the participants' mean TRUE nowcast severity only (labels, no model output): 0 = lowest third, 1 = middle, 2 = highest (ties broken by rank order)."""
    r = np.argsort(np.argsort(severity, kind="stable"), kind="stable")
    return np.minimum(r * 3 // len(severity), 2)


def top_share(improvement, frac=0.10):
    """Descriptive: share of the total net improvement (sum of improvement = -(new - ref) per participant) contributed by the top `frac` of participants (by improvement). NaN if the total is not positive."""
    imp = np.sort(np.asarray(improvement, float))[::-1]
    k = max(1, int(np.ceil(frac * len(imp))))
    tot = imp.sum()
    return float(imp[:k].sum() / tot) if tot > 0 else float("nan")


def weighted_share(flag, C):
    """share of participants with flag, for every resample row of C."""
    return (C @ flag.astype(float)) / C.sum(1)
