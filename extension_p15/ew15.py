"""P1-5 EXTENSION (SYNTHETIC; not part of the original six stages): extended early-warning evaluation. Post-processing of SAVED predictions only; nothing is retrained and no existing
file is modified. This module generalises src/leadtime.py (Stage 7) to horizons h in {7, 14, 21} with matching windows W = h, WITHOUT changing the Stage 7 protocol:
  * threshold tau from CALIBRATION never-onset window maxima only, (floor(f*n)+1)-th largest, alarm iff score > tau (the Stage 7 function lt.tau_from_calibration is reused);
  * onset participant: window [max(27, o-W), min(o-1, T1)], lead = o - first alarm day; never-onset window 27..T1; participant-level false-alarm proportion;
  * episodes separated by >= 7 quiet days, burden per participant-day, participant precision, day-level precision/recall, achievable-lead rates.
T1 = last origin whose outcome (and, for sustained onset, the 3-day confirmation) is fully observed: sustained 117-h (h21 -> 96 = Stage 7), first-crossing 119-h.
The vectorised implementation here is verified against Stage 7's loop implementation and against its saved results (tests/test_p15.py, first test)."""
import functools
import math
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import leadtime as lt  # noqa: E402  (read-only reuse of the Stage 7 tau rule and cohort definition)

PROC = "data/processed_v1"
T0, EP_GAP = 27, 7
LEADS = (1, 3, 7, 14, 21)
TARGETS = (0.10, 0.20)
KS = ("dep", "anx")
HORIZONS = (7, 14, 21)
N_DAYS = 120


def t1_of(kind, h):
    return (117 if kind == "sust" else 119) - h


# ------------------------------------------------------------------ cohorts
def load_cohort(split, k, kind="sust", h=21):
    d = np.load(os.path.join(ROOT, PROC, f"{split}.npz"))
    onset, group = lt.cohort_from_at_risk(d[f"y_at_risk_{kind}_{k}"])
    T1 = t1_of(kind, h)
    lo, hi = np.zeros(len(group), int), np.full(len(group), -1)                     # empty window for excluded participants
    on, nv = group == "onset", group == "never"
    lo[on], hi[on] = np.maximum(T0, onset[on] - h), np.minimum(onset[on] - 1, T1)
    lo[nv], hi[nv] = T0, T1
    return dict(pid=d["pid"], onset=onset, group=group, Y=d[f"y_onset_{kind}_{k}_h{h}"], kind=kind, k=k, h=h, W=h, T1=T1, lo=lo, hi=hi)


def window_mask(coh, T=N_DAYS):
    t = np.arange(T)[None, :]
    return (t >= coh["lo"][:, None]) & (t <= coh["hi"][:, None])


def window_max(S, coh):
    """S (..., N, T) -> (..., N): maximum score inside each participant's evaluation window; NaN for excluded participants."""
    M = np.where(window_mask(coh), S, -np.inf).max(-1)
    return np.where(coh["group"] == "excl", np.nan, M)


def tau_from_calibration(Scal, cal, f):
    m = window_max(Scal, cal)
    return lt.tau_from_calibration(m[cal["group"] == "never"], f)


def alarm_stats(S, tau, coh):
    """S (..., N, T). Returns first_lead (onset participants; 0 = no alarm in window), n_days and n_ep (never-onset alarm days / episodes), all (..., N)."""
    A = (S > tau) & window_mask(coh)
    on, nv = coh["group"] == "onset", coh["group"] == "never"
    first = A.argmax(-1)
    first_lead = np.where(on & A.any(-1), coh["onset"] - first, 0)
    n_days = np.where(nv, A.sum(-1), 0)
    C = np.cumsum(A, -1)
    pad = np.concatenate([np.zeros(A.shape[:-1] + (EP_GAP + 1,), C.dtype), C], -1)
    prior = pad[..., EP_GAP:EP_GAP + A.shape[-1]] - pad[..., :A.shape[-1]]          # alarms in the EP_GAP days before each day: C[t-1] - C[t-1-EP_GAP]
    n_ep = np.where(nv, (A & (prior == 0)).sum(-1), 0)
    return dict(first_lead=first_lead.astype(int), n_days=n_days.astype(int), n_ep=n_ep.astype(int))


def achievable(coh, l):
    """(N,) bool for onset participants: an alarm with lead >= l is possible inside the window (Stage 7 definition)."""
    return (coh["group"] == "onset") & (coh["lo"] <= np.minimum(coh["hi"], coh["onset"] - l))


def leads_for(coh):
    return tuple(l for l in LEADS if l <= coh["W"])


def metrics_from(st, coh, idx=None):
    """Stage 7 `stat_metrics` generalised (same formulas). st fields (N,) arrays; idx optional bootstrap index."""
    g, fl, nd, ne, on = coh["group"], st["first_lead"], st["n_days"], st["n_ep"], coh["onset"]
    if idx is not None:
        g, fl, nd, ne, on = g[idx], fl[idx], nd[idx], ne[idx], on[idx]
    onm, nvm = g == "onset", g == "never"
    n_on, n_nv = int(onm.sum()), int(nvm.sum())
    out = {"n_onset": n_on, "n_never": n_nv}
    exposure = (coh["T1"] - T0 + 1)
    for l in LEADS:
        if l > coh["W"]:
            for s in ("", "_achievable"):
                out[f"det_lead{l}{s}"] = np.nan
            out[f"n_achievable_lead{l}"] = 0
            continue
        out[f"det_lead{l}"] = float((fl[onm] >= l).mean()) if n_on else np.nan
        ach = achievable(dict(coh, group=g, lo=(coh["lo"] if idx is None else coh["lo"][idx]), hi=(coh["hi"] if idx is None else coh["hi"][idx]), onset=on), l)[onm]
        out[f"n_achievable_lead{l}"] = int(ach.sum())
        out[f"det_lead{l}_achievable"] = float((fl[onm][ach] >= l).mean()) if ach.sum() else np.nan
    det = fl[onm] >= 1
    out["median_earliest_lead_detected"] = float(np.median(fl[onm][det])) if det.any() else np.nan
    fp = int((nd[nvm] > 0).sum())
    out["fap"] = fp / n_nv if n_nv else np.nan
    out["episodes_per_100_participant_days"] = float(ne[nvm].sum()) / (n_nv * exposure) * 100 if n_nv else np.nan
    out["alarm_burden"] = float(nd[nvm].sum()) / (n_nv * exposure) if n_nv else np.nan
    tp = int(det.sum())
    out["precision_participant"] = tp / (tp + fp) if tp + fp else np.nan
    out["recall_participant"] = out.get("det_lead1", np.nan)
    return out


def day_level(S, tau, coh):
    Y = coh["Y"]
    ok = ~np.isnan(Y)
    ok[:, :T0] = False
    ok[:, coh["T1"] + 1:] = False
    a, y = S[ok] > tau, Y[ok] == 1
    tp = int((a & y).sum())
    return {"day_precision": tp / a.sum() if a.sum() else np.nan, "day_recall": tp / y.sum() if y.sum() else np.nan, "day_n_origins": int(ok.sum()), "day_n_pos": int(y.sum())}


def participant_auc(S, coh):
    M = window_max(S, coh)
    keep = coh["group"] != "excl"
    return float(roc_auc_score(coh["group"][keep] == "onset", M[keep]))


# ------------------------------------------------------------------ supplementary: exposure-matched blocks
def block_layout(coh):
    """Complete non-overlapping W-day blocks of the never-onset window T0..T1; leftover days at the END of the window (after the last complete block) are not used."""
    total = coh["T1"] - T0 + 1
    nb = total // coh["W"]
    return nb, total - nb * coh["W"]


def matched_exposure(S, tau, coh, st):
    """Supplementary. never-onset participants: share of complete W-day blocks containing >= 1 alarm (block false-alarm rate); onset participants with a FULL W-day pre-onset window
    (o-W >= 27 and o-1 <= T1): detection at lead >= l (all leads <= W achievable) and window-max AUC against never-onset BLOCK maxima (chance 0.5)."""
    nb, left = block_layout(coh)
    nv = coh["group"] == "never"
    W = coh["W"]
    B = S[nv][:, T0:T0 + nb * W].reshape(nv.sum(), nb, W).max(-1)                       # (n_never, nb) block maxima
    full = (coh["group"] == "onset") & (coh["onset"] - W >= T0) & (coh["onset"] - 1 <= coh["T1"])
    Mf = window_max(S, coh)[full]
    out = {"matched_n_blocks": int(B.size), "matched_leftover_days": left, "matched_block_fap": float((B > tau).mean()), "matched_n_full_window_onsets": int(full.sum())}
    for l in leads_for(coh):
        out[f"matched_det_lead{l}"] = float((st["first_lead"][full] >= l).mean()) if full.any() else np.nan
    if full.any():
        Bf = B.ravel()
        gt = (Mf[:, None] > Bf[None, :]).mean() + 0.5 * (Mf[:, None] == Bf[None, :]).mean()
        out["matched_auc_vs_blocks"] = float(gt)
    else:
        out["matched_auc_vs_blocks"] = np.nan
    return out, B, full


# ------------------------------------------------------------------ prediction sources (all existing, read-only)
SRC = {  # name: (dir, file pattern kind, seeds, family)
    "gbt": ("results/stage3", "std", [0], "stage7"), "gru": ("results/stage3", "std", range(5), "stage7"), "kan": ("results/stage3", "std", range(5), "stage7"),
    "kan_reg": ("results/stage3b_kan_reg/final", "std", range(5), "stage7"), "m1": ("results/stage4_m1", "std", range(5), "stage7"),
    "m1_static": ("results/stage4_ablation", "std", range(5), "stage7"), "kangru": ("results/ext_kangru", "std", range(5), "stage7"),
    "kangru_fixed": ("results/ext_kangru", "std", range(5), "stage7"),
    "transformer": ("results/p1_baselines/main", "std", range(5), "p1-1"), "cnnlstm": ("results/p1_baselines/main", "std", range(5), "p1-1"),
    "mlp_m1": ("results/p1_kan_mlp_control/m1_all", "std", range(5), "p1-3"), "mlp_b2": ("results/p1_kan_mlp_control/b2_main", "std", range(5), "p1-3"),
    "mlp_b2_reg": ("results/p1_kan_mlp_control/b2_main", "std", range(5), "p1-3"), "kan_nospline": ("results/p1_kan_mlp_control/b2_main", "std", range(5), "p1-3"),
    "st_task": ("results/p1_multitask_ablation/st_task_main", "head", range(5), "p1-2"), "st_task_matched": ("results/p1_multitask_ablation/st_task_matched_main", "head", range(5), "p1-2"),
    "family": ("results/p1_multitask_ablation/family_main", "family", range(5), "p1-2")}
STAGE7_MODELS = [m for m, v in SRC.items() if v[3] == "stage7"]


def head_name(k, kind, h):
    return f"onset_{kind}_{k}_h{h}"


@functools.lru_cache(maxsize=24)
def _load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["pred"], z["pid"], list(z["heads"])


def load_scores(model, seed, split, k, kind, h, pid):
    d, how, _, _ = SRC[model]
    head = head_name(k, kind, h)
    fn = {"std": f"{model}_seed{seed}_{split}.npz", "head": f"{head}_seed{seed}_{split}.npz", "family": f"family_onset_sust_seed{seed}_{split}.npz"}[how]
    pred, zpid, heads = _load_npz(os.path.join(ROOT, d, "predictions", fn))
    assert np.array_equal(zpid, pid), "participant order mismatch"
    return pred[:, :, heads.index(head)].astype(float)


def has_head(model, kind):
    return not (SRC[model][1] in ("head", "family") and kind == "cross")


def null_scores_iid(draw, shape=(200, N_DAYS)):
    return lt.null_scores(draw, shape)


def shuffle_perms(n_shuffles=200, n=200, seed=15015):
    """Complete-trajectory participant permutations, identical for every model/seed/outcome/horizon (fixed seed)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(n_shuffles)])


# ------------------------------------------------------------------ the ONE evaluation path (calibration -> tau -> test)
def evaluate_one(Scal, Stest, cal, test, f):
    """tau is a function of CALIBRATION scores and cohort only; test scores are used only after tau is fixed."""
    tau = tau_from_calibration(Scal, cal, f)
    st = alarm_stats(Stest, tau, test)
    res = metrics_from(st, test)
    res.update(day_level(Stest, tau, test))
    res["tau"], res["cal_fap_realised"] = tau, metrics_from(alarm_stats(Scal, tau, cal), cal)["fap"]
    m, _, _ = matched_exposure(Stest, tau, test, st)
    res.update(m)
    return res, st


# ------------------------------------------------------------------ participant-trajectory shuffle null (whole score trajectories are permuted; labels, windows and tau stay fixed)
def shuffle_null(Stest, tau, test, perms):
    """Returns dict metric -> (n_shuffles,) for the same metrics as the primary analysis, with participant i assigned the COMPLETE score trajectory of participant perms[b, i]
    (calendar alignment and temporal ordering preserved). Used as a null comparison, not as a confidence interval."""
    S = Stest[perms]                                                                      # (B, N, T)
    st = alarm_stats(S, tau, test)
    on, nv = test["group"] == "onset", test["group"] == "never"
    fl, nd, ne = st["first_lead"], st["n_days"], st["n_ep"]
    out = {}
    for l in leads_for(test):
        out[f"det_lead{l}"] = (fl[:, on] >= l).mean(1)
    fp = (nd[:, nv] > 0).sum(1)
    tp = (fl[:, on] >= 1).sum(1)
    exposure = test["T1"] - T0 + 1
    out["fap"] = fp / nv.sum()
    out["alarm_burden"] = nd[:, nv].sum(1) / (nv.sum() * exposure)
    out["episodes_per_100_participant_days"] = ne[:, nv].sum(1) / (nv.sum() * exposure) * 100
    out["precision_participant"] = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    nb, _ = block_layout(test)
    Bm = S[:, nv][:, :, T0:T0 + nb * test["W"]].reshape(len(perms), nv.sum(), nb, test["W"]).max(-1)
    out["matched_block_fap"] = (Bm > tau).mean((1, 2))
    M = window_max(S, test)
    keep = test["group"] != "excl"
    out["auc_participant"] = np.array([roc_auc_score(test["group"][keep] == "onset", m[keep]) for m in M])
    return out


# ------------------------------------------------------------------ bootstrap helpers (participant-level, stratified; paired across models via shared indices)
def stratified_indices(test, B, seed):
    """Resample onset participants and never-onset participants separately (positions within the evaluable cohort). Same indices for every model -> paired contrasts."""
    rng = np.random.default_rng(seed)
    on, nv = np.where(test["group"] == "onset")[0], np.where(test["group"] == "never")[0]
    return on, nv, rng.integers(0, len(on), (B, len(on))), rng.integers(0, len(nv), (B, len(nv)))


def participant_arrays(st, test):
    on, nv = np.where(test["group"] == "onset")[0], np.where(test["group"] == "never")[0]
    d = {"fl": st["first_lead"][on], "alarm_any": st["n_days"][nv] > 0, "n_days": st["n_days"][nv], "n_ep": st["n_ep"][nv]}
    d["ach"] = {l: achievable(test, l)[on] for l in leads_for(test)}
    return d


def boot_metrics(arrs, bon, bnv, test):
    """arrs: list (over seeds) of participant_arrays; bon (B, n_on), bnv (B, n_nv) position indices. Metric per resample is computed for each seed and then averaged over seeds
    (so the interval reflects participant sampling, conditional on the trained seeds). Returns dict metric -> (B,)."""
    exposure = test["T1"] - T0 + 1
    out = {}
    per = {}
    for a in arrs:
        r = {}
        for l in leads_for(test):
            ind = a["fl"] >= l
            r[f"det_lead{l}"] = ind[bon].mean(1)
            ach = a["ach"][l]
            num, den = (ind & ach)[bon].sum(1), ach[bon].sum(1)
            r[f"det_lead{l}_achievable"] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        fp = a["alarm_any"][bnv].sum(1)
        tp = (a["fl"] >= 1)[bon].sum(1)
        r["fap"] = fp / bnv.shape[1]
        r["alarm_burden"] = a["n_days"][bnv].sum(1) / (bnv.shape[1] * exposure)
        r["episodes_per_100_participant_days"] = a["n_ep"][bnv].sum(1) / (bnv.shape[1] * exposure) * 100
        r["precision_participant"] = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
        for kk, v in r.items():
            per.setdefault(kk, []).append(v)
    for kk, v in per.items():
        out[kk] = np.nanmean(np.stack(v), 0)
    return out


def threshold_bootstrap(Scal, Stest, cal, test, f, B, seed):
    """Uncertainty from the calibration threshold: resample the calibration never-onset participants (with replacement), re-derive tau by the Stage 7 rule, re-evaluate the SAME test scores."""
    rng = np.random.default_rng(seed)
    m = window_max(Scal, cal)[cal["group"] == "never"]
    rows = []
    for _ in range(B):
        tau = lt.tau_from_calibration(m[rng.integers(0, len(m), len(m))], f)
        r = metrics_from(alarm_stats(Stest, tau, test), test)
        rows.append(dict(tau=tau, det_lead1=r["det_lead1"], det_lead7=r.get("det_lead7", np.nan), fap=r["fap"], precision_participant=r["precision_participant"]))
    return rows
