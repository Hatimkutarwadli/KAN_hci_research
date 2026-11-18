"""Synthetic longitudinal wearable-data simulator.  ALL OUTPUT IS SYNTHETIC.

Each participant is followed for n_days daily observations. A hidden latent state
z(t) = (z_dep, z_anx) evolves by a mean-reverting stochastic process with planted
lagged (nonlinear) causal effects from wearable features, a hidden stress
confounder, occasional drift episodes and individual differences.

Outputs are split so that ground truth never mixes with model inputs:
  participants.csv   static inputs (pid, sex, age_group)
  observations.csv   model inputs (noisy wearables with NaN gaps) + analysis-only stress_proxy
  targets.csv        severity, binary status, onset labels (NaN where undefined/censored)
  truth.csv          latent z, true (noise-free) features, true stress  [NEVER a model input]
  truth_participants.csv  person-level truth (baselines, sensitivities, onset days)
  splits.csv         participant-level train/cal/test assignment
  meta.json          config, thresholds, column lists

Usage:  python src/simulate.py --n 200 --out data/pilot
"""
import argparse
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

FEATURES = ["steps", "calories", "sleep_duration", "sleep_efficiency",
            "sleep_variability", "resting_hr", "hrv", "circadian_regularity"]
F = {n: i for i, n in enumerate(FEATURES)}
# (mean, sd, lo, hi) used to map standardized latent features to natural units
UNITS = {"steps": (7000, 2500, 0, 40000), "calories": (2200, 300, 1200, 5000),
         "sleep_duration": (7.0, 0.8, 2, 12), "sleep_efficiency": (0.88, 0.04, 0.5, 1.0),
         "sleep_variability": (40, 15, 2, 200), "resting_hr": (65, 6, 40, 110),
         "hrv": (45, 12, 5, 150), "circadian_regularity": (0.7, 0.1, 0.0, 1.0)}
RHO = np.array([0.75, 0.70, 0.60, 0.50, 0.85, 0.80, 0.75, 0.70])  # AR(1) persistence
MODEL_INPUT_COLUMNS = FEATURES + ["missing_day", "hrv_missing"]  # + static sex, age_group
ANALYSIS_ONLY_COLUMNS = ["stress_proxy"]


@dataclass
class Config:
    n_participants: int = 200
    n_days: int = 120
    burn_in: int = 60
    seed: int = 2026
    # latent dynamics
    a: float = 0.08            # mean-reversion rate
    z_noise: float = 0.20
    coupling: float = 0.03     # dep<->anx comorbidity (must be < a for stability)
    m_sd: float = 0.50         # between-person baseline sd
    base_sd: float = 0.60      # between-person sd of wearable-feature baselines (std units)
    female_shift: float = 0.15
    # drift episodes
    ep_prob: float = 0.006
    ep_len: tuple = (10, 25)
    ep_drift: tuple = (0.08, 0.16)
    ep_anx_share: float = 0.5
    # planted causal effects (per-day increment on z, applied to bounded g(u))
    eff_sv_dep: float = 0.06       # sleep_variability -> dep (+), lag lag_sv
    eff_steps_dep: float = -0.05   # steps -> dep (-), lag lag_steps
    eff_steps_anx: float = -0.03   # steps -> anx (-)
    eff_hrv_anx: float = -0.05     # hrv -> anx (-), lag lag_hrv
    eff_stress_direct: float = 0.03  # hidden confounder, direct path to z
    lag_sv: int = 3
    lag_steps: int = 2
    lag_hrv: int = 1
    sens_sd: float = 0.30          # log-sd of per-person effect sensitivity
    rev_steps: float = -0.15       # state -> feature (reverse path)
    rev_hrv: float = -0.10
    # outcomes
    sev_intercept: float = 8.0
    sev_scale: float = 3.0
    sev_noise: float = 1.0
    z_thr_dep: float = 2.45  # tuned on pilot to ~7.5% prevalence
    z_thr_anx: float = 2.05  # tuned on pilot to ~7.5% prevalence
    # missingness
    miss_mean: float = 0.12
    miss_persist: float = 0.60
    miss_informative: float = 0.50
    partial_hrv_miss: float = 0.04
    # sensors / subgroups
    sensor_sd: float = 0.35
    noisy_group_mult: float = 1.6   # applied to age_group 2 (small, noisier)
    stress_proxy_sd: float = 0.80
    horizons: tuple = (7, 14, 21)
    persistence_days: int = 3        # primary onset = first day of >=3 consecutive positive days
    p_female: float = 0.65
    age_probs: tuple = (0.45, 0.45, 0.10)
    split_fracs: tuple = (0.6, 0.2, 0.2)


def _draw(rng, N, T):
    return {
        "z_eps": rng.standard_normal((N, T, 2)), "s_eps": rng.standard_normal((N, T)),
        "ev_u": rng.random((N, T)), "f_eps": rng.standard_normal((N, T, 8)),
        "sensor": rng.standard_normal((N, T, 8)), "proxy": rng.standard_normal((N, T)),
        "miss_u": rng.random((N, T)), "hrv_miss_u": rng.random((N, T)),
        "ep_u": rng.random((N, T)), "ep_len_u": rng.random((N, T)),
        "ep_drift_u": rng.random((N, T)), "ep_coin": rng.random((N, T)),
        "sev_eps": rng.standard_normal((N, T, 2)),
    }


def simulate(cfg: Config, perturb_after=None, perturb_seed=999):
    """Return dict of DataFrames. If perturb_after=d, all random draws for days >= d
    are replaced by independent ones (used by the leakage test only)."""
    N, D, B = cfg.n_participants, cfg.n_days, cfg.burn_in
    T = B + D
    rs = np.random.default_rng(cfg.seed)          # static participant parameters
    rn = np.random.default_rng(cfg.seed + 1)      # daily noise
    nz = _draw(rn, N, T)
    if perturb_after is not None:
        alt = _draw(np.random.default_rng(perturb_seed), N, T)
        cut = B + perturb_after
        for k in nz:
            nz[k][:, cut:] = alt[k][:, cut:]

    # ---- static participant parameters ----
    female = (rs.random(N) < cfg.p_female).astype(int)
    age_group = rs.choice(3, size=N, p=cfg.age_probs)
    m = rs.normal(0, cfg.m_sd, (N, 2))
    m[:, 0] += cfg.female_shift * female
    base = rs.normal(0, cfg.base_sd, (N, 8))               # person feature baselines (std units)
    phase = rs.uniform(0, 2 * np.pi, N)
    sens = np.exp(rs.normal(0, cfg.sens_sd, (N, 3)))  # sv, steps, hrv sensitivities
    sens_st_anx = np.exp(rs.normal(0, cfg.sens_sd, N))
    p_miss = rs.beta(1.5, 1.5 * (1 - cfg.miss_mean) / cfg.miss_mean, N)
    sensor_mult = np.where(age_group == 2, cfg.noisy_group_mult, 1.0)

    z = np.zeros((N, T, 2))
    u = np.zeros((N, T, 8))
    s = np.zeros((N, T))
    ar = np.zeros((N, T, 8))
    miss = np.zeros((N, T), dtype=bool)
    ep_left = np.zeros(N, dtype=int)
    ep_d = np.zeros((N, 2))
    sd_ar = 0.8
    q_base = p_miss * (1 - cfg.miss_persist) / (1 - p_miss)
    z[:, 0] = m + 0.65 * nz["z_eps"][:, 0]

    def g_sv(x): return np.tanh(x)
    def g_st(x): return np.tanh(0.7 * x)
    def g_hr(x): return np.tanh(0.8 * x)

    for t in range(T):
        ev = (nz["ev_u"][:, t] < 0.01) * 1.5
        s[:, t] = (0.93 * s[:, t - 1] + 0.35 * nz["s_eps"][:, t] + ev) if t > 0 \
            else 0.95 * nz["s_eps"][:, t] + ev
        rho = RHO
        ar[:, t] = (rho * ar[:, t - 1] + np.sqrt(1 - rho ** 2) * sd_ar * nz["f_eps"][:, t]) \
            if t > 0 else sd_ar * nz["f_eps"][:, t]
        week = 0.4 * np.sin(2 * np.pi * t / 7 + phase)
        ut = base + ar[:, t]
        ut[:, F["steps"]] += -0.2 * s[:, t] + cfg.rev_steps * z[:, t, 0] + week
        ut[:, F["sleep_duration"]] += -0.2 * s[:, t]
        ut[:, F["sleep_efficiency"]] += -0.15 * s[:, t]
        ut[:, F["sleep_variability"]] += 0.5 * s[:, t]
        ut[:, F["resting_hr"]] += 0.6 * s[:, t]
        ut[:, F["hrv"]] += cfg.rev_hrv * z[:, t, 1] - 0.2 * s[:, t]
        ut[:, F["calories"]] += 0.7 * ut[:, F["steps"]]
        ut[:, F["circadian_regularity"]] += -0.3 * ut[:, F["sleep_variability"]]
        u[:, t] = ut

        # wear-gap process (Markov, informative: higher gap risk when dep state is high)
        q = np.clip(q_base * (1 + cfg.miss_informative * np.maximum(z[:, t, 0], 0)), 0, 0.95)
        prev = miss[:, t - 1] if t > 0 else np.zeros(N, dtype=bool)
        miss[:, t] = np.where(prev, nz["miss_u"][:, t] < cfg.miss_persist, nz["miss_u"][:, t] < q)

        if t == T - 1:
            break
        # drift episodes
        starting = (ep_left == 0) & (nz["ep_u"][:, t] < cfg.ep_prob)
        ep_left = np.where(starting, cfg.ep_len[0] + (nz["ep_len_u"][:, t] * (cfg.ep_len[1] - cfg.ep_len[0] + 1)).astype(int), ep_left)
        dr = cfg.ep_drift[0] + nz["ep_drift_u"][:, t] * (cfg.ep_drift[1] - cfg.ep_drift[0])
        ep_d[starting, 0] = dr[starting]
        ep_d[starting, 1] = (dr * (nz["ep_coin"][:, t] < cfg.ep_anx_share))[starting]
        drift = np.where((ep_left > 0)[:, None], ep_d, 0.0)
        ep_left = np.maximum(ep_left - 1, 0)

        # z(t+1): lagged effects use features at day t+1-lag (lag>=1 -> already generated)
        i_sv, i_st, i_hr = (max(t + 1 - L, 0) for L in (cfg.lag_sv, cfg.lag_steps, cfg.lag_hrv))
        sv = g_sv(u[:, i_sv, F["sleep_variability"]])
        st = g_st(u[:, i_st, F["steps"]])
        hr = g_hr(u[:, i_hr, F["hrv"]])
        dz = np.zeros((N, 2))
        dz[:, 0] = (cfg.a * (m[:, 0] - z[:, t, 0]) + cfg.eff_sv_dep * sens[:, 0] * sv
                    + cfg.eff_steps_dep * sens[:, 1] * st
                    + cfg.coupling * (z[:, t, 1] - m[:, 1]) + cfg.eff_stress_direct * s[:, t])
        dz[:, 1] = (cfg.a * (m[:, 1] - z[:, t, 1]) + cfg.eff_steps_anx * sens_st_anx * st
                    + cfg.eff_hrv_anx * sens[:, 2] * hr
                    + cfg.coupling * (z[:, t, 0] - m[:, 0]) + cfg.eff_stress_direct * s[:, t])
        z[:, t + 1] = z[:, t] + dz + drift + cfg.z_noise * nz["z_eps"][:, t + 1]

    # ---- crop burn-in ----
    sl = slice(B, T)
    z, u, s, miss = z[:, sl], u[:, sl], s[:, sl], miss[:, sl]
    sensor = nz["sensor"][:, sl]
    proxy_n = nz["proxy"][:, sl]
    hrv_miss = nz["hrv_miss_u"][:, sl] < cfg.partial_hrv_miss
    sev_eps = nz["sev_eps"][:, sl]

    pid = np.repeat(np.arange(N), D)
    day = np.tile(np.arange(D), N)

    # ---- observed wearables (noisy, natural units, NaN gaps) ----
    obs = {"pid": pid, "day": day}
    for f, i in F.items():
        mu, sd, lo, hi = UNITS[f]
        x = mu + sd * (u[:, :, i] + cfg.sensor_sd * sensor_mult[:, None] * sensor[:, :, i])
        x = np.clip(x, lo, hi)
        x[miss] = np.nan
        if f == "hrv":
            x[hrv_miss] = np.nan
        obs[f] = x.reshape(-1)
    obs["missing_day"] = miss.reshape(-1).astype(int)
    # 1 whenever HRV is NaN (whole-day gap OR HRV-only sensor gap); partial = hrv_missing & ~missing_day
    obs["hrv_missing"] = np.isnan(obs["hrv"]).astype(int)
    proxy = s + cfg.stress_proxy_sd * proxy_n
    proxy[miss] = np.nan
    obs["stress_proxy"] = proxy.reshape(-1)
    observations = pd.DataFrame(obs)

    # ---- targets ----
    sev = np.clip(cfg.sev_intercept + cfg.sev_scale * z + cfg.sev_noise * sev_eps, 0, 27)
    thr = np.array([cfg.z_thr_dep, cfg.z_thr_anx])
    status = (z >= thr).astype(int)                       # latent-state threshold
    prevalent = status[:, 0, :] == 1                      # positive on first observed day
    ever = np.maximum.accumulate(status, axis=1) > 0      # positive on any day <= t
    onset_day = np.full((N, 2), np.nan)
    for k in range(2):
        has = status[:, :, k].any(axis=1) & ~prevalent[:, k]
        first = status[:, :, k].argmax(axis=1)
        onset_day[has, k] = first[has]
    # primary (sustained) onset: first day d with status positive on d..d+P-1; prevalent participants excluded
    P = cfg.persistence_days
    run = np.ones((N, D - P + 1, 2), dtype=bool)
    for j in range(P):
        run &= status[:, j:D - P + 1 + j, :] == 1
    first_run = run.argmax(axis=1)
    onset_sust = np.full((N, 2), np.nan)
    for k in range(2):
        okp = run[:, :, k].any(axis=1) & ~prevalent[:, k]
        onset_sust[okp, k] = first_run[okp, k]
    tg = {"pid": pid, "day": day}
    for k, name in enumerate(["dep", "anx"]):
        tg[f"severity_{name}"] = sev[:, :, k].reshape(-1)
        tg[f"status_{name}"] = status[:, :, k].reshape(-1)
        at_risk = ~ever[:, :, k] & ~prevalent[:, k][:, None]
        tg[f"at_risk_{name}"] = at_risk.reshape(-1).astype(int)
        t_idx = np.arange(D)[None, :]
        for h in cfg.horizons:
            lab = ((onset_day[:, k][:, None] > t_idx) &
                   (onset_day[:, k][:, None] <= t_idx + h)).astype(float)
            lab[~at_risk] = np.nan                         # already positive -> not at risk
            lab[:, t_idx[0] + h > D - 1] = np.nan          # right-censored: horizon not fully observed
            tg[f"onset_{name}_h{h}"] = lab.reshape(-1)       # SECONDARY: first-crossing onset
        # PRIMARY: sustained onset. at risk until the sustained-onset day; confirmation needs P-1 extra
        # observed days, so a label also requires t+h+P-1 <= last day (right-censoring).
        os_ = onset_sust[:, k][:, None]
        at_risk_s = ~prevalent[:, k][:, None] & ~(os_ <= t_idx)
        tg[f"at_risk_sust_{name}"] = at_risk_s.reshape(-1).astype(int)
        for h in cfg.horizons:
            lab = ((os_ > t_idx) & (os_ <= t_idx + h)).astype(float)
            lab[~at_risk_s] = np.nan
            lab[:, t_idx[0] + h + P - 1 > D - 1] = np.nan
            tg[f"onset_sust_{name}_h{h}"] = lab.reshape(-1)
    targets = pd.DataFrame(tg)

    # ---- truth (never a model input) ----
    tr = {"pid": pid, "day": day, "z_dep": z[:, :, 0].reshape(-1), "z_anx": z[:, :, 1].reshape(-1),
          "stress": s.reshape(-1)}
    for f, i in F.items():
        tr[f"u_{f}"] = u[:, :, i].reshape(-1)
    truth = pd.DataFrame(tr)
    truth_p = pd.DataFrame({
        "pid": np.arange(N), "m_dep": m[:, 0], "m_anx": m[:, 1], "sens_sv": sens[:, 0],
        "sens_steps": sens[:, 1], "sens_hrv": sens[:, 2], "sens_steps_anx": sens_st_anx,
        "p_miss": p_miss, "sensor_mult": sensor_mult,
        "prevalent_dep": prevalent[:, 0].astype(int), "prevalent_anx": prevalent[:, 1].astype(int),
        "onset_day_dep": onset_day[:, 0], "onset_day_anx": onset_day[:, 1],
        "onset_day_sust_dep": onset_sust[:, 0], "onset_day_sust_anx": onset_sust[:, 1]})
    participants = pd.DataFrame({"pid": np.arange(N), "sex": np.where(female == 1, "F", "M"),
                                 "age_group": age_group})

    # ---- participant-level split ----
    perm = np.random.default_rng(cfg.seed + 2).permutation(N)
    n_tr, n_ca = int(cfg.split_fracs[0] * N), int(cfg.split_fracs[1] * N)
    split = np.empty(N, dtype=object)
    split[perm[:n_tr]] = "train"
    split[perm[n_tr:n_tr + n_ca]] = "cal"
    split[perm[n_tr + n_ca:]] = "test"
    splits = pd.DataFrame({"pid": np.arange(N), "split": split})
    return dict(participants=participants, observations=observations, targets=targets,
                truth=truth, truth_participants=truth_p, splits=splits)


def save(out, dirpath, cfg, final=False):
    if os.path.isdir(dirpath) and os.listdir(dirpath):
        raise SystemExit(f"{dirpath} exists and is not empty; refusing to overwrite (CLAUDE.md Rule 4).")
    os.makedirs(dirpath, exist_ok=True)
    for name, df in out.items():
        df.to_csv(os.path.join(dirpath, f"{name}.csv"), index=False)
    meta = {"SYNTHETIC": True, "schema_version": 2, "final_frozen_dataset": final,
            "config": asdict(cfg), "features": FEATURES,
            "model_input_columns": MODEL_INPUT_COLUMNS + ["sex (static)", "age_group (static)"],
            "analysis_only_columns": ANALYSIS_ONLY_COLUMNS,
            "target_columns": [c for c in out["targets"].columns if c not in ("pid", "day")],
            "primary_onset": "onset_sust_{dep|anx}_h{7|14|21}: first day of >=persistence_days consecutive positive days; "
                             "at_risk_sust_*; label NaN if t+h+persistence_days-1 > last day",
            "secondary_onset": "onset_{dep|anx}_h{7|14|21}: first latent threshold crossing; at_risk_*; label NaN if t+h > last day",
            "truth_files": ["truth.csv", "truth_participants.csv"],
            "notes": ["status and both onset definitions use the latent z threshold; severity = noisy score",
                      "onset NaN where already positive/at_risk=0, prevalent at day 0, or horizon (+confirmation) not fully observed",
                      "stress_proxy is analysis-only; never a model input",
                      "parquet not used: pyarrow not installed"]}
    with open(os.path.join(dirpath, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if final:
        import hashlib
        lines = []
        for fn in sorted(os.listdir(dirpath)):
            with open(os.path.join(dirpath, fn), "rb") as fh:
                lines.append(f"{hashlib.sha256(fh.read()).hexdigest()}  {fn}")
        with open(os.path.join(dirpath, "checksums.sha256"), "w") as f:
            f.write("\n".join(lines) + "\n")
        for fn in os.listdir(dirpath):   # read-only to prevent accidental overwrite
            os.chmod(os.path.join(dirpath, fn), 0o444)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", default=None, help="JSON file overriding Config fields")
    ap.add_argument("--final", action="store_true", help="mark as final frozen dataset: checksums + read-only")
    a = ap.parse_args()
    kw = {"n_participants": a.n, "seed": a.seed}
    if a.config:
        kw.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(a.config)).items()})
    cfg = Config(**kw)
    save(simulate(cfg), a.out, cfg, final=a.final)
    print("saved", a.out)
