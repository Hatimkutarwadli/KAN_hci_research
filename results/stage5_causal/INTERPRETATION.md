# Stage 5 — Causal analysis and planted-effect recovery (SYNTHETIC data)

**What this is.** The simulator contains effects we planted ourselves, so for once the "true answer" is known. This stage asks: *if an analyst saw only the wearable data and the measured severity scores, could they recover those planted effects, and how does the hidden stress confounder get in the way?* Everything here is an **observational association estimated from synthetic data**. It does not show that any signal causes depression or anxiety in people; it shows how well a standard method recovers effects that we know exist in a simulated world.

## Three kinds of numbers (do not mix them up)
1. **Planted effect (ground truth, known from the simulator code):** e.g. sleep variability raises the depression state 3 days later by 0.108 severity points per unit per day (average slope of the planted tanh effect).
2. **Observational estimate (what an analyst could compute):** from observed wearable features, the noisy `stress_proxy` and the measured severity. Variants **V0** (naive, one feature at a time), **V1** (adjusted for the other features and the previous severity, person fixed effects), **V2** (V1 plus the noisy stress proxy).
3. **Evaluation-only variants (use ground truth, flagged `uses_ground_truth=True`):** **E1** (V2 but with the true stress), **E2/E3** (oracle: true features and true latent state, without / with true stress), **E4** (true features, noisy measured outcome), **E5** (noisy features, true outcome). They exist only to explain *why* V2 differs from the truth.

**Method.** Increment model with person fixed effects: change in severity on day τ regressed on severity at τ−1 and each feature at lags 1–5 (lag L = the feature on day τ−L). Participant-clustered standard errors (participants, not days, are the independent units); a participant bootstrap (300 resamples) agrees with them (interval width ratio 0.98). 59,387 complete-case person-days from 975 participants; every variant uses the same rows. Ground truth is never a predictor in V0–V2.

## Headline recovery table (outcome change per day, severity points per unit of the feature; 95% CI)
| relationship (planted lag) | planted effect | V1 observed, no stress adj. | **V2 observed + stress proxy** | V2 largest-|coef| lag / other lags significant | oracle (E3) | oracle largest-|coef| lag |
|---|---|---|---|---|---|---|
| sleep_variability -> dep (lag 3) | +0.108 | +0.060 [+0.036, +0.084] (56%) | **+0.050 [+0.026, +0.073] (46%)** | 5 / 2 | +0.105 [+0.089, +0.121] (98%) | 3 |
| steps -> dep (lag 2) | -0.077 | -0.058 [-0.081, -0.036] (76%) | **-0.055 [-0.077, -0.033] (71%)** | 5 / 4 | -0.062 [-0.076, -0.049] (81%) | 2 |
| steps -> anx (lag 2) | -0.046 | -0.021 [-0.043, +0.001] (47%) | **-0.018 [-0.040, +0.004] (38%)** | 5 / 4 | -0.055 [-0.069, -0.041] (120%) | 2 |
| hrv -> anx (lag 1) | -0.082 | -0.089 [-0.105, -0.073] (109%) | **-0.084 [-0.100, -0.067] (102%)** | 1 / 3 | -0.085 [-0.094, -0.076] (103%) | 1 |

Entries: coefficient at the planted lag [95% participant-clustered CI] (percent of the planted effect). Units: severity points per unit of the feature, per day.

Reading: **V2** is what an analyst could do. **Oracle** shows the estimator itself works: with true variables it recovers all four planted effects, peaks at the planted lag, and is near zero at the other lags. Most of the gap between V2 and the truth is therefore measurement noise, not a flaw in the estimator.

## Relationship by relationship
**1. Sleep variability -> depression (planted: lag 3, positive).**
- Naive (one feature at a time, within person): positive and significant at every lag 1-5 and rising with lag (0.57 -> 0.86). It shows an association but cannot tell the lag, and it contains stress confounding and the persistence of both series.
- V2: lag-3 coefficient +0.050 [0.026, 0.073], 46% of the planted +0.108 (CI excludes it); lag profile 0.002, 0.023, 0.050, 0.037, 0.073, so the largest value is at lag 5. **Direction recovered, lag not, magnitude about half.**
- Oracle: +0.105 [0.089, 0.121] (98%), peak exactly at lag 3, other lags about 0.
- Why: noisy features (E5) keep the lag-3 peak but halve it (0.052); a noisy outcome used as the previous-severity control (E4) creates a spurious lag-5 coefficient (0.064). The true stress changes lag 3 little (E1 0.050; every proxy reliability gives 0.048-0.060).

**2. Steps -> depression (planted: lag 2, negative).**
- V2: lag-2 coefficient -0.055 [-0.077, -0.033], 71% of the planted -0.077 (CI covers it); profile -0.050, -0.055, -0.031, -0.029, -0.058, so every lag is negative and lag 5 is as large as lag 2. **Direction and rough size recovered, lag not unique.**
- Oracle: -0.062 [-0.076, -0.049] (81%), peak at lag 2, others about 0. Noisy features alone (E5) give -0.040 at the right lag.

**3. Steps -> anxiety (planted: lag 2, negative, weaker).**
- V2: lag-2 coefficient -0.018 [-0.040, +0.004], 38% of the planted -0.046 and **not significant**; lags 1 and 5 are larger (-0.051, -0.053). **Not reliably recovered from observable data.** Oracle: -0.055 (120%), peak at lag 2.

**4. HRV -> anxiety (planted: lag 1, negative).**
- V2: lag-1 coefficient -0.084 [-0.100, -0.067], 102% of the planted -0.082, peak at lag 1. **Best-recovered relationship**: correct direction, lag and size. Oracle -0.085 (103%). It is the effect closest in time to the outcome change (little room for persistence to leak across lags) and the largest per unit; V2 still shows a weaker spurious coefficient at lag 5 (-0.054).

**5. Non-causal variables (planted effect 0).**
- Resting heart rate (driven by stress): naive within-person association with depression +0.45 -> +0.58 per unit over lags 1-5 (anxiety +0.37 -> +0.47), large and highly significant. V1: sum of lag coefficients +0.098 (depression) / +0.108 (anxiety) with 3 significant lags; V2: +0.033 / +0.039 (one marginal lag, t about 2.0-2.4), about two thirds removed; with the TRUE stress (E1): +0.004 / +0.004, nothing significant (|t| <= 1.3).
- Calories (0.7 x steps): naive -0.34 -> -0.40 (depression), -0.23 -> -0.28 (anxiety). Adjusted coefficients are small (<= 0.022) but one lag stays significant (t about 2.1-2.8) in V1, V2 and even with true stress (E1); with true features (E2/E3) none (|t| < 1.6). Calories is a second noisy measure of steps, so when steps is mismeasured it borrows part of steps' true effect: a "negative control" that is contaminated by measurement error.
- Sleep variability -> anxiety (no planted effect): max |t| 4.8 (V1) -> 3.1 (V2) -> 1.1 (true stress): confounding by stress, removed once stress is known. HRV -> depression: tiny (max |coef| 0.026, |t| <= 2.9), consistent with the indirect path through the planted coupling.

## Hidden stress and the noisy proxy
- Stress is hidden and moves many features: planted loadings +0.5 sleep variability, +0.6 resting HR, -0.2 steps, -0.2 HRV, -0.14 calories (via steps), plus a direct +0.03 per day on both states (0.09 severity points per stress unit at lag 1). Observed within-person correlations of each feature with the TRUE stress are 0.48, 0.54, -0.24, -0.22, -0.14 (same signs and ordering). Stress therefore pushes a feature and the outcome together, which inflates unadjusted estimates.
- `stress_proxy` is the true stress plus noise: correlation 0.79 with the true stress (reliability R2 = 0.63), so adjustment is only partial.
- Dose-response (evaluation-only: proxies built from the true stress plus noise; depression outcome; resting-HR sum of lag coefficients, truth 0): proxy reliability 0.10 -> 0.082; 0.30 -> 0.062; 0.63 (about our proxy) -> 0.033; 0.87 -> 0.013; 1.00 -> 0.004; no adjustment 0.098. Residual confounding is roughly (1 - reliability) x the unadjusted bias (fig3).
- Stress adjustment does not fix the attenuation of sleep variability and steps (0.048-0.060 and -0.053 to -0.058 at every reliability): that comes from measurement noise, not from stress.

## Reverse paths and depression–anxiety coupling
- **Reverse paths (state -> feature).** Planted: depression -> steps (-0.15 per latent unit, same day = -0.05 feature units per severity point) and anxiety -> HRV (-0.033). With the feature's own previous value controlled, the planted same-day effect appears as a negative lag-0 and a positive lag-1 coefficient (benchmark -0.050 / +0.037 for steps, -0.033 / +0.025 for HRV); the oracle gives -0.032 / +0.038 and -0.025 / +0.022. Observed: lag 0 = -0.017 (t = -9.1) and -0.018 (t = -9.9): right sign and clearly detected, but only 34% and 54% of the planted lag-0 value, and the positive lag-1 signature is lost (-0.001, -0.004).
- **Direction is not identified by timing alone:** same-day associations also appear where no reverse path exists (calories -0.012 with t = -6.2 via steps; sleep variability +0.009, t = 5.6, and resting HR +0.006, t = 3.4 via stress). Steps and HRV have both a forward and a reverse path, so the observed steps-depression association mixes the two directions.
- **Depression-anxiety coupling** (planted 0.03 per severity point, both directions). anx -> dep: naive 0.082, adjusted (features + proxy) 0.056 [0.048, 0.065], oracle without stress 0.048, oracle with true stress 0.041 [0.033, 0.050]; dep -> anx: 0.073, 0.052, 0.044, 0.038 [0.030, 0.046]. The coupling is detected (positive, significant) but overstated by 27-173%, and every CI excludes the planted 0.03 even for the oracle, because the simulator also gives the two conditions shared drift episodes and stress, hidden common causes that no analysis variable captures. The two directions give near-identical estimates, so the direction of the depression-anxiety link is not identifiable.

## What can and cannot be claimed
- **Can claim (synthetic setting):** all planted forward effects have the correct sign in the adjusted observational estimates, HRV → anxiety is recovered accurately at its planted lag, and adjustment for even a noisy stress proxy removes about two thirds of the spurious resting-heart-rate association (99% with a perfect proxy).
- **Cannot claim:** that timing or adjustment proves causation; that the planted lag can be read off the lag profile (it cannot for sleep variability and steps); that magnitudes are unbiased (they are attenuated to roughly 38–72% except HRV); or that real-world effects of these signals exist. The synthetic result shows only that a standard method partly recovers effects we planted.
- **Residual confounding and limitations:** the stress proxy has reliability 0.63 so confounding remains; shared drift episodes inflate the depression–anxiety coupling and are invisible to every analysis variable; missing wearable days are informative by design (complete-case rows may be selected); the estimator is linear while planted effects are saturating (tanh); only five of the eight wearable channels were analysed (the three omitted channels are non-causal distractors); lag-wise CIs are not corrected for multiple comparisons; one simulated dataset.

Files: `effect_lag_table.csv` (all variants, features, lags, CIs), `recovery_summary.csv`, `noncausal_summary.csv`, `naive_lag_association.csv`, `reverse_paths.csv`, `coupling.csv`, `proxy_sensitivity.csv`, `stress_confounding.csv`, `bootstrap_ci_V2.csv`, `analysis_config.json`, figures `fig1`–`fig5`. Code: `src/causal_analysis.py`; tests: `tests/test_causal.py`.
