# P3 Phase 0: feasibility report (SYNTHETIC data; no training; test splits not read)

## What was run

- **Script:** `extension_p3/phase0_feasibility.py`. Runtime about 3 minutes.
- **Data used:** existing raw and processed data for v1, R1 and S-EFF, train and calibration splits only.
- **Counterfactual simulations:** run in memory only, with nothing saved. Before use, each dataset's in-memory factual simulation was checked against its saved files (max difference < 1e-9, which is CSV round-off).
- **Not done:**
  - no model was trained;
  - no test split was read;
  - no existing file was modified.

**Outputs (this folder):**

| File | Content |
|---|---|
| `A_truth_curves_and_mapping.csv` | Scale mapping and truth-curve curvature |
| `B_detectability.csv` | Held-out R² of spline vs linear distributed-lag models |
| `C_estimand_check.csv` | Associational curves, planted share and contamination |
| `D_severity_clipping.csv` | Fraction of severity values at the clip bounds |
| `E_scaler_exactness.csv`, `E_subsets_and_scaler.csv` | Subset-scaler feasibility and subset composition |
| `curves.npz` | All curves |

## Definitions

- **x** is the observed, scaled feature: x = (raw − scaler mean) / scaler SD, on observed days only.
- **u** is the true standardised feature from `truth.csv`.
- **Planted effect:** dz_k += eff · sens · g(u at lag L), with g = tanh(c·u) where c = 1 (sleep variability), 0.7 (steps) or 0.8 (HRV).
- **The two possible window interventions for a response curve:**
  - **"Set":** every observed day in the window is set to v. The interventional truth is **T_set(v) = g(u(v))**, where u(v) = (v − α)/β is the noise-free mapping fitted on train days.
  - **"Shift":** δ is added to every observed day. This is the intervention P1-3 used. The truth is **T_shift(δ) = E_u[g(u + δ/β)] − E[g(u)]**, which is g smoothed by the distribution of u.
- **Curvature share** = 1 − R² of the best straight line through a curve on its grid. For "set" curves it is weighted by the observed density of x.
- **Distributed-lag regression** (the "oracle-structured" detectability test):
  - Target: severity_t, or latent z_t (noise-free).
  - Planted feature: the planted channel at lag L, as an exponentially weighted (1 − a)^k mean of either the value (linear) or a 6-knot cubic B-spline basis applied per day (spline). Days t−L−k inside the 28-day window.
  - Covariates: the 7 other channels (linear, lag 1) and static variables.
  - Fitted on train origins; scored on calibration origins.
  - Two input versions: the true feature u ("oracle") and the observed x.
- **Planted share** = variance of [associational curve(factual) − associational curve(planted effect switched off)] ÷ variance of the factual curve. The two simulations share common random numbers (CRN), so they differ only in the planted effect.
- **Contamination share:** the same, with reverse paths and the stress → z path switched off instead.

## Results for the four planted pairs

### A. Scale mapping and truth curvature

These values are almost identical in v1, R1 and S-EFF.

| Pair | Lag | x-per-u slope | Day reliability R² | Observed beyond ±2 SD | Curvature T_set | Curvature T_shift |
|---|---|---|---|---|---|---|
| sleep variability → dep | 3 | 0.84–0.85 | 0.89–0.90 | 4.8% | 0.081 | 0.013 |
| steps → dep | 2 | 0.82–0.85 | 0.89–0.90 | 4.7% | 0.040–0.044 | 0.008 |
| steps → anx | 2 | same | same | same | 0.040–0.044 | 0.008 |
| HRV → anx | 1 | 0.87–0.89 | 0.89 | 4.5% | 0.050–0.053 | 0.009–0.010 |

**Over the region where the data lie, the planted tanh curves are 92–96% linear for "set" and about 99% linear for "shift".**

### B. Detectability of the curvature

Held-out (calibration) R² gain of spline over linear:

| Pair | Planted linear contribution (ΔR²) | Spline gain: observed x → severity | Spline gain: **oracle** true u → latent z |
|---|---|---|---|
| sleep variability → dep | 0.086–0.135 | −0.0006 / +0.0009 / +0.0045 (v1 / R1 / S-EFF) | +0.0007 / +0.0011 / +0.0040 |
| steps → dep | 0.086–0.131 | −0.0002 / −0.0048 / −0.0016 | −0.0028 / −0.0069 / −0.0033 |
| steps → anx | 0.009–0.094 | −0.0044 / −0.0035 / −0.0009 | −0.0068 / −0.0060 / −0.0030 |
| HRV → anx | 0.132–0.257 | −0.0016 / −0.0018 / +0.0004 | +0.0018 / +0.0002 / +0.0023 |

- **The planted effects themselves are clearly learnable:** the linear term adds 0.01–0.26 R².
- **Their curvature is not detectable.** This holds even for a correctly structured oracle model with access to the true feature and a noise-free target. The gain is between −0.007 and +0.006 in every case, and ≤ +0.002 in v1 and R1.
- The only consistently positive case is sleep variability → depression in S-EFF (+0.003 to +0.006), and S-EFF is the gate-failing dataset.

### C. Estimand check: what a predictive model's response curve can show

| Pair | Planted share of the associational "set" curve | Contamination share (reverse paths + stress → z) | Curvature of the fitted associational curve: set / shift |
|---|---|---|---|
| sleep variability → dep | 0.55–0.83 | 0.003–0.022 | 0.035–0.041 / 0.025–0.060 |
| steps → dep | **0.23–0.42** | 0.07–0.14 | 0.013–0.063 / 0.04–0.18 |
| steps → anx | **0.21–0.28** | 0.05–0.11 | 0.048–0.087 / 0.06–0.29 |
| HRV → anx | 0.52–0.70 | 0.03–0.08 | 0.033–0.051 / 0.003–0.18 |

1. **Most of the learnable steps association is not the planted effect.** Contamination from reverse paths and stress is modest (≤ 14%), so most of the non-planted part comes from **collinearity**: calories = 0.7 · steps + noise, so the steps effect is shared with the calories channel in any partial-dependence view. The shares are also not additive, because switching the effect off changes z, which feeds back into steps.
2. **Curvature estimates are noisier than the true curvature.** The fitted curve's curvature varies across datasets (0.013–0.087 for "set"; up to 0.29 for "shift", driven by extrapolation in the tails) by more than the true curvature itself (0.04–0.08). The data cannot resolve the curvature.
3. **"Shift" curves are mainly an extrapolation artefact.** Shifting the window by ±2 SD pushes the tails out of the data's support. The truth T_shift is about 99% linear, so a shift-based shape comparison cannot discriminate between models. This also explains P1-3's finding that the response curves were "close to linear": that was the expected outcome, not a model failure.
4. **Signs are always identified.** The slope signs of the associational curve and of the planted part agree with the planted signs in all 12 dataset × pair cases.

### D. Severity clipping

| Dataset | Train severity at 0 | At 27 |
|---|---|---|
| v1 | 0.5% | 0 |
| R1 | 0.3% | 0 |
| S-EFF | 2.7% | 0 |

Clipping is not a material source of curvature, except slightly in S-EFF.

### E. RQ1: subset scaler and subset composition

**Exactness.** The pure preprocessing functions (`fit_scaler` and `transform`), run in memory, reproduce the frozen processed train and calibration arrays **exactly** (max difference 0.0) for both v1 and R1. Scaling with a subset-only scaler is therefore feasible in memory, with no new preprocessing outputs and no change to existing files.

**Size of the effect** (mean over 5 nested draws):

| n | Max \|Δ mean\| (full-scaler SD) | Max \|SD ratio − 1\| | Max change in a calibration input value (SD) | Calibration values outside the KAN grid [−3, 3] |
|---|---|---|---|---|
| 50 | 0.13–0.14 (max 0.18) | 6.5% | 0.29–0.36 | 0.1–0.2% (unchanged) |
| 100 | 0.08–0.10 | 4% | 0.19 | unchanged |
| 200 | 0.06 | 2.5–3% | 0.11–0.14 | unchanged |
| 400 | 0.03 | 1–1.5% | 0.06 | unchanged |

**Subset composition.** Nested subsets were drawn with stratification by sex × age group × any sustained onset.

| n | Sustained-onset participants: dep (mean, min) | anx (mean, min) | Age group 2 |
|---|---|---|---|
| 50 | 8.6–9.0, min 7 | 8.6–9.0, min 6 | ≈ 5 |
| 100 | 16–17, min 12 | 16–18, min 15 | ≈ 10–12 |
| 200 | 32–34 | 34–35 | ≈ 18–24 |
| 400 | 69–72 | 66–68 | ≈ 36–47 |

Proportions are preserved for every n. At n = 50, each onset target has only about 6–9 positive training participants.

## Conclusions

- **RQ2a: NOT TESTABLE as a KAN-vs-MLP shape-recovery comparison** on these simulators.
  - Over the data's support, the true curves are 92–99% linear.
  - Even a correctly specified oracle cannot detect the curvature (|ΔR²| ≤ 0.007).
  - For the steps pairs, most of the learnable association is not the planted effect.
  - Only the *direction* and rough *size* of each effect are identifiable, and P1-3 and P1-4 already showed that all models recover them equally.
  - Any KAN-vs-MLP difference in curvature would sit below the estimation noise, so a "win" would not be interpretable.
- **RQ1: FEASIBLE.**
  - A subset-only scaler is exactly implementable in memory, and its effect on inputs is real but modest (≤ 0.36 SD at n = 50).
  - Stratified nested subsets preserve composition.
  - Onset outcomes at n ≤ 100 rest on 6–18 positive participants.
