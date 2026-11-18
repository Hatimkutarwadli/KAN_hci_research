# P2-2 Phase A: replication on new synthetic datasets (SYNTHETIC; calibration split only)

All data and results here are **synthetic**. Nothing here supports clinical or real-world validity, or causal effects in humans.

- **Test labels and outcomes for R1–R3 have not been loaded or inspected.** The single hash-locked test evaluation is deferred to a separately approved step.
- **Analysis plan.** The plan `ANALYSIS_PLAN_PHASE_A.md` (SHA-256 `6fbbf092…3bdf7`) was fixed before any R1–R3 result was viewed. Its hash was re-verified before the analysis was run.
- **Data and code.** Analysis outputs are in `phase_a_cal/`. The analysis code is `extension_p22/analyze_p22.py`.

## 1. Dataset integrity and validation

**Datasets.**
- R1, R2 and R3 use seeds 3101, 3201 and 3301. They use the unchanged frozen config (`configs/frozen_config_v1.json`) and the original frozen thresholds (z_thr 2.45). There was no prevalence matching.
- They are stored read-only and with checksums in `data/synthetic_p22/<ds>`, with processed data in `data/processed_p22/<ds>`. Provenance is in `data/synthetic_p22/_provenance/`.

**What is held constant and what changes.**
- Phase A replicates are *independent* draws.
- Held constant: the simulator code, the config, the thresholds, N, T, the burn-in, the horizons, the split fractions and RHO.
- Changes: all three random streams (static parameters `seed`, daily noise `seed+1`, split `seed+2`). These streams are disjoint from each other and from the reserved v1 streams 2026–2029.
- Common random numbers therefore do **not** link R1–R3 to v1. CRN is relevant to Phase B, where each shift reuses a replicate's seed so that only the shifted mechanism changes. The `test_rho_override_is_temporary_and_crn_holds` test verifies that a sensor-noise shift leaves truth and targets bit-identical.

**Validation.** These checks are simulator-level and use the train and calibration splits only.

| | Daily prevalence dep / anx | Missing-day rate | Smallest cal sustained-onset participant count | Gates |
|---|---|---|---|---|
| v1 (reference) | 6.7% / 7.2% | 15.4% | — | — |
| R1 | 6.9% / 7.2% | 15.8% | 25 (anxiety; exactly at the gate) | pass |
| R2 | 7.6% / 7.7% | 16.0% | 34 | pass |
| R3 | 6.3% / 7.3% | 15.5% | 35 | pass |

Calibration positive rates by family are in `phase_a_cal/cal_prevalence.csv`:

| Family | v1 | R1 | R2 | R3 |
|---|---|---|---|---|
| status_now | 0.0714 | 0.0729 | 0.0774 | 0.0775 |
| status_fc | 0.0715 | 0.0700 | 0.0758 | 0.0799 |
| onset_sust | 0.0372 | 0.0288 | 0.0340 | 0.0403 |
| onset_cross | 0.0467 | 0.0388 | 0.0384 | 0.0499 |

R1's onset prevalence is about 20% lower than v1's, so absolute onset AP is compared across datasets only via lift (`family_summary.csv`, `lift` column).

**Reproduction and integrity tests.** `tests/test_p22.py` passes all 10 tests:
- v1 regenerated through the new wrapper is byte-identical.
- The new runner reproduces the Stage 3 GRU seed-0 calibration predictions exactly (max |diff| 0.0).
- The lag evaluator reproduces the P1-4 per-seed metrics for 36 of 36 cases.
- The augmentation evaluator reproduces the Stage 9 calibration metrics (max diff < 1e-6).
- New datasets are read-only and pass checksum checks.
- Validation is sealed.
- No P2-2 script loads the test split.
- All 3,095 pre-existing files are unchanged (hash manifest).

**Training diagnostics.** All 57 neural runs and the GBT finished without errors. Selected epochs were 1–5. Four R3 runs selected epoch 1: KAN-reg seed 0, M1 seed 1, and B2-MLP seeds 0 and 1. These are regularised models that also peak early on v1, at epochs 2–3.

## 2. Per-dataset model comparisons

Values are calibration seed means (3 seeds; GBT is deterministic). They are averaged over depression and anxiety. The best value in each column is in **bold**. Per-seed values, SDs, per-disease values and lift are in `family_summary.csv` and `family_per_seed.csv`.

**Severity RMSE (lower is better)**

| Model | sev_now v1 | R1 | R2 | R3 | sev_fc v1 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|
| GBT | 2.937 | 2.938 | 2.883 | 2.911 | 3.168 | 3.145 | 3.111 | 3.098 |
| **GRU** | **2.921** | **2.919** | **2.830** | **2.887** | **3.146** | **3.118** | **3.085** | **3.093** |
| KAN-reg | 2.981 | 3.013 | 2.926 | 2.934 | | | | |
| B2-MLP-reg | 2.955 | 2.996 | 2.945 | 2.915 | | | | |
| M1 | 3.058 | 3.061 | 3.037 | 3.037 | 3.257 | 3.224 | 3.221 | 3.200 |
| M1-MLP | 2.990 | 3.041 | 2.987 | 3.002 | 3.196 | 3.214 | 3.196 | 3.167 |

**Status AP (higher is better)**

| Model | status_now v1 | R1 | R2 | R3 | status_fc v1 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|
| GBT | **0.305** | 0.432 | 0.337 | **0.381** | **0.254** | 0.328 | **0.293** | 0.301 |
| GRU | 0.290 | **0.439** | 0.351 | 0.363 | 0.244 | **0.342** | 0.285 | **0.310** |
| KAN-reg | 0.266 | 0.422 | **0.359** | 0.349 | | | | |
| B2-MLP-reg | 0.294 | 0.413 | 0.342 | 0.352 | | | | |
| M1 | 0.249 | 0.388 | 0.317 | 0.306 | 0.229 | 0.321 | 0.263 | 0.254 |
| M1-MLP | 0.279 | 0.426 | 0.340 | 0.337 | 0.249 | 0.337 | 0.277 | 0.285 |

**Onset AP (higher is better)**

| Model | onset_sust v1 | R1 | R2 | R3 | onset_cross v1 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|
| GBT | 0.116 | 0.108 | 0.095 | 0.117 | 0.123 | 0.101 | 0.115 | 0.159 |
| GRU | **0.134** | **0.136** | **0.103** | **0.149** | **0.154** | **0.126** | **0.116** | 0.203 |
| KAN-reg | 0.114 | 0.090 | 0.082 | 0.113 | 0.124 | 0.085 | 0.090 | 0.163 |
| B2-MLP-reg | 0.125 | 0.095 | 0.079 | 0.122 | 0.132 | 0.094 | 0.092 | 0.172 |
| M1 | 0.111 | 0.116 | 0.090 | 0.144 | 0.130 | 0.107 | 0.103 | 0.186 |
| M1-MLP | 0.120 | 0.117 | 0.093 | 0.146 | 0.140 | 0.116 | 0.103 | **0.206** |

**How the GRU was beaten.** In the whole grid, the GRU was beaten *consistently* (3 of 3 seeds and |mean| > SD) in only these cases:

| Dataset | Family | Winning model | AP difference vs GRU |
|---|---|---|---|
| v1 | status_now | GBT | +0.015 |
| v1 | status_fc | GBT | +0.010 |
| R3 | status_now | GBT | +0.018 |
| R2 | status_now | KAN-reg | +0.008 |

All other "best value" entries that are not the GRU are leaning or not distinguishable.

**GBT caveat.** GBT hyperparameters were selected per head on calibration AP and RMSE in Stage 3. GBT's calibration numbers are therefore optimistically biased relative to the neural models, which selected only the epoch on calibration loss.

**Rank agreement across the four datasets.** Kendall's W per family:

| Family | W |
|---|---|
| sev_now | 0.98 |
| sev_fc | 1.00 |
| status_now | 0.59 |
| status_fc | 0.68 |
| onset_sust | 0.69 |
| onset_cross | 0.74 |

Severity rankings are essentially fixed. Status and onset rankings vary between datasets.

## 3. Cross-dataset consistency of the five conclusions (pre-registered rules)

| Conclusion | v1 | R1 | R2 | R3 | Overall (supported if 4/4; not supported if ≤ 1/4) |
|---|---|---|---|---|---|
| C1. GRU is the strongest baseline | mixed (beaten in 2 families) | supported | mixed (1) | mixed (1) | **not supported** |
| C2. KAN is not consistently better than its matched MLP | supported (n_K 0, n_M 9) | supported (0, 6) | supported (1, 0) | supported (0, 3) | **supported** |
| C3. M1 is not better than GRU | supported (0 of 6) | supported (0) | supported (0) | supported (0) | **supported** |
| C4. Directions are recovered more reliably than lags | supported | supported | supported | supported | **supported** |
| C5. Augmentation helps severity more than status or onset | supported | not supported | supported | supported | **mixed** |

**C1.** Under the pre-registered aggregation rule, C1 is *not supported*: it is fully supported only in R1. Two points are needed to read this label:
1. In no dataset was the GRU beaten in more than 2 of 6 families.
2. It led all severity families and both onset families, apart from one nominal non-consistent case (R3 onset_cross, where M1-MLP leads by 0.003).

Every consistent loss was on **status AP**, and 3 of the 4 losses were to the calibration-selected GBT. The honest statement is: *"The GRU is the strongest model for severity and onset in all 4 datasets. It is not uniformly strongest for status: GBT (calibration-selected) or KAN-reg match or exceed it on status AP in 3 of 4 datasets."* Between datasets, the best non-GRU model's status_now margin was +0.009 ± 0.011 AP (range −0.006 to +0.018).

**C2.** Supported in 4 of 4 datasets. The only consistent KAN win was R2 sev_now (KAN-reg RMSE 0.020 lower than B2-MLP-reg). The MLP was consistently better 18 times in total across datasets. KAN − MLP differences between datasets:

| Comparison | Mean ± SD |
|---|---|
| M1 − M1-MLP, sev_now RMSE | +0.043 ± 0.021 (MLP better in all 4 datasets) |
| KAN-reg − B2-MLP-reg, sev_now RMSE | +0.011 ± 0.021 (sign varies) |

**C3.** M1 was never better than the GRU in any family of any dataset. Between datasets, M1 − GRU was:

| Metric | Difference |
|---|---|
| sev_now RMSE | +0.159 ± 0.032 |
| status_now AP | −0.046 ± 0.010 |
| onset_cross AP | −0.019 ± 0.004 |

**C4.** Pooled over GRU, M1 and M1-MLP × 3 seeds × 4 planted pairs (36 cases per dataset):

| | Sign accuracy | Level-peak exact-hit rate | Increment-peak exact-hit rate |
|---|---|---|---|
| v1 | 1.00 | 0.28 | 0.17 |
| R1 | 1.00 | 0.47 | 0.28 |
| R2 | 1.00 | 0.31 | 0.22 |
| R3 | 1.00 | 0.28 | 0.17 |

The constant-guess reference hits 0.50, so no dataset's lag hit rate beats an uninformative guess. Directions are recovered every time. Analytic truth was parameterised from each dataset's config and train+calibration truth (`<ds>/lag_recovery/truth.json`).

**C5.** Relative change in the 10-scenario P1-6 panel, GRU+R0O0T1 vs GRU. Negative values mean augmentation is better.

| Dataset | Severity (3 seeds) | Status (3 seeds) | Onset (3 seeds) |
|---|---|---|---|
| v1 | −1.7, −1.3, −2.2% | +0.6, −0.2, 0.0% | −1.1, −0.2, +0.8% |
| R1 | +0.2, −1.5, −1.5% | −2.9, −4.1, −1.6% | −10.6, −3.2, −0.9% |
| R2 | −2.1, −1.4, −1.6% | −0.7, +3.5, 0.0% | −1.2, +1.5, −0.3% |
| R3 | −1.1, −0.5, −0.9% | +1.3, −1.4, +0.7% | +0.1, −1.0, −0.4% |

The severity gain replicated consistently in v1, R2 and R3. **R1 is the exception:**
- Severity improved in only 2 of 3 seeds.
- Status improved consistently and onset leaned toward improvement.

On R1 the augmented GRU is also better on the *clean* control C0:

| R1, clean C0 | Change |
|---|---|
| Status | −1.5 to −4.4% |
| Onset (seed 0) | −15.0% |

So R1's status and onset gains look like a general difference in training outcome rather than a missingness-specific effect. This is a descriptive observation only, and the pre-registered label for R1 stays "not supported". The overall C5 label is mixed: 3 of 4 datasets support it.

## 4. Runtime and storage

All runs were CPU only, with the three datasets run in parallel. CPU time is reported because parallel contention inflates wall time.

| Item | CPU time (sum over R1–R3) |
|---|---|
| M1 (3 seeds × 3 datasets) | 486 min (65%) |
| KAN-reg | 88 min |
| GBT (main + forecast) | 56 min |
| M1-MLP | 45 min |
| GRU (main + forecast) | 39 min |
| GRU + R0O0T1 | 24 min |
| B2-MLP-reg | 13 min |
| **Training total** | **12.5 CPU-h** (4.3 h summed per-run wall time; about 1.5–2 h elapsed in parallel) |
| Lag recovery (3 datasets in parallel) | 3.5 min elapsed |
| Augmentation evaluation | 1 min elapsed |

| Storage | Size |
|---|---|
| Raw synthetic R1–R3 | 162 MB |
| Processed | 58 MB |
| Results (checkpoints, cal predictions, logs, evaluations) | 109 MB |
| **Total** | **≈ 330 MB** |

Phase B with 5 shifts at the same panel would be about 5/3 × 12.5 ≈ **21 CPU-h** and about 550 MB. Dropping or reducing M1 alone would cut the time by about two thirds.

## 5. Should Phase B proceed unchanged?

**Recommendation: proceed, with the shift design unchanged and three small pre-specifications added before any shifted dataset is generated.** Nothing in Phase A requires changing the approved shifts, models, seeds or thresholds. The pipeline, the CRN wrapper and the sealing all work as intended.

The three pre-specifications are:

1. **Validation-gate failure policy.** R1 sat exactly at the onset-count gate (25). Shifts that reduce effects or increase missingness (S-EFF, S-MISS) may fall below it. Fix in advance that a gate failure is *reported* and the shift is analysed with a "low-event" flag, or dropped, without reseeding or retuning. Otherwise the gate becomes a selection step.

2. **Shift-effect reference interval.** Phase A shows real between-replicate spread: status/onset rank W is about 0.6–0.7, and C1 and C5 flipped on one replicate. Phase B shift effects should therefore be judged against the Phase A between-dataset spread of the same paired difference (`between_dataset_key_diffs.csv`), not against v1 alone. This should be written into the Phase B plan before Phase B results exist.

3. **GBT in C1.** Keep GBT, but report C1 both with and without GBT. Its calibration values are selection-biased, and every C1 failure so far was a GBT or KAN-reg status-AP margin of 0.01–0.02.

**Optional cost reduction (your decision; not recommended unless time is short):** run M1 on Phase B with fewer seeds, or drop it. This saves about 13 of the 21 CPU-h, but leaves C2 and C3 resting on KAN-reg vs B2-MLP alone.

**Phase A test evaluation:** still sealed. I suggest one locked test run covering R1–R3 (and Phase B) together after Phase B, so that no test result can influence the Phase B design.
