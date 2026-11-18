# P3 locked plan: small-data learning curves (SYNTHETIC data)

**Approval.** G1 was approved on 2026-11-17. This plan was written and hash-locked (`P3_PLAN_LOCKED.sha256`) **before R4/R5 were generated and before any P3 model was trained.** It supersedes the draft `P3_PLAN_REVISED.md`, which is kept unchanged. Phase 0 is in `phase0/PHASE0_REPORT.md`. RQ2a and RQ2b are NO-GO and are not part of P3.

**Gates.**
- **G2:** a hard gate after the Phase 1 pilot.
- **G3:** a hard gate before any test access. Test access is not pre-approved.
- Each test split is evaluated exactly once.

## 1. Research question

**RQ1.** When training data are limited, do KAN-based models outperform:
- **RQ1a:** their parameter-matched MLP counterparts?
- **RQ1b:** the GRU?

## 2. Datasets

- **R4:** seed 3401 (streams 3401–3403).
- **R5:** seed 3501 (streams 3501–3503).

Both use the frozen configuration and frozen thresholds, and are generated with the unchanged P2-2 pipeline (`gen_p22.generate`). Configs are `configs/p22/R4.json` and `R5.json`. Data go to `data/synthetic_p22/<ds>` and `data/processed_p22/<ds>` (read-only, checksummed). Validation results go to `results/p3_small_data/datasets/<ds>/validation.json`, using train/cal and simulator-level information only.

## 3. Models

All models use the frozen P2-2 specification (`common_p22.CORE`) and the frozen `fit_nn` / `run_gbt` protocol.

| Model | Task set | Seeds |
|---|---|---|
| GBT | main + forecast | seed 0 (deterministic) |
| GRU | main + forecast | per the draw design (§4) |
| KAN-reg (B2) | main | per the draw design |
| B2-MLP-reg | main | per the draw design |
| M1 | all | per the draw design |
| M1-MLP | all | per the draw design |

## 4. Subsampling

**Training pool.** Each dataset's **train split only** (600 participants). Calibration and test participants are never in a training subset.

**Nested stratified order** (Phase 0 method).
- Strata: sex × age group × "any sustained onset (dep or anx)". The onset indicator comes from train labels only.
- Within each stratum the participants are put in random order. The global order sorts participants by (i + U) / n_stratum.
- RNG: `numpy.random.default_rng(7000 + 100·k + draw)`, with k = 4 for R4 and 5 for R5.
- The subset for size n is the first n participants in this order, **sorted by pid**.
- Subsets are written once to `results/p3_small_data/subsets/<ds>.json` and every model reads them from there.

**Design (Standard tier).**

| n | Draws | Seeds per draw (neural models) | GBT |
|---|---|---|---|
| 50, 100, 200 | 0–4 | 0, 1 | 1 run per draw |
| 400 | 0–2 | 0, 1 | 1 run per draw |
| 600 (all) | 0 | 0, 1, 2 | 1 run |

**Scaling (primary).**
- The scaler is fitted on the subset's own observed training values (`preprocess.fit_scaler`).
- That run's train-subset and calibration inputs are rebuilt in memory with `preprocess.transform`. The test inputs are rebuilt the same way, but only after G3.
- Each run stores its scaler.
- At n = 600 this reproduces the frozen processed arrays exactly.

**Calibration set (primary).** The full 200 calibration participants, used **only** for epoch selection (neural models) and for the frozen GBT selection.

## 5. Outcomes, metrics and evaluation

**Families and metrics** (both diseases averaged, as in P2-2):

| Family | Metric |
|---|---|
| sev_now | severity RMSE |
| sev_fc | severity RMSE |
| status_now | AP |
| status_fc | AP |
| onset_sust | AP |
| onset_cross | AP |

Lift is also reported.

- **Primary families:** sev_now, sev_fc, status_now, status_fc.
- **Onset families:** secondary; **exploratory at n = 50**.
- **Primary evaluation:** each dataset's **test split**, evaluated once after the G3 lock.
- **Calibration results:** secondary and developmental.

**Per-draw value.** d_draw = mean over seeds of (A − B), on the identical subset. GBT has one value per draw. For a GRU comparison, the GRU value is the seed mean within the draw.

**Verdict across draws** (P2-2 rule):

| Verdict | Rule |
|---|---|
| consistent | all draws share a sign and \|mean\| > SD |
| leaning | all draws share a sign but \|mean\| ≤ SD |
| not distinguishable | otherwise |

"A better" means lower RMSE or higher AP.

**Reproducible.** A comparison is reproducible if its verdict is consistent, in the same direction, in **both R4 and R5**, for the **same** model pair, family and n.

## 6. The 10 matched comparisons used in RQ1a

| # | KAN model | Matched MLP | Family | Primary? |
|---|---|---|---|---|
| 1 | KAN-reg (B2) | B2-MLP-reg | sev_now | primary |
| 2 | KAN-reg (B2) | B2-MLP-reg | status_now | primary |
| 3 | KAN-reg (B2) | B2-MLP-reg | onset_sust | secondary |
| 4 | KAN-reg (B2) | B2-MLP-reg | onset_cross | secondary |
| 5 | M1 | M1-MLP | sev_now | primary |
| 6 | M1 | M1-MLP | sev_fc | primary |
| 7 | M1 | M1-MLP | status_now | primary |
| 8 | M1 | M1-MLP | status_fc | primary |
| 9 | M1 | M1-MLP | onset_sust | secondary |
| 10 | M1 | M1-MLP | onset_cross | secondary |

B2 models are trained on the main task set only, so they have no forecast families.

## 7. Pre-registered decision rules

**General.** Each training size is evaluated **independently**. Advantages are never combined across n. RQ1 labels are given separately for n = 50 and for n = 100.

**RQ1a: KAN vs matched MLP** (for a given n; over the 10 comparisons in §6).

| Label | Rule |
|---|---|
| **Supported** | ≥ 3 reproducible KAN advantages, **and** more than the number of reproducible MLP advantages. |
| **Mixed** | 1–2 reproducible KAN advantages, **or** ≥ 3 comparisons with a consistent KAN advantage in one dataset only. |
| **Not supported** | 0 reproducible KAN advantages **and** ≤ 1 consistent KAN advantage in each dataset. |

The same counts are also reported for the 6 primary comparisons alone (descriptive).

**RQ1b: KAN vs GRU** (for a given n; primary families only; each KAN model judged on its own).

| KAN model | Primary families evaluated |
|---|---|
| KAN-reg | sev_now, status_now (2 families: KAN-reg has no forecast heads) |
| M1 | sev_now, sev_fc, status_now, status_fc (4 families) |

| Label | Rule |
|---|---|
| **Supported** | For one KAN model: a reproducible advantage over the GRU in ≥ 2 of its primary families, **and** no primary family at that n where the GRU is reproducibly better than that model. |
| **Mixed** | No model meets "supported", **and** any of: exactly 1 reproducible advantage for a model; or ≥ 2 primary families with a consistent advantage in only one dataset; or reproducible advantages alongside reproducible losses for that model. |
| **Not supported** | Neither KAN model has any reproducible advantage, **and** each has ≤ 1 dataset-specific consistent advantage. |

For KAN-reg, "≥ 2 primary families" therefore means both of its 2 families.

**Secondary results (descriptive only, no labels):**
- the same counts at n = 200, 400 and 600;
- the crossover n per comparison: the smallest n at which the sign of the seed-and-draw mean difference flips, per dataset;
- learning curves;
- GBT as a reference;
- onset families.

## 8. Sensitivity analyses

These run on **R4 only**, **after** the primary experiment, in priority order. Each runs only if the projected finish of all remaining steps is ≤ 11:30 h from the Phase 1 start. Otherwise it is **deferred** and reported as "not run (time budget)". Each is compared descriptively with the primary R4 result at the same n.

| Priority | ID | Change | n | Models | Draws × seeds |
|---|---|---|---|---|---|
| 1 | **S2** | Regularised GRU: weight decay 1e-3 (dropout unchanged) | 50, 100 | GRU (main + forecast) | 5 × 1 (seed 0) |
| 2 | **S3** | Reduced calibration set: 33 calibration participants (stratified draw, RNG 9000 + draw) used for epoch selection and GBT selection; evaluation still on the full calibration set or test | 100 | GRU main, KAN-reg, B2-MLP-reg | 5 × 1 |
| 3 | **S1** | Frozen 600-participant scaler (the existing processed arrays) | 50, 100 | all models; M1 and M1-MLP at 3 draws × 1 seed | 5 × 1 |

## 9. Stop conditions (approved)

**A. Reproduction and integrity: stop.**

| ID | Condition |
|---|---|
| A1 | R4/R5 generation is not deterministic, fails a checksum or provenance check, or v1 does not regenerate byte-identically. |
| A2 | R4 or R5 fails a validation gate. Stop and ask; the dataset is retained and flagged, never reseeded. |
| A3 | The n = 600 subset pipeline does not reproduce the P2-2 R1 GRU seed-0 calibration predictions (max diff > 1e-6), or the scaler exactness check is ≠ 0. |
| A4 | A subset contains calibration or test participants, subsets are not nested, or they differ across models. |
| A5 | Any pre-existing file changes (`manifest_p3_pre.json`, 4,070 files). |
| A6 | Pre-lock regeneration mismatch > 1e-6, or lock verification fails. |
| A7 | A test split is accessed before G3 (hard error by design). |

**B. Training stability.**

| ID | Condition | Action |
|---|---|---|
| B1 | A non-finite loss or prediction | Pause and report. |
| B2 | Pilot: a model at n = 50 is worse than a constant predictor (train-subset-mean RMSE, or AP ≤ prevalence) on ≥ 2 primary families | Stop at G2. |
| B3 | Best epoch = 1 in > 50% of a model's runs at some n | Flag and report; do not stop. |

**C. Runtime.**

| ID | Condition | Action |
|---|---|---|
| C1 | The pilot projects the main run at > 6:00 h elapsed | Stop at G2. |
| C2 | At 50% of main jobs, the projection is > 25% over the pilot projection | Report; continue the primary; defer the sensitivities. |
| C3 | Projected total > 12 h even with all sensitivities deferred | Stop and ask. |
| C4 | Wall time far exceeds CPU time (laptop sleep) | Report; use CPU time for projections. |

**D. Method.**

| ID | Condition | Action |
|---|---|---|
| D1 | Any change needed to this locked plan, the rules or the code | Stop and ask. |
| D2 | A crash or partial job | The queue skips completed jobs; partial jobs are rerun into a clean directory and reported. |
| D3 | Free disk space < 5 GB | Stop. |

**Priority.** The primary Standard experiment is never reduced without approval. Sensitivity analyses are deferred first.

## 10. Execution schedule (elapsed)

| Phase | Steps | Elapsed |
|---|---|---|
| Phase 1 | Lock 0:10; generate R4/R5 0:10; implement 1:00; tests 0:20; pilot 0:25 | ≈ 2:05 |
| **G2** | | |
| Phase 2 | Main experiment ≈ 4:45; S2/S3/S1 ≈ 0:25 (subject to §8); calibration analysis 0:15 | ≈ 5:25 |
| **G3** | | |
| Phase 3 | Lock and pre-lock checks 0:15; test evaluation 0:35; final report 0:50 | ≈ 1:40 |
| **Total** | | ≈ 9:10 plus buffer, ≤ 12 h |

## 11. Pilot (Phase 1)

- **Scope:** R4, draw 0, seed 0, n ∈ {50, 400}, all models (8 model-task sets × 2 sizes = 16 jobs).
- **Location:** `results/p3_small_data/pilot/`. The pilot is not part of the main results.
- **Checked:** B1, B2 and B3; per-job CPU and wall time.
- **Runtime projection:** a linear per-model CPU model in n, summed over all main jobs and divided by the measured CPU-to-elapsed ratio.

## 12. Reporting rules

- Descriptive only: no p-values.
- Every number is labelled as synthetic.
- Results are reported per draw, per seed, as means and SDs, and as reproducibility counts.
- There are no real-world, clinical or causal claims.
