# P3 revised plan: small-data learning curves (SYNTHETIC data) — DRAFT, not yet approved or locked

This revision incorporates the Phase 0 decisions (`phase0/PHASE0_REPORT.md`):

- **RQ2a and RQ2b are removed.** Over the region the data cover, the planted curves are 92–99% linear, and their curvature is undetectable even for an oracle model with the correct structure (|ΔR²| ≤ 0.007). Shape recovery is therefore not testable on the current simulators.
- **What remains is one research question with two parts.**

## 1. Research question

**RQ1.** When training data are limited, do KAN-based models outperform:

- **RQ1a:** their parameter-matched MLP counterparts (KAN-reg vs B2-MLP-reg; M1 vs M1-MLP)?
- **RQ1b:** the GRU?

**Secondary (descriptive):**

- the crossover size: the smallest n at which each paired difference changes sign;
- the shape of each learning curve.

## 2. Models

| Model | Role |
|---|---|
| KAN-reg (B2) and B2-MLP-reg | matched pair 1 |
| M1 and M1-MLP | matched pair 2 |
| GRU | reference to beat (RQ1b) |
| GBT | small-data reference; deterministic, 1 run per subset |

**Excluded,** because they have no bearing on this question: GRU-aug, Transformer, CNN-LSTM, KAN-GRU and KAN-ODE.

All hyperparameters follow the frozen protocol, unchanged from P2-2.

## 3. Datasets (decision pending)

- **Recommended:** two new fresh replicates, R4 (seed 3401) and R5 (seed 3501).
  - They use the frozen configuration and frozen thresholds.
  - They would be generated with the existing P2-2 pipeline and validation gates (the gate-failure policy is unchanged).
  - Their test splits stay sealed until the locked evaluation.
- **Why not reuse R1–R3?** Their test splits were inspected in P2-2. Choosing among them now would also be test-informed: R1's test split is known to be unusual.

## 4. Subsampling (Phase 0 decisions incorporated)

**Pool.** Subsets are drawn from each dataset's **train split only.** Calibration and test participants are never used for training.

**Nesting and stratification.**
- Subsets are nested, 50 ⊂ 100 ⊂ 200 ⊂ 400 ⊂ 600.
- They are stratified by sex × age group × "any sustained onset" using train labels only, with the nested stratified ordering validated in Phase 0.
- Every model in a draw trains on the identical subset.

**Scaling.**
- **Primary: a subset-fitted scaler.** The existing pure functions `preprocess.fit_scaler` and `preprocess.transform` run in memory, fitted on the subset's own observed training values.
  - The same scaler is applied to that run's calibration and test inputs.
  - The fill value 0 equals the subset mean.
  - Phase 0 showed this reproduces the frozen arrays exactly when the subset is the full train set.
  - Nothing new is written to `data/`. Each run stores only its 16 scaler numbers.
- **Sensitivity: the frozen 600-participant scaler.**

**Calibration set.**
- **Primary:** the full 200-participant calibration set, used only for epoch selection, as in the frozen protocol.
- **Sensitivity:** a reduced calibration set of 33 participants at n = 100.

**Draws and seeds.** The Standard tier is shown; the tier choice is pending.

| n | Draws | Seeds per draw |
|---|---|---|
| 50, 100, 200 | 5 | 2 |
| 400 | 3 | 2 |
| 600 (full) | 1 | 3 |

- **Lean tier:** M1 and M1-MLP use 3 draws × 1 seed at n ≤ 400.

## 5. Outcomes and metrics

The families, metrics and code are the same as P2-2: severity RMSE, status AP, onset AP and lift.

- **Primary families:** sev_now, sev_fc, status_now, status_fc.
- **Onset families:** secondary at n ≥ 100 and **exploratory at n = 50**, where each target has only 6–9 positive training participants (Phase 0).
- **Primary evaluation:** each dataset's **test split**, evaluated **once** after a hash lock. No selection is made on test.
- **Development:** calibration results are reported as secondary.

## 6. Paired analysis and verdicts

**Paired difference.** For each dataset, n, family and pair (A, B), compute:

d_draw = mean over seeds of (A − B), computed on the identical subset.

GBT has one value per draw.

**Verdict across draws** (P2-2 rule):

| Verdict | Rule |
|---|---|
| consistent | all draws share a sign and \|mean\| > SD |
| leaning | all draws share a sign but \|mean\| ≤ SD |
| not distinguishable | otherwise |

"A better" means lower RMSE or higher AP.

## 7. Pre-registered decision rules

**RQ1a.** Counted at n ∈ {50, 100}, across the 10 matched comparisons (B2: 4 primary + secondary families; M1: 6), with primary families reported separately:

- **Supported:** the KAN is consistently better in ≥ 3 comparisons, and more often than the MLP, in **both** datasets.
- **Not supported:** the KAN is consistently better in ≤ 1 comparison in each dataset.
- **Mixed:** otherwise.

**RQ1b.**

- **Supported:** KAN-reg or M1 is consistently better than the GRU in ≥ 2 primary families at n ≤ 100, in both datasets.
- **Not supported:** none.
- **Mixed:** otherwise.

**Secondary (descriptive only):**

- the crossover n per comparison;
- whether any KAN advantage shrinks as n grows.

## 8. Sensitivity analyses

These are pre-specified, run on R4 only, and reported next to the primary results, never in place of them.

| ID | Analysis | Scope |
|---|---|---|
| S1 | Frozen 600-participant scaler | n ∈ {50, 100}; all models; 5 draws × 1 seed (M1 pair: 3 draws × 1 seed) |
| S2 | Regularised GRU (weight decay 1e-3) | n ∈ {50, 100}; same draws |
| S3 | Reduced calibration set (33 participants) for epoch selection | n = 100; GRU, KAN-reg, B2-MLP; 5 draws × 1 seed |

## 9. Known confounders and limitations

- **Frozen hyperparameters were chosen at n = 600.** The KAN and MLP variants carry weight decay 1e-3 and dropout 0.3; the GRU has 1e-5 and 0. S2 addresses this partly.
- **Early collapse at small n.** Training may stop at epoch 1–2, so a comparison could reflect initialisation rather than inductive bias. Best epochs will be logged, and runs with best epoch ≤ 1 will be flagged and counted.
- **Head sets differ.** M1 has 28 heads; B2 and the GRU have 16 or 12.
- **Onset estimates are noisy.** There are few onset participants at small n, and test-split onset counts are about 26–35.
- **Multiplicity.** With many comparisons, the results are descriptive. Some consistent verdicts are expected by chance.
- **Single simulator family.** There is no claim to real-world or clinical validity.

## 10. Integrity safeguards (implemented and tested in Phase 1)

- **Subset integrity:**
  - subsets are nested and stratified, and contain no calibration or test participants;
  - every model receives identical subsets;
  - the subset scaler is fitted on subset training data only.
- **Reproduction check:** n = 600 with the subset scaler reproduces the existing P2-2 R1 GRU seed-0 calibration predictions exactly.
- **Protection:**
  - the P2-2 and Phase 0 hash manifests are re-verified;
  - the scripts refuse to write into non-empty directories;
  - no existing file is modified.
- **Before test access:** the plan, code, checkpoints and settings are hash-locked, and each test split is evaluated once.

## 11. Runtime and storage (estimates; to be re-checked by the Phase 1 pilot)

| Item | CPU time | Elapsed |
|---|---|---|
| Standard tier, 2 datasets | ≈ 43 CPU-h | ≈ 4.5–5 h |
| Lean tier | ≈ 25 CPU-h | ≈ 2.7 h |
| One dataset only | half of the above | half of the above |
| Sensitivities S1–S3 | ≈ 2.5 CPU-h | — |
| Subset-scaler preprocessing | seconds per subset | — |
| Locked test evaluation | — | ≈ 1 h |

- **Storage:** about 0.3–0.4 GB in total.
  - Datasets: about 150 MB.
  - Checkpoints and metrics: about 150 MB.
  - Subset-run predictions are not stored; they can be regenerated exactly from checkpoints.

## 12. Phases and approval checkpoints

| Phase | Content | Status |
|---|---|---|
| 0 | Feasibility | **COMPLETE** |
| 1 | Hash-lock this plan. Generate and validate R4/R5. Subset and scaler pipeline in `extension_p3/` with tests. **Pilot** (R4, 1 draw × 1 seed × all n), including timing. | **Pending approval** |
| 2 | Learning-curve training (chosen tier) plus S1–S3; calibration analysis | after the Phase 1 review |
| 3 | Lock, one test evaluation of R4/R5, final report | after the Phase 2 review |

## 13. Decisions needed before Phase 1

1. Datasets: R4 + R5 (recommended), R4 only, or reuse R1–R3?
2. Tier: Standard or Lean?
3. Approve sensitivities S1–S3 as specified?
