# P2-2 single locked test evaluation plan (SYNTHETIC data)

Written 2026-11-17, after approval and **before any test split of R1–R3 or the five shifts was loaded**.

The lock covers this plan, the code, configs, checkpoints, preprocessing artefacts and prediction-generation settings. It is recorded in `test_lock.json`, which `eval_test_p22.py lock` writes. Both the evaluation and the analysis scripts refuse to run if any locked hash differs. The evaluation script also refuses to run a second time: each test split is evaluated exactly once.

## 1. What is evaluated

| Role | Datasets | Notes |
|---|---|---|
| **Fresh replicates (primary)** | R1, R2, R3 | Test splits never inspected. |
| **Shifts** | S-AR, S-MISS, S-HET, S-NOISE, S-EFF | Common-random-number (CRN) pairing with R1 is preserved: the same participants and test split as R1, and seed-matched differences against R1. **S-EFF failed the prevalence gate.** It is retained and reported throughout, and labelled **qualified / exploratory**. It is never silently excluded. |
| **Historical reference** | v1, model seeds 0–2 | **Its test split was inspected in earlier stages.** It is reported for context and used only in sensitivity analyses, never in primary labels. The existing saved v1 test predictions are reused; nothing is regenerated. |

**Models:** the frozen checkpoints from Phases A and B only.
- GBT (seed 0)
- GRU (main and forecast)
- KAN-reg
- B2-MLP-reg
- M1
- M1-MLP
- GRU+R0O0T1 (augmented GRU)

Model seeds are 0, 1 and 2.

**Inference only.** The following are all forbidden after test access:
- retraining
- recalibration
- threshold tuning or any other tuning
- epoch, model or hyperparameter selection
- dataset exclusion, reseeding or regeneration
- analysis changes

**Not performed:** the Stage 5 observational regression on S-EFF and S-AR. It was listed in the original P2-2 proposal but was run in neither phase. It is recorded as not performed and will not be added post hoc.

## 2. Prediction generation (locked settings)

- **Test inputs:** `data/processed_p22/<ds>/test.npz`, from the unchanged Stage 2 preprocessing. The scaler was fitted on each dataset's own training split.
- **Origin set:** `baselines.load_data(splits=("test",))`, the same origin set as the training code.
- **Neural models:** `baselines.predict_nn` with the checkpoint's `ymean` and `ystd`. Each model is built as in training (`C.CORE` architecture, dropout, and heads for the M1 family) and run in eval mode.
- **GBT:** `baselines.gbt_features` and the saved per-head fitted estimators. Regression heads use `predict`; classification heads use `predict_proba[:, 1]`.
- **Threads:** torch threads = 4; threadpool limit = 4.
- **Pre-lock check (calibration only):** the same code regenerates every saved calibration prediction for all 8 datasets × all models × seeds. It must match exactly (max |diff| ≤ 1e-6). This verifies the generation settings before any test access.

## 3. Primary metrics and families

These are unchanged from Phases A and B.

| Family | Heads | Metric |
|---|---|---|
| sev_now | sev_k_t0 | RMSE |
| sev_fc | sev_k_h7/14/21 | RMSE |
| status_now | status_k_t0 | AP |
| status_fc | status_k_h7/14/21 | AP |
| onset_sust | onset_sust_k_h7/14/21 | AP |
| onset_cross | onset_cross_k_h7/14/21 | AP |

- Values are means over heads, then averaged over k ∈ {dep, anx} ("both").
- Per-disease values are also reported.
- AP is also reported as lift (AP divided by that dataset's test prevalence for the family).

**Model sets:**
- Main families: GBT, GRU, KAN-reg, B2-MLP-reg, M1, M1-MLP.
- Forecast families: GBT, GRU, M1, M1-MLP.

## 4. Paired comparisons and verdicts

**Paired differences:** d = A − B per seed, for each dataset and family. GBT has a single seed, so its value is paired with each seed of the other model. Pairs:
- every model vs GRU;
- KAN-reg vs B2-MLP-reg (main families);
- M1 vs M1-MLP (all families).

**Verdicts (3 seeds):**

| Verdict | Rule |
|---|---|
| consistent | all 3 seeds share a sign and \|mean\| > SD |
| leaning | all 3 seeds share a sign but \|mean\| ≤ SD |
| not distinguishable | otherwise |

"A better" means lower RMSE or higher AP.

## 5. Per-dataset conclusions

The rules are identical to `ANALYSIS_PLAN_PHASE_A.md` and `ANALYSIS_PLAN_PHASE_B.md`, applied to the test split.

**C1. GRU is the strongest model.** Two analyses are reported separately:
- **C1 all-model (primary):** the GRU is beaten in a family if any other model is consistently better.
  - Supported: beaten in 0 families. Mixed: 1–2. Not supported: ≥ 3.
- **C1 no-GBT (sensitivity):** the same rule with GBT removed from the challengers.
- GBT's hyperparameters were selected per head on calibration, so its calibration values are optimistically biased. This bias is disclosed. On test, that optimism does not apply in the same way.

**C2. KAN is not consistently better than its matched MLP.**
- n_K = number of the 10 KAN-vs-MLP comparisons where the KAN is consistently better.
- Supported: n_K ≤ 2. Mixed: 3–4. Not supported: ≥ 5.
- n_M (number where the MLP is consistently better) is also reported.

**C3. M1 is not better than the GRU.**
- Count the families where M1 is consistently better.
- Supported: 0. Mixed: 1–2. Not supported: ≥ 3.

**C4. Directions are recovered more reliably than lags.** Uses the P1-4 perturbation profile on **1,000 test origins**, with the same selection rule as calibration (RandomState(0), N = 1,000 without replacement). Models: GRU, M1, M1-MLP; seeds 0–2. Analytic truth comes from the dataset config, with train+cal truth for the benchmark slope, as in Phases A and B.
- S = sign accuracy. H = max(H_lvl, H_inc), where H_lvl and H_inc are the level-peak and increment-peak exact-hit rates.
- Supported: S ≥ 0.90 and H ≤ 0.50.
- Not supported: S − H ≤ 0.10.
- Mixed: otherwise.

**C5. Augmentation improves severity more consistently than status or onset.** Uses the Stage 8 scenario panel on the **test split** (Stage 8/9 test mask seeds, windows and replicates), comparing GRU+R0O0T1 with GRU. Relative changes are as in Phase A.
- Supported: severity improvement is consistent, and neither status nor onset improvement is consistent.
- Not supported: severity improved in ≤ 1 of 3 seeds, or status/onset improvement is consistent while severity is not.
- Mixed: otherwise.
- The clean control C0 is reported separately.

## 6. Replication conclusions (test)

**Primary (fresh replicates R1, R2, R3).** For each of C1 all-model, C1 no-GBT, C2, C3, C4 and C5:
- **Supported:** supported in 3/3.
- **Mixed:** supported in 2/3.
- **Not supported:** supported in ≤ 1/3.

**Sensitivity (v1 + R1–R3, historical reference included).** The Phase A 4-dataset rule:
- Supported: 4/4.
- Mixed: 2–3.
- Not supported: ≤ 1.

**Also reported:**
- Kendall's W of model ranks per family, for R1–R3 (primary) and v1+R1–R3.
- The between-dataset mean, SD and range of the key paired differences.

## 7. Shift-effect analysis (test)

These are the Phase B rules on the test split.

**Effect.** E = seed-mean of (A − B) per family, for 36 effects per shift. It is computed on two scales:
- raw;
- normalised: AP difference ÷ test family prevalence, or RMSE difference ÷ test target SD (pooled over the family's heads, averaged over diseases).

**Replicate yardstick:**
- **Primary:** the [min, max] range of R1, R2 and R3 test.
- **Sensitivity:** the range of v1, R1, R2 and R3 test. This is the like-for-like comparison with the calibration analysis.

**CRN change.** Δ_s = d_S(s) − d_R1(s) on test, seed-matched.

**Labels:**
- **shift-sensitive (with direction):** outside the range on both scales, on the same side, *and* Δ is consistent in that direction;
- **suggestive:** outside on both scales but Δ not consistent;
- **inconclusive (scale-dependent):** outside on exactly one scale;
- **within replication noise:** inside on both scales.

**Lag recovery and augmentation.** The test-split C4 quantities and C5 panel relative changes are compared with the replicate range in the same way, as in Phase B.

**Rank robustness.** For each family, compute the Spearman correlation between the shift's ranks and the mean replicate rank. A family is **rank-shifted** if this is below the minimum leave-one-out replicate correlation. With 3 replicates, each leave-one-out comparison is against the mean of the other 2.

## 8. Research-question labels (test)

The rules are identical to Phase B, with the primary yardstick from section 7 and the sensitivity yardstick also reported.

| Question | Rule |
|---|---|
| **RQ3** (S-EFF, qualified / exploratory): do KAN models gain with stronger effects? | Of the 10 KAN-vs-MLP effects: if 3 or more are scale-dependent → **inconclusive**. Otherwise, with n+ / n− the effects shift-sensitive for / against the KAN: **supported** if n+ ≥ 3 and n+ > n−; **not supported** if n+ = 0; **mixed** otherwise. |
| **RQ4a** (S-AR) and **RQ4b** (S-EFF, qualified): does lag recovery improve? | **Supported:** H_max exceeds the replicate maximum and 0.50. **Mixed:** exceeds the replicate maximum but ≤ 0.50. **Not supported:** otherwise. |
| **RQ5** (S-MISS): does augmentation still help severity under heavier missingness? | **Supported:** consistent severity improvement. **Mixed:** 2/3 seeds improve, or leaning. **Not supported:** otherwise. The magnitude is compared with the replicate range. |
| **RQ6a** (S-HET): does M1 gain under heterogeneity? | Same rule as RQ3, over the 6 M1-vs-GRU effects. |

**Robustness of C1 (both versions), C2, C3, C4 and C5 across shifts:**
- **Supported:** 5/5. **Mixed:** 3–4. **Not supported:** ≤ 2.
- **Sensitivity:** excluding S-EFF (4 shifts): supported 4/4, mixed 2–3, not supported ≤ 1. This is shown alongside the primary result, never instead of it.

## 9. Test event counts (reported, never acted on)

**Counts per dataset × head (test split):**
- prevalence;
- positive origins;
- positive participants.

**Flag.** A dataset is flagged **"below minimum (test)"** if any sustained-onset head has fewer than 25 positive participants on test (the same threshold as the calibration gate). Onset-family results for flagged datasets are reported with that qualification.

No dataset, seed or model is excluded, reseeded or retuned because of test counts or results.

## 10. Calibration-versus-test comparison

- **Family means:** for every dataset × family × model, the calibration and test seed means and their difference.
- **Paired comparisons:**
  - the calibration verdict and the test verdict for each comparison;
  - the sign agreement of the mean difference;
  - the share of calibration-consistent comparisons that are also consistent, in the same direction, on test.
- **Labels:** per-dataset C1–C5 labels, replication labels and RQ labels on calibration versus test. The like-for-like RQ comparison uses the 4-dataset (v1-included) test yardstick, since calibration used v1 + R1–R3.

## 11. Integrity checks recorded with the results

- **Lock:** every locked hash is re-verified, both immediately before test access and before analysis.
- **Protection:** the full test-suite pre- and post-evaluation manifests of existing files must be unchanged (3,095 pre-existing files and 278 Phase A files). The Phase B outputs are also hashed into the lock.
- **Test access log:** `test/test_access_log.json` records the start and end times and the datasets accessed. The evaluation refuses to run if `results/p2_datasets/test/` already exists.
- **Reporting:** descriptive evidence only; no p-values. There are no real-world, clinical or causal-in-humans claims.
