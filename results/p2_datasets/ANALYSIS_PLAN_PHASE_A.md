# P2-2 Phase A (replication) analysis plan: SYNTHETIC data

Written 2026-11-17 while R1–R3 were training and **before any R1–R3 model result was viewed**. Its SHA-256 is recorded in `plan_phase_a_sha256.txt` at the time of writing. Phase A uses the **calibration split only**. Test labels and outcomes for R1–R3 stay sealed. Their single, hash-locked test evaluation is deferred to a separate, approved step, so that Phase B design cannot be influenced by test results.

## Datasets
- v1: the frozen original. Existing predictions and checkpoints are reused; model seeds 0–2 only, for like-for-like comparison.
- R1, R2, R3: seeds 3101 / 3201 / 3301, unchanged frozen configuration and thresholds.
- Replication results are reported separately from any Phase B shift result.

## Models
Six core models and one augmentation model:
- GBT (seed 0, deterministic)
- GRU (main + forecast)
- KAN-reg B2 (main)
- B2-MLP-reg (main)
- M1 (28 heads)
- M1-MLP (28 heads)
- GRU + Stage 9 augmentation R0O0T1 (main; pre-specified, not reselected)

Frozen protocol; model seeds 0, 1, 2.

## Families (calibration, means over heads; dep and anx averaged unless stated)
| Family | Heads | Metric |
|---|---|---|
| sev_now | sev_k_t0 | RMSE |
| sev_fc | sev_k_h7/14/21 | RMSE |
| status_now | status_k_t0 | AP |
| status_fc | status_k_h* | AP |
| onset_sust | onset_sust_k_h* | AP |
| onset_cross | onset_cross_k_h* | AP |

- Main-family models: GBT, GRU, KAN-reg, B2-MLP-reg, M1, M1-MLP.
- Forecast-family models: GBT, GRU, M1, M1-MLP.
- Prevalence context: calibration positive-origin rate per family; AP is also reported as lift = AP / prevalence.

## Paired comparisons and verdict labels (3 seeds)
- d = A − B per seed (same seed index). GBT has a single deterministic value, so it is compared with each GRU seed.
- **Consistent:** all 3 seeds share a sign *and* |mean d| > SD(d).
- **Leaning:** all 3 seeds share a sign but |mean d| ≤ SD(d).
- **Not distinguishable:** otherwise.
- "A better" means lower RMSE or higher AP.

## Replication conclusions: per-dataset rules (fixed in advance)
**C1. GRU remains the strongest predictive baseline.**
- In each of the 6 families, the GRU is "not beaten" unless some other model is *consistently better* than it.
- Supported: not beaten in all 6 families. Mixed: beaten in 1–2. Not supported: beaten in ≥ 3.
- Also reported: the GRU's rank by seed-mean.

**C2. KAN does not consistently outperform its matched MLP control.**
- 10 comparisons: KAN-reg vs B2-MLP-reg in 4 main families; M1 vs M1-MLP in 6 families.
- n_K = number of comparisons where the KAN is consistently better.
- Supported: n_K ≤ 2. Mixed: n_K = 3–4. Not supported: n_K ≥ 5.
- Also reported: n_M, the number where the MLP is consistently better.

**C3. M1 does not outperform GRU** (6 families).
- Supported: M1 consistently better in 0 families. Mixed: 1–2. Not supported: ≥ 3.
- Caveat: M1 has 28 heads, the GRU 16 / 12.

**C4. Predictive models recover planted directions more reliably than planted temporal lags.**
- Method: P1-4 single-day perturbation profile (lags 0–10, 1,000 fixed calibration origins, severity nowcast heads) for GRU, M1 and M1-MLP, seeds 0–2.
- Analytic truth is parameterised from each dataset's actual simulator configuration: lags `lag_*`, signs of `eff_*`, mean reversion `a`, and the benchmark slope from that dataset's train+calibration truth.
- Pooled over model × seed × 4 planted pairs:
  - S = sign accuracy
  - H_lvl = exact-hit rate of the level-peak lag
  - H_inc = exact-hit rate of the increment-peak lag over lags 1–5
- Uninformative reference: the best constant guess, which hits 2 of the 4 planted lags (0.50) with the unchanged lags 3/2/2/1.
- Supported: S ≥ 0.90 and max(H_lvl, H_inc) ≤ 0.50. Not supported: S − max(H_lvl, H_inc) ≤ 0.10. Mixed: otherwise.

**C5. Missing-data augmentation improves severity more consistently than status or onset.**
- GRU + R0O0T1 vs GRU, seed-matched, on the P1-6 panel of 10 Stage 8 scenarios: S1_r40, S2_r40_L7, S3_k3, S3_k7, S3_k14, S4_activity_absent, S4_sleep_absent, S4_cardiac_absent, S5_rand40, S5_blk40. Same windows, origins and mask replicates as Stage 8; calibration split.
- Per seed: relative change per scenario (severity: (RMSE_aug − RMSE_gru)/RMSE_gru; status and onset: (AP_gru − AP_aug)/AP_gru; negative = augmentation better), averaged over the 10 scenarios.
- Families: severity (sev_dep/anx_t0), status (status_dep/anx_t0), sustained onset (6 heads).
- Supported: severity improvement consistent, and neither status nor onset improvement consistent.
- Not supported: severity improved in ≤ 1 of 3 seeds, or status/onset improvement consistent while severity is not.
- Mixed: otherwise.
- The clean control C0 is reported separately.

## Cross-dataset aggregation (4 replicate datasets: v1, R1, R2, R3)
- For each conclusion: **supported** if supported in 4/4 datasets; **not supported** if supported in ≤ 1/4; **mixed** otherwise.
- Also reported:
  - per-dataset labels and counts
  - Kendall's W of model ranks per family across the 4 datasets
  - between-dataset mean, SD and range of key paired differences (M1 − GRU, KAN − MLP, best other model − GRU)
  - per-seed values
- Descriptive evidence only: no p-values, no dataset-level significance claims.
- Absolute metrics are not compared across datasets without prevalence / lift context.

## Reproduction checks required before interpretation
- Wrapper regeneration of v1: byte-identical. **Done.**
- New runner GRU seed 0 on v1 equals the Stage 3 predictions. **Done.**
- Lag profiles on v1 reproduce P1-4's saved per-seed metrics (seeds 0–2).
- The augmentation evaluation on v1 calibration reproduces Stage 9's saved calibration metrics.
