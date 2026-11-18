# P3 locked test evaluation plan (2026-11-18)

**SYNTHETIC data only.** This plan was written after the user approved G3, and **before any R4 or R5 test-split data was read**. It is hash-locked, together with all code, checkpoints, scalers, data and settings, in `test_lock.json`.

`P3_PLAN_LOCKED.md` (SHA-256 `f084f9e0…a4c1`), `P3_PLAN_AMENDMENT_1.md` and `P3_PLAN_AMENDMENT_2.md` remain in force and unchanged. This plan adds nothing to the research questions, models, metrics or decision rules. It only fixes how the single test evaluation is carried out.

## 1. Scope

| Item | Setting |
|---|---|
| Models evaluated | The 416 primary (main) jobs of amendment 1: R4 and R5; all 8 model-task sets; n = 50, 100, 200, 400 and 600; the amendment-1 draws and seeds |
| Not evaluated on test | Sensitivity runs S1, S2 and S3. They remain **calibration-only**, as the locked plan §8 specifies. |
| Test participants | All test-split participants of R4 and of R5, the same set for every job |
| Inference | Each job's saved checkpoint, used with its own `scaler.json` (subset-fitted scaler, the primary analysis) |
| Procedure | Inference only: no training, retraining, recalibration, retuning, threshold change, model exclusion or seed selection |
| Frequency | **Exactly once.** `eval_test_p3.py run` refuses to start if `results/p3_small_data/test/` already exists. |

## 2. Pre-lock checks (no test access)

The lock is written only if all of the following pass. If any fails, I stop and the test split is not accessed.

1. **Manifest.** All 4,070 pre-existing files are unchanged (`tests/test_p3.py`, 8 tests).
2. **Calibration reproduction.** Every one of the 416 models is rebuilt from its checkpoint and `scaler.json` using the **same inference code that the test step uses**. Its per-head calibration metrics must match the stored `metrics.json` within 1e-6 (`test_prelock_cal_regeneration.csv`).
3. **Analysis reproduction.** The test-analysis code (`analyze_test_p3.py`), run on the calibration metrics, must reproduce the following from `analysis_cal/` exactly:
   - every RQ1a and RQ1b label and count at n = 50 and 100;
   - the secondary counts;
   - every paired comparison.

## 3. Analysis

The test analysis is fixed before test access. It applies the calibration rules unchanged, by running the `analyze_p3` functions on the test metrics.

- **Family values.** Each family's metric is the mean over its heads, for dep and anx.
- **Verdict units.**
  - n ≤ 400: one unit per draw, which is the seed mean of that draw.
  - n = 600: one unit per seed.
  - GBT: one run per draw.
- **Verdicts.** A comparison is *consistent* if all units share a sign and |mean| > SD. Otherwise it is not distinguishable.
- **Reproducibility.** An advantage is reproducible only if it is consistent, in the same direction, in **both** R4 and R5.
- **Labels.**
  - **RQ1a**, KAN vs matched MLP over the 10 enumerated comparisons: the locked rule plus amendment 2.
  - **RQ1b**, KAN vs GRU: per KAN model, as in the locked plan.
  - Both are evaluated **independently at n = 50 and at n = 100.**
- **Descriptive results.** n = 200, 400 and 600 are reported descriptively: learning curves, counts and crossover. They do not enter any label.
- **Calibration vs test.** Every paired comparison is compared between the two splits:
  - whether the mean difference has the same sign;
  - whether the "consistently better" label is the same;
  - whether a consistent advantage reverses direction.

  Labels are reported side by side. **Test results are primary for the final conclusions.** Every disagreement with calibration is reported.
- **Reporting.** R4 and R5 are reported separately before any combined conclusion.

## 4. Outputs

All outputs are kept and never overwritten.

| Location | Content |
|---|---|
| `results/p3_small_data/test/predictions/<ds>/<job>.npz` | Test predictions, with participant ID and day for each row |
| `results/p3_small_data/test/metrics/<ds>/<job>.json` | Per-head test metrics |
| `results/p3_small_data/test/test_jobs.csv` | Per-job record of the test run |
| `results/p3_small_data/test/test_access_log.json` | Lock hash, start and end times, finite check |
| `results/p3_small_data/analysis_test/` | Test analysis |

## 5. Integrity rules

- The test step runs only after `verify_lock()` passes. If any locked file or setting has changed, it stops.
- Intermediate test outputs are not inspected for methodological decisions.
- If a test job fails or returns non-finite values, the failure is reported as it is. Nothing is rerun after test results are seen.
