# P3 Phase 1 pilot report (SYNTHETIC; calibration only; the test split is sealed)

## Scope

The pilot followed `P3_PLAN_LOCKED.md` §11:

- **Data:** R4, draw 0, seed 0.
- **Sizes:** n = 50 and 400.
- **Models:** all 8 model-task sets.
- **Jobs:** 16, on 3 workers × 4 threads.
- **Timing:** 18:12:53–18:31:04, 1,091 s elapsed, 9,661 CPU-s.

Outputs:

| File | Content |
|---|---|
| `pilot_jobs.csv` | per-job runtime and training details |
| `pilot_families.csv` | family-level calibration metrics |
| `pilot_checks.json` | stop-condition checks and projection |
| `R4/<job>/` | checkpoint, scaler, metrics and run log for each job |

The pilot is not part of the main results.

## Stability checks

| Check | Result |
|---|---|
| **B1: non-finite loss or prediction** | None. All 16 jobs finite. |
| **B2: worse than a constant predictor on ≥ 2 primary families at n = 50** | **None.** Every model beats the train-subset-mean RMSE and the prevalence AP on every family. |
| **B3: best epoch ≤ 1** | None. Best epochs are 2–12. |

At n = 50 the models train *longer* than at n = 400. For example, M1 runs 16 epochs at n = 50 against 8 at n = 400, and KAN-reg runs 14 against 6.

**n = 50, calibration, illustrative only (one draw, one seed):**

| Model | sev_now RMSE (constant 3.64) | status_now AP (prevalence 0.065) |
|---|---|---|
| GRU | 3.020 | 0.303 |
| GBT | 3.005 | 0.194 |
| KAN-reg | 3.065 | 0.322 |
| B2-MLP-reg | 3.033 | 0.269 |
| M1 | 3.176 | 0.215 |
| M1-MLP | 3.162 | 0.209 |

These are single-run calibration numbers and must not be interpreted.

## Runtime: **stop condition C1 triggered**

Per-job CPU time in seconds:

| Model | n = 50 | n = 400 |
|---|---|---|
| M1 | 2,321 | 3,064 |
| KAN-reg | 776 | 632 |
| GBT main | 363 | 776 |
| GBT forecast | 232 | 546 |
| M1-MLP | 106 | 252 |
| GRU main | 57 | 161 |
| GRU forecast | 58 | 157 |
| B2-MLP-reg | 51 | 108 |

**Why the estimate was wrong.** My pre-pilot estimate (43 CPU-h) assumed cost scales with n. It does not:
- Every epoch evaluates the fixed 200-participant calibration set.
- Smaller subsets train for more epochs.

So an n = 50 M1 run costs 72% of a full-data M1 run.

**Projection for the approved Standard design** (544 jobs, R4 + R5): **91.5 CPU-h**, which is **9.8–10.3 h elapsed** at the measured 8.9–9.3 CPU-h per hour. M1 accounts for 57 CPU-h of this.

- That exceeds the C1 threshold of 6:00 h.
- With the remaining steps (≈ 2 h), it would also exceed the 12-hour total (C3).

As required, **the primary design has not been changed.** The options below need your decision.

## Options

The projection uses the pilot's per-model linear cost curves.

| Option | Design | Main CPU-h | Main elapsed | Keeps the primary decision sizes (n = 50, 100) at 5 draws × 2 seeds for all models? |
|---|---|---|---|---|
| Standard (approved) | as locked | 91.5 | 9.8–10.3 h | yes; total ≈ 12.5–13 h, **over budget** |
| **F** | n = 50, 100 Standard; n = 200 and 400 at 3 draws × 1 seed; n = 600 at 3 seeds | 68.4 | 7.4–7.7 h | **yes** |
| **G** | n = 50, 100 Standard; n = 200 at 3 × 1; n = 400 dropped; n = 600 at 3 seeds | 58.9 | 6.3–6.7 h | **yes** |
| **D** | n = 50, 100 Standard; n = 200 and 400 dropped; n = 600 at 3 seeds | 51.0 | 5.5–5.8 h | **yes** |
| A | M1 and M1-MLP at 1 seed per draw at every n (all draws kept); others Standard | 64.3 | 6.9–7.3 h | no (M1 pair 1 seed) |
| B | Lean tier (M1 pair: 3 draws × 1 seed at n ≤ 400) | 55.6 | 6.0–6.3 h | no |

**Sensitivity analyses** (CPU-h): S2 0.36, S3 1.23, S1 8.6 (≈ 1 h elapsed).

**Approximate total time from the Phase 1 start (18:03)**, excluding your review time at G2 and G3. The total is Phase 1 (0:35) + main + calibration analysis, lock, test and report (≈ 2:00) + sensitivities.

| Option | Total |
|---|---|
| F | ≈ 10.1–10.3 h with S2 and S3; S1 adds ≈ 1 h (≈ 11.1–11.3 h), so S1 is likely deferred |
| G | ≈ 9.1–9.5 h with S2 and S3; ≈ 10.1–10.5 h with all three |
| D | ≈ 8.3–8.6 h with S2 and S3; ≈ 9.3–9.6 h with all three |

**Recommendation.** Option **F** if you want the full learning-curve shape: every size is retained, and the primary decision sizes are untouched. Option **G** for more margin. The RQ1a and RQ1b decision rules use only n = 50 and n = 100, and those stay at the full Standard design in F, G and D. Only the descriptive sizes, n = 200 and 400, are reduced.

## Integrity

- **Tests:** `tests/test_p3.py` 8/8 pass before the pilot; 4/4 re-run after it (pilot subset identity, plan lock, test sealing, manifest).
- **Plan lock:** `P3_PLAN_LOCKED.md`, SHA-256 `f084f9e0…a4c1`, unchanged and read-only.
- **Reproduction:** the n = 600 subset pipeline reproduces the P2-2 R1 GRU seed-0 calibration predictions exactly.
- **Scaling:** the in-memory scaler reproduces the processed arrays exactly (R4 and R5, train and calibration).
- **Subsets:** nested, stratified within ±1.5 participants, train-only and deterministic. All pilot models received identical subsets.
- **Datasets:** R4 and R5 pass all validation gates.

  | Dataset | Daily prevalence dep / anx | Missing-day rate | Calibration onset minimum |
  |---|---|---|---|
  | R4 | 6.1 / 6.7% | 14.8% | 35 |
  | R5 | 6.2 / 7.5% | 15.7% | 27 |

- **Test sealing:** no test split was read; `split_data` and `all_pids` refuse "test" until G3.
- **Pre-existing files:** all 4,070 unchanged (`manifest_p3_pre.json`).
- **Storage so far:** R4 + R5 raw and processed data 149 MB; pilot 17 MB (about 1 MB per job). The main run would add about 0.4–0.55 GB.
