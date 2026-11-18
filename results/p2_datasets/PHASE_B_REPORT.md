# P2-2 Phase B: distribution shifts (SYNTHETIC data, calibration split only)

All data and results are **synthetic**. They make no clinical, real-world, or causal-in-humans claims.

**Test splits are sealed.** No test split of any new dataset has been loaded or inspected. The single test evaluation needs a further hash lock (plan, code, model configurations, comparison rules, primary and sensitivity analyses) and your approval.

**Locked plan.** The analysis follows `ANALYSIS_PLAN_PHASE_B.md` (SHA-256 `6769f042…65fc`). The plan was stamped at 10:58, before any Phase B model had finished, and its hash was re-verified before analysis.

- Code: `extension_p22/analyze_p22b.py`
- Outputs: `phase_b_cal/`

Replication results (Phase A, `PHASE_A_REPORT.md`) are kept separate and are only used here as the reference range.

## 1. Integrity report

`tests/test_p22.py` reports **14/14 pass**:

- **Shift configs:** each shift differs from R1 only in its declared field(s), and the thresholds are frozen.
- **Common random numbers (CRN):**
  - Every shift has the same splits and participants as R1.
  - S-MISS and S-NOISE have byte-identical truth and targets to R1, because they change observations only.
  - S-EFF, S-AR and S-HET have different truth, as intended.
  - All shifts have different observations from R1.
- **Integrity checks:**
  - S-AR's RHO provenance is × 0.5.
  - All new data is read-only and checksummed.
  - Validation is sealed.
  - No P2-2 script loads a test split.
  - The gate-fail flag is recorded.
- **Reproduction:**
  - v1 regenerates byte-identically through the wrapper.
  - The runner reproduces the Stage 3 GRU exactly.
  - The lag and augmentation evaluators reproduce P1-4 and Stage 9.
- **Protection:** all 3,095 pre-existing files and all 278 Phase A files are unchanged (hash manifests).

**What CRN holds constant and what it changes:**

| Shift | Same as R1 | Changed |
|---|---|---|
| All shifts | seed 3101 → identical static-parameter, daily-noise and split random streams; same participants and same split | — |
| S-MISS, S-NOISE | identical latent trajectories, severity and labels | observations only |
| S-EFF, S-AR, S-HET | the same noise draws | the latent dynamics (via the shifted mechanism), and therefore the labels |

**Code changes made in this phase:**

- **New option in `run_p22.py`:** `--allow-gate-fail`, which writes `GATE_FAIL_FLAG.json`. This implements the approved retention policy.
- **New script:** `analyze_p22b.py`.
- **Tests extended:** `tests/test_p22.py` gained four new tests.
- **Bug fix (disclosed):** in the first analysis run, `E.shift` resolved to the pandas `shift` method, so the RQ3 and RQ6a filters were empty (n = 0). I changed it to `E["shift"]` and reran. No rule changed. The buggy output was moved to the scratchpad and is not used.

## 2. Validation results

Simulator-level checks use train and calibration data only.

| Dataset | Change | Daily prevalence dep / anx | Missing-day rate (mean gap) | Calibration minimum sustained-onset participants | Gate |
|---|---|---|---|---|---|
| R1 (reference) | — | 6.9% / 7.2% | 15.8% (2.5 d) | 25 | pass |
| S-EFF | 4 planted effects × 2 | **17.7% / 16.9%** | 17.0% (2.5 d) | 32 | **FAIL (prevalence > 10%)**, retained and flagged |
| S-AR | RHO × 0.5 | 6.5% / 6.9% | 15.8% | 27 | pass |
| S-MISS | miss_mean 0.25, persist 0.75 | 6.9% / 7.2% (= R1) | **30.2% (3.9 d; max 34 d)** | 25 | pass (at the boundary) |
| S-HET | sens_sd 0.70 | 9.7% / 9.8% | 16.2% | 26 | pass |
| S-NOISE | sensor_sd 0.70 | 6.9% / 7.2% (= R1) | 15.8% | 25 | pass (at the boundary) |

**S-EFF gate failure.** With the frozen thresholds, doubling the effects raises latent variance. That more than doubles status prevalence and raises onset prevalence by about 50%. Per the approved policy, S-EFF was not reseeded, retuned or excluded. Every conclusion that depends on it is annotated, and the shift-effect rules require agreement on a prevalence-normalised scale (section 4).

**S-HET** prevalence also rose, to 9.7%, which is within the gate.

**Calibration prevalence by family:**

| Family | R1 | S-EFF | S-AR | S-MISS | S-HET | S-NOISE |
|---|---|---|---|---|---|---|
| status_now | 0.073 | 0.168 | 0.070 | 0.073 | 0.094 | 0.073 |
| onset_sust | 0.029 | 0.043 | 0.030 | 0.029 | 0.031 | 0.029 |
| onset_cross | 0.039 | 0.052 | 0.037 | 0.039 | 0.042 | 0.039 |

Test-split prevalence and event counts have **not** been looked at. They are reported only in the locked final evaluation.

## 3. Per-dataset model comparisons

These are the Phase A rules applied to each shifted dataset on its own. Absolute metrics are not compared across datasets.

Full tables (per seed, means, SD, lift, ranks):

- `phase_b_cal/family_summary.csv`
- `phase_b_cal/family_per_seed.csv`
- `phase_b_cal/paired.csv`
- `phase_b_cal/ranks.csv`

| Dataset | C1 primary (all models) | C1 no-GBT (sensitivity) | GRU rank-1 families (of 6) | C2: KAN / MLP consistently better (of 10) | C3: M1 better than GRU (of 6) | C4 | C5 |
|---|---|---|---|---|---|---|---|
| S-EFF ⚑ | mixed (GBT on onset_cross, +0.012 AP) | supported | 5 | supported (0 / 5) | supported (0) | supported | mixed |
| S-AR | supported | supported | 3 | supported (0 / 3) | supported (0) | supported | supported |
| S-MISS | supported | supported | 5 | supported (0 / 7) | supported (0) | supported | mixed |
| S-HET | supported | supported | 6 | supported (0 / 1) | supported (0) | supported | supported |
| S-NOISE | supported | supported | 6 | supported (0 / 6) | supported (0) | supported | supported |

⚑ = gate-flagged.

**C1 caveat on GBT.** GBT hyperparameters were selected per head on calibration AP and RMSE. Its calibration values are therefore optimistically biased relative to the neural models, which only had their epoch selected on calibration loss.

**No-GBT sensitivity for the replicates** (new in Phase B):

| Dataset | C1 no-GBT |
|---|---|
| v1 | supported |
| R1 | supported |
| R2 | mixed (KAN-reg on status_now) |
| R3 | supported |

The overall replication label for C1 changes from "not supported" (primary) to "mixed" (no-GBT).

**Notes:**

- In S-AR the GRU was rank 1 in only 3 of 6 families, but it was never *consistently* beaten.
- The KAN was never consistently better than its matched MLP in any shift.
- M1 was never better than the GRU in any family of any shift.

## 4. Shift-effect analysis (primary)

**Method.** I compared 36 paired effects per shift with the range of the same effect across the four replicates (v1, R1, R2, R3). An effect is labelled:

| Label | Rule |
|---|---|
| **shift-sensitive** | outside the replicate range on both the raw and the normalised scale (AP / prevalence, RMSE / target SD), and the seed-matched change versus R1 is consistent (3/3 seeds and \|mean\| > SD) |
| **suggestive** | outside on both scales, but the seed-matched change is not consistent |
| **inconclusive (scale-dependent)** | outside on one scale only |

**Label counts per shift (of 36):**

| Shift | Shift-sensitive | Suggestive | Inconclusive (scale) | Within replication noise |
|---|---|---|---|---|
| S-EFF ⚑ | 9 | 5 | **12** | 10 |
| S-AR | 7 | 8 | 3 | 18 |
| S-MISS | 5 | 9 | 1 | 21 |
| S-HET | 5 | 11 | 3 | 17 |
| S-NOISE | 3 | 8 | 0 | 25 |

**Caution on these counts.** With only four replicates, a fifth draw from the *same* distribution falls outside their range about 40% of the time. The seed-matched consistency requirement removes most of these, but with 36 effects per shift, a few shift-sensitive labels are expected by chance. S-NOISE has the fewest (3). These are descriptive patterns, not tests.

S-EFF's 12 scale-dependent effects reflect its prevalence change. Those effects cannot be attributed to the shift itself.

**Effects labelled shift-sensitive** (full list: `phase_b_cal/shift_effects.csv`):

- **S-EFF ⚑ (stronger effects):** the GRU's severity advantage widens.
  - M1 − GRU sev_now RMSE is +0.343 (replicates +0.137 to +0.206).
  - M1 − GRU sev_fc RMSE is +0.267 (+0.106 to +0.136).
  - KAN-reg − GRU sev_now is +0.173 (+0.047 to +0.096).
  - M1 − M1-MLP sev_now is +0.153 and sev_fc +0.131 (+0.020 to +0.068): the MLP's severity advantage over the KAN *widens*.
  - KAN-reg and B2-MLP fall further behind the GRU on onset_cross.
  - GBT closes the gap on sustained onset, and beats the GRU on onset_cross.
- **S-HET (heterogeneity):** M1, M1-MLP and KAN-reg fall further behind the GRU on status_now. For M1 the gap is −0.097 AP (replicates −0.057 to −0.034). M1-MLP also falls further behind on onset_sust, and on sev_fc.
- **S-AR (lower autocorrelation):**
  - M1 − GRU severity gap narrows: +0.124 (replicates +0.137 to +0.206).
  - KAN-reg − B2-MLP turns slightly positive on both onset families: +0.005 AP (replicates −0.011 to +0.003).
  - Rank order changed in status_fc.
- **S-MISS (heavier missingness):**
  - GBT matches the GRU on sev_now: −0.001 (replicates +0.016 to +0.052).
  - KAN-reg and M1 fall further behind on onset_cross; GBT falls further behind on onset_sust.
- **S-NOISE (sensor noise):** GBT falls further behind on sev_fc. M1 falls further behind on status_now. GBT's status_now gap sits at the edge of the replicate range.

**Rank robustness.** Rank order was shifted (Spearman correlation with the replicate mean rank below every replicate's leave-one-out value) in only 2 of 30 shift × family cells: S-AR status_fc and S-EFF onset_cross.

## 5. Updated conclusions

### Research questions (pre-registered labels)

| Question | Label | Evidence |
|---|---|---|
| **RQ3:** Do KANs gain relative to matched MLPs when planted effects are stronger? (S-EFF ⚑) | **Inconclusive** | Of the 10 KAN–MLP effects: 0 shift-sensitive in the KAN's favour, 2 shift-sensitive *against* it (M1 vs M1-MLP severity), 3 scale-dependent, 3 suggestive (mixed directions). The pre-registered "inconclusive" rule (n_inc ≥ 3) applies, compounded by the gate failure. There is **no** evidence of a KAN gain; the clear signals point the other way. |
| **RQ4a:** Does lag recovery improve with lower autocorrelation? (S-AR) | **Not supported** | Sign accuracy 1.00. Best exact-lag hit rate 0.42 is within the replicate range (0.28–0.47) and below the 0.50 constant-guess reference. |
| **RQ4b:** Does lag recovery improve with stronger effects? (S-EFF ⚑) | **Not supported** | Sign 1.00; hit rate 0.36, within range. |
| **RQ5:** Does the augmentation's severity benefit persist under heavier missingness? (S-MISS) | **Mixed** | Severity improved in 2/3 seeds (+0.54, −0.62, −0.59%; mean −0.22%). The magnitude is shift-sensitive, so **augmentation helps less**: replicates range −1.72 to −0.82%. Status and onset not distinguishable. C5 for S-MISS is mixed. |
| **RQ6a:** Does M1's per-participant latent help under heterogeneity? (S-HET) | **Not supported** | 0 of 6 M1-vs-GRU effects shift in M1's favour. 1 shifts against it (status_now, shift-sensitive). 1 is scale-dependent and 2 are suggestive. |

**Robustness of the replication conclusions across shifts.** Primary counts include all five shifts. The sensitivity column excludes S-EFF ⚑ (reported alongside, not instead).

| Conclusion | Replication result (Phase A, 4 datasets) | Shifts supported | Shift robustness | Excluding S-EFF |
|---|---|---|---|---|
| C1 GRU strongest (primary) | not supported (supported in 1/4; every loss on status AP) | 4/5 | **mixed** | supported (4/4) |
| C1 no-GBT (sensitivity) | mixed | 5/5 | supported | supported |
| C2 KAN not better than matched MLP | supported (4/4) | 5/5 | **supported** | supported |
| C3 M1 not better than GRU | supported (4/4) | 5/5 | **supported** | supported |
| C4 direction recovered more reliably than lags | supported (4/4) | 5/5 | **supported** | supported |
| C5 augmentation helps severity more than status or onset | mixed (3/4) | 3/5 | **mixed** | mixed (3/4) |

**C1 needs care.** It did *not* replicate across seeds, so a "mixed" label under shift is not evidence of a shift effect. In the shifts, the only consistent loss of the GRU is to the calibration-tuned GBT on S-EFF onset_cross. In the replicates, the losses were on status AP. Both are small (≤ 0.02 AP). The defensible statement is:

> The GRU is the strongest model for severity and onset in every dataset, and its status-AP lead is not robust to the dataset draw.

**C5 split in the shifts:**

- *Supported:* S-AR, S-HET, S-NOISE.
- *Mixed:* S-EFF, where onset also improved consistently.
- *Mixed:* S-MISS, where severity improved in only 2/3 seeds.

The severity benefit was larger than the replicate range under S-AR and S-NOISE, and smaller under S-MISS.

### Summary by evidence strength

- **Supported:**
  - C2, C3 and C4 replicate (4/4) and hold in every shift (5/5).
  - KANs do not beat matched MLPs.
  - M1 does not beat the GRU.
  - Directions are recovered but lags are not.
- **Mixed:**
  - C1 primary: the GRU's status lead depends on the draw, and GBT tuning bias is involved.
  - C5: the augmentation's severity benefit is shift-dependent and shrinks under heavier natural missingness (RQ5).
- **Not supported:**
  - RQ4a and RQ4b: neither lower autocorrelation nor stronger effects made lags recoverable.
  - RQ6a: M1 gains no advantage under heterogeneity; its status gap widens.
- **Inconclusive:** RQ3 (gate-flagged S-EFF; no KAN gain observed).

## 6. Runtime and storage

All runs were on CPU. Training used a 3-worker queue that scheduled the M1 jobs first.

| Item | Value |
|---|---|
| Training total | **23.2 CPU-h**, 10:57–13:26 (2 h 29 min elapsed) |
| — M1 (3 seeds × 5 datasets) | 964 CPU-min (69%) |
| — KAN-reg | 142 CPU-min |
| — M1-MLP | 78 CPU-min |
| — GBT | 85 CPU-min |
| — GRU (main + forecast) | 67 CPU-min |
| — GRU+aug | 34 CPU-min |
| — B2-MLP | 21 CPU-min |
| Lag recovery (5 datasets) | 4.6 min elapsed |
| Augmentation evaluation | 1.3 min elapsed |
| Generation, validation and analysis | ≈ 15 min |
| Storage: raw data | 266 MB |
| Storage: processed data | 97 MB |
| Storage: results | 184 MB |
| **Storage: Phase B total** | **≈ 550 MB** |
| **Storage: P2-2 total (Phases A + B)** | **≈ 880 MB** |

## 7. Limitations

- **One simulator family.** Replication and shifts show stability *within this generative model*, not external validity.
- **Single magnitude per shift on a single seed.** Results show the direction of sensitivity, not a dose–response.
- **Coarse replicate yardstick.** Four replicates give a coarse range, and some shift-sensitive labels are expected by chance (section 4).
- **S-EFF failed the prevalence gate.** Its absolute and AP-based effects are confounded with prevalence. That is why the scale-agreement rule applies and RQ3 is inconclusive.
- **Small onset samples.** S-MISS and S-NOISE (like R1) sit at the 25-participant onset gate in calibration, so their onset results are noisy.
- **Three model seeds per dataset.**
- **GBT optimism.** GBT calibration values are optimistically biased (section 3).
- **Calibration only.** None of these results has been confirmed on held-out test data.

## 8. Not done / next step (needs your approval)

- **Single locked test evaluation** (R1–R3 and the five shifts, plus v1 seeds 0–2 for reference). Before any test access, this needs:
  - a test-evaluation plan
  - test-evaluation code
  - a hash lock covering the plans, code, configs and checkpoints
- **Stage 5 observational regression on S-EFF and S-AR.** This was in the original P2-2 proposal table but was not run in either phase. The lag analysis used the P1-4 perturbation method only.
- **P2-3 and later packages** have not been started.
