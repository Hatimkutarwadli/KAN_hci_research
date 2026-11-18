# P2-2 Phase B (distribution shifts) analysis plan: SYNTHETIC data

**Written:** 2026-11-17, while the Phase B models were training and **before any Phase B model result was viewed**.

**Plan hash:** SHA-256 recorded in `plan_phase_b_sha256.txt` at the time of writing.

**Scope:**
- Phase B uses the **calibration split only**.
- The test splits of all eight new datasets (R1–R3 and the five shifts) stay sealed.
- A single test evaluation needs separate approval and a further hash lock. Before any test access, that lock must cover the analysis plan, code, model configurations, comparison rules, and primary and sensitivity analyses.
- Each test split will be evaluated exactly once.

**Approved Phase B decisions (2026-11-17), applied here:**
- **Shifts and models:** all five shifts, the original model panel, frozen thresholds, and model seeds 0, 1, 2. M1 runs at three seeds on every dataset.
- **Gate failures:** every generated dataset is retained, flagged and reported. No dataset is reseeded, retuned or excluded.
- **Shift-effect yardstick:** shift effects are judged against the spread across **all four** Phase A replicate datasets (v1, R1, R2, R3).
- **C1 reporting:** the all-model result is primary. A no-GBT sensitivity is reported separately, with disclosure that GBT's calibration values are optimistically biased.
- **Separation:** replication conclusions and shift conclusions are kept separate. Absolute performance is not compared across shifted datasets.

## Datasets

All shifts use seed 3101 (streams 3101–3103) and are paired with R1 by common random numbers (CRN).

| ID | Change versus R1 | Held identical to R1 (verified by test) | Gate |
|---|---|---|---|
| S-EFF | the four planted `eff_*` × 2 | participants, split, noise draws | **fails** the prevalence gate (17.7% / 16.9%); retained and flagged |
| S-AR | RHO × 0.5 | participants, split, noise draws | pass |
| S-MISS | `miss_mean` 0.25, `miss_persist` 0.75 | participants, split, **truth and targets** | pass |
| S-HET | `sens_sd` 0.70 | participants, split, noise draws | pass |
| S-NOISE | `sensor_sd` 0.70 | participants, split, **truth and targets** | pass |

## Per-dataset analysis

Each shift is analysed with exactly the Phase A code and rules (`analyze_p22.py`):
- families and verdict labels (consistent / leaning / not distinguishable)
- paired comparisons
- C1–C5 labels; C4 uses lag recovery and C5 uses augmentation, both run on every shift
- prevalence and lift

**C1 has two versions:**
- **Primary (all models):** as in Phase A.
- **Sensitivity (no GBT):** GBT is removed from the set of models that can beat the GRU. It is also computed for the four replicates.

## Shift-effect analysis (primary)

**Effects.** For each dataset, family (both diseases averaged) and pair (A, B), the effect is E = seed-mean of (A − B). Pairs:
- every model vs GRU: 5 main-family pairs and 3 forecast-family pairs per family
- KAN-reg vs B2-MLP-reg (main families)
- M1 vs M1-MLP (all 6 families)

That gives 36 effects per dataset.

**Two scales**, because shifts change prevalence and task difficulty:
- (a) **Raw:** the metric difference.
- (b) **Normalised:**
  - AP: the difference divided by that dataset's calibration family prevalence (lift difference).
  - RMSE: the difference divided by that dataset's calibration target SD, pooled over the family's heads.

**Replicate reference.** The four replicate values of the same effect (v1, R1, R2, R3) on the same scale, using their [min, max] range.

**CRN-paired change.** Δ_s = d_S(seed s) − d_R1(seed s) for s = 0, 1, 2, judged with the standard verdict rule. GBT is deterministic, so its single value is paired with each GRU seed.

**Labels for each effect in each shift:**

| Label | Rule |
|---|---|
| **Shift-sensitive** (direction stated: favours A or favours B, relative to the replicates) | E_S lies outside the replicate range on **both** scales, on the same side, **and** Δ is consistent (3/3 same sign and \|mean\| > SD) in that direction |
| **Suggestive** | outside the range on both scales, but Δ is not consistent |
| **Inconclusive (scale-dependent)** | outside the range on exactly one scale |
| **Within replication noise** | inside the range on both scales |

Lag recovery (C4 quantities) and augmentation (C5 panel changes) are compared with the replicate range in the same way. Lag uses its single scale. Augmentation uses the relative-change percentage, which is already scale-free.

## Research-question conclusions

Pre-registered labels are **supported / mixed / not supported / inconclusive**. Any conclusion that depends on S-EFF also carries the "gate-flagged (prevalence 17%)" annotation.

**RQ3 (S-EFF): do KANs gain relative to their matched MLPs when planted effects are stronger?**
- Uses the 10 KAN–MLP effects in S-EFF.
- n+ = number shift-sensitive in the KAN's favour. n− = number shift-sensitive against the KAN. n_inc = number inconclusive.
- **Inconclusive** if n_inc ≥ 3.
- Otherwise:
  - **Supported:** n+ ≥ 3 and n+ > n−.
  - **Not supported:** n+ = 0.
  - **Mixed:** anything else.

**RQ4 (S-AR = RQ4a, S-EFF = RQ4b): does lag recovery improve when autocorrelation is lower or when effects are stronger?**
- H_max = max(level-peak exact-hit rate, increment-peak exact-hit rate), pooled over GRU, M1 and M1-MLP × 3 seeds × 4 pairs.
- **Supported:** H_max is above the replicate maximum **and** above 0.50 (the constant-guess reference).
- **Mixed:** above the replicate maximum but ≤ 0.50.
- **Not supported:** H_max ≤ the replicate maximum.
- Sign accuracy is reported alongside.

**RQ5 (S-MISS): does the Stage 9 augmentation's severity benefit persist under heavier natural missingness?**
- **Supported:** the severity panel improvement in S-MISS is consistent (3/3 seeds and |mean| > SD).
- **Mixed:** 2/3 seeds improved, or leaning.
- **Not supported:** ≤ 1/3 seeds improved.
- The magnitude is also compared with the replicate range (shift-sensitive or within noise). The status and onset effects and the S-MISS C5 label are reported.

**RQ6a (S-HET): does M1's per-participant latent help under stronger heterogeneity?**
- Uses the 6 M1-vs-GRU effects in S-HET.
- n+ = number shift-sensitive in M1's favour; n− and n_inc as for RQ3.
- **Inconclusive** if n_inc ≥ 3.
- Otherwise:
  - **Supported:** n+ ≥ 3 and n+ > n−.
  - **Mixed:** n+ = 1–2, or n+ ≥ 3 but n+ ≤ n−.
  - **Not supported:** n+ = 0.

**RQ1/2/6b: are the replication conclusions robust to each shift?**
- For each shift and each conclusion, report the per-dataset label (C1 primary and C1 no-GBT, C2, C3, C4, C5).
- Rank robustness is also reported:
  - For each family, compute the Spearman correlation between the shift's seed-mean model ranks and the mean rank over the four replicates.
  - For reference, compute each replicate's leave-one-out correlation with the mean of the other three.
  - A family is **rank-shifted** if the shift's correlation is below the minimum leave-one-out correlation.
- **Robustness of Ci across shifts:**
  - **Supported:** Ci is supported in 5/5 shifts.
  - **Mixed:** supported in 3–4.
  - **Not supported:** supported in ≤ 2.
- **Sensitivity:** the same counts excluding the gate-flagged S-EFF, shown alongside, never instead.
- Shift robustness is interpreted against the replication result for the same Ci. For example, C1 was already "not supported" across replicates, so a non-supported C1 under a shift is not evidence of a shift effect.

## Reporting rules
- Report per-seed values, seed means and SDs, replicate ranges and CRN Δ.
- Descriptive evidence only: no p-values.
- The four replicates and five single-seed shifts allow only coarse statements.
- Absolute metrics are never compared across datasets. Only within-dataset paired effects, normalised effects, ranks and lifts are compared.
- No real-world, clinical or causal-in-humans claims.
