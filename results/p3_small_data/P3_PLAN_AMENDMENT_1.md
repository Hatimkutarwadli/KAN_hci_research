# P3 plan amendment 1 (2026-11-17): G2 approval, Option F

**Basis.** The Phase 1 pilot triggered stop condition C1: the Standard design projected to 91.5 CPU-h, 9.8–10.3 h elapsed (`pilot/PILOT_REPORT.md`). At G2 the user approved **Option F**.

**Relation to the locked plan.** This amendment changes only the items listed below. Everything else in `P3_PLAN_LOCKED.md` (SHA-256 `f084f9e0…a4c1`, unchanged and read-only) remains in force: research questions, models, datasets, subsets, scaling, metrics, the 10 matched comparisons, decision rules, stop conditions and gates.

The amendment was written **before any main-experiment job was run** and is hash-stamped in `P3_PLAN_AMENDMENT_1.sha256`.

## 1. Design (replaces the table in §4 of the locked plan)

| n | Draws | Seeds per draw (neural models) | GBT | Role |
|---|---|---|---|---|
| 50 | 0–4 | 0, 1 | 1 run per draw | **primary decision size** (unchanged) |
| 100 | 0–4 | 0, 1 | 1 run per draw | **primary decision size** (unchanged) |
| 200 | 0–2 | 0 | 1 run per draw | descriptive learning-curve point (reduced from 5 × 2) |
| 400 | 0–2 | 0 | 1 run per draw | descriptive learning-curve point (reduced from 3 × 2) |
| 600 | 0 | 0, 1, 2 | 1 run | full-data anchor (unchanged) |

- All models are kept.
- Both datasets, R4 and R5, are kept.
- Subsets are unchanged: draws 0–2 at n = 200 and 400 are the same nested subsets as before.
- The projected cost is 68.4 CPU-h, about 7.4–7.7 h elapsed.

## 2. Verdict units for the descriptive sizes (clarification)

- **n = 200 and 400:** d_draw is the single-seed difference, and the verdict is taken across the 3 draws.
- **n = 600:** there is one draw, so the verdict is taken across the 3 seeds (the P2-2 convention).

These sizes are descriptive only. They enter no RQ1 label. RQ1a and RQ1b are evaluated at n = 50 and n = 100 only, each size independently (locked plan §7).

## 3. Sensitivity analyses

- **Order:** S2, then S3.
- **S1** runs only if the deferral rule permits it: the projected finish of all remaining steps must be ≤ 11:30 h after the Phase 1 start (18:03), i.e. by **05:33**.
- **The primary experiment is never reduced to make room for sensitivity analyses.**
- The specifications of S1–S3 are unchanged (locked plan §8).

## 4. Gates

- **G3 remains a hard gate.** Calibration experiments and analyses stop at G3.
- No test split is accessed without explicit approval.
