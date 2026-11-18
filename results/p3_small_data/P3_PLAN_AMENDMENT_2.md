# P3 plan amendment 2 (2026-11-17): RQ1a rule-gap classification

**Basis.** While I was writing the calibration analysis code, and **before any main-experiment result existed or was analysed**, I found that the locked RQ1a rule (`P3_PLAN_LOCKED.md` §7) did not classify two possible outcome patterns. The user approved the following classifications.

**What stays unchanged.** `P3_PLAN_LOCKED.md` (SHA-256 `f084f9e0…a4c1`) and `P3_PLAN_AMENDMENT_1.md` (SHA-256 `75cfd693…79db`) remain in force and unchanged. This amendment is hash-stamped in `P3_PLAN_AMENDMENT_2.sha256`.

## RQ1a: complete rule (applied separately at n = 50 and at n = 100)

| Symbol | Meaning |
|---|---|
| nK | number of the 10 matched comparisons with a **reproducible** KAN advantage (consistent in both R4 and R5) |
| nM | the same count for reproducible **MLP** advantages |
| c_R4, c_R5 | number of comparisons with a consistent KAN advantage in that dataset |
| s | number of comparisons with a consistent KAN advantage in exactly one dataset |

Labels are assigned in this order:

1. **Supported:** nK ≥ 3 and nK > nM. *(unchanged)*
2. **Mixed:** any of the following.
   - 1 ≤ nK ≤ 2 *(unchanged)*
   - s ≥ 3 *(unchanged)*
   - **(new, b)** nK ≥ 3 and nM ≥ nK
   - **(new, a)** nK = 0, and c_R4 ≥ 2 or c_R5 ≥ 2. That is, at least two consistent KAN wins within a single dataset, none of them reproduced.
3. **Not supported:** nK = 0, c_R4 ≤ 1 and c_R5 ≤ 1. *(unchanged)*

With these additions the rule is exhaustive: every outcome receives exactly one label.

RQ1b, all other rules, the design (amendment 1) and the gates are unchanged. G3 remains a hard stop.
