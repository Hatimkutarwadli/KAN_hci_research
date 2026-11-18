# Stage 10 / Phase 1 (P0-1 to P0-4) - summary (SYNTHETIC data; saved predictions only, nothing retrained)

Code: `extension_stage10/{common,p01_subgroup,p02_calibration,p03_ensemble,p04_oracle}.py`; tests: `tests/test_stage10.py` (14 pass). Outputs: `results/stage10_subgroup/`, `stage10_calibration/`, `stage10_ensemble/`, `stage10_oracle/` (each with config.json). Test split used for final reporting only; no selection used it. All 1,722 pre-existing files hash-verified unchanged.

## P0-1 Subgroup performance (test, 200 participants: F 133 / M 67; age 0/1/2 = 84/104/12)
- Severity RMSE is near-identical across sex and age (GRU nowcast 2.66-3.03; gaps <=0.05 by sex). The age-2 group (noisier sensors) is NOT worse (12 participants: UNRELIABLE).
- Onset/status AP differ by sex: female higher in all 6 onset heads and status (e.g. onset_anx_h14 AP 0.169 F vs 0.086 M; same sign in 5/5 seeds), consistent with the simulator's higher female baseline/prevalence; AP depends on prevalence, so read with the prevalence and AUROC columns. Few bootstrap gap CIs exclude 0 (27/500, mostly age group 2 with 9-12 participants).
- Lead time (20% false-alarm target, GRU): sex F 0.41/0.29 vs M 0.22/0.28 (anx/dep); age 2 and male-dep not estimable (<10 onset participants).
- Calibration (ECE, Brier) is similar across groups except age 2 (UNRELIABLE).
## P0-2 Calibration and conformal sets (status + sustained onset; 14 heads)
- Raw GRU probabilities are already well calibrated (ECE 0.007-0.013, slope 0.91-1.02). M1 status_dep_t0 is over-confident (slope 1.36); Platt fitted on calibration fixes it. Isotonic recalibration reduces AP (tied scores) and is not recommended.
- Marginal conformal sets are degenerate for rare events: coverage of POSITIVES only 0.00-0.18 (overall coverage 0.90-0.97 is carried by negatives; sets almost always {0} or empty).
- Mondrian (class-conditional) sets reach positive coverage 0.88-0.99 (GRU, 90% nominal) but at a large cost: 37-62% of sets are {0,1} (mean size 1.37-1.62) - i.e. the sets are often uninformative. About 13-20 calibration positives per head, so positive-class quantiles are coarse; one-origin-per-participant draws are noisy (see repeat mean vs primary draw).
## P0-3 Seed-ensemble intervals (severity; 5 families x 8 tasks)
- Ensemble mean slightly lowers RMSE (e.g. GRU 3.067 vs 3.075 single).
- Seed disagreement does NOT predict error: Spearman(sigma, |residual|) = -0.01...0.01. Normalised intervals therefore give no benefit: same coverage (0.896-0.899) but wider (GRU 10.6 vs 10.2) and worse Winkler score (13.4 vs 12.7); conditional coverage is skewed (low-sigma tertile 0.83, high 0.96). Constant-width ensemble intervals are equivalent to the single-seed Stage 6 intervals. Severity uncertainty is dominated by irreducible noise (sd 1.0 plus future shocks), not epistemic variance. Negative result.
## P0-4 Oracle ceiling (HistGradientBoosting on true latent state; features from days <= t only; trained on train, n_iter chosen on calibration)
- Nowcast severity RMSE: oracle 1.00 (= simulator noise floor) vs GRU 2.97/2.91. Status at t0: oracle AP=1.0 (status is a deterministic threshold of z). Forecast RMSE 1.7-2.8 vs GRU 2.9-3.3.
- Sustained-onset AP: oracle 0.21-0.39 vs GRU 0.08-0.15; AUROC 0.83-0.97 vs 0.75-0.83. GRU closes roughly 27-63% of the AP gap and 66-84% of the AUROC gap from chance to oracle. Lead time (h21, 10% false alarms, lead >=1 d): oracle A 0.59/0.72 vs GRU 0.15/0.19 (dep/anx).
- So the weak onset performance is mostly an INFORMATION limit of noisy wearables vs the true latent, but a sizeable gap remains that a better model/features could in principle narrow; the oracle is an estimate of headroom, not a proven bound (Tier B sometimes scores below Tier A on onset = overfit, so the headroom table that uses Tier B understates it; Tier A is better for onset).
## Limitations
One synthetic dataset; 5 seeds; 32 onset participants/target in test; many subgroup cells are underpowered (flagged); calibration split was also used for model selection/thresholds in earlier stages; conformal sets assume exchangeability of one origin per participant, not a time-series guarantee; Venn-Abers and adaptive/time-series conformal were not implemented (P2-5); oracle uses simulator internals and cannot be obtained in practice; no clinical or causal claims.
