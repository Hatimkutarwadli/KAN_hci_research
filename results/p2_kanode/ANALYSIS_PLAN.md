# P2-1 five-seed analysis plan (fixed BEFORE any test-split access; SYNTHETIC data)

Written 2026-11-16 while seeds 1-4 were training, after the seed-0 pilot (calibration only). Its SHA-256, and that of `extension_p21/analyze_p21.py`, are stored in `results/p2_kanode/plan_lock.json` before the test evaluation. The test command refuses to run if either file has changed since locking.

## Models (5 seeds each: 0 from `pilot/`, unchanged and hash-checked; 1-4 from `full/`)
- KAN-ODE (`kanode`, 28 heads), no-flow ablation (`kanode_noflow`, 28 heads), GRU-D (`grud`, main 16 heads + forecast 12 heads).
- References, existing saved predictions only, never retrained: M1 (`results/stage4_m1`, 28 heads), GRU main (`results/stage3`), GRU forecast (`results/stage3c_forecast`).
- Architecture, hyperparameters and S = 1 are fixed. Checkpoints are those selected on calibration loss; nothing is re-selected after test.
- **Deviation from the originally approved proposal:** vector field kappa(zbar - z + 2 tanh(K_f(z))) instead of -kappa(z - zbar) + tanh(K_f(z)), to keep the exact flow inside the spline grid.

## Metrics and families (computed from predictions for every model, including references)
Per seed and per disease (dep, anx, and their mean):
- Severity nowcast RMSE (`sev_k_t0`)
- Severity forecast RMSE (`sev_k_h7/14/21`, mean)
- Status nowcast AP and AUROC (`status_k_t0`)
- Status forecast AP and AUROC (`status_k_h*`, mean)
- Sustained-onset AP and AUROC (`onset_sust_k_h*`, mean)
- First-crossing onset AP and AUROC (`onset_cross_k_h*`, mean)

GRU / GRU-D: nowcast and onset families from the main models, forecast families from the forecast models.

## Missing-day strata
Days since the last observed day (missing_day = 0) at the origin, computed causally from the full record: 0, 1-2, >= 3. Severity nowcast RMSE and status nowcast AP, separately for depression and anxiety. Onset is not stratified (too few positives).

## Comparisons (seed-matched pairs, same seed index)
1. KAN-ODE vs no-flow: **primary** (flow vs gating).
2. KAN-ODE vs M1.
3. No-flow vs M1.
4. GRU-D vs GRU.

Paired difference d = A - B per seed. Reported: per-seed values, mean, SD (ddof 1), min, max, and the number of seeds (of 5) in which A is better (lower RMSE, higher AP/AUROC). Same seed index means the same RNG seed, not an identical initialisation of different architectures.

## Inference rules
No p-values and no significance claims. A difference is called **consistent** only if all 5 seeds have the same sign *and* |mean d| > SD(d). It is called **leaning** if 4 of 5 seeds have the same sign, or if all 5 do but |mean d| <= SD(d). Everything else is **not distinguishable**. Five seeds is limited evidence; calibration and test are both reported; calibration was also used for epoch selection.

## Stability (all seeds)
Per seed: max|z| (train, max over epochs; calibration at the final model), fraction of K_f / K_g latent inputs outside [-3, 3] (max over epochs), non-finite counts, kappa (min/max at the selected checkpoint and max over training), S=1 vs S=4 latent/output discrepancy (selected checkpoint and max over epochs), best epoch, epochs run, wall and CPU time.

## Test evaluation (once)
After all 20 new runs exist, the plan is locked, and the calibration analysis is written: the 20 selected checkpoints (5 seeds x {kanode, kanode_noflow, grud main, grud forecast}) predict the test split once. The same analysis code then runs on test. No checkpoint, threshold or setting is changed afterwards.
