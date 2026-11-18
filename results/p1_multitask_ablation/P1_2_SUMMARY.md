# P1-2: multitask vs single-task GRU ablation (SYNTHETIC; EXTENSION, not part of the original six stages)

Code `extension_p12/{common_p12,run_p12,compare_p12}.py`; tests `tests/test_p12.py` (9 pass); results `results/p1_multitask_ablation/{st_task_*,st_task_matched_*,family_*,comparison}`; full tables `comparison/p12_comparison.md`, `group_paired.csv`, `per_task_paired.csv`, `group_bootstrap_seed0.csv`, `cost_convergence.csv`.

Conditions (all: frozen data/preprocessing/splits, frozen GRUNet, frozen `fit_nn` protocol, 5 seeds, seed-paired):
- **MT**: existing multitask GRU (16 main / 12 forecast heads; not retrained).
- **ST**: one GRU per head (n_out = 1) with the frozen early stopping on its own calibration loss.
- **ST-matched**: one GRU per head trained for exactly the epochs the same-seed MT run selected (no early stopping: equal gradient steps, no selection advantage).
- **FAM**: one GRU per task family (nowcast severity [2 heads], nowcast status [2], sustained onset [6]; forecast severity [6], forecast status [6]).

Strict single-task covered 22 heads (10 main + 12 forecast); first-crossing onset heads were not run as single-task. Parameters: MT 10,864 / 10,732, ST 10,369 (-4.6%), FAM 10,446 / 10,534: only the last linear layer differs. Reproducibility: the harness reproduces the frozen Stage 3 epoch-1 losses exactly (tests), `fit_fixed` reproduces `fit_nn` trajectories exactly, and seeds/initialisation/batch order are the same for every condition.

## Findings (test; paired worse% of A vs B, NEGATIVE = A better; seed-0 participant bootstrap in brackets, * = excludes 0)
| Family | ST vs MT | ST-matched vs MT | ST vs ST-matched | FAM vs MT | ST vs FAM |
|---|---|---|---|---|---|
| Severity nowcast (RMSE) | +0.46 (0/5 seeds better) [+0.09, +2.07]* | +0.54 (0/5) | -0.08 (2/5) | -0.03 (2/5) | +0.49 (0/5) |
| Status nowcast (AP) | -2.93 (4/5) | -1.75 (3/5) | -1.19 | -4.43 (5/5) | +1.43 (1/5) |
| Sustained onset (AP) | +13.7 (0/5) [-3.3, +31.0] | +12.9 (0/5) | +0.8 | +11.4 (0/5) | +2.5 (1/5) |
| Sustained onset (AUROC) | +0.98 (0/5) | +0.72 (0/5) | +0.26 | +0.04 (2/5) | +0.93 (0/5) |
| Severity forecast (RMSE) | +0.76 (0/5) [+0.12, +1.45]* | +0.73 (0/5) | +0.03 | +0.30 (1/5) | +0.45 (0/5) |
| Status forecast (AP) | +7.24 (0/5) [+1.76, +14.0]* | +4.00 (0/5) | +3.37 (1/5) [+0.46, +9.4]* | -1.06 (3/5) | +8.13 (0/5) [+4.3, +18.6]* |

1. **Multitask learning helps, modestly, on four of five families; this is not explained by architecture, budget or epoch selection.** Single-task GRUs are worse than the multitask GRU on severity (+0.5 to +0.8% RMSE, 0/5 seeds better; nowcast and forecast bootstrap intervals exclude 0), on sustained-onset AP and AUROC (AP +13.7%, 0/5 seeds; intervals wide) and on forecast status AP (+7.2%). The budget-matched single-task models (same epochs and steps, no selection) show the same gaps, so extra epochs or selection are not the cause. The parameter difference is only the last layer (-4.6%).
2. **Where the benefit comes from:**
   - Severity nowcast and forecast status: sharing **within the family** (FAM) recovers the multitask level (FAM vs MT -0.03% / -1.06%), while ST is worse (+0.49% / +8.1%; the latter interval excludes 0).
   - Sustained onset AP: FAM is as far behind MT as ST is (+11.4% vs +13.7%), so the onset heads sharing a representation among themselves does not help; **the extra supervision from the other task families (nowcast severity/status and first-crossing onset heads) is what helps** (not separable further here). Onset AUROC differs: FAM matches MT (+0.04%) while ST is worse (+0.98%), so within-family sharing explains the AUROC difference.
   - Nowcast status AP: single-task and family models were nominally **better** than MT (ST -2.9%, FAM -4.4%, 5/5 seeds for FAM on test; also better on calibration), but AUROC is unchanged (±0.3%) and the bootstrap intervals include 0: at most weak evidence that the shared representation slightly hurts nowcast status AP (possible negative transfer).
3. **Selection/budget confound (forecast status):** about half of the single-task deficit (+7.2%) disappears when the epoch count is matched to MT (+4.0%). Single-head early stopping on its own calibration loss was worse on test than MT's epoch (ST vs ST-matched +3.4%, interval excludes 0) although better on calibration (-1.8%, 5/5 seeds). Selecting the epoch on one head's calibration loss is noisier and overfits the calibration split: a training-protocol effect, not a representation effect. For other families the selection effect is within noise.
4. **Not controlled:** a single-head model receives gradient only from the labelled share of each batch (onset heads 59-70% of rows, forecast heads 77-92%), whereas the multitask model receives gradient from every row through at least one head; ST-matched equalises gradient steps, not supervised samples per step. Head weighting inside MT (equal weights) was not varied. Calibration and test disagree on some effects (severity nowcast ST vs MT: -0.23%, 4/5 seeds better on calibration vs +0.46% on test), so effects below about 1% should be regarded as not established.

## Cost and convergence
CPU, 4 torch threads. MT/GRU figures are from the original runs; ST/FAM runs shared the laptop, throttled at times and included a hibernation gap, so CPU seconds are approximate and wall seconds unreliable.
- To cover all heads, MT needs **1 training per task set** (95 CPU-s main, 126 forecast). ST needs 10 / 12 trainings (about 1,434 / 1,697 CPU-s per seed, about 15x and 13x). FAM needs 3 / 2 trainings (386 / 239 CPU-s). ST-matched (fewer epochs, no selection) costs about 619 / 595 CPU-s per seed.
- Convergence: all runs stopped within 3-10 epochs. Single-task models overfit faster: calibration loss rises by +0.017 (main) and +0.035 (forecast) after the best epoch vs +0.004 / +0.013 for MT; 42% (main) and 53% (forecast) of ST models peaked at epoch <= 2 vs 0% / 60% for MT. Consistent with the multitask objective acting as a regulariser, but not separated from the noisier single-head selection signal.

## Limitations
One synthetic dataset, one architecture (the frozen GRU), 5 seeds, seed-0 bootstrap only (wide intervals; onset has 32 events per target); the strict single-task set excludes first-crossing onset; family groupings and equal head weights are our choices; calibration data used for selection; the test split was inspected in earlier stages and not used for any decision here. An initial attempt with three parallel processes was stopped because of CPU contention; its partial outputs are kept in `results/p1_multitask_ablation/_interrupted_parallel_attempt/` and are not used.
