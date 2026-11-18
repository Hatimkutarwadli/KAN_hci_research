# KAN regularization pass (SYNTHETIC; selection on calibration only)

Pre-declared grid: weight decay {1e-5(original), 1e-3, 1e-2} x dropout {0, 0.3}; selection seeds {0,1}; criterion = mean best CALIBRATION loss (all 16 heads). The test split was never loaded by the selection code.

| weight decay | dropout | mean best cal loss | mean best epoch | train-minus-cal loss at last epoch | cal AP status (mean of 2) | cal AP sustained onset (mean of 6) |
|---|---|---|---|---|---|---|
| 1e-05 | 0 (original) | 0.2272 | 1.5 | -0.162 | 0.280 | 0.113 |
| 1e-05 | 0.3 | 0.2269 | 2.5 | -0.078 | 0.286 | 0.116 |
| 0.001 | 0 | 0.2278 | 1.5 | -0.133 | 0.281 | 0.114 |
| 0.001 | 0.3 **(chosen)** | 0.2262 | 2.5 | -0.055 | 0.272 | 0.114 |
| 0.01 | 0 | 0.2335 | 5.0 | -0.081 | 0.251 | 0.121 |
| 0.01 | 0.3 | 0.2373 | 1.5 | -0.010 | 0.272 | 0.109 |

## Original (wd 1e-5, no dropout) vs regularized (wd 1e-3, dropout 0.3), TEST split, mean ±SD over 5 seeds (final evaluation only)

| task | metric | original | regularized |
|---|---|---|---|
| sev_dep_t0 | rmse | 3.019 ±0.026 | 3.025 ±0.013 |
| sev_anx_t0 | rmse | 2.930 ±0.012 | 2.946 ±0.024 |
| status_dep_t0 | ap | 0.350 ±0.016 | 0.346 ±0.013 |
| status_dep_t0 | auroc | 0.869 ±0.005 | 0.867 ±0.003 |
| status_anx_t0 | ap | 0.315 ±0.020 | 0.305 ±0.009 |
| status_anx_t0 | auroc | 0.836 ±0.005 | 0.833 ±0.003 |
| onset_sust_dep_h7 | ap | 0.042 ±0.004 | 0.048 ±0.006 |
| onset_sust_dep_h7 | auroc | 0.746 ±0.039 | 0.762 ±0.013 |
| onset_sust_dep_h14 | ap | 0.070 ±0.005 | 0.074 ±0.007 |
| onset_sust_dep_h14 | auroc | 0.741 ±0.010 | 0.745 ±0.006 |
| onset_sust_dep_h21 | ap | 0.096 ±0.004 | 0.099 ±0.006 |
| onset_sust_dep_h21 | auroc | 0.727 ±0.008 | 0.730 ±0.007 |
| onset_sust_anx_h7 | ap | 0.080 ±0.020 | 0.091 ±0.007 |
| onset_sust_anx_h7 | auroc | 0.705 ±0.109 | 0.750 ±0.014 |
| onset_sust_anx_h14 | ap | 0.104 ±0.012 | 0.106 ±0.007 |
| onset_sust_anx_h14 | auroc | 0.744 ±0.031 | 0.751 ±0.010 |
| onset_sust_anx_h21 | ap | 0.123 ±0.009 | 0.125 ±0.006 |
| onset_sust_anx_h21 | auroc | 0.738 ±0.027 | 0.739 ±0.006 |
| onset_cross_dep_h7 | ap | 0.061 ±0.008 | 0.068 ±0.009 |
| onset_cross_dep_h7 | auroc | 0.760 ±0.016 | 0.765 ±0.013 |
| onset_cross_dep_h14 | ap | 0.095 ±0.007 | 0.100 ±0.009 |
| onset_cross_dep_h14 | auroc | 0.745 ±0.010 | 0.751 ±0.006 |
| onset_cross_dep_h21 | ap | 0.115 ±0.008 | 0.120 ±0.006 |
| onset_cross_dep_h21 | auroc | 0.708 ±0.007 | 0.718 ±0.006 |
| onset_cross_anx_h7 | ap | 0.057 ±0.005 | 0.066 ±0.006 |
| onset_cross_anx_h7 | auroc | 0.706 ±0.019 | 0.711 ±0.005 |
| onset_cross_anx_h14 | ap | 0.093 ±0.007 | 0.096 ±0.003 |
| onset_cross_anx_h14 | auroc | 0.713 ±0.003 | 0.705 ±0.005 |
| onset_cross_anx_h21 | ap | 0.119 ±0.004 | 0.122 ±0.006 |
| onset_cross_anx_h21 | auroc | 0.709 ±0.008 | 0.701 ±0.008 |

## Summary over the six primary onset tasks (test)

| metric | original | regularized |
|---|---|---|
| mean AP | 0.0860 | 0.0904 |
| mean AUROC | 0.7333 | 0.7462 |
| mean across-seed SD of AUROC | 0.037 | 0.009 |