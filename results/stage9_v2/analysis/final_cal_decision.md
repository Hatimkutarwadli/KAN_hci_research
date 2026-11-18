# Stage 9 final selection - CALIBRATION only (SYNTHETIC; 5 paired seeds). Reference = R0O0T1. No overall winner is declared.

worse% = (new-ref)/ref (RMSE) or (ref-new)/ref (AP/AUROC); NEGATIVE = better. n = seeds (of 5) better.


## severity (nowcast)  (part=main; clean margins: sev_nowcast_rmse 0.21%)

| config | clean worse% vs original (margin test) | panel worse% vs R0O0T1 (n better) | S3_k14 worse% vs R0O0T1 | S3_k14 worse% vs original | clean worse% R0O0T1-ref |
|---|---|---|---|---|---|
| R0O0T1 | sev_nowcast_rmse -0.12 -> pass | (reference) | +nan | -4.29 | |
| R0O2T1 | sev_nowcast_rmse -0.12 -> pass | -0.29 (5/5) | -0.70 | -4.96 | |
| R0O3T1 | sev_nowcast_rmse -0.12 -> pass | -0.17 (5/5) | +0.02 | -4.27 | |

**Selected for severity (nowcast): R0O2T1** (met rule vs R0O0T1: ['R0O2T1', 'R0O3T1'])

## status (nowcast)  (part=main; clean margins: status_ap 2.22%)

| config | clean worse% vs original (margin test) | panel worse% vs R0O0T1 (n better) | S3_k14 worse% vs R0O0T1 | S3_k14 worse% vs original | clean worse% R0O0T1-ref |
|---|---|---|---|---|---|
| R0O0T1 | status_ap -0.05 -> pass | (reference) | +nan | -0.66 | |
| R0O2T1 | status_ap -0.64 -> pass | -2.22 (5/5) | +1.77 | +1.13 | |
| R0O3T1 | status_ap -0.17 -> pass | -4.66 (5/5) | -6.15 | -6.83 | |

**Selected for status (nowcast): R0O3T1** (met rule vs R0O0T1: ['R0O2T1', 'R0O3T1'])

## onset  (part=main; clean margins: onset_ap 5.21%, onset_auroc 0.28%)

| config | clean worse% vs original (margin test) | panel worse% vs R0O0T1 (n better) | S3_k14 worse% vs R0O0T1 | S3_k14 worse% vs original | clean worse% R0O0T1-ref |
|---|---|---|---|---|---|
| R0O0T1 | onset_ap -2.73, onset_auroc +0.05 -> pass | (reference) | +nan | -2.61 | |
| R0O2T1 | onset_ap -1.96, onset_auroc -0.02 -> pass | -0.28 (3/5) | -0.81 | -3.44 | |
| R0O3T1 | onset_ap -1.35, onset_auroc +0.22 -> pass | -0.02 (2/5) | -2.46 | -5.16 | |

**Selected for onset: R0O0T1** (default: no finalist met the pre-specified improvement rule)

## forecast severity  (part=forecast; clean margins: sev_forecast_rmse 0.28%)

| config | clean worse% vs original (margin test) | panel worse% vs R0O0T1 (n better) | S3_k14 worse% vs R0O0T1 | S3_k14 worse% vs original | clean worse% R0O0T1-ref |
|---|---|---|---|---|---|
| R0O0T1 | sev_forecast_rmse -0.06 -> pass | (reference) | +nan | -2.43 | |
| R0O2T1 | sev_forecast_rmse -0.07 -> pass | -0.36 (5/5) | -0.86 | -3.27 | |
| R0O3T1 | sev_forecast_rmse -0.05 -> pass | -0.42 (5/5) | -0.94 | -3.35 | |

**Selected for forecast severity: R0O2T1 or R0O3T1 (not separable; both beat R0O0T1)** (met rule vs R0O0T1: ['R0O2T1', 'R0O3T1'])

## forecast status  (part=forecast; clean margins: status_ap 2.00%)

| config | clean worse% vs original (margin test) | panel worse% vs R0O0T1 (n better) | S3_k14 worse% vs R0O0T1 | S3_k14 worse% vs original | clean worse% R0O0T1-ref |
|---|---|---|---|---|---|
| R0O0T1 | status_ap -0.58 -> pass | (reference) | +nan | -1.79 | |
| R0O2T1 | status_ap -0.78 -> pass | -2.01 (5/5) | -1.41 | -3.22 | |
| R0O3T1 | status_ap +1.20 -> pass | -3.88 (5/5) | -8.88 | -10.82 | |

**Selected for forecast status: R0O3T1** (met rule vs R0O0T1: ['R0O2T1', 'R0O3T1'])

## Caveat (not a selection criterion): onset risk inflation, ratio of mean predicted onset probability to own clean control

| scenario | original GRU | R0O0T1 | R0O2T1 | R0O3T1 |
|---|---|---|---|---|
| S1_r40 | 1.18 | 1.14 | 1.13 | 1.10 |
| S2_r40_L7 | 1.09 | 1.04 | 1.07 | 1.08 |
| S3_k7 | 0.97 | 0.92 | 1.04 | 1.04 |
| S3_k14 | 1.06 | 0.96 | 1.03 | 1.15 |
| S5_blk40 | 0.91 | 0.89 | 0.93 | 0.98 |
