# P1-2: multitask vs single-task GRU ablation (SYNTHETIC; EXTENSION, not part of the original six stages)

MT = existing multitask GRU (16 main heads / 12 forecast heads; not retrained). ST = one GRU per head (n_out=1), frozen early stopping on its own calibration loss. ST-matched = one GRU per head trained for exactly the epochs the same-seed MT run selected (no early stopping; equal gradient steps). FAM = one GRU per task family (heads of one family share a representation). Same data, splits, protocol, 5 seeds, seed-paired.
Cell `x ±sd`: mean ±SD over 5 seeds. worse% of A vs B: NEGATIVE = A better; `(n/5)` = seeds where A is better.


## Per-task test results (mean ±SD over seeds) and paired worse% vs MT

| task | metric | MT | ST | ST-matched | FAM | ST vs MT | ST-matched vs MT | FAM vs MT |
|---|---|---|---|---|---|---|---|---|
| sev_dep_t0 | rmse | 2.967 ±0.011 | 2.984 ±0.008 | 2.986 ±0.012 | 2.976 ±0.008 | +0.6 (0/5) | +0.6 (1/5) | +0.3 (1/5) |
| sev_anx_t0 | rmse | 2.907 ±0.008 | 2.917 ±0.004 | 2.920 ±0.012 | 2.896 ±0.010 | +0.3 (1/5) | +0.4 (0/5) | -0.4 (4/5) |
| status_dep_t0 | ap | 0.361 ±0.010 | 0.368 ±0.018 | 0.360 ±0.012 | 0.385 ±0.007 | -1.9 (3/5) | +0.1 (3/5) | -6.6 (5/5) |
| status_dep_t0 | auroc | 0.870 ±0.002 | 0.874 ±0.003 | 0.873 ±0.002 | 0.874 ±0.002 | -0.5 (5/5) | -0.3 (4/5) | -0.5 (5/5) |
| status_anx_t0 | ap | 0.324 ±0.002 | 0.337 ±0.005 | 0.337 ±0.008 | 0.331 ±0.007 | -4.1 (5/5) | -3.9 (5/5) | -2.1 (3/5) |
| status_anx_t0 | auroc | 0.849 ±0.002 | 0.844 ±0.002 | 0.842 ±0.003 | 0.844 ±0.005 | +0.6 (0/5) | +0.9 (0/5) | +0.6 (1/5) |
| onset_sust_dep_h7 | ap | 0.083 ±0.011 | 0.066 ±0.003 | 0.064 ±0.004 | 0.060 ±0.006 | +18.9 (0/5) | +21.8 (0/5) | +27.4 (0/5) |
| onset_sust_dep_h7 | auroc | 0.798 ±0.005 | 0.780 ±0.005 | 0.779 ±0.006 | 0.789 ±0.005 | +2.2 (0/5) | +2.4 (0/5) | +1.2 (0/5) |
| onset_sust_dep_h14 | ap | 0.105 ±0.007 | 0.093 ±0.012 | 0.091 ±0.013 | 0.086 ±0.005 | +10.8 (1/5) | +12.6 (1/5) | +17.6 (0/5) |
| onset_sust_dep_h14 | auroc | 0.770 ±0.005 | 0.756 ±0.005 | 0.756 ±0.002 | 0.760 ±0.004 | +1.9 (0/5) | +1.9 (0/5) | +1.3 (0/5) |
| onset_sust_dep_h21 | ap | 0.123 ±0.003 | 0.102 ±0.010 | 0.100 ±0.008 | 0.110 ±0.003 | +17.3 (0/5) | +18.6 (0/5) | +10.4 (0/5) |
| onset_sust_dep_h21 | auroc | 0.749 ±0.005 | 0.731 ±0.008 | 0.731 ±0.004 | 0.742 ±0.007 | +2.4 (0/5) | +2.4 (0/5) | +1.0 (1/5) |
| onset_sust_anx_h7 | ap | 0.100 ±0.007 | 0.092 ±0.015 | 0.099 ±0.010 | 0.093 ±0.016 | +7.7 (2/5) | +0.7 (3/5) | +6.5 (2/5) |
| onset_sust_anx_h7 | auroc | 0.825 ±0.006 | 0.823 ±0.006 | 0.829 ±0.007 | 0.836 ±0.005 | +0.2 (2/5) | -0.5 (5/5) | -1.3 (5/5) |
| onset_sust_anx_h14 | ap | 0.132 ±0.006 | 0.111 ±0.013 | 0.114 ±0.011 | 0.121 ±0.012 | +16.0 (0/5) | +13.9 (0/5) | +8.6 (0/5) |
| onset_sust_anx_h14 | auroc | 0.798 ±0.002 | 0.802 ±0.010 | 0.806 ±0.003 | 0.807 ±0.006 | -0.5 (4/5) | -1.0 (5/5) | -1.2 (5/5) |
| onset_sust_anx_h21 | ap | 0.152 ±0.010 | 0.135 ±0.005 | 0.137 ±0.009 | 0.145 ±0.008 | +10.4 (1/5) | +9.1 (1/5) | +4.6 (2/5) |
| onset_sust_anx_h21 | auroc | 0.781 ±0.004 | 0.784 ±0.008 | 0.786 ±0.006 | 0.785 ±0.004 | -0.3 (3/5) | -0.6 (4/5) | -0.5 (5/5) |
| sev_dep_h7 | rmse | 3.045 ±0.007 | 3.062 ±0.026 | 3.052 ±0.018 | 3.058 ±0.017 | +0.6 (1/5) | +0.2 (2/5) | +0.4 (1/5) |
| sev_dep_h14 | rmse | 3.190 ±0.007 | 3.220 ±0.024 | 3.212 ±0.015 | 3.200 ±0.010 | +0.9 (1/5) | +0.7 (0/5) | +0.3 (1/5) |
| sev_dep_h21 | rmse | 3.334 ±0.008 | 3.374 ±0.019 | 3.369 ±0.012 | 3.345 ±0.004 | +1.2 (0/5) | +1.1 (0/5) | +0.3 (1/5) |
| sev_anx_h7 | rmse | 2.932 ±0.007 | 2.947 ±0.012 | 2.949 ±0.018 | 2.944 ±0.020 | +0.5 (1/5) | +0.6 (1/5) | +0.4 (1/5) |
| sev_anx_h14 | rmse | 3.054 ±0.003 | 3.074 ±0.013 | 3.081 ±0.013 | 3.060 ±0.005 | +0.7 (0/5) | +0.9 (0/5) | +0.2 (1/5) |
| sev_anx_h21 | rmse | 3.174 ±0.005 | 3.192 ±0.004 | 3.201 ±0.019 | 3.177 ±0.007 | +0.6 (0/5) | +0.9 (0/5) | +0.1 (1/5) |
| status_dep_h7 | ap | 0.351 ±0.014 | 0.320 ±0.021 | 0.331 ±0.009 | 0.352 ±0.015 | +8.8 (1/5) | +5.7 (0/5) | -0.5 (3/5) |
| status_dep_h7 | auroc | 0.854 ±0.003 | 0.850 ±0.005 | 0.852 ±0.002 | 0.854 ±0.003 | +0.4 (1/5) | +0.2 (2/5) | +0.1 (2/5) |
| status_dep_h14 | ap | 0.288 ±0.011 | 0.266 ±0.011 | 0.268 ±0.014 | 0.295 ±0.024 | +7.5 (0/5) | +6.8 (0/5) | -2.6 (3/5) |
| status_dep_h14 | auroc | 0.828 ±0.001 | 0.821 ±0.005 | 0.823 ±0.004 | 0.826 ±0.003 | +0.8 (0/5) | +0.6 (0/5) | +0.2 (2/5) |
| status_dep_h21 | ap | 0.234 ±0.008 | 0.179 ±0.012 | 0.208 ±0.014 | 0.240 ±0.017 | +23.3 (0/5) | +10.8 (0/5) | -2.8 (4/5) |
| status_dep_h21 | auroc | 0.796 ±0.003 | 0.768 ±0.010 | 0.782 ±0.004 | 0.793 ±0.003 | +3.6 (0/5) | +1.8 (0/5) | +0.4 (1/5) |
| status_anx_h7 | ap | 0.300 ±0.014 | 0.309 ±0.004 | 0.306 ±0.005 | 0.308 ±0.020 | -3.2 (4/5) | -2.4 (3/5) | -2.9 (5/5) |
| status_anx_h7 | auroc | 0.826 ±0.008 | 0.829 ±0.005 | 0.829 ±0.004 | 0.826 ±0.006 | -0.3 (2/5) | -0.3 (3/5) | +0.1 (2/5) |
| status_anx_h14 | ap | 0.265 ±0.009 | 0.256 ±0.006 | 0.262 ±0.016 | 0.262 ±0.014 | +3.5 (0/5) | +1.3 (1/5) | +1.4 (2/5) |
| status_anx_h14 | auroc | 0.803 ±0.005 | 0.798 ±0.005 | 0.802 ±0.007 | 0.798 ±0.006 | +0.5 (2/5) | +0.2 (3/5) | +0.6 (0/5) |
| status_anx_h21 | ap | 0.233 ±0.006 | 0.220 ±0.004 | 0.228 ±0.013 | 0.230 ±0.012 | +5.5 (0/5) | +2.2 (1/5) | +1.0 (4/5) |
| status_anx_h21 | auroc | 0.780 ±0.004 | 0.779 ±0.005 | 0.781 ±0.007 | 0.774 ±0.004 | +0.2 (2/5) | -0.1 (3/5) | +0.7 (1/5) |

## Group-level paired comparisons, test split (mean over the group's tasks; worse% of A vs B, NEGATIVE = A better; seeds where A better)

| group | metric | ST vs MT | ST-matched vs MT | ST vs ST-matched | FAM vs MT | ST vs FAM |
|---|---|---|---|---|---|---|
| Severity nowcast | rmse | +0.46 ±0.37 (0/5) | +0.54 ±0.36 (0/5) | -0.08 ±0.29 (2/5) | -0.03 ±0.22 (2/5) | +0.49 ±0.32 (0/5) |
| Status nowcast | ap | -2.93 ±3.55 (4/5) | -1.75 ±3.43 (3/5) | -1.19 ±2.78 (3/5) | -4.43 ±1.72 (5/5) | +1.43 ±3.08 (1/5) |
| Status nowcast | auroc | +0.04 ±0.22 (2/5) | +0.28 ±0.27 (1/5) | -0.24 ±0.25 (4/5) | +0.05 ±0.27 (2/5) | -0.01 ±0.17 (2/5) |
| Sustained onset | ap | +13.67 ±5.73 (0/5) | +12.86 ±4.86 (0/5) | +0.81 ±6.33 (2/5) | +11.36 ±6.02 (0/5) | +2.52 ±4.30 (1/5) |
| Sustained onset | auroc | +0.98 ±0.30 (0/5) | +0.72 ±0.37 (0/5) | +0.26 ±0.31 (1/5) | +0.04 ±0.43 (2/5) | +0.93 ±0.41 (0/5) |
| Severity forecast | rmse | +0.76 ±0.37 (0/5) | +0.73 ±0.30 (0/5) | +0.03 ±0.45 (2/5) | +0.30 ±0.25 (1/5) | +0.45 ±0.19 (0/5) |
| Status forecast | ap | +7.24 ±2.38 (0/5) | +4.00 ±0.61 (0/5) | +3.37 ±2.82 (1/5) | -1.06 ±2.61 (3/5) | +8.13 ±4.56 (0/5) |
| Status forecast | auroc | +0.87 ±0.58 (0/5) | +0.41 ±0.33 (0/5) | +0.47 ±0.40 (1/5) | +0.35 ±0.34 (1/5) | +0.52 ±0.47 (1/5) |

## Group-level paired comparisons, cal split (mean over the group's tasks; worse% of A vs B, NEGATIVE = A better; seeds where A better)

| group | metric | ST vs MT | ST-matched vs MT | ST vs ST-matched | FAM vs MT | ST vs FAM |
|---|---|---|---|---|---|---|
| Severity nowcast | rmse | -0.23 ±0.24 (4/5) | +0.57 ±0.44 (0/5) | -0.79 ±0.23 (5/5) | -0.02 ±0.28 (2/5) | -0.21 ±0.24 (4/5) |
| Status nowcast | ap | -5.35 ±2.27 (5/5) | -5.46 ±1.62 (5/5) | +0.10 ±1.69 (2/5) | -4.54 ±5.00 (4/5) | -0.96 ±5.13 (3/5) |
| Status nowcast | auroc | -0.27 ±0.36 (3/5) | -0.26 ±0.27 (4/5) | -0.00 ±0.20 (3/5) | -0.02 ±0.34 (3/5) | -0.25 ±0.18 (4/5) |
| Sustained onset | ap | +6.20 ±3.75 (0/5) | +6.85 ±5.38 (1/5) | -0.80 ±3.12 (2/5) | +1.64 ±7.71 (2/5) | +4.08 ±9.73 (2/5) |
| Sustained onset | auroc | +1.37 ±0.68 (0/5) | +1.73 ±0.47 (0/5) | -0.37 ±0.48 (3/5) | +0.26 ±0.54 (2/5) | +1.12 ±0.71 (0/5) |
| Severity forecast | rmse | +0.16 ±0.24 (2/5) | +0.69 ±0.39 (0/5) | -0.53 ±0.35 (5/5) | +0.41 ±0.28 (0/5) | -0.25 ±0.13 (5/5) |
| Status forecast | ap | -0.12 ±1.15 (2/5) | +1.62 ±0.96 (0/5) | -1.77 ±0.76 (5/5) | +0.82 ±3.23 (3/5) | -1.02 ±2.68 (3/5) |
| Status forecast | auroc | +0.30 ±0.14 (0/5) | +0.59 ±0.30 (0/5) | -0.30 ±0.30 (4/5) | +0.54 ±0.22 (0/5) | -0.25 ±0.30 (4/5) |

Pair meanings: **ST vs MT** = single-task vs multitask (total effect); **ST-matched vs MT** = single-task at MT's epochs vs multitask (equal budget); **ST vs ST-matched** = ST selection advantage (own early stopping vs MT's epochs); **FAM vs MT** = family-only multitask vs full multitask (cross-family sharing); **ST vs FAM** = single-task vs family multitask (within-family sharing).

## Seed-0 participant-bootstrap 95% intervals (test, B=200, paired resamples) for the group-level worse% of A vs B; `*` = interval excludes 0

| group | metric | ST vs MT | ST-matched vs MT | ST vs ST-matched | FAM vs MT | ST vs FAM |
|---|---|---|---|---|---|---|
| Severity nowcast | rmse | +1.03 [+0.09, +2.07]* | +1.12 [-0.12, +2.25] | -0.09 [-1.27, +0.99] | +0.18 [-0.71, +0.97] | +0.85 [-0.04, +1.70] |
| Status nowcast | ap | +2.02 [-8.29, +10.02] | +0.29 [-11.96, +10.40] | +1.74 [-2.87, +5.63] | -3.24 [-11.42, +3.75] | +5.10 [-3.56, +12.56] |
| Status nowcast | auroc | +0.29 [-0.63, +1.30] | +0.28 [-0.90, +1.54] | +0.01 [-0.78, +0.67] | +0.09 [-0.94, +1.07] | +0.20 [-0.33, +0.77] |
| Sustained onset | ap | +16.84 [-3.31, +30.98] | +9.66 [-9.77, +25.88] | +7.95 [-5.37, +17.56] | +14.32 [-7.63, +26.43] | +2.94 [-12.67, +22.46] |
| Sustained onset | auroc | +1.07 [-0.98, +3.01] | +0.59 [-1.64, +2.62] | +0.48 [-0.56, +1.42] | +0.66 [-1.06, +2.30] | +0.41 [-1.63, +2.27] |
| Severity forecast | rmse | +0.78 [+0.12, +1.45]* | +1.12 [+0.38, +1.74]* | -0.34 [-0.96, +0.37] | +0.52 [-0.28, +1.39] | +0.25 [-0.26, +0.78] |
| Status forecast | ap | +9.33 [+1.76, +14.01]* | +3.82 [-1.92, +8.95] | +5.73 [+0.46, +9.38]* | -2.86 [-11.42, +2.74] | +11.85 [+4.32, +18.57]* |
| Status forecast | auroc | +0.48 [-1.01, +1.63] | +0.11 [-0.82, +0.97] | +0.37 [-0.50, +1.03] | -0.10 [-1.07, +0.97] | +0.58 [-0.94, +2.23] |

## Training cost and convergence (CPU, 4 torch threads per process; 2 processes ran concurrently and the laptop throttled and hibernated at times, so wall seconds are unreliable; CPU seconds are the fairer cost figure but also vary with load)

| part | condition | trainings per seed | params per model | CPU s per model | median wall s per model (one run includes a laptop-hibernation gap, so the median is used) | CPU s for ALL heads per seed | mean best epoch | share best epoch <= 2 | mean cal-loss rise after best |
|---|---|---|---|---|---|---|---|---|---|
| forecast | FAM | 2 | 10534 | 119 | 33 | 239 | 2.0 | 0.90 | +0.0120 |
| forecast | MT | 1 | 10732 | 126 | 32 | 126 | 2.6 | 0.60 | +0.0128 |
| forecast | ST | 12 | 10369 | 141 | 47 | 1697 | 2.9 | 0.53 | +0.0347 |
| forecast | ST-matched | 12 | 10369 | 50 | 13 | 595 | 2.6 | 0.60 | +0.0000 |
| main | FAM | 3 | 10446 | 129 | 38 | 386 | 2.5 | 0.60 | +0.0105 |
| main | MT | 1 | 10864 | 95 | 24 | 95 | 3.4 | 0.00 | +0.0039 |
| main | ST | 10 | 10369 | 143 | 43 | 1434 | 2.9 | 0.42 | +0.0171 |
| main | ST-matched | 10 | 10369 | 62 | 18 | 619 | 3.4 | 0.00 | +0.0000 |

## Supervision context: share of the 55,800 training origins with a defined label, per head (a single-head model gets gradient from only this share of each batch; the multitask model gets gradient from every row through at least one head)

| head | labelled share of training origins |
|---|---|
| sev_dep_t0 | 1.00 |
| sev_anx_t0 | 1.00 |
| status_dep_t0 | 1.00 |
| status_anx_t0 | 1.00 |
| onset_sust_dep_h7 | 0.70 |
| onset_sust_dep_h14 | 0.64 |
| onset_sust_dep_h21 | 0.59 |
| onset_sust_anx_h7 | 0.69 |
| onset_sust_anx_h14 | 0.64 |
| onset_sust_anx_h21 | 0.59 |
| sev_dep_h7 | 0.92 |
| sev_dep_h14 | 0.85 |
| sev_dep_h21 | 0.77 |
| sev_anx_h7 | 0.92 |
| sev_anx_h14 | 0.85 |
| sev_anx_h21 | 0.77 |
| status_dep_h7 | 0.92 |
| status_dep_h14 | 0.85 |
| status_dep_h21 | 0.77 |
| status_anx_h7 | 0.92 |
| status_anx_h14 | 0.85 |
| status_anx_h21 | 0.77 |
