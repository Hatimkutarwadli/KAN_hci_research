# Updated baseline comparison (SYNTHETIC data; test = final evaluation only)

NN cells: mean ±SD over seeds; gbt: single deterministic run. 'best on CAL' is the decision-grade ranking.


## NOWCAST severity (RMSE)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (rmse) | best on TEST (rmse) |
|---|---|---|---|---|---|---|---|---|
| sev_dep_t0 | 18,600 | 3.965 | 2.996 | 2.967 ±0.011 | 3.019 ±0.026 | 3.025 ±0.013 | gru | gru |
| sev_anx_t0 | 18,600 | 3.710 | 2.952 | 2.907 ±0.008 | 2.930 ±0.012 | 2.946 ±0.024 | gru | gru |


## NOWCAST severity (R2)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (r2) | best on TEST (r2) |
|---|---|---|---|---|---|---|---|---|
| sev_dep_t0 | 18,600 | -0.000 | 0.429 | 0.440 ±0.004 | 0.420 ±0.010 | 0.418 ±0.005 | gru | gru |
| sev_anx_t0 | 18,600 | -0.000 | 0.367 | 0.386 ±0.004 | 0.376 ±0.005 | 0.369 ±0.010 | gru | gru |


## NOWCAST status (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|
| status_dep_t0 | 18,600 / 1,343 pos (69 ppl) | 0.072 | 0.362 | 0.361 ±0.010 | 0.350 ±0.016 | 0.346 ±0.013 | gbt | gbt |
| status_anx_t0 | 18,600 / 1,466 pos (73 ppl) | 0.079 | 0.273 | 0.324 ±0.002 | 0.315 ±0.020 | 0.305 ±0.009 | gru | gru |


## NOWCAST status (AUROC)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (auroc) | best on TEST (auroc) |
|---|---|---|---|---|---|---|---|---|
| status_dep_t0 | 18,600 / 1,343 pos (69 ppl) | 0.500 | 0.862 | 0.870 ±0.002 | 0.869 ±0.005 | 0.867 ±0.003 | gru | gru |
| status_anx_t0 | 18,600 / 1,466 pos (73 ppl) | 0.500 | 0.812 | 0.849 ±0.002 | 0.836 ±0.005 | 0.833 ±0.003 | gru | gru |


## PRIMARY sustained onset (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|
| onset_sust_dep_h7 | 12,933 / 202 pos (32 ppl) | 0.016 | 0.054 | 0.083 ±0.011 | 0.042 ±0.004 | 0.048 ±0.006 | gru | gru |
| onset_sust_dep_h14 | 11,925 / 370 pos (32 ppl) | 0.031 | 0.086 | 0.105 ±0.007 | 0.070 ±0.005 | 0.074 ±0.007 | gru | gru |
| onset_sust_dep_h21 | 10,917 / 511 pos (32 ppl) | 0.047 | 0.096 | 0.123 ±0.003 | 0.096 ±0.004 | 0.099 ±0.006 | gru | gru |
| onset_sust_anx_h7 | 12,290 / 206 pos (32 ppl) | 0.017 | 0.069 | 0.100 ±0.007 | 0.080 ±0.020 | 0.091 ±0.007 | gru | gru |
| onset_sust_anx_h14 | 11,338 / 373 pos (32 ppl) | 0.033 | 0.122 | 0.132 ±0.006 | 0.104 ±0.012 | 0.106 ±0.007 | gru | gru |
| onset_sust_anx_h21 | 10,372 / 508 pos (32 ppl) | 0.049 | 0.100 | 0.152 ±0.010 | 0.123 ±0.009 | 0.125 ±0.006 | kan_reg | gru |


## PRIMARY sustained onset (AUROC)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (auroc) | best on TEST (auroc) |
|---|---|---|---|---|---|---|---|---|
| onset_sust_dep_h7 | 12,933 / 202 pos (32 ppl) | 0.500 | 0.800 | 0.798 ±0.005 | 0.746 ±0.039 | 0.762 ±0.013 | gru | gbt |
| onset_sust_dep_h14 | 11,925 / 370 pos (32 ppl) | 0.500 | 0.741 | 0.770 ±0.005 | 0.741 ±0.010 | 0.745 ±0.006 | gru | gru |
| onset_sust_dep_h21 | 10,917 / 511 pos (32 ppl) | 0.500 | 0.717 | 0.749 ±0.005 | 0.727 ±0.008 | 0.730 ±0.007 | kan_reg | gru |
| onset_sust_anx_h7 | 12,290 / 206 pos (32 ppl) | 0.500 | 0.794 | 0.825 ±0.006 | 0.705 ±0.109 | 0.750 ±0.014 | gru | gru |
| onset_sust_anx_h14 | 11,338 / 373 pos (32 ppl) | 0.500 | 0.791 | 0.798 ±0.002 | 0.744 ±0.031 | 0.751 ±0.010 | gru | gru |
| onset_sust_anx_h21 | 10,372 / 508 pos (32 ppl) | 0.500 | 0.745 | 0.781 ±0.004 | 0.738 ±0.027 | 0.739 ±0.006 | gru | gru |


## SECONDARY first-crossing onset (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|
| onset_cross_dep_h7 | 12,444 / 276 pos (41 ppl) | 0.022 | 0.088 | 0.086 ±0.006 | 0.061 ±0.008 | 0.068 ±0.009 | gru | gbt |
| onset_cross_dep_h14 | 11,532 / 481 pos (41 ppl) | 0.042 | 0.098 | 0.115 ±0.004 | 0.095 ±0.007 | 0.100 ±0.009 | gru | gru |
| onset_cross_dep_h21 | 10,598 / 637 pos (41 ppl) | 0.060 | 0.121 | 0.134 ±0.003 | 0.115 ±0.008 | 0.120 ±0.006 | gbt | gru |
| onset_cross_anx_h7 | 11,556 / 258 pos (41 ppl) | 0.022 | 0.060 | 0.082 ±0.006 | 0.057 ±0.005 | 0.066 ±0.006 | gru | gru |
| onset_cross_anx_h14 | 10,703 / 469 pos (41 ppl) | 0.044 | 0.093 | 0.111 ±0.007 | 0.093 ±0.007 | 0.096 ±0.003 | gru | gru |
| onset_cross_anx_h21 | 9,831 / 623 pos (41 ppl) | 0.063 | 0.098 | 0.136 ±0.002 | 0.119 ±0.004 | 0.122 ±0.006 | gru | gru |


## FORECAST severity at t+h (RMSE)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (rmse) | best on TEST (rmse) |
|---|---|---|---|---|---|---|---|---|
| sev_dep_h7 | 17,200 | 3.943 | 3.062 | 3.045 ±0.007 | 3.147 ±0.032 | 3.169 ±0.027 | gru | gru |
| sev_dep_h14 | 15,800 | 3.915 | 3.216 | 3.190 ±0.007 | 3.315 ±0.032 | 3.327 ±0.025 | gru | gru |
| sev_dep_h21 | 14,400 | 3.893 | 3.371 | 3.334 ±0.008 | 3.458 ±0.022 | 3.456 ±0.020 | gru | gru |
| sev_anx_h7 | 17,200 | 3.706 | 3.000 | 2.932 ±0.007 | 3.027 ±0.014 | 3.042 ±0.015 | gru | gru |
| sev_anx_h14 | 15,800 | 3.695 | 3.119 | 3.054 ±0.003 | 3.154 ±0.010 | 3.165 ±0.014 | gru | gru |
| sev_anx_h21 | 14,400 | 3.683 | 3.239 | 3.174 ±0.005 | 3.267 ±0.015 | 3.270 ±0.012 | gru | gru |


## FORECAST severity at t+h (R2)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (r2) | best on TEST (r2) |
|---|---|---|---|---|---|---|---|---|
| sev_dep_h7 | 17,200 | -0.000 | 0.397 | 0.404 ±0.003 | 0.363 ±0.013 | 0.354 ±0.011 | gru | gru |
| sev_dep_h14 | 15,800 | -0.000 | 0.325 | 0.336 ±0.003 | 0.283 ±0.014 | 0.278 ±0.011 | gru | gru |
| sev_dep_h21 | 14,400 | -0.000 | 0.250 | 0.266 ±0.004 | 0.211 ±0.010 | 0.212 ±0.009 | gru | gru |
| sev_anx_h7 | 17,200 | -0.000 | 0.345 | 0.374 ±0.003 | 0.333 ±0.006 | 0.326 ±0.007 | gru | gru |
| sev_anx_h14 | 15,800 | -0.000 | 0.288 | 0.317 ±0.002 | 0.271 ±0.005 | 0.266 ±0.006 | gru | gru |
| sev_anx_h21 | 14,400 | -0.000 | 0.227 | 0.258 ±0.003 | 0.213 ±0.007 | 0.212 ±0.006 | gru | gru |


## FORECAST status at t+h (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|
| status_dep_h7 | 17,200 / 1,239 pos (68 ppl) | 0.072 | 0.313 | 0.351 ±0.014 | 0.328 ±0.033 | 0.320 ±0.015 | gbt | gru |
| status_dep_h14 | 15,800 / 1,119 pos (61 ppl) | 0.071 | 0.257 | 0.288 ±0.011 | 0.268 ±0.025 | 0.265 ±0.009 | gbt | gru |
| status_dep_h21 | 14,400 / 1,011 pos (59 ppl) | 0.070 | 0.183 | 0.234 ±0.008 | 0.234 ±0.014 | 0.235 ±0.012 | gbt | kan_reg |
| status_anx_h7 | 17,200 / 1,345 pos (69 ppl) | 0.078 | 0.236 | 0.300 ±0.014 | 0.279 ±0.016 | 0.270 ±0.010 | gru | gru |
| status_anx_h14 | 15,800 / 1,221 pos (65 ppl) | 0.077 | 0.204 | 0.265 ±0.009 | 0.237 ±0.012 | 0.227 ±0.009 | gru | gru |
| status_anx_h21 | 14,400 / 1,123 pos (60 ppl) | 0.078 | 0.186 | 0.233 ±0.006 | 0.219 ±0.016 | 0.214 ±0.011 | gru | gru |


## FORECAST status at t+h (AUROC)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | best on CAL (auroc) | best on TEST (auroc) |
|---|---|---|---|---|---|---|---|---|
| status_dep_h7 | 17,200 / 1,239 pos (68 ppl) | 0.500 | 0.847 | 0.854 ±0.003 | 0.843 ±0.007 | 0.840 ±0.002 | gru | gru |
| status_dep_h14 | 15,800 / 1,119 pos (61 ppl) | 0.500 | 0.802 | 0.828 ±0.001 | 0.811 ±0.005 | 0.807 ±0.002 | gru | gru |
| status_dep_h21 | 14,400 / 1,011 pos (59 ppl) | 0.500 | 0.765 | 0.796 ±0.003 | 0.779 ±0.003 | 0.775 ±0.004 | gru | gru |
| status_anx_h7 | 17,200 / 1,345 pos (69 ppl) | 0.500 | 0.792 | 0.826 ±0.008 | 0.809 ±0.005 | 0.806 ±0.003 | gru | gru |
| status_anx_h14 | 15,800 / 1,221 pos (65 ppl) | 0.500 | 0.775 | 0.803 ±0.005 | 0.783 ±0.010 | 0.782 ±0.006 | gru | gru |
| status_anx_h21 | 14,400 / 1,123 pos (60 ppl) | 0.500 | 0.759 | 0.780 ±0.004 | 0.764 ±0.007 | 0.760 ±0.006 | gru | gru |


## Strongest baseline per group (mean rank over the group's tasks; metric = RMSE for severity, AP for status/onset)

| group | best on CAL | best on TEST |
|---|---|---|
| nowcast severity | gru (1.00), gbt (2.00), kan (3.50), kan_reg (3.50) | gru (1.00), kan (2.50), gbt (3.00), kan_reg (3.50) |
| nowcast status | gru (1.50), gbt (2.50), kan (2.50), kan_reg (3.50) | gru (1.50), gbt (2.50), kan (2.50), kan_reg (3.50) |
| sustained onset | gru (1.33), kan (2.50), kan_reg (2.83), gbt (3.33) | gru (1.00), kan_reg (2.50), gbt (3.00), kan (3.50) |
| first-crossing onset | gru (1.17), kan (2.17), gbt (3.33), kan_reg (3.33) | gru (1.17), kan_reg (2.33), gbt (2.67), kan (3.83) |
| forecast severity | gru (1.00), gbt (2.00), kan_reg (3.00), kan (4.00) | gru (1.00), gbt (2.00), kan (3.17), kan_reg (3.83) |
| forecast status | gru (1.50), gbt (2.50), kan_reg (2.50), kan (3.50) | gru (1.33), kan (2.00), kan_reg (2.67), gbt (4.00) |