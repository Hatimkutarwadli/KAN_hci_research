# KAN-GRU extension: comparison on matching tasks (SYNTHETIC; EXTENSION, not part of the original six stages)

Cells: mean ±SD over 5 seeds (gbt: one deterministic run). Test = final evaluation only. Every model below was evaluated on each listed task.


## NOWCAST severity (RMSE)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | best on CAL (rmse) | best on TEST (rmse) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sev_dep_t0 | 18,600 | 3.965 | 2.996 | 2.967 ±0.011 | 3.019 ±0.026 | 3.025 ±0.013 | 3.142 ±0.039 | 3.139 ±0.030 | 2.999 ±0.017 | 3.003 ±0.018 | gru | gru |
| sev_anx_t0 | 18,600 | 3.710 | 2.952 | 2.907 ±0.008 | 2.930 ±0.012 | 2.946 ±0.024 | 3.063 ±0.043 | 3.044 ±0.064 | 2.922 ±0.027 | 2.923 ±0.040 | gru | gru |


## NOWCAST status (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| status_dep_t0 | 18,600 / 1,343 pos (69 ppl) | 0.072 | 0.362 | 0.361 ±0.010 | 0.350 ±0.016 | 0.346 ±0.013 | 0.334 ±0.013 | 0.325 ±0.014 | 0.341 ±0.013 | 0.335 ±0.024 | gbt | gbt |
| status_anx_t0 | 18,600 / 1,466 pos (73 ppl) | 0.079 | 0.273 | 0.324 ±0.002 | 0.315 ±0.020 | 0.305 ±0.009 | 0.268 ±0.011 | 0.280 ±0.011 | 0.296 ±0.006 | 0.302 ±0.014 | m1 | gru |


## PRIMARY sustained onset (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_sust_dep_h7 | 12,933 / 202 pos (32 ppl) | 0.016 | 0.054 | 0.083 ±0.011 | 0.042 ±0.004 | 0.048 ±0.006 | 0.043 ±0.006 | 0.061 ±0.013 | 0.063 ±0.009 | 0.064 ±0.010 | kangru_fixed | gru |
| onset_sust_dep_h14 | 11,925 / 370 pos (32 ppl) | 0.031 | 0.086 | 0.105 ±0.007 | 0.070 ±0.005 | 0.074 ±0.007 | 0.068 ±0.010 | 0.087 ±0.010 | 0.095 ±0.010 | 0.099 ±0.011 | gru | gru |
| onset_sust_dep_h21 | 10,917 / 511 pos (32 ppl) | 0.047 | 0.096 | 0.123 ±0.003 | 0.096 ±0.004 | 0.099 ±0.006 | 0.094 ±0.011 | 0.116 ±0.006 | 0.116 ±0.009 | 0.122 ±0.016 | gru | gru |
| onset_sust_anx_h7 | 12,290 / 206 pos (32 ppl) | 0.017 | 0.069 | 0.100 ±0.007 | 0.080 ±0.020 | 0.091 ±0.007 | 0.063 ±0.014 | 0.057 ±0.003 | 0.088 ±0.014 | 0.077 ±0.010 | kangru_fixed | gru |
| onset_sust_anx_h14 | 11,338 / 373 pos (32 ppl) | 0.033 | 0.122 | 0.132 ±0.006 | 0.104 ±0.012 | 0.106 ±0.007 | 0.094 ±0.006 | 0.092 ±0.001 | 0.112 ±0.009 | 0.107 ±0.008 | gru | gru |
| onset_sust_anx_h21 | 10,372 / 508 pos (32 ppl) | 0.049 | 0.100 | 0.152 ±0.010 | 0.123 ±0.009 | 0.125 ±0.006 | 0.123 ±0.006 | 0.125 ±0.004 | 0.137 ±0.008 | 0.136 ±0.009 | kan_reg | gru |


## PRIMARY sustained onset (AUROC)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | best on CAL (auroc) | best on TEST (auroc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_sust_dep_h7 | 12,933 / 202 pos (32 ppl) | 0.500 | 0.800 | 0.798 ±0.005 | 0.746 ±0.039 | 0.762 ±0.013 | 0.777 ±0.022 | 0.805 ±0.006 | 0.798 ±0.009 | 0.804 ±0.004 | gru | m1 |
| onset_sust_dep_h14 | 11,925 / 370 pos (32 ppl) | 0.500 | 0.741 | 0.770 ±0.005 | 0.741 ±0.010 | 0.745 ±0.006 | 0.736 ±0.036 | 0.781 ±0.005 | 0.777 ±0.003 | 0.778 ±0.004 | kangru_fixed | m1 |
| onset_sust_dep_h21 | 10,917 / 511 pos (32 ppl) | 0.500 | 0.717 | 0.749 ±0.005 | 0.727 ±0.008 | 0.730 ±0.007 | 0.722 ±0.021 | 0.757 ±0.004 | 0.752 ±0.012 | 0.752 ±0.011 | m1 | m1 |
| onset_sust_anx_h7 | 12,290 / 206 pos (32 ppl) | 0.500 | 0.794 | 0.825 ±0.006 | 0.705 ±0.109 | 0.750 ±0.014 | 0.759 ±0.048 | 0.809 ±0.003 | 0.822 ±0.004 | 0.815 ±0.008 | gru | gru |
| onset_sust_anx_h14 | 11,338 / 373 pos (32 ppl) | 0.500 | 0.791 | 0.798 ±0.002 | 0.744 ±0.031 | 0.751 ±0.010 | 0.762 ±0.006 | 0.784 ±0.003 | 0.797 ±0.003 | 0.795 ±0.004 | gru | gru |
| onset_sust_anx_h21 | 10,372 / 508 pos (32 ppl) | 0.500 | 0.745 | 0.781 ±0.004 | 0.738 ±0.027 | 0.739 ±0.006 | 0.742 ±0.028 | 0.770 ±0.004 | 0.782 ±0.003 | 0.779 ±0.005 | gru | kangru_fixed |


## SECONDARY first-crossing onset (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_cross_dep_h7 | 12,444 / 276 pos (41 ppl) | 0.022 | 0.088 | 0.086 ±0.006 | 0.061 ±0.008 | 0.068 ±0.009 | 0.061 ±0.008 | 0.080 ±0.007 | 0.079 ±0.005 | 0.078 ±0.008 | kangru | gbt |
| onset_cross_dep_h14 | 11,532 / 481 pos (41 ppl) | 0.042 | 0.098 | 0.115 ±0.004 | 0.095 ±0.007 | 0.100 ±0.009 | 0.092 ±0.009 | 0.114 ±0.005 | 0.114 ±0.003 | 0.116 ±0.010 | gru | kangru |
| onset_cross_dep_h21 | 10,598 / 637 pos (41 ppl) | 0.060 | 0.121 | 0.134 ±0.003 | 0.115 ±0.008 | 0.120 ±0.006 | 0.116 ±0.011 | 0.141 ±0.007 | 0.135 ±0.005 | 0.139 ±0.014 | gbt | m1 |
| onset_cross_anx_h7 | 11,556 / 258 pos (41 ppl) | 0.022 | 0.060 | 0.082 ±0.006 | 0.057 ±0.005 | 0.066 ±0.006 | 0.054 ±0.003 | 0.061 ±0.002 | 0.073 ±0.002 | 0.074 ±0.011 | gru | gru |
| onset_cross_anx_h14 | 10,703 / 469 pos (41 ppl) | 0.044 | 0.093 | 0.111 ±0.007 | 0.093 ±0.007 | 0.096 ±0.003 | 0.093 ±0.002 | 0.097 ±0.002 | 0.105 ±0.006 | 0.106 ±0.004 | kangru_fixed | gru |
| onset_cross_anx_h21 | 9,831 / 623 pos (41 ppl) | 0.063 | 0.098 | 0.136 ±0.002 | 0.119 ±0.004 | 0.122 ±0.006 | 0.124 ±0.003 | 0.126 ±0.004 | 0.134 ±0.003 | 0.133 ±0.003 | kangru_fixed | gru |


## FORECAST severity at t+h (RMSE)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | best on CAL (rmse) | best on TEST (rmse) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sev_dep_h7 | 17,200 | 3.943 | 3.062 | 3.045 ±0.007 | 3.147 ±0.032 | 3.169 ±0.027 | 3.240 ±0.023 | 3.192 ±0.029 | 3.065 ±0.013 | 3.067 ±0.015 | gru | gru |
| sev_dep_h14 | 15,800 | 3.915 | 3.216 | 3.190 ±0.007 | 3.315 ±0.032 | 3.327 ±0.025 | 3.379 ±0.023 | 3.317 ±0.024 | 3.212 ±0.011 | 3.212 ±0.017 | gru | gru |
| sev_dep_h21 | 14,400 | 3.893 | 3.371 | 3.334 ±0.008 | 3.458 ±0.022 | 3.456 ±0.020 | 3.484 ±0.014 | 3.429 ±0.017 | 3.350 ±0.010 | 3.353 ±0.027 | gru | gru |
| sev_anx_h7 | 17,200 | 3.706 | 3.000 | 2.932 ±0.007 | 3.027 ±0.014 | 3.042 ±0.015 | 3.132 ±0.044 | 3.081 ±0.057 | 2.963 ±0.019 | 2.963 ±0.034 | gru | gru |
| sev_anx_h14 | 15,800 | 3.695 | 3.119 | 3.054 ±0.003 | 3.154 ±0.010 | 3.165 ±0.014 | 3.245 ±0.046 | 3.188 ±0.053 | 3.080 ±0.006 | 3.089 ±0.017 | gru | gru |
| sev_anx_h21 | 14,400 | 3.683 | 3.239 | 3.174 ±0.005 | 3.267 ±0.015 | 3.270 ±0.012 | 3.335 ±0.023 | 3.281 ±0.036 | 3.201 ±0.006 | 3.204 ±0.017 | gru | gru |


## FORECAST status at t+h (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| status_dep_h7 | 17,200 / 1,239 pos (68 ppl) | 0.072 | 0.313 | 0.351 ±0.014 | 0.328 ±0.033 | 0.320 ±0.015 | 0.299 ±0.019 | 0.313 ±0.011 | 0.323 ±0.018 | 0.323 ±0.019 | gbt | gru |
| status_dep_h14 | 15,800 / 1,119 pos (61 ppl) | 0.071 | 0.257 | 0.288 ±0.011 | 0.268 ±0.025 | 0.265 ±0.009 | 0.258 ±0.011 | 0.278 ±0.011 | 0.274 ±0.015 | 0.273 ±0.018 | gbt | gru |
| status_dep_h21 | 14,400 / 1,011 pos (59 ppl) | 0.070 | 0.183 | 0.234 ±0.008 | 0.234 ±0.014 | 0.235 ±0.012 | 0.231 ±0.009 | 0.247 ±0.007 | 0.227 ±0.014 | 0.231 ±0.017 | gbt | m1 |
| status_anx_h7 | 17,200 / 1,345 pos (69 ppl) | 0.078 | 0.236 | 0.300 ±0.014 | 0.279 ±0.016 | 0.270 ±0.010 | 0.235 ±0.008 | 0.262 ±0.010 | 0.291 ±0.005 | 0.293 ±0.013 | m1 | gru |
| status_anx_h14 | 15,800 / 1,221 pos (65 ppl) | 0.077 | 0.204 | 0.265 ±0.009 | 0.237 ±0.012 | 0.227 ±0.009 | 0.205 ±0.006 | 0.218 ±0.008 | 0.240 ±0.009 | 0.240 ±0.010 | m1 | gru |
| status_anx_h21 | 14,400 / 1,123 pos (60 ppl) | 0.078 | 0.186 | 0.233 ±0.006 | 0.219 ±0.016 | 0.214 ±0.011 | 0.183 ±0.008 | 0.194 ±0.008 | 0.216 ±0.007 | 0.213 ±0.010 | m1 | gru |


## Mean rank over each group's tasks (excluding B0; lower = better). Decision-grade = CAL.

| group | CAL ranking | TEST ranking |
|---|---|---|
| nowcast severity | gru (1.00), gbt (2.00), kangru_fixed (3.00), kangru (4.00), kan (5.50), kan_reg (5.50), m1 (7.00), m1_static (8.00) | gru (1.00), kangru_fixed (2.50), kangru (3.50), gbt (4.00), kan (4.50), kan_reg (5.50), m1 (7.00), m1_static (8.00) |
| nowcast status | gru (2.00), kangru_fixed (4.00), gbt (4.50), kan (4.50), m1 (4.50), kan_reg (5.50), kangru (5.50), m1_static (5.50) | gru (1.50), kan (2.50), kan_reg (3.50), gbt (4.00), kangru (5.00), kangru_fixed (5.00), m1 (7.00), m1_static (7.50) |
| sustained onset | gru (1.83), kangru_fixed (2.83), kangru (3.33), kan (4.00), kan_reg (4.50), gbt (5.67), m1 (6.00), m1_static (7.83) | gru (1.00), kangru (3.00), kangru_fixed (3.00), kan_reg (4.67), m1 (5.33), gbt (5.50), kan (6.17), m1_static (7.33) |
| first-crossing onset | gru (1.83), kangru_fixed (2.00), kangru (2.67), m1 (4.50), kan (4.83), gbt (6.17), kan_reg (6.33), m1_static (7.67) | gru (1.83), kangru (2.50), kangru_fixed (3.17), m1 (3.33), gbt (5.33), kan_reg (5.33), kan (7.17), m1_static (7.33) |
| forecast severity | gru (1.00), gbt (2.50), kangru_fixed (2.67), kangru (3.83), kan_reg (5.33), m1 (5.67), kan (7.00), m1_static (8.00) | gru (1.00), kangru_fixed (2.33), kangru (3.00), gbt (3.67), kan (5.33), kan_reg (6.17), m1 (6.50), m1_static (8.00) |
| forecast status | gru (3.00), kangru_fixed (3.00), kangru (3.67), m1 (4.33), gbt (4.50), m1_static (5.33), kan_reg (5.50), kan (6.67) | gru (1.50), kan (3.33), kangru_fixed (3.50), kangru (4.00), kan_reg (4.50), m1 (4.67), m1_static (7.17), gbt (7.33) |
| ALL 28 tasks (rmse for severity, AP otherwise) | gru (1.86), kangru_fixed (2.75), kangru (3.57), gbt (4.50), m1 (5.21), kan_reg (5.43), kan (5.54), m1_static (7.14) | gru (1.32), kangru_fixed (3.11), kangru (3.29), kan_reg (5.07), kan (5.21), gbt (5.25), m1 (5.25), m1_static (7.50) |