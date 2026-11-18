# P1-1: Transformer-encoder and CNN-LSTM baselines vs existing models (SYNTHETIC; EXTENSION, not part of the original six stages)

Frozen protocol, 5 seeds (gbt: one deterministic run), no tuning. Cells: mean ±SD over seeds. Test = final evaluation only; CAL = decision-grade. Blank = that model was not trained on that task set (e.g. the GRU/KAN/GBT forecast heads come from the 12-head forecast runs; first-crossing heads exist only in the 16-head main runs and in M1/KAN-GRU 28-head runs).


## NOWCAST severity (RMSE)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (rmse) | best on TEST (rmse) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sev_dep_t0 | 18,600 | 3.965 | 2.996 | 2.967 ±0.011 | 3.019 ±0.026 | 3.025 ±0.013 | 3.142 ±0.039 | 3.139 ±0.030 | 2.999 ±0.017 | 3.003 ±0.018 | 2.976 ±0.009 | 2.993 ±0.012 | gru | gru |
| sev_anx_t0 | 18,600 | 3.710 | 2.952 | 2.907 ±0.008 | 2.930 ±0.012 | 2.946 ±0.024 | 3.063 ±0.043 | 3.044 ±0.064 | 2.922 ±0.027 | 2.923 ±0.040 | 2.901 ±0.015 | 2.915 ±0.017 | gru | transformer |


## NOWCAST status (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| status_dep_t0 | 18,600 / 1,343 pos (69 ppl) | 0.072 | 0.362 | 0.361 ±0.010 | 0.350 ±0.016 | 0.346 ±0.013 | 0.334 ±0.013 | 0.325 ±0.014 | 0.341 ±0.013 | 0.335 ±0.024 | 0.342 ±0.013 | 0.359 ±0.018 | gbt | gbt |
| status_anx_t0 | 18,600 / 1,466 pos (73 ppl) | 0.079 | 0.273 | 0.324 ±0.002 | 0.315 ±0.020 | 0.305 ±0.009 | 0.268 ±0.011 | 0.280 ±0.011 | 0.296 ±0.006 | 0.302 ±0.014 | 0.342 ±0.007 | 0.322 ±0.005 | m1 | transformer |


## NOWCAST status (AUROC)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (auroc) | best on TEST (auroc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| status_dep_t0 | 18,600 / 1,343 pos (69 ppl) | 0.500 | 0.862 | 0.870 ±0.002 | 0.869 ±0.005 | 0.867 ±0.003 | 0.861 ±0.004 | 0.860 ±0.003 | 0.868 ±0.003 | 0.867 ±0.004 | 0.870 ±0.003 | 0.868 ±0.004 | transformer | transformer |
| status_anx_t0 | 18,600 / 1,466 pos (73 ppl) | 0.500 | 0.812 | 0.849 ±0.002 | 0.836 ±0.005 | 0.833 ±0.003 | 0.818 ±0.011 | 0.830 ±0.004 | 0.843 ±0.007 | 0.843 ±0.005 | 0.852 ±0.003 | 0.847 ±0.003 | gru | transformer |


## PRIMARY sustained onset (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_sust_dep_h7 | 12,933 / 202 pos (32 ppl) | 0.016 | 0.054 | 0.083 ±0.011 | 0.042 ±0.004 | 0.048 ±0.006 | 0.043 ±0.006 | 0.061 ±0.013 | 0.063 ±0.009 | 0.064 ±0.010 | 0.067 ±0.011 | 0.063 ±0.012 | kangru_fixed | gru |
| onset_sust_dep_h14 | 11,925 / 370 pos (32 ppl) | 0.031 | 0.086 | 0.105 ±0.007 | 0.070 ±0.005 | 0.074 ±0.007 | 0.068 ±0.010 | 0.087 ±0.010 | 0.095 ±0.010 | 0.099 ±0.011 | 0.093 ±0.009 | 0.091 ±0.008 | gru | gru |
| onset_sust_dep_h21 | 10,917 / 511 pos (32 ppl) | 0.047 | 0.096 | 0.123 ±0.003 | 0.096 ±0.004 | 0.099 ±0.006 | 0.094 ±0.011 | 0.116 ±0.006 | 0.116 ±0.009 | 0.122 ±0.016 | 0.115 ±0.008 | 0.116 ±0.006 | gru | gru |
| onset_sust_anx_h7 | 12,290 / 206 pos (32 ppl) | 0.017 | 0.069 | 0.100 ±0.007 | 0.080 ±0.020 | 0.091 ±0.007 | 0.063 ±0.014 | 0.057 ±0.003 | 0.088 ±0.014 | 0.077 ±0.010 | 0.089 ±0.010 | 0.088 ±0.010 | transformer | gru |
| onset_sust_anx_h14 | 11,338 / 373 pos (32 ppl) | 0.033 | 0.122 | 0.132 ±0.006 | 0.104 ±0.012 | 0.106 ±0.007 | 0.094 ±0.006 | 0.092 ±0.001 | 0.112 ±0.009 | 0.107 ±0.008 | 0.126 ±0.009 | 0.117 ±0.008 | transformer | gru |
| onset_sust_anx_h21 | 10,372 / 508 pos (32 ppl) | 0.049 | 0.100 | 0.152 ±0.010 | 0.123 ±0.009 | 0.125 ±0.006 | 0.123 ±0.006 | 0.125 ±0.004 | 0.137 ±0.008 | 0.136 ±0.009 | 0.140 ±0.012 | 0.144 ±0.005 | transformer | gru |


## PRIMARY sustained onset (AUROC)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (auroc) | best on TEST (auroc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_sust_dep_h7 | 12,933 / 202 pos (32 ppl) | 0.500 | 0.800 | 0.798 ±0.005 | 0.746 ±0.039 | 0.762 ±0.013 | 0.777 ±0.022 | 0.805 ±0.006 | 0.798 ±0.009 | 0.804 ±0.004 | 0.798 ±0.005 | 0.794 ±0.005 | cnnlstm | m1 |
| onset_sust_dep_h14 | 11,925 / 370 pos (32 ppl) | 0.500 | 0.741 | 0.770 ±0.005 | 0.741 ±0.010 | 0.745 ±0.006 | 0.736 ±0.036 | 0.781 ±0.005 | 0.777 ±0.003 | 0.778 ±0.004 | 0.768 ±0.003 | 0.769 ±0.004 | cnnlstm | m1 |
| onset_sust_dep_h21 | 10,917 / 511 pos (32 ppl) | 0.500 | 0.717 | 0.749 ±0.005 | 0.727 ±0.008 | 0.730 ±0.007 | 0.722 ±0.021 | 0.757 ±0.004 | 0.752 ±0.012 | 0.752 ±0.011 | 0.749 ±0.005 | 0.752 ±0.003 | transformer | m1 |
| onset_sust_anx_h7 | 12,290 / 206 pos (32 ppl) | 0.500 | 0.794 | 0.825 ±0.006 | 0.705 ±0.109 | 0.750 ±0.014 | 0.759 ±0.048 | 0.809 ±0.003 | 0.822 ±0.004 | 0.815 ±0.008 | 0.815 ±0.005 | 0.826 ±0.006 | gru | cnnlstm |
| onset_sust_anx_h14 | 11,338 / 373 pos (32 ppl) | 0.500 | 0.791 | 0.798 ±0.002 | 0.744 ±0.031 | 0.751 ±0.010 | 0.762 ±0.006 | 0.784 ±0.003 | 0.797 ±0.003 | 0.795 ±0.004 | 0.793 ±0.008 | 0.801 ±0.006 | transformer | cnnlstm |
| onset_sust_anx_h21 | 10,372 / 508 pos (32 ppl) | 0.500 | 0.745 | 0.781 ±0.004 | 0.738 ±0.027 | 0.739 ±0.006 | 0.742 ±0.028 | 0.770 ±0.004 | 0.782 ±0.003 | 0.779 ±0.005 | 0.774 ±0.012 | 0.784 ±0.004 | gru | cnnlstm |


## SECONDARY first-crossing onset (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_cross_dep_h7 | 12,444 / 276 pos (41 ppl) | 0.022 | 0.088 | 0.086 ±0.006 | 0.061 ±0.008 | 0.068 ±0.009 | 0.061 ±0.008 | 0.080 ±0.007 | 0.079 ±0.005 | 0.078 ±0.008 | 0.075 ±0.004 | 0.074 ±0.009 | kangru | gbt |
| onset_cross_dep_h14 | 11,532 / 481 pos (41 ppl) | 0.042 | 0.098 | 0.115 ±0.004 | 0.095 ±0.007 | 0.100 ±0.009 | 0.092 ±0.009 | 0.114 ±0.005 | 0.114 ±0.003 | 0.116 ±0.010 | 0.109 ±0.004 | 0.106 ±0.004 | gru | kangru |
| onset_cross_dep_h21 | 10,598 / 637 pos (41 ppl) | 0.060 | 0.121 | 0.134 ±0.003 | 0.115 ±0.008 | 0.120 ±0.006 | 0.116 ±0.011 | 0.141 ±0.007 | 0.135 ±0.005 | 0.139 ±0.014 | 0.128 ±0.005 | 0.130 ±0.004 | gbt | m1 |
| onset_cross_anx_h7 | 11,556 / 258 pos (41 ppl) | 0.022 | 0.060 | 0.082 ±0.006 | 0.057 ±0.005 | 0.066 ±0.006 | 0.054 ±0.003 | 0.061 ±0.002 | 0.073 ±0.002 | 0.074 ±0.011 | 0.077 ±0.007 | 0.075 ±0.006 | gru | gru |
| onset_cross_anx_h14 | 10,703 / 469 pos (41 ppl) | 0.044 | 0.093 | 0.111 ±0.007 | 0.093 ±0.007 | 0.096 ±0.003 | 0.093 ±0.002 | 0.097 ±0.002 | 0.105 ±0.006 | 0.106 ±0.004 | 0.112 ±0.004 | 0.107 ±0.006 | kangru_fixed | transformer |
| onset_cross_anx_h21 | 9,831 / 623 pos (41 ppl) | 0.063 | 0.098 | 0.136 ±0.002 | 0.119 ±0.004 | 0.122 ±0.006 | 0.124 ±0.003 | 0.126 ±0.004 | 0.134 ±0.003 | 0.133 ±0.003 | 0.138 ±0.006 | 0.131 ±0.003 | kangru_fixed | transformer |


## FORECAST severity at t+h (RMSE)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (rmse) | best on TEST (rmse) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sev_dep_h7 | 17,200 | 3.943 | 3.062 | 3.045 ±0.007 | 3.147 ±0.032 | 3.169 ±0.027 | 3.240 ±0.023 | 3.192 ±0.029 | 3.065 ±0.013 | 3.067 ±0.015 | 3.056 ±0.015 | 3.071 ±0.008 | gru | gru |
| sev_dep_h14 | 15,800 | 3.915 | 3.216 | 3.190 ±0.007 | 3.315 ±0.032 | 3.327 ±0.025 | 3.379 ±0.023 | 3.317 ±0.024 | 3.212 ±0.011 | 3.212 ±0.017 | 3.195 ±0.019 | 3.208 ±0.003 | gru | gru |
| sev_dep_h21 | 14,400 | 3.893 | 3.371 | 3.334 ±0.008 | 3.458 ±0.022 | 3.456 ±0.020 | 3.484 ±0.014 | 3.429 ±0.017 | 3.350 ±0.010 | 3.353 ±0.027 | 3.342 ±0.014 | 3.341 ±0.009 | gru | gru |
| sev_anx_h7 | 17,200 | 3.706 | 3.000 | 2.932 ±0.007 | 3.027 ±0.014 | 3.042 ±0.015 | 3.132 ±0.044 | 3.081 ±0.057 | 2.963 ±0.019 | 2.963 ±0.034 | 2.950 ±0.017 | 2.959 ±0.029 | gru | gru |
| sev_anx_h14 | 15,800 | 3.695 | 3.119 | 3.054 ±0.003 | 3.154 ±0.010 | 3.165 ±0.014 | 3.245 ±0.046 | 3.188 ±0.053 | 3.080 ±0.006 | 3.089 ±0.017 | 3.066 ±0.006 | 3.072 ±0.016 | gru | gru |
| sev_anx_h21 | 14,400 | 3.683 | 3.239 | 3.174 ±0.005 | 3.267 ±0.015 | 3.270 ±0.012 | 3.335 ±0.023 | 3.281 ±0.036 | 3.201 ±0.006 | 3.204 ±0.017 | 3.186 ±0.006 | 3.190 ±0.010 | gru | gru |


## FORECAST status at t+h (AP)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (ap) | best on TEST (ap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| status_dep_h7 | 17,200 / 1,239 pos (68 ppl) | 0.072 | 0.313 | 0.351 ±0.014 | 0.328 ±0.033 | 0.320 ±0.015 | 0.299 ±0.019 | 0.313 ±0.011 | 0.323 ±0.018 | 0.323 ±0.019 | 0.357 ±0.025 | 0.348 ±0.019 | gbt | transformer |
| status_dep_h14 | 15,800 / 1,119 pos (61 ppl) | 0.071 | 0.257 | 0.288 ±0.011 | 0.268 ±0.025 | 0.265 ±0.009 | 0.258 ±0.011 | 0.278 ±0.011 | 0.274 ±0.015 | 0.273 ±0.018 | 0.289 ±0.024 | 0.290 ±0.014 | gbt | cnnlstm |
| status_dep_h21 | 14,400 / 1,011 pos (59 ppl) | 0.070 | 0.183 | 0.234 ±0.008 | 0.234 ±0.014 | 0.235 ±0.012 | 0.231 ±0.009 | 0.247 ±0.007 | 0.227 ±0.014 | 0.231 ±0.017 | 0.245 ±0.014 | 0.243 ±0.014 | gbt | m1 |
| status_anx_h7 | 17,200 / 1,345 pos (69 ppl) | 0.078 | 0.236 | 0.300 ±0.014 | 0.279 ±0.016 | 0.270 ±0.010 | 0.235 ±0.008 | 0.262 ±0.010 | 0.291 ±0.005 | 0.293 ±0.013 | 0.308 ±0.010 | 0.292 ±0.012 | m1 | transformer |
| status_anx_h14 | 15,800 / 1,221 pos (65 ppl) | 0.077 | 0.204 | 0.265 ±0.009 | 0.237 ±0.012 | 0.227 ±0.009 | 0.205 ±0.006 | 0.218 ±0.008 | 0.240 ±0.009 | 0.240 ±0.010 | 0.269 ±0.011 | 0.250 ±0.011 | m1 | transformer |
| status_anx_h21 | 14,400 / 1,123 pos (60 ppl) | 0.078 | 0.186 | 0.233 ±0.006 | 0.219 ±0.016 | 0.214 ±0.011 | 0.183 ±0.008 | 0.194 ±0.008 | 0.216 ±0.007 | 0.213 ±0.010 | 0.242 ±0.008 | 0.223 ±0.009 | m1 | transformer |


## FORECAST status at t+h (AUROC)

| task | test eligible origins / positives | B0_reference | gbt | gru | kan | kan_reg | m1_static | m1 | kangru_fixed | kangru | transformer | cnnlstm | best on CAL (auroc) | best on TEST (auroc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| status_dep_h7 | 17,200 / 1,239 pos (68 ppl) | 0.500 | 0.847 | 0.854 ±0.003 | 0.843 ±0.007 | 0.840 ±0.002 | 0.833 ±0.007 | 0.846 ±0.002 | 0.856 ±0.002 | 0.854 ±0.003 | 0.855 ±0.003 | 0.851 ±0.004 | cnnlstm | kangru_fixed |
| status_dep_h14 | 15,800 / 1,119 pos (61 ppl) | 0.500 | 0.802 | 0.828 ±0.001 | 0.811 ±0.005 | 0.807 ±0.002 | 0.806 ±0.005 | 0.821 ±0.003 | 0.833 ±0.002 | 0.832 ±0.003 | 0.829 ±0.003 | 0.825 ±0.004 | transformer | kangru_fixed |
| status_dep_h21 | 14,400 / 1,011 pos (59 ppl) | 0.500 | 0.765 | 0.796 ±0.003 | 0.779 ±0.003 | 0.775 ±0.004 | 0.776 ±0.005 | 0.788 ±0.003 | 0.801 ±0.002 | 0.799 ±0.005 | 0.795 ±0.003 | 0.792 ±0.005 | transformer | kangru_fixed |
| status_anx_h7 | 17,200 / 1,345 pos (69 ppl) | 0.500 | 0.792 | 0.826 ±0.008 | 0.809 ±0.005 | 0.806 ±0.003 | 0.793 ±0.016 | 0.816 ±0.002 | 0.830 ±0.002 | 0.829 ±0.002 | 0.828 ±0.003 | 0.827 ±0.002 | kangru_fixed | kangru_fixed |
| status_anx_h14 | 15,800 / 1,221 pos (65 ppl) | 0.500 | 0.775 | 0.803 ±0.005 | 0.783 ±0.010 | 0.782 ±0.006 | 0.776 ±0.010 | 0.788 ±0.004 | 0.801 ±0.001 | 0.798 ±0.002 | 0.802 ±0.004 | 0.799 ±0.004 | gru | gru |
| status_anx_h21 | 14,400 / 1,123 pos (60 ppl) | 0.500 | 0.759 | 0.780 ±0.004 | 0.764 ±0.007 | 0.760 ±0.006 | 0.750 ±0.011 | 0.768 ±0.004 | 0.779 ±0.002 | 0.777 ±0.004 | 0.779 ±0.001 | 0.777 ±0.004 | transformer | gru |


## Mean rank over each group's tasks (excluding B0; lower = better; ties not broken)

| group | CAL ranking | TEST ranking |
|---|---|---|
| nowcast severity | gru (1.00), transformer (2.00), gbt (3.00), cnnlstm (4.00), kangru_fixed (5.00), kangru (6.00), kan (7.50), kan_reg (7.50), m1 (9.00), m1_static (10.00) | gru (1.50), transformer (1.50), cnnlstm (3.00), kangru_fixed (4.50), kangru (5.50), gbt (6.00), kan (6.50), kan_reg (7.50), m1 (9.00), m1_static (10.00) |
| nowcast status | cnnlstm (3.00), gru (3.00), transformer (3.00), gbt (5.50), m1 (5.50), kangru_fixed (6.00), kan (6.50), kan_reg (7.50), kangru (7.50), m1_static (7.50) | gru (2.00), cnnlstm (3.00), transformer (3.50), kan (4.00), gbt (5.00), kan_reg (5.00), kangru (7.00), kangru_fixed (7.00), m1 (9.00), m1_static (9.50) |
| sustained onset | transformer (2.33), gru (2.83), cnnlstm (3.83), kangru_fixed (4.17), kangru (4.83), kan (5.50), kan_reg (6.33), gbt (7.33), m1 (8.00), m1_static (9.83) | gru (1.00), transformer (3.33), kangru_fixed (4.00), kangru (4.17), cnnlstm (4.33), kan_reg (6.33), m1 (7.00), gbt (7.33), kan (8.17), m1_static (9.33) |
| first-crossing onset | gru (2.17), kangru_fixed (2.67), transformer (3.00), kangru (3.50), cnnlstm (4.67), m1 (6.50), kan (6.67), gbt (7.83), kan_reg (8.33), m1_static (9.67) | gru (2.17), kangru (3.33), transformer (3.50), kangru_fixed (4.00), m1 (4.33), cnnlstm (4.83), gbt (7.00), kan_reg (7.33), kan (9.17), m1_static (9.33) |
| forecast severity | gru (1.00), transformer (2.67), cnnlstm (3.50), gbt (3.50), kangru_fixed (4.50), kangru (5.83), kan_reg (7.33), m1 (7.67), kan (9.00), m1_static (10.00) | gru (1.00), transformer (2.17), cnnlstm (3.33), kangru_fixed (4.17), kangru (4.83), gbt (5.50), kan (7.33), kan_reg (8.17), m1 (8.50), m1_static (10.00) |
| forecast status | cnnlstm (3.00), transformer (3.83), gru (4.50), kangru_fixed (4.50), kangru (5.33), m1 (5.33), gbt (5.50), m1_static (6.83), kan_reg (7.50), kan (8.67) | transformer (1.33), cnnlstm (2.83), gru (2.83), kan (5.33), kangru_fixed (5.50), kangru (5.83), m1 (6.33), kan_reg (6.50), m1_static (9.17), gbt (9.33) |

## Paired comparison with the GRU, same seed (mean over the group's tasks; worse% = degradation relative to the GRU, NEGATIVE = better than GRU; n = seeds better)

| group | split | transformer: worse% ±SD (n better) | cnnlstm: worse% ±SD (n better) |
|---|---|---|---|
| NOWCAST severity (RMSE) | cal | +0.2 ±0.1 (0/5) | +0.7 ±0.4 (0/5) |
| NOWCAST severity (RMSE) | test | +0.1 ±0.3 (3/5) | +0.6 ±0.3 (0/5) |
| NOWCAST status (AP) | cal | -3.1 ±6.4 (3/5) | +1.6 ±2.8 (3/5) |
| NOWCAST status (AP) | test | +0.1 ±2.2 (1/5) | +0.5 ±2.7 (1/5) |
| NOWCAST status (AUROC) | cal | +0.0 ±0.3 (2/5) | +0.3 ±0.2 (1/5) |
| NOWCAST status (AUROC) | test | -0.2 ±0.3 (3/5) | +0.3 ±0.2 (1/5) |
| PRIMARY sustained onset (AP) | cal | -0.7 ±8.3 (2/5) | +3.9 ±5.6 (1/5) |
| PRIMARY sustained onset (AP) | test | +9.2 ±4.0 (0/5) | +10.8 ±4.6 (0/5) |
| PRIMARY sustained onset (AUROC) | cal | -0.5 ±0.3 (5/5) | -0.1 ±0.3 (3/5) |
| PRIMARY sustained onset (AUROC) | test | +0.5 ±0.6 (1/5) | -0.1 ±0.4 (3/5) |
| SECONDARY first-crossing onset (AP) | cal | +1.8 ±4.8 (2/5) | +7.0 ±2.5 (0/5) |
| SECONDARY first-crossing onset (AP) | test | +3.6 ±2.2 (0/5) | +5.8 ±5.4 (0/5) |
| FORECAST severity at t+h (RMSE) | cal | +0.5 ±0.4 (1/5) | +0.6 ±0.4 (1/5) |
| FORECAST severity at t+h (RMSE) | test | +0.4 ±0.5 (2/5) | +0.6 ±0.4 (0/5) |
| FORECAST status at t+h (AP) | cal | -1.1 ±3.5 (4/5) | -1.6 ±5.3 (3/5) |
| FORECAST status at t+h (AP) | test | -2.4 ±4.8 (3/5) | +1.4 ±3.0 (2/5) |
| FORECAST status at t+h (AUROC) | cal | -0.1 ±0.3 (3/5) | +0.2 ±0.5 (3/5) |
| FORECAST status at t+h (AUROC) | test | +0.0 ±0.3 (3/5) | +0.4 ±0.6 (1/5) |

## Computational cost and convergence (CPU, 4 torch threads; GRU values from the original runs)

| part | model | params | mean wall s/seed | mean CPU s/seed | mean best epoch | seeds with best epoch <= 2 | runs hitting 25 epochs | train loss at best | cal loss at best | mean cal-loss rise after best | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forecast | cnnlstm | 10780 | 170 | 183 | 3.2 | 1/5 | 0 | 0.4259 | 0.4841 | +0.0110 | 5 |
| forecast | gru | 10732 | 34 | 126 | 2.6 | 3/5 | 0 | 0.4224 | 0.4791 | +0.0128 | 5 |
| forecast | transformer | 11380 | 170 | 312 | 2.4 | 4/5 | 0 | 0.4255 | 0.4826 | +0.0105 | 5 |
| main | cnnlstm | 10912 | 69 | 229 | 3.2 | 1/5 | 0 | 0.2034 | 0.2198 | +0.0028 | 5 |
| main | gru | 10864 | 25 | 95 | 3.4 | 0/5 | 0 | 0.1992 | 0.2180 | +0.0039 | 5 |
| main | transformer | 11512 | 103 | 345 | 3.4 | 1/5 | 0 | 0.1988 | 0.2187 | +0.0068 | 5 |

## Seed-0 participant-bootstrap 95% CIs, test (main task set; point [lo, hi]); overlap with the GRU interval means the difference is not resolvable

| task | metric | gru | transformer | cnnlstm |
|---|---|---|---|---|
| sev_dep_t0 | rmse | 2.951 [2.789, 3.096] | 2.978 [2.805, 3.134] | 2.998 [2.846, 3.143] |
| sev_anx_t0 | rmse | 2.907 [2.781, 3.057] | 2.906 [2.785, 3.064] | 2.919 [2.802, 3.063] |
| status_dep_t0 | ap | 0.367 [0.234, 0.476] | 0.336 [0.220, 0.457] | 0.360 [0.236, 0.473] |
| status_dep_t0 | auroc | 0.872 [0.832, 0.904] | 0.869 [0.832, 0.903] | 0.870 [0.832, 0.901] |
| status_anx_t0 | ap | 0.324 [0.199, 0.435] | 0.351 [0.210, 0.490] | 0.328 [0.199, 0.443] |
| status_anx_t0 | auroc | 0.847 [0.807, 0.880] | 0.849 [0.806, 0.880] | 0.849 [0.811, 0.881] |
| onset_sust_dep_h7 | ap | 0.095 [0.036, 0.205] | 0.071 [0.031, 0.159] | 0.065 [0.031, 0.158] |
| onset_sust_dep_h7 | auroc | 0.799 [0.732, 0.862] | 0.802 [0.739, 0.865] | 0.795 [0.727, 0.862] |
| onset_sust_dep_h14 | ap | 0.110 [0.049, 0.212] | 0.093 [0.050, 0.196] | 0.102 [0.049, 0.202] |
| onset_sust_dep_h14 | auroc | 0.770 [0.688, 0.835] | 0.768 [0.695, 0.833] | 0.768 [0.693, 0.839] |
| onset_sust_dep_h21 | ap | 0.123 [0.067, 0.214] | 0.117 [0.067, 0.213] | 0.125 [0.064, 0.227] |
| onset_sust_dep_h21 | auroc | 0.749 [0.663, 0.818] | 0.755 [0.677, 0.818] | 0.752 [0.673, 0.815] |
| onset_sust_anx_h7 | ap | 0.098 [0.041, 0.200] | 0.091 [0.036, 0.198] | 0.075 [0.041, 0.152] |
| onset_sust_anx_h7 | auroc | 0.829 [0.780, 0.879] | 0.808 [0.753, 0.857] | 0.832 [0.787, 0.882] |
| onset_sust_anx_h14 | ap | 0.132 [0.064, 0.247] | 0.122 [0.060, 0.233] | 0.121 [0.063, 0.230] |
| onset_sust_anx_h14 | auroc | 0.797 [0.733, 0.863] | 0.780 [0.710, 0.839] | 0.806 [0.741, 0.864] |
| onset_sust_anx_h21 | ap | 0.151 [0.084, 0.249] | 0.126 [0.071, 0.229] | 0.140 [0.082, 0.253] |
| onset_sust_anx_h21 | auroc | 0.779 [0.705, 0.843] | 0.752 [0.678, 0.815] | 0.790 [0.718, 0.849] |

## Seed-0 participant-bootstrap 95% CIs, test (forecast task set; point [lo, hi]); overlap with the GRU interval means the difference is not resolvable

| task | metric | gru | transformer | cnnlstm |
|---|---|---|---|---|
| sev_dep_h7 | rmse | 3.050 [2.883, 3.220] | 3.048 [2.887, 3.189] | 3.079 [2.917, 3.232] |
| sev_dep_h14 | rmse | 3.191 [3.010, 3.366] | 3.177 [3.007, 3.328] | 3.213 [3.037, 3.384] |
| sev_dep_h21 | rmse | 3.336 [3.150, 3.504] | 3.327 [3.136, 3.485] | 3.346 [3.164, 3.512] |
| sev_anx_h7 | rmse | 2.936 [2.818, 3.082] | 2.942 [2.818, 3.081] | 2.967 [2.855, 3.113] |
| sev_anx_h14 | rmse | 3.054 [2.930, 3.201] | 3.062 [2.923, 3.206] | 3.075 [2.959, 3.212] |
| sev_anx_h21 | rmse | 3.175 [3.049, 3.328] | 3.182 [3.035, 3.332] | 3.186 [3.049, 3.333] |
| status_dep_h7 | ap | 0.346 [0.227, 0.453] | 0.372 [0.235, 0.490] | 0.322 [0.212, 0.422] |
| status_dep_h7 | auroc | 0.851 [0.804, 0.887] | 0.856 [0.810, 0.895] | 0.845 [0.804, 0.881] |
| status_dep_h14 | ap | 0.288 [0.174, 0.390] | 0.297 [0.183, 0.404] | 0.271 [0.162, 0.377] |
| status_dep_h14 | auroc | 0.827 [0.773, 0.871] | 0.830 [0.774, 0.878] | 0.820 [0.771, 0.862] |
| status_dep_h21 | ap | 0.234 [0.124, 0.339] | 0.256 [0.143, 0.363] | 0.222 [0.123, 0.322] |
| status_dep_h21 | auroc | 0.793 [0.733, 0.844] | 0.795 [0.736, 0.849] | 0.784 [0.729, 0.836] |
| status_anx_h7 | ap | 0.303 [0.178, 0.435] | 0.294 [0.164, 0.427] | 0.290 [0.170, 0.424] |
| status_anx_h7 | auroc | 0.823 [0.776, 0.862] | 0.823 [0.773, 0.862] | 0.826 [0.786, 0.860] |
| status_anx_h14 | ap | 0.261 [0.132, 0.405] | 0.254 [0.134, 0.399] | 0.249 [0.129, 0.399] |
| status_anx_h14 | auroc | 0.795 [0.737, 0.840] | 0.796 [0.736, 0.841] | 0.800 [0.755, 0.839] |
| status_anx_h21 | ap | 0.233 [0.112, 0.385] | 0.233 [0.114, 0.382] | 0.232 [0.115, 0.384] |
| status_anx_h21 | auroc | 0.775 [0.714, 0.824] | 0.778 [0.715, 0.829] | 0.781 [0.719, 0.831] |
