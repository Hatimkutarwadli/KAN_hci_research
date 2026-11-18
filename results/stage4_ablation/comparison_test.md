
### ALL: severity, nowcast + forecast (test)

**rmse**

| task | B0_reference | m1_static |
|---|---|---|
| sev_dep_t0 | 3.965 | 3.142 ± 0.039 |
| sev_dep_h7 | 3.943 | 3.240 ± 0.023 |
| sev_dep_h14 | 3.915 | 3.379 ± 0.023 |
| sev_dep_h21 | 3.893 | 3.484 ± 0.014 |
| sev_anx_t0 | 3.710 | 3.063 ± 0.043 |
| sev_anx_h7 | 3.706 | 3.132 ± 0.044 |
| sev_anx_h14 | 3.695 | 3.245 ± 0.046 |
| sev_anx_h21 | 3.683 | 3.335 ± 0.023 |

**mae**

| task | B0_reference | m1_static |
|---|---|---|
| sev_dep_t0 | 3.159 | 2.494 ± 0.033 |
| sev_dep_h7 | 3.144 | 2.577 ± 0.020 |
| sev_dep_h14 | 3.121 | 2.700 ± 0.025 |
| sev_dep_h21 | 3.101 | 2.787 ± 0.017 |
| sev_anx_t0 | 2.967 | 2.443 ± 0.033 |
| sev_anx_h7 | 2.970 | 2.505 ± 0.031 |
| sev_anx_h14 | 2.967 | 2.602 ± 0.037 |
| sev_anx_h21 | 2.961 | 2.685 ± 0.018 |

**r2**

| task | B0_reference | m1_static |
|---|---|---|
| sev_dep_t0 | -0.000 | 0.372 ± 0.016 |
| sev_dep_h7 | -0.000 | 0.325 ± 0.009 |
| sev_dep_h14 | -0.000 | 0.255 ± 0.010 |
| sev_dep_h21 | -0.000 | 0.199 ± 0.006 |
| sev_anx_t0 | -0.000 | 0.318 ± 0.019 |
| sev_anx_h7 | -0.000 | 0.285 ± 0.020 |
| sev_anx_h14 | -0.000 | 0.229 ± 0.022 |
| sev_anx_h21 | -0.000 | 0.180 ± 0.011 |


### ALL: status, nowcast + forecast (test)

**ap**

| task | B0_reference | m1_static |
|---|---|---|
| status_dep_t0 | 0.072 | 0.334 ± 0.013 |
| status_dep_h7 | 0.072 | 0.299 ± 0.019 |
| status_dep_h14 | 0.071 | 0.258 ± 0.011 |
| status_dep_h21 | 0.070 | 0.231 ± 0.009 |
| status_anx_t0 | 0.079 | 0.268 ± 0.011 |
| status_anx_h7 | 0.078 | 0.235 ± 0.008 |
| status_anx_h14 | 0.077 | 0.205 ± 0.006 |
| status_anx_h21 | 0.078 | 0.183 ± 0.008 |

**auroc**

| task | B0_reference | m1_static |
|---|---|---|
| status_dep_t0 | 0.500 | 0.861 ± 0.004 |
| status_dep_h7 | 0.500 | 0.833 ± 0.007 |
| status_dep_h14 | 0.500 | 0.806 ± 0.005 |
| status_dep_h21 | 0.500 | 0.776 ± 0.005 |
| status_anx_t0 | 0.500 | 0.818 ± 0.011 |
| status_anx_h7 | 0.500 | 0.793 ± 0.016 |
| status_anx_h14 | 0.500 | 0.776 ± 0.010 |
| status_anx_h21 | 0.500 | 0.750 ± 0.011 |

**f1**

| task | B0_reference | m1_static |
|---|---|---|
| status_dep_t0 | 0.135 | 0.392 ± 0.005 |
| status_dep_h7 | 0.134 | 0.359 ± 0.014 |
| status_dep_h14 | 0.132 | 0.310 ± 0.011 |
| status_dep_h21 | 0.131 | 0.276 ± 0.011 |
| status_anx_t0 | 0.146 | 0.348 ± 0.015 |
| status_anx_h7 | 0.145 | 0.315 ± 0.011 |
| status_anx_h14 | 0.143 | 0.282 ± 0.012 |
| status_anx_h21 | 0.145 | 0.261 ± 0.015 |

**precision**

| task | B0_reference | m1_static |
|---|---|---|
| status_dep_t0 | 0.072 | 0.326 ± 0.014 |
| status_dep_h7 | 0.072 | 0.326 ± 0.022 |
| status_dep_h14 | 0.071 | 0.274 ± 0.015 |
| status_dep_h21 | 0.070 | 0.213 ± 0.042 |
| status_anx_t0 | 0.079 | 0.286 ± 0.011 |
| status_anx_h7 | 0.078 | 0.263 ± 0.027 |
| status_anx_h14 | 0.077 | 0.228 ± 0.028 |
| status_anx_h21 | 0.078 | 0.217 ± 0.013 |

**recall**

| task | B0_reference | m1_static |
|---|---|---|
| status_dep_t0 | 1.000 | 0.494 ± 0.038 |
| status_dep_h7 | 1.000 | 0.405 ± 0.057 |
| status_dep_h14 | 1.000 | 0.362 ± 0.056 |
| status_dep_h21 | 1.000 | 0.446 ± 0.140 |
| status_anx_t0 | 1.000 | 0.446 ± 0.048 |
| status_anx_h7 | 1.000 | 0.405 ± 0.066 |
| status_anx_h14 | 1.000 | 0.387 ± 0.069 |
| status_anx_h21 | 1.000 | 0.337 ± 0.073 |


### ALL: sustained onset (test)

**ap**

| task | B0_reference | m1_static |
|---|---|---|
| onset_sust_dep_h7 | 0.016 | 0.043 ± 0.006 |
| onset_sust_dep_h14 | 0.031 | 0.068 ± 0.010 |
| onset_sust_dep_h21 | 0.047 | 0.094 ± 0.011 |
| onset_sust_anx_h7 | 0.017 | 0.063 ± 0.014 |
| onset_sust_anx_h14 | 0.033 | 0.094 ± 0.006 |
| onset_sust_anx_h21 | 0.049 | 0.123 ± 0.006 |

**auroc**

| task | B0_reference | m1_static |
|---|---|---|
| onset_sust_dep_h7 | 0.500 | 0.777 ± 0.022 |
| onset_sust_dep_h14 | 0.500 | 0.736 ± 0.036 |
| onset_sust_dep_h21 | 0.500 | 0.722 ± 0.021 |
| onset_sust_anx_h7 | 0.500 | 0.759 ± 0.048 |
| onset_sust_anx_h14 | 0.500 | 0.762 ± 0.006 |
| onset_sust_anx_h21 | 0.500 | 0.742 ± 0.028 |

**f1**

| task | B0_reference | m1_static |
|---|---|---|
| onset_sust_dep_h7 | 0.031 | 0.077 ± 0.011 |
| onset_sust_dep_h14 | 0.060 | 0.116 ± 0.014 |
| onset_sust_dep_h21 | 0.089 | 0.163 ± 0.032 |
| onset_sust_anx_h7 | 0.033 | 0.105 ± 0.010 |
| onset_sust_anx_h14 | 0.064 | 0.144 ± 0.017 |
| onset_sust_anx_h21 | 0.093 | 0.187 ± 0.012 |

**precision**

| task | B0_reference | m1_static |
|---|---|---|
| onset_sust_dep_h7 | 0.016 | 0.054 ± 0.010 |
| onset_sust_dep_h14 | 0.031 | 0.079 ± 0.004 |
| onset_sust_dep_h21 | 0.047 | 0.111 ± 0.015 |
| onset_sust_anx_h7 | 0.017 | 0.076 ± 0.008 |
| onset_sust_anx_h14 | 0.033 | 0.100 ± 0.014 |
| onset_sust_anx_h21 | 0.049 | 0.136 ± 0.015 |

**recall**

| task | B0_reference | m1_static |
|---|---|---|
| onset_sust_dep_h7 | 1.000 | 0.145 ± 0.044 |
| onset_sust_dep_h14 | 1.000 | 0.226 ± 0.080 |
| onset_sust_dep_h21 | 1.000 | 0.330 ± 0.149 |
| onset_sust_anx_h7 | 1.000 | 0.200 ± 0.085 |
| onset_sust_anx_h14 | 1.000 | 0.263 ± 0.049 |
| onset_sust_anx_h21 | 1.000 | 0.322 ± 0.079 |


### ALL: first-crossing onset (test, separate)

**ap**

| task | B0_reference | m1_static |
|---|---|---|
| onset_cross_dep_h7 | 0.022 | 0.061 ± 0.008 |
| onset_cross_dep_h14 | 0.042 | 0.092 ± 0.009 |
| onset_cross_dep_h21 | 0.060 | 0.116 ± 0.011 |
| onset_cross_anx_h7 | 0.022 | 0.054 ± 0.003 |
| onset_cross_anx_h14 | 0.044 | 0.093 ± 0.002 |
| onset_cross_anx_h21 | 0.063 | 0.124 ± 0.003 |

**auroc**

| task | B0_reference | m1_static |
|---|---|---|
| onset_cross_dep_h7 | 0.500 | 0.777 ± 0.016 |
| onset_cross_dep_h14 | 0.500 | 0.750 ± 0.016 |
| onset_cross_dep_h21 | 0.500 | 0.720 ± 0.017 |
| onset_cross_anx_h7 | 0.500 | 0.714 ± 0.011 |
| onset_cross_anx_h14 | 0.500 | 0.702 ± 0.014 |
| onset_cross_anx_h21 | 0.500 | 0.697 ± 0.012 |

**f1**

| task | B0_reference | m1_static |
|---|---|---|
| onset_cross_dep_h7 | 0.043 | 0.100 ± 0.022 |
| onset_cross_dep_h14 | 0.080 | 0.148 ± 0.033 |
| onset_cross_dep_h21 | 0.113 | 0.188 ± 0.026 |
| onset_cross_anx_h7 | 0.044 | 0.097 ± 0.028 |
| onset_cross_anx_h14 | 0.084 | 0.159 ± 0.007 |
| onset_cross_anx_h21 | 0.119 | 0.190 ± 0.008 |

**precision**

| task | B0_reference | m1_static |
|---|---|---|
| onset_cross_dep_h7 | 0.022 | 0.074 ± 0.008 |
| onset_cross_dep_h14 | 0.042 | 0.101 ± 0.014 |
| onset_cross_dep_h21 | 0.060 | 0.138 ± 0.015 |
| onset_cross_anx_h7 | 0.022 | 0.062 ± 0.009 |
| onset_cross_anx_h14 | 0.044 | 0.100 ± 0.005 |
| onset_cross_anx_h21 | 0.063 | 0.117 ± 0.010 |

**recall**

| task | B0_reference | m1_static |
|---|---|---|
| onset_cross_dep_h7 | 1.000 | 0.229 ± 0.189 |
| onset_cross_dep_h14 | 1.000 | 0.293 ± 0.137 |
| onset_cross_dep_h21 | 1.000 | 0.295 ± 0.069 |
| onset_cross_anx_h7 | 1.000 | 0.283 ± 0.137 |
| onset_cross_anx_h14 | 1.000 | 0.391 ± 0.046 |
| onset_cross_anx_h21 | 1.000 | 0.532 ± 0.098 |
