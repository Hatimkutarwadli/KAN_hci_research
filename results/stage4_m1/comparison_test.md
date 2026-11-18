
### ALL: severity, nowcast + forecast (test)

**rmse**

| task | B0_reference | m1 |
|---|---|---|
| sev_dep_t0 | 3.965 | 3.139 ± 0.030 |
| sev_dep_h7 | 3.943 | 3.192 ± 0.029 |
| sev_dep_h14 | 3.915 | 3.317 ± 0.024 |
| sev_dep_h21 | 3.893 | 3.429 ± 0.017 |
| sev_anx_t0 | 3.710 | 3.044 ± 0.064 |
| sev_anx_h7 | 3.706 | 3.081 ± 0.057 |
| sev_anx_h14 | 3.695 | 3.188 ± 0.053 |
| sev_anx_h21 | 3.683 | 3.281 ± 0.036 |

**mae**

| task | B0_reference | m1 |
|---|---|---|
| sev_dep_t0 | 3.159 | 2.493 ± 0.022 |
| sev_dep_h7 | 3.144 | 2.531 ± 0.022 |
| sev_dep_h14 | 3.121 | 2.639 ± 0.020 |
| sev_dep_h21 | 3.101 | 2.729 ± 0.016 |
| sev_anx_t0 | 2.967 | 2.426 ± 0.046 |
| sev_anx_h7 | 2.970 | 2.462 ± 0.041 |
| sev_anx_h14 | 2.967 | 2.549 ± 0.040 |
| sev_anx_h21 | 2.961 | 2.635 ± 0.025 |

**r2**

| task | B0_reference | m1 |
|---|---|---|
| sev_dep_t0 | -0.000 | 0.373 ± 0.012 |
| sev_dep_h7 | -0.000 | 0.344 ± 0.012 |
| sev_dep_h14 | -0.000 | 0.282 ± 0.010 |
| sev_dep_h21 | -0.000 | 0.224 ± 0.008 |
| sev_anx_t0 | -0.000 | 0.327 ± 0.028 |
| sev_anx_h7 | -0.000 | 0.308 ± 0.026 |
| sev_anx_h14 | -0.000 | 0.256 ± 0.025 |
| sev_anx_h21 | -0.000 | 0.207 ± 0.017 |


### ALL: status, nowcast + forecast (test)

**ap**

| task | B0_reference | m1 |
|---|---|---|
| status_dep_t0 | 0.072 | 0.325 ± 0.014 |
| status_dep_h7 | 0.072 | 0.313 ± 0.011 |
| status_dep_h14 | 0.071 | 0.278 ± 0.011 |
| status_dep_h21 | 0.070 | 0.247 ± 0.007 |
| status_anx_t0 | 0.079 | 0.280 ± 0.011 |
| status_anx_h7 | 0.078 | 0.262 ± 0.010 |
| status_anx_h14 | 0.077 | 0.218 ± 0.008 |
| status_anx_h21 | 0.078 | 0.194 ± 0.008 |

**auroc**

| task | B0_reference | m1 |
|---|---|---|
| status_dep_t0 | 0.500 | 0.860 ± 0.003 |
| status_dep_h7 | 0.500 | 0.846 ± 0.002 |
| status_dep_h14 | 0.500 | 0.821 ± 0.003 |
| status_dep_h21 | 0.500 | 0.788 ± 0.003 |
| status_anx_t0 | 0.500 | 0.830 ± 0.004 |
| status_anx_h7 | 0.500 | 0.816 ± 0.002 |
| status_anx_h14 | 0.500 | 0.788 ± 0.004 |
| status_anx_h21 | 0.500 | 0.768 ± 0.004 |

**f1**

| task | B0_reference | m1 |
|---|---|---|
| status_dep_t0 | 0.135 | 0.404 ± 0.010 |
| status_dep_h7 | 0.134 | 0.385 ± 0.012 |
| status_dep_h14 | 0.132 | 0.349 ± 0.016 |
| status_dep_h21 | 0.131 | 0.314 ± 0.015 |
| status_anx_t0 | 0.146 | 0.365 ± 0.007 |
| status_anx_h7 | 0.145 | 0.350 ± 0.008 |
| status_anx_h14 | 0.143 | 0.301 ± 0.008 |
| status_anx_h21 | 0.145 | 0.279 ± 0.005 |

**precision**

| task | B0_reference | m1 |
|---|---|---|
| status_dep_t0 | 0.072 | 0.329 ± 0.024 |
| status_dep_h7 | 0.072 | 0.353 ± 0.066 |
| status_dep_h14 | 0.071 | 0.309 ± 0.046 |
| status_dep_h21 | 0.070 | 0.278 ± 0.043 |
| status_anx_t0 | 0.079 | 0.289 ± 0.022 |
| status_anx_h7 | 0.078 | 0.287 ± 0.016 |
| status_anx_h14 | 0.077 | 0.232 ± 0.024 |
| status_anx_h21 | 0.078 | 0.200 ± 0.010 |

**recall**

| task | B0_reference | m1 |
|---|---|---|
| status_dep_t0 | 1.000 | 0.529 ± 0.041 |
| status_dep_h7 | 1.000 | 0.449 ± 0.081 |
| status_dep_h14 | 1.000 | 0.414 ± 0.055 |
| status_dep_h21 | 1.000 | 0.374 ± 0.052 |
| status_anx_t0 | 1.000 | 0.501 ± 0.049 |
| status_anx_h7 | 1.000 | 0.452 ± 0.040 |
| status_anx_h14 | 1.000 | 0.439 ± 0.049 |
| status_anx_h21 | 1.000 | 0.462 ± 0.032 |


### ALL: sustained onset (test)

**ap**

| task | B0_reference | m1 |
|---|---|---|
| onset_sust_dep_h7 | 0.016 | 0.061 ± 0.013 |
| onset_sust_dep_h14 | 0.031 | 0.087 ± 0.010 |
| onset_sust_dep_h21 | 0.047 | 0.116 ± 0.006 |
| onset_sust_anx_h7 | 0.017 | 0.057 ± 0.003 |
| onset_sust_anx_h14 | 0.033 | 0.092 ± 0.001 |
| onset_sust_anx_h21 | 0.049 | 0.125 ± 0.004 |

**auroc**

| task | B0_reference | m1 |
|---|---|---|
| onset_sust_dep_h7 | 0.500 | 0.805 ± 0.006 |
| onset_sust_dep_h14 | 0.500 | 0.781 ± 0.005 |
| onset_sust_dep_h21 | 0.500 | 0.757 ± 0.004 |
| onset_sust_anx_h7 | 0.500 | 0.809 ± 0.003 |
| onset_sust_anx_h14 | 0.500 | 0.784 ± 0.003 |
| onset_sust_anx_h21 | 0.500 | 0.770 ± 0.004 |

**f1**

| task | B0_reference | m1 |
|---|---|---|
| onset_sust_dep_h7 | 0.031 | 0.107 ± 0.015 |
| onset_sust_dep_h14 | 0.060 | 0.112 ± 0.014 |
| onset_sust_dep_h21 | 0.089 | 0.164 ± 0.020 |
| onset_sust_anx_h7 | 0.033 | 0.107 ± 0.017 |
| onset_sust_anx_h14 | 0.064 | 0.156 ± 0.013 |
| onset_sust_anx_h21 | 0.093 | 0.189 ± 0.020 |

**precision**

| task | B0_reference | m1 |
|---|---|---|
| onset_sust_dep_h7 | 0.016 | 0.099 ± 0.021 |
| onset_sust_dep_h14 | 0.031 | 0.120 ± 0.023 |
| onset_sust_dep_h21 | 0.047 | 0.130 ± 0.009 |
| onset_sust_anx_h7 | 0.017 | 0.078 ± 0.009 |
| onset_sust_anx_h14 | 0.033 | 0.114 ± 0.007 |
| onset_sust_anx_h21 | 0.049 | 0.139 ± 0.011 |

**recall**

| task | B0_reference | m1 |
|---|---|---|
| onset_sust_dep_h7 | 1.000 | 0.121 ± 0.014 |
| onset_sust_dep_h14 | 1.000 | 0.115 ± 0.036 |
| onset_sust_dep_h21 | 1.000 | 0.247 ± 0.092 |
| onset_sust_anx_h7 | 1.000 | 0.183 ± 0.067 |
| onset_sust_anx_h14 | 1.000 | 0.273 ± 0.094 |
| onset_sust_anx_h21 | 1.000 | 0.324 ± 0.099 |


### ALL: first-crossing onset (test, separate)

**ap**

| task | B0_reference | m1 |
|---|---|---|
| onset_cross_dep_h7 | 0.022 | 0.080 ± 0.007 |
| onset_cross_dep_h14 | 0.042 | 0.114 ± 0.005 |
| onset_cross_dep_h21 | 0.060 | 0.141 ± 0.007 |
| onset_cross_anx_h7 | 0.022 | 0.061 ± 0.002 |
| onset_cross_anx_h14 | 0.044 | 0.097 ± 0.002 |
| onset_cross_anx_h21 | 0.063 | 0.126 ± 0.004 |

**auroc**

| task | B0_reference | m1 |
|---|---|---|
| onset_cross_dep_h7 | 0.500 | 0.799 ± 0.003 |
| onset_cross_dep_h14 | 0.500 | 0.785 ± 0.003 |
| onset_cross_dep_h21 | 0.500 | 0.757 ± 0.006 |
| onset_cross_anx_h7 | 0.500 | 0.742 ± 0.010 |
| onset_cross_anx_h14 | 0.500 | 0.716 ± 0.006 |
| onset_cross_anx_h21 | 0.500 | 0.710 ± 0.006 |

**f1**

| task | B0_reference | m1 |
|---|---|---|
| onset_cross_dep_h7 | 0.043 | 0.109 ± 0.012 |
| onset_cross_dep_h14 | 0.080 | 0.121 ± 0.030 |
| onset_cross_dep_h21 | 0.113 | 0.220 ± 0.026 |
| onset_cross_anx_h7 | 0.044 | 0.130 ± 0.010 |
| onset_cross_anx_h14 | 0.084 | 0.177 ± 0.005 |
| onset_cross_anx_h21 | 0.119 | 0.212 ± 0.006 |

**precision**

| task | B0_reference | m1 |
|---|---|---|
| onset_cross_dep_h7 | 0.022 | 0.109 ± 0.014 |
| onset_cross_dep_h14 | 0.042 | 0.131 ± 0.009 |
| onset_cross_dep_h21 | 0.060 | 0.157 ± 0.012 |
| onset_cross_anx_h7 | 0.022 | 0.081 ± 0.007 |
| onset_cross_anx_h14 | 0.044 | 0.118 ± 0.009 |
| onset_cross_anx_h21 | 0.063 | 0.139 ± 0.009 |

**recall**

| task | B0_reference | m1 |
|---|---|---|
| onset_cross_dep_h7 | 1.000 | 0.110 ± 0.020 |
| onset_cross_dep_h14 | 1.000 | 0.125 ± 0.055 |
| onset_cross_dep_h21 | 1.000 | 0.373 ± 0.080 |
| onset_cross_anx_h7 | 1.000 | 0.335 ± 0.023 |
| onset_cross_anx_h14 | 1.000 | 0.368 ± 0.060 |
| onset_cross_anx_h21 | 1.000 | 0.461 ± 0.077 |
