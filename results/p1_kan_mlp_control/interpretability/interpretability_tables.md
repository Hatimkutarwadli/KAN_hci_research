# P1-3 interpretability tables (SYNTHETIC; EXTENSION). Calibration split, fixed 2,000 origins, window-shift analysis. Recovered sign/shape/rank describe the PREDICTOR, not a causal effect.


## 1. Recovery of the planted structure, severity heads (dep: sleep variability +, steps -; anx: steps -, HRV -); mean ±SD over 5 seeds

| model | sign accuracy (chance 0.5) | shape r with planted g | mean rank of planted channels (chance 4.5; 1 = best) | planted in top 3 (chance 0.375) | calories ranked above steps |
|---|---|---|---|---|---|
| gru | 1.00 ±0.00 | 0.99 ±0.00 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| kan | 1.00 ±0.00 | 0.99 ±0.00 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| kan_reg | 1.00 ±0.00 | 0.99 ±0.00 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| kan_nospline | 1.00 ±0.00 | 0.98 ±0.00 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| mlp_b2 | 1.00 ±0.00 | 0.99 ±0.00 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| mlp_b2_reg | 1.00 ±0.00 | 0.99 ±0.00 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| m1 | 1.00 ±0.00 | 0.99 ±0.00 | 1.70 ±0.11 | 1.00 ±0.00 | 0.00 ±0.00 |
| mlp_m1 | 1.00 ±0.00 | 0.99 ±0.00 | 1.65 ±0.14 | 1.00 ±0.00 | 0.00 ±0.00 |
| gru_untrained | 0.60 ±0.29 | 0.20 ±0.54 | 4.45 ±0.69 | 0.35 ±0.14 | 0.60 ±0.42 |
| kan_untrained | 0.55 ±0.21 | 0.04 ±0.37 | 4.75 ±1.43 | 0.35 ±0.29 | 0.60 ±0.42 |
| kan_nospline_untrained | 0.50 ±0.25 | -0.05 ±0.40 | 5.20 ±1.55 | 0.25 ±0.25 | 0.60 ±0.42 |
| mlp_b2_untrained | 0.35 ±0.14 | -0.39 ±0.26 | 4.90 ±1.10 | 0.40 ±0.22 | 0.60 ±0.22 |
| m1_untrained | 0.40 ±0.22 | -0.19 ±0.36 | 4.20 ±0.86 | 0.40 ±0.22 | 0.70 ±0.27 |
| mlp_m1_untrained | 0.60 ±0.29 | 0.20 ±0.47 | 5.10 ±0.84 | 0.35 ±0.14 | 0.50 ±0.35 |

## 1b. Same, status (probability) heads

| model | sign accuracy (chance 0.5) | shape r with planted g | mean rank of planted channels (chance 4.5; 1 = best) | planted in top 3 (chance 0.375) | calories ranked above steps |
|---|---|---|---|---|---|
| gru | 1.00 ±0.00 | 0.95 ±0.01 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| kan | 1.00 ±0.00 | 0.98 ±0.00 | 1.65 ±0.14 | 1.00 ±0.00 | 0.00 ±0.00 |
| kan_reg | 1.00 ±0.00 | 0.98 ±0.00 | 1.85 ±0.22 | 1.00 ±0.00 | 0.00 ±0.00 |
| kan_nospline | 1.00 ±0.00 | 0.97 ±0.00 | 1.95 ±0.21 | 0.90 ±0.14 | 0.00 ±0.00 |
| mlp_b2 | 1.00 ±0.00 | 0.95 ±0.01 | 1.50 ±0.00 | 1.00 ±0.00 | 0.00 ±0.00 |
| mlp_b2_reg | 1.00 ±0.00 | 0.95 ±0.00 | 1.55 ±0.11 | 1.00 ±0.00 | 0.00 ±0.00 |
| m1 | 1.00 ±0.00 | 0.99 ±0.00 | 1.80 ±0.21 | 0.95 ±0.11 | 0.00 ±0.00 |
| mlp_m1 | 1.00 ±0.00 | 0.97 ±0.00 | 1.60 ±0.14 | 1.00 ±0.00 | 0.00 ±0.00 |
| gru_untrained | 0.65 ±0.29 | 0.33 ±0.50 | 4.60 ±1.13 | 0.30 ±0.21 | 0.70 ±0.45 |
| kan_untrained | 0.50 ±0.18 | -0.00 ±0.37 | 3.95 ±1.08 | 0.45 ±0.21 | 0.10 ±0.22 |
| kan_nospline_untrained | 0.50 ±0.18 | -0.00 ±0.34 | 3.90 ±1.40 | 0.55 ±0.27 | 0.00 ±0.00 |
| mlp_b2_untrained | 0.35 ±0.22 | -0.25 ±0.38 | 5.35 ±0.78 | 0.15 ±0.14 | 0.80 ±0.27 |
| m1_untrained | 0.70 ±0.33 | 0.32 ±0.54 | 5.15 ±0.91 | 0.20 ±0.21 | 0.60 ±0.42 |
| mlp_m1_untrained | 0.65 ±0.22 | 0.33 ±0.35 | 4.90 ±0.89 | 0.30 ±0.11 | 0.60 ±0.42 |

## 2. Paired seed comparisons (severity heads). KAN vs matched MLP = interpretability difference at matched parameters; KAN vs no-spline = contribution of the splines

| comparison (first vs second) | metric | mean difference (first - second) | seeds where first is better |
|---|---|---|---|
| kan vs mlp_b2 | sign_acc | +0.000 (SD 0.000) | 0/5 |
| kan vs mlp_b2 | shape_r | +0.002 (SD 0.002) | 5/5 |
| kan vs mlp_b2 | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| kan vs mlp_b2 | mean_rank_planted | +0.000 (SD 0.000) | 0/5 |
| kan_reg vs mlp_b2_reg | sign_acc | +0.000 (SD 0.000) | 0/5 |
| kan_reg vs mlp_b2_reg | shape_r | -0.001 (SD 0.001) | 3/5 |
| kan_reg vs mlp_b2_reg | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| kan_reg vs mlp_b2_reg | mean_rank_planted | +0.000 (SD 0.000) | 0/5 |
| m1 vs mlp_m1 | sign_acc | +0.000 (SD 0.000) | 0/5 |
| m1 vs mlp_m1 | shape_r | -0.001 (SD 0.002) | 3/5 |
| m1 vs mlp_m1 | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| m1 vs mlp_m1 | mean_rank_planted | +0.050 (SD 0.209) | 1/5 |
| kan vs kan_nospline | sign_acc | +0.000 (SD 0.000) | 0/5 |
| kan vs kan_nospline | shape_r | +0.013 (SD 0.004) | 5/5 |
| kan vs kan_nospline | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| kan vs kan_nospline | mean_rank_planted | +0.000 (SD 0.000) | 0/5 |
| kan vs gru | sign_acc | +0.000 (SD 0.000) | 0/5 |
| kan vs gru | shape_r | +0.003 (SD 0.002) | 5/5 |
| kan vs gru | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| kan vs gru | mean_rank_planted | +0.000 (SD 0.000) | 0/5 |
| mlp_b2 vs gru | sign_acc | +0.000 (SD 0.000) | 0/5 |
| mlp_b2 vs gru | shape_r | +0.000 (SD 0.001) | 3/5 |
| mlp_b2 vs gru | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| mlp_b2 vs gru | mean_rank_planted | +0.000 (SD 0.000) | 0/5 |
| m1 vs gru | sign_acc | +0.000 (SD 0.000) | 0/5 |
| m1 vs gru | shape_r | +0.000 (SD 0.002) | 3/5 |
| m1 vs gru | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| m1 vs gru | mean_rank_planted | +0.200 (SD 0.112) | 0/5 |
| mlp_m1 vs gru | sign_acc | +0.000 (SD 0.000) | 0/5 |
| mlp_m1 vs gru | shape_r | +0.001 (SD 0.001) | 5/5 |
| mlp_m1 vs gru | recall_top3 | +0.000 (SD 0.000) | 0/5 |
| mlp_m1 vs gru | mean_rank_planted | +0.150 (SD 0.137) | 0/5 |

## 3. Per planted pair, severity heads: seeds with correct slope sign (of 5) / mean shape r / mean rank among 8 channels

| model | dep: sleep_variability | dep: steps | anx: steps | anx: hrv |
|---|---|---|---|---|
| gru | 5/5, r=0.98, rank 2.0 | 5/5, r=0.99, rank 1.0 | 5/5, r=0.99, rank 1.6 | 5/5, r=0.99, rank 1.4 |
| kan | 5/5, r=0.99, rank 2.0 | 5/5, r=1.00, rank 1.0 | 5/5, r=1.00, rank 1.8 | 5/5, r=0.99, rank 1.2 |
| kan_reg | 5/5, r=0.98, rank 2.0 | 5/5, r=1.00, rank 1.0 | 5/5, r=1.00, rank 1.0 | 5/5, r=0.99, rank 2.0 |
| kan_nospline | 5/5, r=0.97, rank 2.0 | 5/5, r=0.99, rank 1.0 | 5/5, r=0.99, rank 1.4 | 5/5, r=0.98, rank 1.6 |
| mlp_b2 | 5/5, r=0.98, rank 2.0 | 5/5, r=0.99, rank 1.0 | 5/5, r=0.99, rank 2.0 | 5/5, r=0.99, rank 1.0 |
| mlp_b2_reg | 5/5, r=0.99, rank 2.0 | 5/5, r=1.00, rank 1.0 | 5/5, r=1.00, rank 1.2 | 5/5, r=0.99, rank 1.8 |
| m1 | 5/5, r=0.97, rank 2.0 | 5/5, r=1.00, rank 1.0 | 5/5, r=1.00, rank 1.0 | 5/5, r=0.99, rank 2.8 |
| mlp_m1 | 5/5, r=0.98, rank 2.2 | 5/5, r=1.00, rank 1.0 | 5/5, r=1.00, rank 1.0 | 5/5, r=0.99, rank 2.4 |
| gru_untrained | 2/5, r=-0.05, rank 5.0 | 3/5, r=0.16, rank 4.6 | 3/5, r=0.19, rank 3.6 | 4/5, r=0.49, rank 4.6 |
| kan_untrained | 2/5, r=-0.19, rank 3.8 | 4/5, r=0.51, rank 4.4 | 1/5, r=-0.58, rank 5.2 | 4/5, r=0.42, rank 5.6 |
| kan_nospline_untrained | 2/5, r=-0.19, rank 3.6 | 3/5, r=-0.03, rank 5.4 | 1/5, r=-0.53, rank 5.8 | 4/5, r=0.54, rank 6.0 |
| mlp_b2_untrained | 0/5, r=-0.97, rank 3.8 | 4/5, r=0.37, rank 5.8 | 1/5, r=-0.65, rank 5.8 | 2/5, r=-0.30, rank 4.2 |
| m1_untrained | 2/5, r=-0.30, rank 4.4 | 2/5, r=-0.17, rank 5.0 | 1/5, r=-0.49, rank 4.6 | 3/5, r=0.20, rank 2.8 |
| mlp_m1_untrained | 1/5, r=-0.48, rank 6.4 | 2/5, r=-0.08, rank 5.8 | 5/5, r=0.79, rank 3.2 | 4/5, r=0.57, rank 5.0 |

## 4. Most sensitive channel (window-shift range / SD of predictions): counts over 5 seeds, severity heads

| model | head | steps | calories | sleep_duration | sleep_efficiency | sleep_variability | resting_hr | hrv | circadian_regularity |
|---|---|---|---|---|---|---|---|---|---|
| gru | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gru | sev_anx_t0 | 2 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| kan | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kan | sev_anx_t0 | 1 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| kan_reg | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kan_reg | sev_anx_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kan_nospline | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kan_nospline | sev_anx_t0 | 3 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| mlp_b2 | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mlp_b2 | sev_anx_t0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| mlp_b2_reg | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mlp_b2_reg | sev_anx_t0 | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| m1 | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| m1 | sev_anx_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mlp_m1 | sev_dep_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mlp_m1 | sev_anx_t0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gru_untrained | sev_dep_t0 | 0 | 2 | 1 | 1 | 0 | 1 | 0 | 0 |
| gru_untrained | sev_anx_t0 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 |
| kan_untrained | sev_dep_t0 | 1 | 1 | 2 | 0 | 0 | 0 | 1 | 0 |
| kan_untrained | sev_anx_t0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| kan_nospline_untrained | sev_dep_t0 | 1 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| kan_nospline_untrained | sev_anx_t0 | 0 | 1 | 2 | 1 | 1 | 0 | 0 | 0 |
| mlp_b2_untrained | sev_dep_t0 | 0 | 0 | 1 | 1 | 0 | 2 | 1 | 0 |
| mlp_b2_untrained | sev_anx_t0 | 0 | 1 | 0 | 0 | 0 | 1 | 2 | 1 |
| m1_untrained | sev_dep_t0 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 |
| m1_untrained | sev_anx_t0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 1 |
| mlp_m1_untrained | sev_dep_t0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| mlp_m1_untrained | sev_anx_t0 | 0 | 1 | 0 | 2 | 0 | 1 | 0 | 1 |

### Association-only reference (no model): Pearson r between each channel's window mean and the severity target, calibration origins

| head | steps | calories | sleep_duration | sleep_efficiency | sleep_variability | resting_hr | hrv | circadian_regularity |
|---|---|---|---|---|---|---|---|---|
| sev_anx_t0 | -0.333 | -0.274 | -0.086 | +0.021 | +0.381 | +0.281 | -0.472 | -0.123 |
| sev_dep_t0 | -0.532 | -0.393 | -0.085 | -0.036 | +0.486 | +0.338 | -0.349 | -0.159 |

Association-only baseline vs planted pairs: dep/sleep_variability: sign ok, |r| rank 2; dep/steps: sign ok, |r| rank 1; anx/steps: sign ok, |r| rank 3; anx/hrv: sign ok, |r| rank 1


## 5. KAN first-layer edge importance (mean |phi| over x in [-3, 3], summed over outputs, lags and the 3 scales), share of total, mean over 5 seeds

| model | steps | calories | sleep_duration | sleep_efficiency | sleep_variability | resting_hr | hrv | circadian_regularity | missing_day | hrv_missing |
|---|---|---|---|---|---|---|---|---|---|---|
| kan | 0.108 | 0.099 | 0.101 | 0.098 | 0.100 | 0.101 | 0.112 | 0.103 | 0.089 | 0.089 |
| kan_nospline | 0.114 | 0.098 | 0.096 | 0.090 | 0.103 | 0.098 | 0.111 | 0.097 | 0.096 | 0.096 |
| m1 | 0.138 | 0.090 | 0.099 | 0.095 | 0.126 | 0.114 | 0.116 | 0.090 | 0.069 | 0.064 |

Spearman correlation between first-layer edge importance and window-shift sensitivity over the 8 value channels (per seed, mean ±SD):

| model | head | rho |
|---|---|---|
| kan | sev_dep_t0 | +0.34 ±0.29 |
| kan | sev_anx_t0 | +0.56 ±0.16 |
| kan_nospline | sev_dep_t0 | +0.65 ±0.11 |
| kan_nospline | sev_anx_t0 | +0.77 ±0.15 |
| m1 | sev_dep_t0 | +0.87 ±0.10 |
| m1 | sev_anx_t0 | +0.89 ±0.06 |

## 6. Lag profile: mean change of the severity prediction for +1 SD on ONE day t-lag (mean over 5 seeds); planted lags are 1-2 days and the latent state integrates them, so a profile concentrated on recent days is plausible but not a lag estimate


sev_dep_t0 / sleep_variability

| model | lag 0 | lag 1 | lag 2 | lag 3 | lag 5 | lag 7 | lag 10 | lag 14 | lag 20 | lag 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| gru | +0.048 | +0.075 | +0.079 | +0.077 | +0.071 | +0.064 | +0.055 | +0.046 | +0.042 | +0.048 |
| kan | +0.031 | +0.048 | +0.059 | +0.063 | +0.065 | +0.065 | +0.063 | +0.062 | +0.069 | +0.012 |
| kan_reg | +0.029 | +0.044 | +0.054 | +0.056 | +0.057 | +0.055 | +0.054 | +0.053 | +0.059 | +0.011 |
| kan_nospline | +0.031 | +0.049 | +0.059 | +0.068 | +0.074 | +0.073 | +0.072 | +0.074 | +0.080 | +0.008 |
| mlp_b2 | +0.024 | +0.044 | +0.052 | +0.054 | +0.058 | +0.055 | +0.055 | +0.053 | +0.059 | +0.012 |
| mlp_b2_reg | +0.032 | +0.045 | +0.054 | +0.057 | +0.057 | +0.054 | +0.054 | +0.053 | +0.059 | +0.012 |
| m1 | +0.058 | +0.069 | +0.066 | +0.059 | +0.047 | +0.045 | +0.031 | +0.021 | +0.012 | +0.031 |
| mlp_m1 | +0.069 | +0.080 | +0.083 | +0.069 | +0.057 | +0.051 | +0.036 | +0.025 | +0.016 | +0.028 |

sev_dep_t0 / steps

| model | lag 0 | lag 1 | lag 2 | lag 3 | lag 5 | lag 7 | lag 10 | lag 14 | lag 20 | lag 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| gru | -0.134 | -0.136 | -0.130 | -0.122 | -0.109 | -0.096 | -0.081 | -0.066 | -0.057 | -0.061 |
| kan | -0.035 | -0.062 | -0.085 | -0.091 | -0.096 | -0.091 | -0.090 | -0.088 | -0.095 | -0.023 |
| kan_reg | -0.035 | -0.056 | -0.074 | -0.078 | -0.081 | -0.076 | -0.075 | -0.073 | -0.079 | -0.019 |
| kan_nospline | -0.041 | -0.070 | -0.101 | -0.111 | -0.117 | -0.113 | -0.111 | -0.110 | -0.120 | -0.026 |
| mlp_b2 | -0.036 | -0.060 | -0.074 | -0.080 | -0.085 | -0.082 | -0.081 | -0.079 | -0.087 | -0.017 |
| mlp_b2_reg | -0.046 | -0.067 | -0.080 | -0.085 | -0.085 | -0.079 | -0.078 | -0.076 | -0.083 | -0.013 |
| m1 | -0.090 | -0.117 | -0.111 | -0.100 | -0.092 | -0.079 | -0.056 | -0.038 | -0.023 | -0.060 |
| mlp_m1 | -0.112 | -0.133 | -0.134 | -0.119 | -0.105 | -0.088 | -0.064 | -0.045 | -0.029 | -0.054 |

sev_anx_t0 / steps

| model | lag 0 | lag 1 | lag 2 | lag 3 | lag 5 | lag 7 | lag 10 | lag 14 | lag 20 | lag 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| gru | -0.054 | -0.074 | -0.076 | -0.074 | -0.067 | -0.060 | -0.051 | -0.043 | -0.038 | -0.043 |
| kan | -0.021 | -0.043 | -0.059 | -0.063 | -0.063 | -0.059 | -0.059 | -0.058 | -0.064 | -0.013 |
| kan_reg | -0.026 | -0.042 | -0.055 | -0.056 | -0.057 | -0.054 | -0.053 | -0.052 | -0.057 | -0.014 |
| kan_nospline | -0.024 | -0.050 | -0.074 | -0.079 | -0.079 | -0.076 | -0.073 | -0.074 | -0.081 | -0.016 |
| mlp_b2 | -0.024 | -0.041 | -0.049 | -0.053 | -0.054 | -0.051 | -0.050 | -0.049 | -0.055 | -0.010 |
| mlp_b2_reg | -0.035 | -0.050 | -0.059 | -0.060 | -0.059 | -0.056 | -0.055 | -0.053 | -0.058 | -0.009 |
| m1 | -0.065 | -0.085 | -0.083 | -0.076 | -0.073 | -0.064 | -0.046 | -0.032 | -0.019 | -0.050 |
| mlp_m1 | -0.083 | -0.099 | -0.100 | -0.088 | -0.078 | -0.067 | -0.049 | -0.035 | -0.023 | -0.043 |

sev_anx_t0 / hrv

| model | lag 0 | lag 1 | lag 2 | lag 3 | lag 5 | lag 7 | lag 10 | lag 14 | lag 20 | lag 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| gru | -0.122 | -0.104 | -0.091 | -0.081 | -0.069 | -0.059 | -0.049 | -0.039 | -0.033 | -0.037 |
| kan | -0.023 | -0.042 | -0.059 | -0.064 | -0.067 | -0.064 | -0.064 | -0.063 | -0.068 | -0.020 |
| kan_reg | -0.022 | -0.035 | -0.046 | -0.046 | -0.047 | -0.046 | -0.046 | -0.044 | -0.048 | -0.016 |
| kan_nospline | -0.024 | -0.049 | -0.064 | -0.069 | -0.073 | -0.070 | -0.070 | -0.068 | -0.074 | -0.020 |
| mlp_b2 | -0.024 | -0.038 | -0.048 | -0.053 | -0.056 | -0.055 | -0.054 | -0.054 | -0.058 | -0.014 |
| mlp_b2_reg | -0.032 | -0.046 | -0.053 | -0.051 | -0.052 | -0.051 | -0.051 | -0.050 | -0.054 | -0.013 |
| m1 | -0.031 | -0.035 | -0.037 | -0.034 | -0.039 | -0.039 | -0.028 | -0.020 | -0.012 | -0.035 |
| mlp_m1 | -0.039 | -0.043 | -0.049 | -0.049 | -0.055 | -0.048 | -0.036 | -0.025 | -0.016 | -0.039 |
