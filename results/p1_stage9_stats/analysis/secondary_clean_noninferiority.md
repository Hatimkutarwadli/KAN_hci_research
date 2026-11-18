# P1-6 SECONDARY DESCRIPTIVE: clean-control non-inferiority vs the original GRU (SYNTHETIC; EXTENSION)

Not part of the confirmatory family and not adjusted. Margin = Stage 9's pre-specified rule: the original GRU's seed SD as a percentage of its mean on the clean control, computed from the CALIBRATION Stage 8 metrics before any test result (value fixed, not tuned). Selected finalists are those chosen by the calibration decision file (forecast severity: both R0O2T1 and R0O3T1, 'not separable'). worse% > 0 = worse than the original. 'Upper 95%' = 95th percentile of the one-sided participant-bootstrap distribution (conditional on the 5 seeds).

| family | finalist | group | clean worse% (Stage 9 point rule: <= margin) | margin | participant 95% interval | upper 95% bound | upper bound <= margin? | seeds worse than original |
|---|---|---|---|---|---|---|---|---|
| severity (nowcast) | R0O2T1 | sev_nowcast_rmse | +0.00 (pass) | 0.21 | [-0.32, +0.33] | +0.28 | no | 3/5 |
| status (nowcast) | R0O3T1 | status_ap | +2.70 (above margin) | 2.22 | [+0.14, +5.30] | +4.86 | no | 5/5 |
| onset | R0O0T1 | onset_ap | +2.08 (pass) | 5.21 | [-0.32, +3.90] | +3.56 | yes | 3/5 |
| onset | R0O0T1 | onset_auroc | +0.16 (pass) | 0.28 | [-0.12, +0.50] | +0.43 | no | 3/5 |
| forecast severity | R0O2T1 | sev_forecast_rmse | +0.07 (pass) | 0.28 | [-0.15, +0.29] | +0.26 | yes | 5/5 |
| forecast severity | R0O3T1 | sev_forecast_rmse | +0.26 (pass) | 0.28 | [-0.04, +0.58] | +0.52 | no | 5/5 |
| forecast status | R0O3T1 | status_ap | +2.33 (above margin) | 2.00 | [-2.15, +5.73] | +5.14 | no | 5/5 |
