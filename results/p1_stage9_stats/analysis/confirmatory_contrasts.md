# P1-6 CONFIRMATORY panel-mean contrasts (SYNTHETIC; EXTENSION). Test split, 200 participants, 10 pre-specified panel scenarios, equal weights

worse% (NEGATIVE = new configuration better). Point = mean over the 5 seeds of the per-seed panel-mean worse% (the Stage 9 report cell). **Participant-level 95% interval** = percentile interval of the participant-cluster bootstrap (B = 2000, shared resamples), CONDITIONAL on the five trained seeds; **Bonferroni interval** = simultaneous over the 25 contrasts (99.8%). **p** = two-sided centred bootstrap test of H0: effect = 0; **Holm** = adjusted over the 25 confirmatory contrasts. Seed variability (separate): SD, range and n seeds better of the per-seed estimates. Supplementary two-level interval resamples participants and seeds together (5 seeds: unstable, shown for orientation only).

| family | contrast | point | participant 95% interval | Bonferroni (25) | p | Holm p | seeds: SD, [min, max], n better | two-level 95% (suppl.) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| severity (nowcast) | R0O0T1 vs orig GRU | -1.40 | [-2.03, -0.80] | [-2.23, -0.52] | 0.0005 | 0.0125 | 0.42, [-1.99, -0.96], 5/5 | [-2.12, -0.75] | Holm-significant |
| severity (nowcast) | R0O2T1 vs orig GRU | -1.66 | [-2.28, -1.07] | [-2.55, -0.77] | 0.0005 | 0.0125 | 0.37, [-2.18, -1.25], 5/5 | [-2.40, -1.04] | Holm-significant |
| severity (nowcast) | R0O3T1 vs orig GRU | -1.52 | [-2.28, -0.80] | [-2.63, -0.22] | 0.0005 | 0.0125 | 0.35, [-1.99, -1.05], 5/5 | [-2.35, -0.75] | Holm-significant |
| severity (nowcast) | R0O2T1 vs R0O0T1 | -0.27 | [-0.39, -0.15] | [-0.45, -0.09] | 0.0005 | 0.0125 | 0.10, [-0.44, -0.20], 5/5 | [-0.43, -0.14] | Holm-significant |
| severity (nowcast) | R0O3T1 vs R0O0T1 | -0.13 | [-0.49, +0.25] | [-0.70, +0.41] | 0.5102 | 1.0000 | 0.15, [-0.38, +0.00], 4/5 | [-0.52, +0.27] | not distinguishable from 0 |
| status (nowcast) | R0O0T1 vs orig GRU | +0.49 | [-0.73, +1.66] | [-1.35, +2.15] | 0.4428 | 1.0000 | 1.81, [-1.11, +3.60], 2/5 | [-1.21, +2.92] | not distinguishable from 0 |
| status (nowcast) | R0O2T1 vs orig GRU | +0.16 | [-1.46, +1.74] | [-2.96, +3.02] | 0.8351 | 1.0000 | 1.55, [-1.31, +2.70], 2/5 | [-1.80, +2.50] | not distinguishable from 0 |
| status (nowcast) | R0O3T1 vs orig GRU | -0.52 | [-3.22, +2.41] | [-4.36, +4.81] | 0.7301 | 1.0000 | 1.99, [-1.92, +3.01], 4/5 | [-3.70, +2.90] | not distinguishable from 0 |
| status (nowcast) | R0O2T1 vs R0O0T1 | -0.33 | [-1.47, +0.81] | [-2.11, +1.61] | 0.5572 | 1.0000 | 0.55, [-0.92, +0.34], 4/5 | [-1.62, +0.97] | not distinguishable from 0 |
| status (nowcast) | R0O3T1 vs R0O0T1 | -1.00 | [-3.45, +1.66] | [-4.65, +3.19] | 0.4668 | 1.0000 | 0.32, [-1.37, -0.61], 5/5 | [-3.59, +1.81] | not distinguishable from 0 |
| onset | R0O0T1 vs orig GRU | +0.89 | [-0.02, +1.77] | [-0.71, +2.19] | 0.0530 | 0.8476 | 1.67, [-0.77, +3.69], 1/5 | [-0.61, +3.02] | not distinguishable from 0 |
| onset | R0O2T1 vs orig GRU | +0.65 | [-0.97, +2.05] | [-1.84, +2.91] | 0.3993 | 1.0000 | 1.65, [-1.35, +3.23], 1/5 | [-1.43, +2.87] | not distinguishable from 0 |
| onset | R0O3T1 vs orig GRU | +2.53 | [-0.01, +4.53] | [-1.30, +5.82] | 0.0355 | 0.6032 | 1.61, [+0.76, +5.15], 0/5 | [-0.47, +5.03] | not distinguishable from 0 |
| onset | R0O2T1 vs R0O0T1 | -0.25 | [-1.56, +1.06] | [-2.38, +1.83] | 0.6917 | 1.0000 | 0.29, [-0.56, +0.11], 4/5 | [-1.66, +1.15] | not distinguishable from 0 |
| onset | R0O3T1 vs R0O0T1 | +1.66 | [-0.72, +3.50] | [-1.67, +4.75] | 0.1274 | 1.0000 | 0.40, [+1.23, +2.29], 0/5 | [-0.85, +3.61] | not distinguishable from 0 |
| forecast severity | R0O0T1 vs orig GRU | -0.75 | [-1.21, -0.31] | [-1.44, -0.07] | 0.0020 | 0.0380 | 0.22, [-1.11, -0.57], 5/5 | [-1.27, -0.28] | Holm-significant |
| forecast severity | R0O2T1 vs orig GRU | -1.05 | [-1.54, -0.58] | [-1.85, -0.33] | 0.0010 | 0.0200 | 0.22, [-1.34, -0.80], 5/5 | [-1.62, -0.55] | Holm-significant |
| forecast severity | R0O3T1 vs orig GRU | -0.85 | [-1.67, -0.13] | [-2.03, +0.20] | 0.0300 | 0.5397 | 0.31, [-1.30, -0.52], 5/5 | [-1.72, -0.04] | unadjusted interval excludes 0 only |
| forecast severity | R0O2T1 vs R0O0T1 | -0.31 | [-0.44, -0.16] | [-0.52, -0.10] | 0.0005 | 0.0125 | 0.08, [-0.42, -0.24], 5/5 | [-0.46, -0.15] | Holm-significant |
| forecast severity | R0O3T1 vs R0O0T1 | -0.11 | [-0.67, +0.43] | [-0.96, +0.72] | 0.6912 | 1.0000 | 0.12, [-0.25, +0.05], 4/5 | [-0.69, +0.48] | not distinguishable from 0 |
| forecast status | R0O0T1 vs orig GRU | -0.08 | [-0.98, +0.78] | [-1.57, +1.27] | 0.8696 | 1.0000 | 0.34, [-0.48, +0.31], 3/5 | [-1.17, +0.85] | not distinguishable from 0 |
| forecast status | R0O2T1 vs orig GRU | -0.41 | [-2.31, +1.39] | [-3.47, +2.36] | 0.6567 | 1.0000 | 0.49, [-0.69, +0.45], 4/5 | [-2.46, +1.53] | not distinguishable from 0 |
| forecast status | R0O3T1 vs orig GRU | +0.82 | [-2.65, +3.86] | [-4.97, +5.14] | 0.6057 | 1.0000 | 0.70, [+0.05, +1.56], 0/5 | [-3.24, +4.09] | not distinguishable from 0 |
| forecast status | R0O2T1 vs R0O0T1 | -0.33 | [-1.63, +0.99] | [-2.49, +1.57] | 0.6377 | 1.0000 | 0.31, [-0.71, +0.13], 4/5 | [-1.72, +1.08] | not distinguishable from 0 |
| forecast status | R0O3T1 vs R0O0T1 | +0.91 | [-1.94, +3.43] | [-3.81, +4.86] | 0.4848 | 1.0000 | 0.72, [-0.18, +1.66], 1/5 | [-2.64, +3.68] | not distinguishable from 0 |

7 of 25 confirmatory contrasts are Holm-significant at 0.05; 8 have an unadjusted 95% interval excluding 0. Variance comparison (participant sampling vs seeds): see `confirmatory_contrasts.csv` columns var_participant (bootstrap variance of the seed-mean statistic) and var_seedmean (between-seed variance / 5).

