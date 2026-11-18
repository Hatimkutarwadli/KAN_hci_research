# P1-5 PRIMARY contrast (SYNTHETIC; EXTENSION): sustained onset, h21 (W = 21), 20% false-alarm target, detection at lead >= 7 days

Overall rate = share of ALL evaluable onset participants alarmed >= 7 days before onset (primary); achievable rate = among participants for whom a lead of 7 days was possible inside their window. Values are seed means; [ ] = stratified participant bootstrap 95% interval (B = 1000, thresholds fixed, conditional on the trained seeds). Difference = model minus GRU, same resampled participants. Null columns: shuffle null (complete score trajectories permuted across participants, 200 shared permutations; mean [2.5-97.5 percentile range], not a CI) and the iid U(0,1) null of Stage 7.


## dep: 32 onset participants, 139 never-onset participants (test); lead 7 achievable for 30 of 32

| model | overall det >= 7 d | seed SD | achievable-lead det >= 7 d | diff vs GRU (overall) | resamples better than GRU | shuffle null | detection >= 1 d | FAP (target 0.20) |
|---|---|---|---|---|---|---|---|---|
| gru | 0.19 [0.07, 0.33] | 0.04 | 0.20 [0.07, 0.34] | (reference) |  | 0.14 [0.06, 0.26]; obs >= null in 71% of shuffles | 0.29 [0.14, 0.44] | 0.19 [0.13, 0.25] |
| transformer | 0.18 [0.07, 0.32] | 0.04 | 0.19 [0.07, 0.33] | -0.01 [-0.06, 0.05] | 33% | 0.13 [0.04, 0.23]; obs >= null in 74% of shuffles | 0.26 [0.12, 0.39] | 0.16 [0.11, 0.21] |
| cnnlstm | 0.19 [0.07, 0.32] | 0.04 | 0.21 [0.08, 0.34] | 0.01 [-0.07, 0.09] | 50% | 0.14 [0.05, 0.24]; obs >= null in 74% of shuffles | 0.30 [0.16, 0.44] | 0.17 [0.12, 0.22] |
| m1 | 0.20 [0.09, 0.34] | 0.02 | 0.21 [0.09, 0.35] | 0.01 [-0.06, 0.09] | 61% | 0.14 [0.04, 0.25]; obs >= null in 76% of shuffles | 0.29 [0.13, 0.44] | 0.18 [0.13, 0.24] |
| mlp_m1 | 0.17 [0.07, 0.30] | 0.04 | 0.19 [0.07, 0.32] | -0.01 [-0.09, 0.06] | 36% | 0.14 [0.04, 0.25]; obs >= null in 62% of shuffles | 0.31 [0.16, 0.46] | 0.17 [0.12, 0.23] |
| kangru | 0.19 [0.08, 0.30] | 0.05 | 0.20 [0.09, 0.32] | 0.00 [-0.08, 0.07] | 47% | 0.15 [0.05, 0.26]; obs >= null in 65% of shuffles | 0.31 [0.17, 0.46] | 0.19 [0.13, 0.24] |
| kan *(secondary group)* | 0.21 [0.09, 0.33] | 0.04 | 0.22 [0.10, 0.35] | 0.02 [-0.06, 0.09] | 68% | 0.15 [0.06, 0.27]; obs >= null in 76% of shuffles | 0.24 [0.11, 0.37] | 0.17 [0.12, 0.22] |
| kan_reg *(secondary group)* | 0.20 [0.08, 0.33] | 0.05 | 0.21 [0.09, 0.35] | 0.01 [-0.08, 0.11] | 56% | 0.15 [0.05, 0.27]; obs >= null in 69% of shuffles | 0.23 [0.11, 0.36] | 0.19 [0.14, 0.26] |
| mlp_b2 *(secondary group)* | 0.22 [0.10, 0.36] | 0.02 | 0.23 [0.11, 0.37] | 0.03 [-0.05, 0.11] | 76% | 0.16 [0.06, 0.26]; obs >= null in 78% of shuffles | 0.29 [0.16, 0.44] | 0.19 [0.14, 0.25] |
| mlp_b2_reg *(secondary group)* | 0.16 [0.07, 0.28] | 0.04 | 0.17 [0.07, 0.29] | -0.02 [-0.10, 0.04] | 22% | 0.14 [0.04, 0.26]; obs >= null in 52% of shuffles | 0.22 [0.11, 0.34] | 0.18 [0.13, 0.23] |
| kan_nospline *(secondary group)* | 0.22 [0.11, 0.34] | 0.07 | 0.23 [0.11, 0.36] | 0.03 [-0.06, 0.13] | 70% | 0.14 [0.04, 0.25]; obs >= null in 78% of shuffles | 0.26 [0.14, 0.39] | 0.18 [0.13, 0.23] |
| m1_static *(secondary group)* | 0.20 [0.09, 0.33] | 0.08 | 0.21 [0.10, 0.34] | 0.01 [-0.08, 0.11] | 55% | 0.16 [0.06, 0.26]; obs >= null in 63% of shuffles | 0.26 [0.13, 0.41] | 0.19 [0.13, 0.25] |
| kangru_fixed *(secondary group)* | 0.18 [0.07, 0.30] | 0.03 | 0.19 [0.08, 0.32] | -0.01 [-0.07, 0.07] | 39% | 0.14 [0.06, 0.25]; obs >= null in 65% of shuffles | 0.31 [0.16, 0.46] | 0.19 [0.13, 0.25] |
| gbt *(secondary group)* | 0.25 [0.09, 0.41] | nan | 0.27 [0.11, 0.44] | 0.06 [-0.07, 0.21] | 80% | 0.13 [0.03, 0.25]; obs >= null in 97% of shuffles | 0.28 [0.12, 0.44] | 0.21 [0.14, 0.28] |
| st_task *(secondary group)* | 0.17 [0.06, 0.32] | 0.03 | 0.19 [0.06, 0.33] | -0.01 [-0.05, 0.02] | 17% | 0.13 [0.04, 0.25]; obs >= null in 72% of shuffles | 0.20 [0.08, 0.33] | 0.20 [0.14, 0.25] |
| st_task_matched *(secondary group)* | 0.16 [0.06, 0.29] | 0.01 | 0.17 [0.06, 0.31] | -0.02 [-0.06, 0.01] | 3% | 0.13 [0.04, 0.25]; obs >= null in 68% of shuffles | 0.20 [0.08, 0.33] | 0.19 [0.13, 0.24] |
| family *(secondary group)* | 0.18 [0.07, 0.32] | 0.04 | 0.19 [0.07, 0.33] | -0.01 [-0.03, 0.02] | 23% | 0.12 [0.03, 0.23]; obs >= null in 78% of shuffles | 0.27 [0.13, 0.41] | 0.18 [0.13, 0.24] |
| random_null_iid | 0.04 [0.01, 0.06] | 0.03 | 0.04 [0.01, 0.07] | -0.15 [-0.28, -0.03] | 0% | n/a | 0.05 [0.03, 0.08] | 0.19 [0.16, 0.21] |

## anx: 32 onset participants, 132 never-onset participants (test); lead 7 achievable for 30 of 32

| model | overall det >= 7 d | seed SD | achievable-lead det >= 7 d | diff vs GRU (overall) | resamples better than GRU | shuffle null | detection >= 1 d | FAP (target 0.20) |
|---|---|---|---|---|---|---|---|---|
| gru | 0.30 [0.16, 0.46] | 0.03 | 0.32 [0.18, 0.48] | (reference) |  | 0.14 [0.03, 0.24]; obs >= null in 99% of shuffles | 0.33 [0.19, 0.49] | 0.20 [0.14, 0.26] |
| transformer | 0.29 [0.16, 0.44] | 0.03 | 0.31 [0.18, 0.47] | -0.01 [-0.04, 0.03] | 30% | 0.14 [0.04, 0.25]; obs >= null in 99% of shuffles | 0.36 [0.23, 0.53] | 0.22 [0.16, 0.29] |
| cnnlstm | 0.26 [0.13, 0.41] | 0.02 | 0.28 [0.14, 0.43] | -0.04 [-0.10, 0.01] | 6% | 0.13 [0.03, 0.25]; obs >= null in 97% of shuffles | 0.31 [0.17, 0.45] | 0.18 [0.12, 0.25] |
| m1 | 0.28 [0.16, 0.44] | 0.04 | 0.30 [0.17, 0.45] | -0.02 [-0.09, 0.04] | 30% | 0.15 [0.04, 0.26]; obs >= null in 95% of shuffles | 0.31 [0.17, 0.46] | 0.18 [0.13, 0.25] |
| mlp_m1 | 0.26 [0.14, 0.41] | 0.05 | 0.28 [0.15, 0.43] | -0.04 [-0.11, 0.02] | 13% | 0.13 [0.03, 0.24]; obs >= null in 96% of shuffles | 0.36 [0.22, 0.52] | 0.17 [0.11, 0.23] |
| kangru | 0.31 [0.17, 0.46] | 0.07 | 0.33 [0.20, 0.48] | 0.01 [-0.05, 0.06] | 55% | 0.15 [0.04, 0.26]; obs >= null in 96% of shuffles | 0.36 [0.21, 0.52] | 0.20 [0.13, 0.26] |
| kan *(secondary group)* | 0.25 [0.12, 0.40] | 0.02 | 0.27 [0.13, 0.42] | -0.05 [-0.14, 0.04] | 10% | 0.16 [0.06, 0.28]; obs >= null in 88% of shuffles | 0.31 [0.18, 0.46] | 0.20 [0.14, 0.26] |
| kan_reg *(secondary group)* | 0.24 [0.12, 0.39] | 0.03 | 0.26 [0.13, 0.42] | -0.06 [-0.14, 0.03] | 7% | 0.15 [0.04, 0.28]; obs >= null in 86% of shuffles | 0.30 [0.17, 0.46] | 0.20 [0.14, 0.27] |
| mlp_b2 *(secondary group)* | 0.29 [0.16, 0.45] | 0.04 | 0.31 [0.18, 0.48] | -0.01 [-0.12, 0.11] | 38% | 0.16 [0.04, 0.29]; obs >= null in 94% of shuffles | 0.35 [0.21, 0.51] | 0.21 [0.15, 0.27] |
| mlp_b2_reg *(secondary group)* | 0.26 [0.14, 0.41] | 0.03 | 0.27 [0.15, 0.43] | -0.04 [-0.14, 0.06] | 18% | 0.15 [0.04, 0.29]; obs >= null in 89% of shuffles | 0.31 [0.18, 0.47] | 0.20 [0.14, 0.26] |
| kan_nospline *(secondary group)* | 0.29 [0.17, 0.43] | 0.07 | 0.31 [0.18, 0.46] | -0.01 [-0.10, 0.09] | 37% | 0.13 [0.04, 0.26]; obs >= null in 97% of shuffles | 0.34 [0.21, 0.49] | 0.19 [0.14, 0.25] |
| m1_static *(secondary group)* | 0.26 [0.14, 0.41] | 0.03 | 0.28 [0.16, 0.43] | -0.04 [-0.15, 0.06] | 22% | 0.14 [0.04, 0.28]; obs >= null in 94% of shuffles | 0.32 [0.19, 0.47] | 0.19 [0.13, 0.26] |
| kangru_fixed *(secondary group)* | 0.31 [0.17, 0.47] | 0.03 | 0.33 [0.19, 0.49] | 0.01 [-0.05, 0.06] | 55% | 0.15 [0.04, 0.26]; obs >= null in 98% of shuffles | 0.36 [0.21, 0.52] | 0.21 [0.15, 0.27] |
| gbt *(secondary group)* | 0.16 [0.03, 0.28] | nan | 0.17 [0.03, 0.32] | -0.14 [-0.31, 0.01] | 4% | 0.11 [0.03, 0.22]; obs >= null in 75% of shuffles | 0.25 [0.09, 0.41] | 0.20 [0.13, 0.27] |
| st_task *(secondary group)* | 0.24 [0.14, 0.37] | 0.03 | 0.26 [0.15, 0.39] | -0.06 [-0.14, 0.01] | 5% | 0.11 [0.02, 0.22]; obs >= null in 97% of shuffles | 0.29 [0.17, 0.44] | 0.21 [0.15, 0.27] |
| st_task_matched *(secondary group)* | 0.26 [0.14, 0.39] | 0.05 | 0.27 [0.16, 0.41] | -0.04 [-0.14, 0.04] | 14% | 0.11 [0.02, 0.22]; obs >= null in 98% of shuffles | 0.33 [0.21, 0.47] | 0.20 [0.14, 0.26] |
| family *(secondary group)* | 0.25 [0.13, 0.39] | 0.05 | 0.27 [0.14, 0.41] | -0.05 [-0.12, 0.00] | 0% | 0.12 [0.03, 0.23]; obs >= null in 95% of shuffles | 0.28 [0.15, 0.42] | 0.20 [0.13, 0.26] |
| random_null_iid | 0.05 [0.02, 0.09] | 0.06 | 0.05 [0.02, 0.09] | -0.25 [-0.41, -0.11] | 0% | n/a | 0.06 [0.03, 0.11] | 0.18 [0.15, 0.21] |

Only this contrast is primary; everything else in the other files is descriptive. 'Resamples better than GRU' is the share of bootstrap resamples with a positive difference; it is not a p-value. With 32 onset participants each participant moves a rate by 3.1 points, and several models are compared with the GRU, so intervals that exclude 0 should be read as indicative only.

