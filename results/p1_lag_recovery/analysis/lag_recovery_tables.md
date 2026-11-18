# P1-4 model-based lag recovery (SYNTHETIC; EXTENSION). Calibration split, 1,000 fixed origins. Predictive attribution of what the models learned; NOT causal identification.


## 1a. Lag-recovery metrics, single-day +1 SD impulse (profile over lags 0-10; mean over seeds; peak lags listed per seed)

Truth row = analytic response of the planted simulator. exact = seeds whose peak equals the planted lag. pre-lag mass = share of |profile| at lags before the planted lag (truth 0). spike = |increment at planted lag| / mean |other increments| (truth large). r = Pearson with the analytic truth profile; magnitude = profile sum / truth sum (sensor noise and the model's own attenuation not removed); split-half r = correlation between two disjoint participant halves.

| outcome | feature | planted lag | model | level-peak lag per seed (exact) | increment-peak lag per seed, lags 0-10 (exact) | increment-peak lag per seed, lags 1-5 as in Stage 5 (exact) | onset lag = first lag reaching 50% of max (exact) | pre-lag mass | spike ratio | r with truth | magnitude vs truth | split-half r | sign ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dep | sleep_variability | 3 | **truth** | 3 | 3 | 3 | 3 | 0.00 | 22.6 | 1.00 | 1.00 | | |
| dep | sleep_variability | 3 | gru | [1, 2, 2, 2, 2] (0/5) | [0, 0, 0, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [0, 0, 0, 1, 1] (0/5) | 0.28 ±0.03 | 0.3 | +0.03 ±0.39 | 0.97 | +0.99 | 100% |
| dep | sleep_variability | 3 | m1 | [1, 1, 1, 1, 2] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 3, 4, 4] (1/5) | [0, 0, 0, 0, 0] (0/5) | 0.35 ±0.03 | 0.7 | -0.50 ±0.19 | 0.73 | +0.98 | 100% |
| dep | sleep_variability | 3 | mlp_m1 | [1, 1, 2, 2, 2] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 3, 3, 3] (3/5) | [0, 0, 0, 0, 0] (0/5) | 0.36 ±0.03 | 1.1 | -0.53 ±0.06 | 0.86 | +0.99 | 100% |
| dep | sleep_variability | 3 | kangru | [1, 2, 2, 2, 6] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 2, 5, 5] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.31 ±0.03 | 0.4 | -0.09 ±0.25 | 1.13 | +0.99 | 100% |
| dep | sleep_variability | 3 | gru (untrained null) | [0, 0, 0] (0/3) | [0, 0, 0] (0/3) | [1, 1, 2] (0/3) | [0, 0, 0] (0/3) | 0.89 ±0.04 | 0.9 | +0.26 ±0.88 | -0.00 | +1.00 | 33% |
| dep | sleep_variability | 3 | m1 (untrained null) | [0, 1, 2] (0/3) | [1, 1, 2] (0/3) | [1, 1, 2] (0/3) | [0, 0, 1] (0/3) | 0.39 ±0.05 | 1.0 | +0.24 ±0.44 | -0.01 | +0.97 | 33% |
| dep | sleep_variability | 3 | mlp_m1 (untrained null) | [2, 2, 4] (0/3) | [2, 3, 4] (1/3) | [2, 3, 4] (1/3) | [1, 2, 2] (0/3) | 0.35 ±0.16 | 3.5 | -0.70 ±0.23 | -0.00 | +0.99 | 33% |
| dep | sleep_variability | 3 | kangru (untrained null) | [2, 4, 4] (0/3) | [2, 2, 4] (0/3) | [2, 2, 4] (0/3) | [0, 0, 0] (0/3) | 0.47 ±0.07 | 1.7 | +0.56 ±0.05 | 0.00 | +0.89 | 33% |
| dep | steps | 2 | **truth** | 2 | 2 | 2 | 2 | 0.00 | 20.5 | 1.00 | 1.00 | | |
| dep | steps | 2 | gru | [0, 0, 1, 1, 2] (1/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 3] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.23 ±0.02 | 0.3 | -0.21 ±0.19 | 2.06 | +0.99 | 100% |
| dep | steps | 2 | m1 | [1, 1, 1, 1, 1] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 1] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.22 ±0.01 | 0.3 | +0.02 ±0.09 | 1.64 | +0.98 | 100% |
| dep | steps | 2 | mlp_m1 | [1, 2, 2, 2, 2] (4/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 3] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.22 ±0.02 | 0.3 | +0.03 ±0.13 | 1.93 | +0.98 | 100% |
| dep | steps | 2 | kangru | [1, 2, 4, 4, 5] (1/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 4] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.19 ±0.02 | 0.4 | +0.22 ±0.28 | 1.89 | +0.98 | 100% |
| dep | steps | 2 | gru (untrained null) | [0, 0, 0] (0/3) | [0, 0, 0] (0/3) | [1, 1, 1] (0/3) | [0, 0, 0] (0/3) | 0.72 ±0.19 | 1.4 | -0.32 ±0.76 | 0.00 | +1.00 | 67% |
| dep | steps | 2 | m1 (untrained null) | [0, 1, 5] (0/3) | [0, 1, 5] (0/3) | [1, 1, 5] (0/3) | [0, 0, 0] (0/3) | 0.33 ±0.17 | 2.0 | -0.49 ±0.19 | -0.00 | +0.97 | 33% |
| dep | steps | 2 | mlp_m1 (untrained null) | [1, 3, 4] (0/3) | [1, 2, 5] (1/3) | [1, 2, 5] (1/3) | [0, 1, 2] (1/3) | 0.26 ±0.24 | 4.4 | +0.15 ±0.72 | 0.01 | +0.98 | 67% |
| dep | steps | 2 | kangru (untrained null) | [0, 1, 3] (0/3) | [0, 1, 1] (0/3) | [1, 1, 2] (1/3) | [0, 0, 0] (0/3) | 0.45 ±0.21 | 2.2 | -0.50 ±0.32 | -0.03 | +0.99 | 33% |
| anx | steps | 2 | **truth** | 2 | 2 | 2 | 2 | 0.00 | 20.5 | 1.00 | 1.00 | | |
| anx | steps | 2 | gru | [2, 2, 2, 2, 2] (5/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 1] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.18 ±0.01 | 0.2 | +0.33 ±0.10 | 2.01 | +0.97 | 100% |
| anx | steps | 2 | m1 | [1, 1, 1, 1, 2] (1/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 1] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.20 ±0.01 | 0.2 | +0.14 ±0.15 | 2.13 | +0.98 | 100% |
| anx | steps | 2 | mlp_m1 | [1, 2, 2, 2, 2] (4/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 3] (0/5) | [0, 0, 0, 0, 0] (0/5) | 0.22 ±0.01 | 0.3 | +0.02 ±0.10 | 2.41 | +0.98 | 100% |
| anx | steps | 2 | kangru | [2, 2, 4, 6, 6] (2/5) | [0, 0, 0, 0, 1] (0/5) | [1, 1, 2, 2, 4] (2/5) | [0, 0, 0, 0, 1] (0/5) | 0.15 ±0.02 | 1.5 | +0.56 ±0.14 | 1.94 | +0.97 | 100% |
| anx | steps | 2 | gru (untrained null) | [0, 0, 0] (0/3) | [0, 0, 0] (0/3) | [1, 1, 1] (0/3) | [0, 0, 0] (0/3) | 0.75 ±0.07 | 1.5 | -0.26 ±0.82 | 0.01 | +1.00 | 67% |
| anx | steps | 2 | m1 (untrained null) | [1, 5, 6] (0/3) | [0, 1, 4] (0/3) | [1, 2, 4] (1/3) | [0, 0, 2] (1/3) | 0.35 ±0.35 | 2.4 | +0.01 ±0.75 | -0.02 | +0.98 | 33% |
| anx | steps | 2 | mlp_m1 (untrained null) | [1, 4, 4] (0/3) | [1, 2, 4] (1/3) | [1, 2, 4] (1/3) | [1, 1, 2] (1/3) | 0.14 ±0.06 | 2.2 | +0.02 ±0.44 | 0.02 | +0.99 | 67% |
| anx | steps | 2 | kangru (untrained null) | [1, 2, 3] (1/3) | [4, 4, 4] (0/3) | [4, 4, 4] (0/3) | [1, 1, 1] (0/3) | 0.27 ±0.11 | 1.6 | -0.09 ±0.18 | 0.03 | +0.99 | 67% |
| anx | hrv | 1 | **truth** | 1 | 1 | 1 | 1 | 0.00 | 18.9 | 1.00 | 1.00 | | |
| anx | hrv | 1 | gru | [0, 0, 0, 0, 0] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 2, 2, 2] (2/5) | [0, 0, 0, 0, 0] (0/5) | 0.15 ±0.01 | 0.8 | -0.05 ±0.10 | 1.27 | +0.99 | 100% |
| anx | hrv | 1 | m1 | [6, 6, 6, 6, 6] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 5] (4/5) | [0, 0, 0, 0, 1] (1/5) | 0.08 ±0.02 | 1.3 | +0.29 ±0.36 | 0.61 | +0.95 | 100% |
| anx | hrv | 1 | mlp_m1 | [1, 4, 6, 6, 6] (1/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 2, 2, 4] (2/5) | [0, 0, 0, 0, 0] (0/5) | 0.08 ±0.01 | 0.6 | +0.48 ±0.13 | 0.80 | +0.95 | 100% |
| anx | hrv | 1 | kangru | [2, 2, 4, 4, 4] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 3, 5, 5] (2/5) | [0, 0, 0, 0, 0] (0/5) | 0.08 ±0.01 | 1.5 | +0.62 ±0.16 | 1.25 | +0.98 | 100% |
| anx | hrv | 1 | gru (untrained null) | [0, 0, 2] (0/3) | [0, 0, 1] (1/3) | [1, 1, 2] (2/3) | [0, 0, 0] (0/3) | 0.34 ±0.16 | 4.3 | +0.39 ±0.58 | -0.00 | +0.94 | 67% |
| anx | hrv | 1 | m1 (untrained null) | [2, 2, 5] (0/3) | [2, 2, 6] (0/3) | [2, 2, 5] (0/3) | [0, 1, 2] (1/3) | 0.10 ±0.09 | 1.6 | -0.00 ±0.63 | 0.02 | +0.96 | 67% |
| anx | hrv | 1 | mlp_m1 (untrained null) | [0, 2, 4] (0/3) | [1, 1, 2] (2/3) | [1, 1, 2] (2/3) | [0, 1, 2] (1/3) | 0.09 ±0.13 | 5.4 | +0.74 ±0.02 | 0.01 | +0.97 | 67% |
| anx | hrv | 1 | kangru (untrained null) | [0, 3, 6] (0/3) | [0, 2, 3] (0/3) | [2, 3, 3] (0/3) | [0, 1, 1] (2/3) | 0.13 ±0.18 | 1.8 | +0.11 ±0.36 | -0.01 | +0.98 | 33% |

## 1b. Lag-recovery metrics, impulse persisting as the channel's own AR(1) (profile over lags 0-10; mean over seeds; peak lags listed per seed)

Truth row = analytic response of the planted simulator. exact = seeds whose peak equals the planted lag. pre-lag mass = share of |profile| at lags before the planted lag (truth 0). spike = |increment at planted lag| / mean |other increments| (truth large). r = Pearson with the analytic truth profile; magnitude = profile sum / truth sum (sensor noise and the model's own attenuation not removed); split-half r = correlation between two disjoint participant halves.

| outcome | feature | planted lag | model | level-peak lag per seed (exact) | increment-peak lag per seed, lags 0-10 (exact) | increment-peak lag per seed, lags 1-5 as in Stage 5 (exact) | onset lag = first lag reaching 50% of max (exact) | pre-lag mass | spike ratio | r with truth | magnitude vs truth | split-half r | sign ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dep | sleep_variability | 3 | **truth** | 10 | 3 | 3 | 4 | 0.00 | 4.1 | 1.00 | 1.00 | | |
| dep | sleep_variability | 3 | gru | [6, 7, 7, 7, 8] (0/5) | [0, 0, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 2, 2] (0/5) | 0.15 ±0.02 | 1.5 | +0.83 ±0.05 | 0.82 | +1.00 | 100% |
| dep | sleep_variability | 3 | m1 | [3, 6, 6, 6, 6] (1/5) | [0, 0, 0, 0, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | 0.19 ±0.02 | 0.9 | +0.62 ±0.13 | 0.62 | +0.99 | 100% |
| dep | sleep_variability | 3 | mlp_m1 | [4, 5, 6, 6, 7] (0/5) | [0, 0, 0, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | 0.20 ±0.01 | 0.9 | +0.56 ±0.18 | 0.75 | +1.00 | 100% |
| dep | sleep_variability | 3 | kangru | [4, 4, 6, 6, 6] (0/5) | [0, 1, 1, 2, 2] (0/5) | [1, 1, 1, 2, 2] (0/5) | [1, 1, 2, 2, 2] (0/5) | 0.17 ±0.01 | 1.3 | +0.67 ±0.16 | 0.93 | +1.00 | 100% |
| dep | steps | 2 | **truth** | 7 | 2 | 2 | 3 | 0.00 | 5.9 | 1.00 | 1.00 | | |
| dep | steps | 2 | gru | [3, 4, 4, 4, 5] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 1] (0/5) | [0, 1, 1, 1, 1] (0/5) | 0.13 ±0.02 | 1.5 | +0.66 ±0.22 | 1.61 | +1.00 | 100% |
| dep | steps | 2 | m1 | [4, 6, 6, 6, 6] (0/5) | [0, 0, 0, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | 0.12 ±0.01 | 1.7 | +0.68 ±0.10 | 1.31 | +0.99 | 100% |
| dep | steps | 2 | mlp_m1 | [4, 4, 4, 4, 5] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | 0.12 ±0.01 | 1.9 | +0.62 ±0.12 | 1.56 | +0.99 | 100% |
| dep | steps | 2 | kangru | [4, 4, 5, 6, 6] (0/5) | [0, 0, 0, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 2] (1/5) | 0.10 ±0.01 | 1.5 | +0.81 ±0.14 | 1.51 | +1.00 | 100% |
| anx | steps | 2 | **truth** | 7 | 2 | 2 | 3 | 0.00 | 5.9 | 1.00 | 1.00 | | |
| anx | steps | 2 | gru | [4, 5, 5, 5, 6] (0/5) | [0, 0, 0, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | 0.10 ±0.00 | 2.2 | +0.87 ±0.04 | 1.54 | +1.00 | 100% |
| anx | steps | 2 | m1 | [6, 6, 6, 6, 6] (0/5) | [0, 0, 0, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | 0.11 ±0.01 | 1.8 | +0.78 ±0.08 | 1.69 | +0.99 | 100% |
| anx | steps | 2 | mlp_m1 | [4, 4, 4, 4, 4] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 1] (0/5) | [1, 1, 1, 1, 1] (0/5) | 0.12 ±0.01 | 2.0 | +0.65 ±0.12 | 1.95 | +0.98 | 100% |
| anx | steps | 2 | kangru | [5, 6, 6, 7, 7] (0/5) | [0, 0, 0, 0, 1] (0/5) | [1, 1, 1, 2, 2] (2/5) | [1, 1, 1, 2, 2] (2/5) | 0.09 ±0.01 | 2.2 | +0.96 ±0.04 | 1.50 | +1.00 | 100% |
| anx | hrv | 1 | **truth** | 6 | 1 | 1 | 2 | 0.00 | 5.5 | 1.00 | 1.00 | | |
| anx | hrv | 1 | gru | [2, 3, 3, 3, 3] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 1] (5/5) | [0, 0, 0, 0, 0] (0/5) | 0.06 ±0.01 | 2.9 | +0.45 ±0.15 | 0.97 | +1.00 | 100% |
| anx | hrv | 1 | m1 | [6, 6, 6, 6, 7] (0/5) | [0, 0, 0, 1, 1] (2/5) | [1, 1, 1, 1, 1] (5/5) | [1, 1, 1, 1, 2] (4/5) | 0.04 ±0.01 | 2.5 | +0.96 ±0.03 | 0.45 | +1.00 | 100% |
| anx | hrv | 1 | mlp_m1 | [6, 6, 6, 6, 6] (0/5) | [0, 0, 0, 0, 0] (0/5) | [1, 1, 1, 1, 2] (4/5) | [1, 1, 2, 2, 2] (2/5) | 0.03 ±0.00 | 2.2 | +0.97 ±0.01 | 0.60 | +1.00 | 100% |
| anx | hrv | 1 | kangru | [4, 4, 4, 6, 6] (0/5) | [0, 0, 0, 1, 1] (2/5) | [1, 1, 1, 1, 2] (4/5) | [1, 1, 1, 1, 1] (5/5) | 0.04 ±0.00 | 2.7 | +0.87 ±0.08 | 0.95 | +1.00 | 100% |

## 1c. Pooled lag-hit rates over all model-seeds and the 4 planted pairs (single-day impulse). Exact = estimated lag equals the planted lag; within 1 day = |error| <= 1. Chance is for guessing uniformly among the estimator's candidate lags (11 for lags 0-10, 5 for lags 1-5). Untrained nulls use the same architectures with random weights.

| estimator | population | exact hits | within 1 day | n | chance (exact) |
|---|---|---|---|---|---|
| level peak (lags 0-10) | trained | 19/80 (24%) | 51/80 (64%) | 80 | 9% |
| level peak (lags 0-10) | untrained | 1/48 (2%) | 25/48 (52%) | 48 | 9% |
| increment peak (lags 0-10) | trained | 0/80 (0%) | 21/80 (26%) | 80 | 9% |
| increment peak (lags 0-10) | untrained | 6/48 (12%) | 25/48 (52%) | 48 | 9% |
| increment peak (lags 1-5, as Stage 5) | trained | 16/80 (20%) | 60/80 (75%) | 80 | 20% |
| increment peak (lags 1-5, as Stage 5) | untrained | 9/48 (19%) | 34/48 (71%) | 48 | 20% |
| onset lag (50% of max) | trained | 1/80 (1%) | 21/80 (26%) | 80 | 9% |
| onset lag (50% of max) | untrained | 7/48 (15%) | 23/48 (48%) | 48 | 9% |

Mean signed error (estimated minus planted lag), trained models, level peak: gru: -0.8, m1: +0.3, mlp_m1: +0.5, kangru: +1.2


### 1d. Ordering test (does the estimate change with the planted lag?). Pairs of planted pairs with the SAME outcome and different planted lags: hrv->anx (1) vs steps->anx (2); steps->dep (2) vs sleep_variability->dep (3). Counts of model-seeds where the estimated lags are in the planted order (<), tied (=) or reversed (>). The planted lags are clustered at 1-3, so a recency-weighted profile peaking at 1-2 earns partial exact hits without carrying lag information; the ordering test removes that advantage.

| estimator | population | hrv<steps (anx) | hrv=steps | hrv>steps | steps<sv (dep) | steps=sv | steps>sv |
|---|---|---|---|---|---|---|---|
| level peak | trained | 7 | 4 | 9 | 8 | 8 | 4 |
| level peak | untrained | 4 | 4 | 4 | 5 | 4 | 3 |
| increment peak 1-5 | trained | 2 | 9 | 9 | 9 | 9 | 2 |
| increment peak 1-5 | untrained | 5 | 4 | 3 | 6 | 4 | 2 |
| onset lag | trained | 1 | 18 | 1 | 2 | 18 | 0 |
| onset lag | untrained | 4 | 6 | 2 | 3 | 9 | 0 |


## 2. Controls across all 8 channels (single-day profile; mean over seeds). total |P| = sum over lags 0-10 of |change in severity points per +1 SD|. Planted channels marked *. Negative controls: channels with no planted effect on that outcome. Reverse-path/confounder channels: HRV and resting HR for depression (driven by anxiety state / stress), calories (= 0.7 x steps) and circadian regularity (= -0.3 x sleep variability) are collinear correlates.


### outcome: dep

| model | *steps | calories | sleep_duration | sleep_efficiency | *sleep_variability | resting_hr | hrv | circadian_regularity |
|---|---|---|---|---|---|---|---|---|
| gru | 1.22 (incr. peak lag 1) | 0.04 (incr. peak lag 1) | 0.05 (incr. peak lag 1) | 0.12 (incr. peak lag 1) | 0.74 (incr. peak lag 1) | 0.27 (incr. peak lag 1) | 0.25 (incr. peak lag 1) | 0.13 (incr. peak lag 1) |
| m1 | 0.97 (incr. peak lag 1) | 0.06 (incr. peak lag 2) | 0.12 (incr. peak lag 2) | 0.16 (incr. peak lag 2) | 0.56 (incr. peak lag 1) | 0.25 (incr. peak lag 4) | 0.48 (incr. peak lag 1) | 0.07 (incr. peak lag 4) |
| mlp_m1 | 1.14 (incr. peak lag 1) | 0.09 (incr. peak lag 2) | 0.08 (incr. peak lag 3) | 0.16 (incr. peak lag 1) | 0.66 (incr. peak lag 3) | 0.35 (incr. peak lag 2) | 0.50 (incr. peak lag 1) | 0.09 (incr. peak lag 1) |
| kangru | 1.12 (incr. peak lag 1) | 0.08 (incr. peak lag 2) | 0.11 (incr. peak lag 1) | 0.23 (incr. peak lag 1) | 0.87 (incr. peak lag 1) | 0.23 (incr. peak lag 3) | 0.26 (incr. peak lag 2) | 0.11 (incr. peak lag 1) |

### outcome: anx

| model | *steps | calories | sleep_duration | sleep_efficiency | sleep_variability | resting_hr | *hrv | circadian_regularity |
|---|---|---|---|---|---|---|---|---|
| gru | 0.71 (incr. peak lag 1) | 0.03 (incr. peak lag 1) | 0.18 (incr. peak lag 1) | 0.05 (incr. peak lag 1) | 0.22 (incr. peak lag 1) | 0.36 (incr. peak lag 1) | 0.83 (incr. peak lag 2) | 0.03 (incr. peak lag 1) |
| m1 | 0.76 (incr. peak lag 1) | 0.04 (incr. peak lag 2) | 0.09 (incr. peak lag 1) | 0.11 (incr. peak lag 2) | 0.41 (incr. peak lag 1) | 0.21 (incr. peak lag 4) | 0.40 (incr. peak lag 1) | 0.05 (incr. peak lag 4) |
| mlp_m1 | 0.85 (incr. peak lag 1) | 0.08 (incr. peak lag 2) | 0.09 (incr. peak lag 3) | 0.07 (incr. peak lag 1) | 0.44 (incr. peak lag 3) | 0.32 (incr. peak lag 1) | 0.52 (incr. peak lag 1) | 0.08 (incr. peak lag 2) |
| kangru | 0.69 (incr. peak lag 1) | 0.07 (incr. peak lag 1) | 0.21 (incr. peak lag 2) | 0.06 (incr. peak lag 2) | 0.29 (incr. peak lag 2) | 0.31 (incr. peak lag 3) | 0.81 (incr. peak lag 1) | 0.07 (incr. peak lag 1) |

Increment-peak lag is the lag in 1-5 with the largest change of the profile (mode over seeds). Planted lags: sleep_variability->dep: 3, steps->dep: 2, steps->anx: 2, hrv->anx: 1. A channel with no planted lag still has a peak somewhere: its position and magnitude are the false-positive reference.


### Correlated-channel control: steps (resp. sleep variability) moved together with the channel the generator couples to it (calories +0.7 x steps; circadian regularity -0.3 x sleep variability)

| model | pair | single-channel total |P| | coupled total |P| | ratio | increment-peak lag single -> coupled |
|---|---|---|---|---|---|
| gru | steps->dep | 1.22 | 1.23 | 1.01 | 0 -> 0 |
| gru | steps->anx | 0.71 | 0.73 | 1.03 | 0 -> 0 |
| gru | sleep_variability->dep | 0.74 | 0.79 | 1.06 | 0 -> 0 |
| m1 | steps->dep | 0.97 | 0.96 | 0.99 | 0 -> 0 |
| m1 | steps->anx | 0.76 | 0.74 | 0.99 | 0 -> 0 |
| m1 | sleep_variability->dep | 0.56 | 0.59 | 1.05 | 0 -> 0 |
| mlp_m1 | steps->dep | 1.14 | 1.10 | 0.97 | 0 -> 0 |
| mlp_m1 | steps->anx | 0.85 | 0.83 | 0.97 | 0 -> 0 |
| mlp_m1 | sleep_variability->dep | 0.66 | 0.67 | 1.02 | 0 -> 0 |
| kangru | steps->dep | 1.12 | 1.10 | 0.99 | 0 -> 0 |
| kangru | steps->anx | 0.69 | 0.68 | 0.98 | 0 -> 0 |
| kangru | sleep_variability->dep | 0.87 | 0.91 | 1.05 | 0 -> 0 |

## 3. Comparison with the Stage 5 distributed-lag regression (increment of severity on lagged feature, lags 1-5, coefficients as multiples of the planted benchmark; V1 = observed features adjusted for other features, V2 = + stress proxy, E2 = oracle true features without stress). Model value = increment of the single-day profile at lag j, converted to simulator units (divided by standardised-units-per-u) and by the benchmark.

| pair | planted lag | source | normalised coefficient at lags 1,2,3,4,5 | peak lag (1-5) | r with truth-increment profile |
|---|---|---|---|---|---|
| sleep_variability->dep | 3 | analytic truth | +0.00, +0.00, +1.00, -0.08, -0.07 | 3 | 1.00 |
| sleep_variability->dep | 3 | Stage 5 V1 | +0.13, +0.28, +0.56, +0.42, +0.79 | 5 | +0.20 |
| sleep_variability->dep | 3 | Stage 5 V2 | +0.02, +0.22, +0.46, +0.34, +0.68 | 5 | +0.20 |
| sleep_variability->dep | 3 | Stage 5 E2 | +0.23, +0.03, +0.97, +0.05, -0.05 | 3 | +0.98 |
| sleep_variability->dep | 3 | model gru (mean of 5 seeds) | +0.21, +0.04, -0.03, -0.02, -0.03 | [1, 1, 1, 1, 1] | -0.27 |
| sleep_variability->dep | 3 | model m1 (mean of 5 seeds) | +0.10, -0.03, -0.07, -0.09, +0.00 | [1, 1, 4, 4, 3] | -0.34 |
| sleep_variability->dep | 3 | model mlp_m1 (mean of 5 seeds) | +0.10, +0.02, -0.11, -0.08, -0.03 | [3, 1, 1, 3, 3] | -0.55 |
| sleep_variability->dep | 3 | model kangru (mean of 5 seeds) | +0.14, +0.12, -0.04, -0.04, -0.12 | [1, 1, 5, 5, 2] | -0.17 |
| steps->dep | 2 | analytic truth | +0.00, +1.00, -0.08, -0.07, -0.07 | 2 | 1.00 |
| steps->dep | 2 | Stage 5 V1 | +0.74, +0.76, +0.46, +0.42, +0.84 | 5 | +0.38 |
| steps->dep | 2 | Stage 5 V2 | +0.65, +0.71, +0.41, +0.38, +0.76 | 5 | +0.45 |
| steps->dep | 2 | Stage 5 E2 | +0.15, +0.83, +0.10, +0.05, +0.07 | 2 | +1.00 |
| steps->dep | 2 | model gru (mean of 5 seeds) | +0.02, -0.07, -0.11, -0.08, -0.08 | [1, 1, 1, 3, 1] | +0.02 |
| steps->dep | 2 | model m1 (mean of 5 seeds) | +0.33, -0.06, -0.15, -0.06, -0.05 | [1, 1, 1, 1, 1] | -0.13 |
| steps->dep | 2 | model mlp_m1 (mean of 5 seeds) | +0.28, +0.03, -0.19, -0.10, -0.09 | [1, 1, 3, 1, 1] | +0.20 |
| steps->dep | 2 | model kangru (mean of 5 seeds) | +0.29, -0.03, +0.01, +0.07, -0.12 | [1, 1, 1, 4, 1] | -0.21 |
| steps->anx | 2 | analytic truth | +0.00, +1.00, -0.08, -0.07, -0.07 | 2 | 1.00 |
| steps->anx | 2 | Stage 5 V1 | +1.29, +0.47, +0.85, +0.62, +1.30 | 5 | -0.61 |
| steps->anx | 2 | Stage 5 V2 | +1.11, +0.38, +0.77, +0.52, +1.15 | 5 | -0.63 |
| steps->anx | 2 | Stage 5 E2 | +0.17, +1.19, +0.10, +0.07, +0.40 | 2 | +0.96 |
| steps->anx | 2 | model gru (mean of 5 seeds) | +0.38, +0.04, -0.08, -0.05, -0.07 | [1, 1, 1, 1, 1] | +0.05 |
| steps->anx | 2 | model m1 (mean of 5 seeds) | +0.40, -0.03, -0.16, -0.03, -0.04 | [1, 1, 1, 1, 1] | -0.08 |
| steps->anx | 2 | model mlp_m1 (mean of 5 seeds) | +0.35, +0.04, -0.26, -0.11, -0.11 | [1, 1, 3, 1, 1] | +0.21 |
| steps->anx | 2 | model kangru (mean of 5 seeds) | +0.11, +0.22, +0.01, +0.02, +0.06 | [1, 2, 1, 2, 4] | +0.92 |
| hrv->anx | 1 | analytic truth | +1.00, -0.08, -0.07, -0.07, -0.06 | 1 | 1.00 |
| hrv->anx | 1 | Stage 5 V1 | +1.09, +0.38, +0.20, +0.44, +0.74 | 1 | +0.84 |
| hrv->anx | 1 | Stage 5 V2 | +1.02, +0.34, +0.17, +0.38, +0.66 | 1 | +0.85 |
| hrv->anx | 1 | Stage 5 E2 | +1.15, -0.07, -0.03, +0.01, +0.11 | 1 | +0.99 |
| hrv->anx | 1 | model gru (mean of 5 seeds) | -0.16, -0.16, -0.13, -0.06, -0.06 | [2, 1, 2, 2, 1] | -0.49 |
| hrv->anx | 1 | model m1 (mean of 5 seeds) | +0.05, +0.03, -0.05, -0.01, +0.05 | [1, 1, 1, 5, 1] | +0.48 |
| hrv->anx | 1 | model mlp_m1 (mean of 5 seeds) | +0.05, +0.07, -0.01, +0.04, +0.02 | [4, 1, 2, 1, 2] | +0.34 |
| hrv->anx | 1 | model kangru (mean of 5 seeds) | +0.23, +0.12, -0.11, +0.06, -0.16 | [5, 3, 1, 5, 1] | +0.70 |

## 4. Integrated-gradients attribution (baseline = all-zero standardised window, 16 midpoint steps, 600 origins; mean |IG| by lag and channel; mean over seeds)

| model | pair | planted lag | mean-|IG| peak lag (per seed) | pre-lag share of |IG| over lags 0-10 | planted channel's rank among 8 by total |IG| (per seed) | seed-to-seed r of the day profile | r(|IG| profile, |perturbation| profile) | mean signed IG sign ok | completeness error (pts) |
|---|---|---|---|---|---|---|---|---|---|
| gru | sleep_variability->dep | 3 | [2, 2, 4, 2, 1] | 0.27 | [2, 2, 2, 2, 2] | +0.72 | +0.99 | 0% | 0.007 |
| gru | steps->dep | 2 | [2, 1, 2, 0, 0] | 0.21 | [1, 1, 1, 1, 1] | +0.77 | +0.95 | 100% | 0.007 |
| gru | steps->anx | 2 | [2, 2, 2, 2, 4] | 0.16 | [2, 2, 2, 2, 2] | +0.90 | +0.85 | 100% | 0.007 |
| gru | hrv->anx | 1 | [0, 0, 0, 0, 0] | 0.14 | [1, 1, 1, 1, 1] | +0.98 | +0.99 | 100% | 0.007 |
| m1 | sleep_variability->dep | 3 | [1, 2, 1, 1, 0] | 0.37 | [3, 2, 3, 3, 3] | +0.85 | +0.96 | 20% | 0.153 |
| m1 | steps->dep | 2 | [1, 1, 1, 1, 1] | 0.22 | [1, 1, 1, 1, 1] | +0.95 | +0.97 | 20% | 0.153 |
| m1 | steps->anx | 2 | [5, 1, 1, 1, 1] | 0.21 | [1, 1, 1, 1, 1] | +0.78 | +0.96 | 0% | 0.153 |
| m1 | hrv->anx | 1 | [5, 6, 6, 6, 2] | 0.08 | [2, 3, 2, 2, 2] | +0.59 | +0.87 | 60% | 0.153 |
| mlp_m1 | sleep_variability->dep | 3 | [2, 1, 1, 0, 2] | 0.40 | [3, 3, 2, 3, 2] | +0.92 | +0.99 | 60% | 0.008 |
| mlp_m1 | steps->dep | 2 | [2, 1, 2, 2, 1] | 0.24 | [1, 1, 1, 1, 1] | +0.97 | +0.98 | 60% | 0.008 |
| mlp_m1 | steps->anx | 2 | [2, 1, 2, 2, 1] | 0.23 | [1, 1, 1, 1, 1] | +0.97 | +0.98 | 60% | 0.008 |
| mlp_m1 | hrv->anx | 1 | [6, 3, 4, 6, 5] | 0.08 | [2, 2, 2, 2, 2] | +0.71 | +0.96 | 40% | 0.008 |
| kangru | sleep_variability->dep | 3 | [2, 6, 2, 1, 2] | 0.31 | [2, 2, 2, 2, 2] | +0.81 | +0.99 | 0% | 0.000 |
| kangru | steps->dep | 2 | [4, 5, 2, 4, 4] | 0.18 | [1, 1, 1, 1, 1] | +0.57 | +0.94 | 60% | 0.000 |
| kangru | steps->anx | 2 | [6, 5, 2, 7, 4] | 0.14 | [2, 2, 2, 2, 2] | +0.22 | +0.83 | 60% | 0.000 |
| kangru | hrv->anx | 1 | [4, 2, 2, 4, 4] | 0.08 | [1, 1, 1, 1, 1] | +0.82 | +0.98 | 100% | 0.000 |

## 5. Stability

Seed-to-seed: mean pairwise Pearson correlation between seeds of the single-day level profile (lags 0-10), and SD of the increment-peak lag over seeds. Cross-model: correlation between the seed-mean profiles of two architectures.

| model | pair | seed-to-seed r | increment-peak lag SD over seeds | split-half r (within seed) |
|---|---|---|---|---|
| gru | sleep_variability->dep | +0.70 | 0.55 | +0.99 |
| gru | steps->dep | +0.93 | 0.00 | +0.99 |
| gru | steps->anx | +0.95 | 0.00 | +0.97 |
| gru | hrv->anx | +0.99 | 0.00 | +0.99 |
| m1 | sleep_variability->dep | +0.88 | 0.00 | +0.98 |
| m1 | steps->dep | +0.95 | 0.00 | +0.98 |
| m1 | steps->anx | +0.91 | 0.00 | +0.98 |
| m1 | hrv->anx | +0.57 | 0.00 | +0.95 |
| mlp_m1 | sleep_variability->dep | +0.91 | 0.00 | +0.99 |
| mlp_m1 | steps->dep | +0.97 | 0.00 | +0.98 |
| mlp_m1 | steps->anx | +0.98 | 0.00 | +0.98 |
| mlp_m1 | hrv->anx | +0.73 | 0.00 | +0.95 |
| kangru | sleep_variability->dep | +0.83 | 0.00 | +0.99 |
| kangru | steps->dep | +0.59 | 0.00 | +0.98 |
| kangru | steps->anx | +0.31 | 0.45 | +0.97 |
| kangru | hrv->anx | +0.79 | 0.00 | +0.98 |

| pair | gru vs m1 | gru vs mlp_m1 | gru vs kangru | m1 vs mlp_m1 | m1 vs kangru | mlp_m1 vs kangru |
|---|---|---|---|---|---|---|
| sleep_variability->dep | +0.60 | +0.62 | +0.85 | +0.97 | +0.81 | +0.86 |
| steps->dep | +0.89 | +0.94 | +0.73 | +0.98 | +0.90 | +0.88 |
| steps->anx | +0.93 | +0.89 | +0.50 | +0.95 | +0.43 | +0.20 |
| hrv->anx | +0.01 | -0.04 | +0.54 | +0.82 | +0.47 | +0.67 |
