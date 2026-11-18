# P2-1 KAN-ODE: one-seed feasibility pilot (SYNTHETIC data, CALIBRATION split only)

Extension, not part of the original six stages. One seed (0), frozen protocol (`baselines.fit_nn`), frozen data `data/processed_v1`, unchanged 600/200/200 split. **The test split was never loaded** (`test_split_loaded: false` in every run log); no test metric exists for these models. One seed cannot establish superiority or inferiority; everything below is a feasibility assessment.

## 1. What was built
- **Hybrid KAN-ODE** (`extension_p21/models_p21.py`). This is *not* the input-conditioned formulation dz/dt = KAN([z, x(t)]). The latent evolves **autonomously between observation days**, dz/dt = kappa * (zbar - z + 2 tanh(K_f(z))), with K_f = KANLayer(8, 8), zbar = tanh(W0 s + b0) and kappa in [0.05, 2.0]. It is integrated with fixed-step RK4, h = 1 day (S = 1). Wearable inputs act on the state only through **gated discrete updates on observed days**: z+ = m * tanh(z- + K_g([z-, xt])) + (1 - m) * z-, with m = 1 - missing_day. Encoder, K_g (M1's transition) and heads are M1's. 21,580 parameters (M1 20,932; +3.1%).
- **Correction to the approved proposal (boundedness):** the originally proposed field -kappa(z - zbar) + tanh(K_f(z)) only gives |z - zbar| <= 1/kappa, which allows |z| up to about 21 at kappa = 0.05, far outside the spline grid [-3, 3]. It was reparameterised as above, so that the exact flow satisfies |z - zbar| <= 2, hence |z| < 3. The analytic bound does **not** carry over to the S = 1 discretisation under stiff forcing. In an adversarial test (kappa = 2, K_f weights x20), RK4 with S = 1 reached |z| = 16.3, while S = 16 stayed at 2.91. So extrapolation was monitored during training, not assumed away.
- **No-flow ablation:** same model with z held constant between observation days (gated updates only), 20,932 parameters.
- **GRU-D-style GRU** (own simplified version after Che et al., 2018; citation to be verified before use): within-window deltas for two mask groups (whole day, HRV), input decay towards the train mean, hidden-state decay, GRUCell(10, 48), GRU readout; 11,024 / 10,892 parameters (GRU 10,864 / 10,732; +1.5%).
- Tests `tests/test_p21.py`: **9 pass**, covering RK4 order about 4 and stability for kappa*h <= 2, exact reduction to M1, missing-day gating, causality, boundedness, GRU-D deltas and imputation against hand-computed values, GRU-D training determinism, pipeline smoke test and existing-file integrity. Pre-run manifest: 2,913 files; after the pilot, 0 changed.

## 2. Runtime (gate G2: <= 75 min per seed)
| Run | Wall | CPU-s | Epochs (best) |
|---|---|---|---|
| Benchmark (before training) | KAN-ODE S=1 2.9x M1 per batch, projected 32-43 min/seed; S=2 4.9x | | |
| KAN-ODE (28 heads) | **15.3 min** | 3,619 | 9 (5) |
| No-flow (28 heads) | 7.5 min | 1,773 | 9 (5) |
| GRU-D main / forecast | 36 s / 37 s | | 8 (4) / 8 (4) |
| Reference: M1 saved, 5 seeds | 8.8-14.6 min | 2,067-3,346 | 7-11 (3-7) |

G2 passes. The measured wall time was below the benchmark projection; the projection assumed M1's epoch count and load.

## 3. Numerical stability (gate G1)
- No non-finite latent or output values in any epoch.
- Latent range: max|z| in training 2.55 (epoch 1), then 1.38-1.66; on calibration <= 1.39. **Fraction of K_f and K_g latent inputs outside the spline grid [-3, 3]: 0 in every epoch** (train and calibration). No-flow: max|z| < 0.995.
- kappa per epoch: min 0.52-0.57, max 0.58 -> 0.92, **rising monotonically through all 9 epochs** (not plateaued). kappa*h <= 0.92, inside RK4's accurate range (the test shows accuracy degrades towards kappa*h = 2).
- Solver error, S = 1 vs S = 4 on a fixed calibration batch: max|dz| 0.0030 -> 0.0109 and max|d output| 0.0018 -> 0.0120 over epochs; **0.0062 / 0.0045 at the selected epoch (5)**. Small but growing with kappa; S = 1 is adequate for this run, not guaranteed for longer training.
- G1 passes.

## 4. Calibration comparison (gate G3; feasibility only)
Best calibration loss (selection criterion; same protocol and heads as the reference):

| Model | Best cal loss | Reference 5-seed mean ± SD (min-max) | z | % vs mean |
|---|---|---|---|---|
| KAN-ODE | 0.3427 | M1 0.3470 ± 0.0037 (0.3431-0.3530) | -1.14 | -1.2% |
| No-flow | 0.3422 | M1 as above | -1.28 | -1.4% |
| GRU-D main | 0.2189 | GRU 0.2180 ± 0.0005 (0.2175-0.2187) | +1.88 | +0.4% |
| GRU-D forecast | 0.4841 | GRU 0.4791 ± 0.0021 (0.4768-0.4814) | +2.35 | +1.1% |

G3: KAN-ODE is not worse than M1 (below M1's 5-seed mean, about 1 SD). **Feasible.** This is not evidence of superiority: one seed, and the selection metric itself.

Calibration metrics by family (mean over heads; seed 0 vs reference 5-seed mean ± SD):

| Family | KAN-ODE | No-flow | M1 | GRU-D | GRU |
|---|---|---|---|---|---|
| Severity nowcast RMSE | 3.008 | 3.006 | 3.054 ± 0.033 | 2.909 | 2.923 ± 0.009 |
| Severity all (t0 + forecast) RMSE | 3.222 | 3.220 | 3.252 ± 0.025 | 3.164 (forecast) | 3.147 ± 0.011 (forecast) |
| Status nowcast AP | **0.204** | 0.240 | 0.252 ± 0.015 | 0.296 | 0.287 ± 0.006 |
| Status forecast AP | **0.192** | 0.235 | 0.231 ± 0.017 | 0.236 | 0.242 ± 0.009 |
| Sustained onset AP | 0.121 | 0.125 | 0.113 ± 0.005 | 0.124 | 0.132 ± 0.009 |
| First-crossing onset AP | 0.142 | 0.144 | 0.132 ± 0.006 | 0.147 | 0.153 ± 0.006 |

(M1, KAN-ODE and no-flow: 28-head models; GRU and GRU-D: separate main / forecast models. Compare within families, not across.)

**Gap-stratified (days since last observed day at the origin; nowcast; n origins 15,551 / 1,971 / 1,078 for 0 / 1-2 / >=3):**
- Depression status AP: KAN-ODE 0.218 / 0.251 / 0.243; no-flow 0.221 / 0.318 / 0.421; M1 0.260 / 0.318 / 0.358 (SD 0.016-0.046).
- Depression severity RMSE: KAN-ODE 2.957 / 3.118 / 3.149; no-flow 2.960 / 3.115 / 3.087; M1 3.020 / 3.243 / 3.251.
- GRU-D vs GRU: within about 1-2 GRU SD in every stratum; no consistent gap-specific benefit (e.g. severity at >=3 days: dep 2.980 vs 2.998 ± 0.015, anx 3.124 vs 3.135 ± 0.012).
- Full tables: `analysis/cal_gap_strata_summary.csv`.

## 5. Interpretation (seed 0, calibration; not decision-grade)
1. The KAN-ODE trains stably within the spline grid at about 1.4x M1's wall time. The technical feasibility questions are answered positively.
2. **The small loss, severity and onset gains over M1 are also present in the no-flow ablation.** On this seed, they come from skipping updates on missing days (gating), not from continuous dynamics. Adding the flow does not improve loss (0.3427 vs 0.3422).
3. **The continuous flow coincides with markedly lower status AP:** nowcast 0.204 vs 0.252 ± 0.015 for M1 (about 3 SD), and vs 0.240 for no-flow. The deficit is present even when the origin day is observed and is largest after gaps (depression). A plausible mechanism is that mean reversion towards zbar between updates erodes status-relevant state. This is an untested hypothesis, and one seed cannot exclude seed noise.
4. GRU-D is not better than the GRU on calibration loss (+0.4% / +1.1%, about 2 SD). Its family metrics are mixed and within noise. There is no evidence here that decay-aware handling helps the GRU on this simulator.

## 6. Limitations
One seed; calibration split only (also used for epoch selection, so the best-loss comparison is optimistic for every model equally); kappa still rising when training stopped; integer-day gaps, so S = 1 KAN-ODE is a structured discrete map; the M1 encoder's patches include forward-filled neighbouring days; GRU-D is a simplified re-implementation; head sets differ between the M1 and GRU families. No clinical, causal, interpretability or foundation-model claim.

## Files
`configs/p2_kanode_pilot.json`; `extension_p21/{models_p21,run_p21}.py`; `tests/test_p21.py`; `results/p2_kanode/manifest_pre.json`; `results/p2_kanode/pilot/{benchmark.json, kanode_all, kanode_noflow_all, grud_main, grud_forecast, analysis/}` (each run: `run_log.json` with per-epoch diagnostics, `train_log.txt`, `metrics_cal.csv`, calibration predictions, checkpoint).
