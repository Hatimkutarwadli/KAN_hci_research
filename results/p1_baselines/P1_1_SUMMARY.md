# P1-1: Transformer-encoder and CNN-LSTM baselines (SYNTHETIC; EXTENSION, not part of the original six stages)

Code `extension_p1/{models_p1,run_p1,compare_p1}.py`; tests `tests/test_p1.py` (8 pass); results `results/p1_baselines/{main,forecast,comparison}/`; full tables `comparison/p1_comparison.md`, `paired_vs_gru.csv`, `cost_and_convergence.csv`.
Protocol: frozen data/preprocessing/splits/heads and unchanged `baselines.fit_nn` (25 epochs max, patience 4, batch 256, Adam lr 2e-3, wd 1e-5, half-epoch sampling, epoch chosen on calibration loss, thresholds from calibration, test predicted once after selection), 5 seeds, no tuning, no sweeps. Same task sets as the GRU baseline: 16-head main set (nowcast severity/status, sustained and first-crossing onset) and 12-head forecast set. Architecture sizes were fixed in advance to the GRU budget (GRU 10,864 / 10,732 parameters; Transformer 11,512 / 11,380; CNN-LSTM 10,912 / 10,780).
Reproducibility: re-running Transformer seed 0 (main) reproduced all 444 saved metric values exactly (max difference 0); unit tests cover determinism, seed dependence, causal convolution, window-ends-at-origin, no test input in `fit_nn`, parameter budget, registry unchanged. Existing files hash-verified unchanged.

## Results (test mean over 5 seeds; paired worse% vs the same-seed GRU, negative = better; n = seeds better; "within noise" = seed SD or seed-0 bootstrap intervals overlap)
| Family (metric) | GRU | Transformer | CNN-LSTM |
|---|---|---|---|
| Nowcast severity (RMSE) | 2.937 | 2.939 (+0.1%, 3/5) | 2.954 (+0.6%, 0/5) |
| Nowcast status (AP / AUROC) | 0.343 / 0.860 | 0.342 / 0.861 | 0.341 / 0.858 |
| Sustained onset (AP) | 0.116 | 0.105 (+9.2%, 0/5) | 0.103 (+10.8%, 0/5) |
| Sustained onset (AUROC) | 0.787 | 0.783 (+0.5%, 1/5) | 0.788 (-0.1%, 3/5) |
| First-crossing onset (AP) | 0.111 | 0.107 (+3.6%, 0/5) | 0.104 (+5.8%, 0/5) |
| Forecast severity (RMSE) | 3.121 | 3.132 (+0.4%, 2/5) | 3.140 (+0.6%, 0/5) |
| Forecast status (AP / AUROC) | 0.278 / 0.815 | 0.285 / 0.815 | 0.275 / 0.812 |

- **Neither baseline beats the GRU on any family.** They are statistically indistinguishable from it on severity, status and forecast status (differences within seed SD; seed-0 bootstrap intervals overlap almost entirely), and nominally worse on onset AP (test: Transformer +9%, CNN-LSTM +11%, 0/5 seeds better; AUROC is on par).
- **Calibration vs test disagree for onset AP:** on the calibration split the Transformer ranked first for sustained onset (mean rank 2.33 vs GRU 2.83) and was on par with the GRU (-0.7% ±8.3), but on test it fell behind (+9.2%). Onset AP rests on 32 onset participants per target and has large seed SD (up to 8 points), so this is a noisy, not a decisive, difference.
- Ranking by mean rank (test): GRU first in nowcast severity (tied with the Transformer), nowcast status, sustained onset and first-crossing onset and forecast severity; forecast status: Transformer 1.33, CNN-LSTM 2.83, GRU 2.83 (not resolvable; seed-0 CIs overlap). The new models rank ahead of M1/KAN-GRU/KAN/GBT in most severity and onset groups; comparing with M1 and KAN-GRU is not configuration-matched (different head sets and regularisation).
- Interpretation: on this synthetic dataset a 28-day recurrent summary already captures what is extractable; neither attention nor convolution + LSTM adds information. This does not show that these architectures are inferior in general (one small configuration each, untuned, small data, short window).

## Computational cost (CPU, 4 torch threads) and convergence
| part | model | params | mean CPU s/seed | mean wall s/seed | mean best epoch | best epoch <= 2 | cal-loss rise after best |
|---|---|---|---|---|---|---|---|
| main | GRU (original run) | 10,864 | 95 | 25 | 3.4 | 0/5 | +0.0039 |
| main | Transformer | 11,512 | 345 | 103 | 3.4 | 1/5 | +0.0068 |
| main | CNN-LSTM | 10,912 | 229 | 69 | 3.2 | 1/5 | +0.0028 |
| forecast | GRU (original run) | 10,732 | 126 | 34 | 2.6 | 3/5 | +0.0128 |
| forecast | Transformer | 11,380 | 312 | 170* | 2.4 | 4/5 | +0.0105 |
| forecast | CNN-LSTM | 10,780 | 183 | 170* | 3.2 | 1/5 | +0.0110 |
Transformer is about 4x and CNN-LSTM about 2.5x the GRU's wall time (main). Total for both task sets, 20 trainings: roughly 35 min wall. *Forecast wall times are inflated by other processes running at the same time (wall > CPU for several runs); CPU seconds are the reliable cost figure. The GRU wall figures come from the original runs on possibly different load, so wall-time ratios are approximate.
Convergence/overfitting: every run stopped by patience (none reached 25 epochs); best epochs 2-5; calibration loss rises after the best epoch while train loss keeps falling (mild early overfitting, as for the GRU and the KAN models); no divergence or NaNs. 4/5 forecast Transformer seeds peaked at epoch <= 2, i.e. the best model is reached after very little training, with no room for a longer schedule under this protocol.

## Limitations
One small configuration per architecture, fixed in advance and untuned (a tuned Transformer or CNN-LSTM might do better; not tested by design); frozen training protocol chosen for the GRU; 5 seeds; seed-0 bootstrap only; 32 onset participants per target; one synthetic dataset; test split inspected in earlier stages (not used for any decision here). Own re-implementations, not the published Interpretable-MTLNet comparators (CNN-LSTM, Transformer baselines from the paper); no claim about the paper's reported numbers. Lead time, Stage 8 missingness robustness and conformal analyses were NOT repeated for these models (roadmap P1-5 and later).
