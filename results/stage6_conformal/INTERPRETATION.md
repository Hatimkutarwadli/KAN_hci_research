# Stage 6 - Conformal uncertainty and subgroup reliability (SYNTHETIC data)

## What was done
Split conformal prediction intervals for **depression and anxiety severity** (nowcast and forecast at 7, 14, 21 days: 8 tasks) on the **saved predictions** of six existing models (GBT, GRU, KAN, KAN-reg, M1, M1-ablation; neural models 5 seeds each). No model was retrained. All models use identical calibration rules, so the comparison is like for like.

## Procedure (fixed before looking at results)
- **Nonconformity score:** the absolute residual |y - yhat|. Intervals are yhat +- q: symmetric and of constant width 2q.
- **Calibration:** the 200 existing calibration participants; the 200 test participants are used only for final evaluation. Test labels never influence q (a test checks this).
- **Participant-level calibration:** each participant contributes exactly **one** (origin, target) pair, so the 200 scores are not 18,600 correlated daily observations. The origin is drawn uniformly from days 27-98 (28-day lookback satisfied; day +21 still observed, so every horizon is eligible) with a fixed seed (20261114), identical for every model, task and level.
- **Quantile:** q is the ceil((n+1)(1-alpha))-th smallest calibration score (finite-sample rule), alpha = 0.10 and 0.05.
- **Primary result:** that single pre-specified origin draw. **Robustness:** the same rule repeated with 1000 further random origin draws (reported as the mean, and the 2.5-97.5% spread, of coverage).
- **Test counts:** 200 test participants and 200 calibration participants per task and level (one origin each); 8 tasks x 2 levels x (5 seeds for neural models, 1 for GBT).

## Overall coverage and width (test; mean over the 8 severity tasks; neural models averaged over seeds)
| model | 90%: pre-specified draw | 90%: mean of 1000 draws | 90%: width | 95%: pre-specified draw | 95%: mean of 1000 draws | 95%: width (clipped to 0-27) |
|---|---|---|---|---|---|---|
| GBT | 0.894 | 0.898 | 10.1 | 0.948 | 0.952 | 12.4 (12.4) |
| GRU | 0.897 | 0.900 | 10.1 | 0.953 | 0.952 | 12.4 (12.3) |
| KAN | 0.903 | 0.902 | 10.5 | 0.954 | 0.954 | 12.6 (12.6) |
| KAN-reg | 0.893 | 0.898 | 10.3 | 0.953 | 0.952 | 12.6 (12.6) |
| M1-ablation | 0.893 | 0.896 | 10.6 | 0.953 | 0.950 | 13.0 (13.0) |
| **M1** | 0.899 | 0.894 | 10.4 | 0.947 | 0.949 | 12.6 (12.6) |

Nominal 90% and 95%. Width is in severity points (the score runs 0-27); clipping intervals to 0-27 shortens them only slightly.

Per-task detail for the strongest baseline (GRU) and the proposed model (M1); all models and seeds in `coverage_width_summary.csv` / `coverage_width_by_seed.csv`:

| task | model | 90%: coverage (draw) / mean of 1000 draws / width | 95%: coverage (draw) / mean of 1000 draws / width | test RMSE (all origins) |
|---|---|---|---|---|
| sev_dep_t0 | GRU | 0.901 / 0.889 / 9.9 | 0.950 / 0.944 / 12.3 | 2.97 |
| sev_dep_t0 | **M1** | 0.889 / 0.876 / 10.5 | 0.932 / 0.934 / 12.5 | 3.14 |
| sev_dep_h7 | GRU | 0.820 / 0.890 / 9.3 | 0.904 / 0.944 / 11.1 | 3.05 |
| sev_dep_h7 | **M1** | 0.842 / 0.885 / 10.0 | 0.905 / 0.943 / 11.6 | 3.19 |
| sev_dep_h14 | GRU | 0.864 / 0.901 / 9.8 | 0.926 / 0.951 / 11.5 | 3.19 |
| sev_dep_h14 | **M1** | 0.872 / 0.896 / 10.3 | 0.931 / 0.952 / 12.2 | 3.32 |
| sev_dep_h21 | GRU | 0.930 / 0.906 / 11.3 | 0.980 / 0.956 / 14.5 | 3.33 |
| sev_dep_h21 | **M1** | 0.945 / 0.903 / 11.8 | 0.978 / 0.956 / 15.3 | 3.43 |
| sev_anx_t0 | GRU | 0.893 / 0.902 / 9.5 | 0.969 / 0.951 / 12.4 | 2.91 |
| sev_anx_t0 | **M1** | 0.900 / 0.897 / 9.7 | 0.958 / 0.948 / 11.8 | 3.04 |
| sev_anx_h7 | GRU | 0.935 / 0.904 / 10.4 | 0.974 / 0.957 / 13.6 | 2.93 |
| sev_anx_h7 | **M1** | 0.926 / 0.895 / 10.2 | 0.971 / 0.950 / 13.0 | 3.08 |
| sev_anx_h14 | GRU | 0.938 / 0.907 / 10.4 | 0.972 / 0.958 / 12.0 | 3.05 |
| sev_anx_h14 | **M1** | 0.934 / 0.897 / 11.0 | 0.953 / 0.952 / 12.2 | 3.19 |
| sev_anx_h21 | GRU | 0.895 / 0.904 / 10.1 | 0.946 / 0.955 / 11.7 | 3.17 |
| sev_anx_h21 | **M1** | 0.884 / 0.899 / 10.1 | 0.947 / 0.953 / 12.1 | 3.28 |

## Subgroup reliability (test participants; sex and age group)
| subgroup | test n | calibration n | RMSE (all test origins) | 95%: mean coverage over 1000 origin draws (gap vs overall) | 95%: pre-specified draw [Wilson 95% CI] | reliability |
|---|---|---|---|---|---|---|
| **GRU** | | | | | | |
| sex F | 133 | 125 | 3.08 | 0.953 (+0.001) | 0.961 [0.91, 0.98] | ok |
| sex M | 67 | 75 | 3.07 | 0.951 (-0.001) | 0.937 [0.86, 0.97] | ok |
| age group 0 | 84 | 89 | 3.03 | 0.957 (+0.005) | 0.951 [0.88, 0.98] | ok |
| age group 1 | 104 | 96 | 3.13 | 0.947 (-0.005) | 0.954 [0.90, 0.98] | ok |
| age group 2 | 12 | 15 | 2.88 | 0.965 (+0.013) | 0.952 [0.69, 0.99] | UNRELIABLE (n=12<30) |
| **M1** | | | | | | |
| sex F | 133 | 125 | 3.23 | 0.947 (-0.002) | 0.955 [0.91, 0.98] | ok |
| sex M | 67 | 75 | 3.16 | 0.952 (+0.004) | 0.931 [0.85, 0.97] | ok |
| age group 0 | 84 | 89 | 3.16 | 0.953 (+0.005) | 0.944 [0.87, 0.98] | ok |
| age group 1 | 104 | 96 | 3.26 | 0.943 (-0.005) | 0.946 [0.89, 0.97] | ok |
| age group 2 | 12 | 15 | 3.10 | 0.962 (+0.013) | 0.971 [0.72, 0.99] | UNRELIABLE (n=12<30) |

Cells are averaged over the 8 severity tasks (and over seeds for the neural models). Per-task, per-model, per-level tables: `subgroup_summary.csv`, `subgroup_by_seed.csv`.

## What the results mean
- **Overall calibration works on average.** Averaged over many random origin draws, empirical coverage is within about 1 point of nominal for every model (0.894-0.902 at 90%, 0.949-0.954 at 95%), and the repeated-draw mean at 95% lies inside the +-1.96 SE band for 200 participants for every task. Intervals are wide: about +-5 points at 90% and +-6 at 95% on a 0-27 scale (width about 10 and 12.5), because the models' RMSE is about 3; the width equals about 1.64 x RMSE (90%) and 1.99 x RMSE (95%) on each side. Intervals widen with the forecast horizon for depression (95% width about 12-12.7 for the nowcast and 14.5-15.6 at 21 days) and only slightly for anxiety (about 12-13 throughout). Models with lower error give narrower intervals (GRU and GBT narrowest; the ablation widest), so conformal calibration keeps coverage the same while the error differences show up as width.
- **A single origin draw is noisy.** The pre-specified draw gives coverage as low as 0.81-0.86 (depression, 7 days, 90%) and as high as 0.98 (depression, 21 days, 95%), while the mean over 1000 draws is 0.89 and 0.955. With one origin per participant and n = 200 the coverage standard deviation is about 0.03 at 90% (calibration noise plus test noise); for example the GRU's 0.82 sits in the lowest 1.6% of the 1000 draws. Because the same origins are used for every model and task, these deviations are shared by all models (a property of the draw, not of a model). I did not redraw to obtain nicer numbers; both views are reported.
- **No clear subgroup coverage gap in this synthetic dataset, but small groups cannot be judged.** For groups with enough participants (sex F/M, age groups 0 and 1) the repeated-draw coverage differs from the overall value by at most 0.014 (90%) and 0.011 (95%); error levels are also similar (RMSE about 3.0-3.3). Age group 2 (the small, noisier-sensor group) has only 12 test and 15 calibration participants: its coverage (0.964 at 95%, mean of draws) is not distinguishable from nominal, but its Wilson 95% interval is about [0.7, 0.99], and gaps including it reach 0.044 (90%) / 0.028 (95%). These estimates are flagged UNRELIABLE (n < 30) and no conclusion is drawn from them. The hypothesis that the noisy small group would be under-covered is **not supported** here, but the data cannot confirm or refute it.
- **Model comparison:** coverage is essentially the same for all models; the models differ in interval width, which follows their RMSE (GRU/GBT narrowest).

## Assumptions and limitations (no formal time-series guarantee is claimed)
- **Exchangeability is assumed at the participant level:** calibration and test participants are drawn independently by the simulator, and one randomly drawn origin per participant removes the within-participant dependence from the calibration scores. Under these conditions split conformal gives about 1-alpha coverage on average over calibration and test draws (verified on iid residuals in a unit test); it does **not** guarantee coverage for a given participant's own future days, for sequences of intervals along a trajectory (consecutive days are highly dependent), or for any single calibration/test sample (hence the noise above).
- **Calibration data were also used to select the models** (early stopping, GBT depth/iterations, F1 thresholds), which breaks strict independence between the fitted model and the calibration scores and can make q slightly optimistic. The average coverage here is at nominal, so no optimism is visible, but it cannot be ruled out.
- **Distribution shift:** calibration and test come from the same stationary simulator. Real data with changing populations, devices or behaviour would break exchangeability, and the intervals would need re-calibration; nothing here supports real-world coverage.
- **Constant-width intervals** cannot adapt to heteroscedasticity. Marginal coverage does not imply group-conditional coverage: group-conditional (Mondrian) conformal was not run because calibration groups are too small (age group 2 has 15 calibration participants, but a 95% interval needs at least 19; sex and large age groups could be done but were out of scope).
- **Subgroups:** sample sizes of 12-15 give very wide uncertainty; no subgroup guarantee is claimed. The pre-specified-draw subgroup numbers are noisier than the repeated-draw averages, which remove origin-sampling noise but not participant-sampling noise (the same 12 participants).
- Severity is bounded at 0-27 but unclipped intervals can extend outside it (clipped widths reported).
- One simulated dataset; synthetic results only.

Files in `results/stage6_conformal/`: `intervals_primary_draw.csv` (per participant: origin, y, prediction, lo, hi, covered, sex, age group, for calibration and test rows), `coverage_width_by_seed.csv`, `coverage_width_summary.csv`, `subgroup_by_seed.csv`, `subgroup_summary.csv`, `config.json`, figures `fig1_coverage_overall.png`, `fig2_interval_width.png`, `fig3_subgroup_coverage.png`.
