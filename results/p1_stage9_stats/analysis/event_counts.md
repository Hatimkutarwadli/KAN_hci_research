# P1-6 event counts and degenerate resamples (SYNTHETIC; EXTENSION)

Test split, 200 participants, origins 27,29,...,119; labels are identical across scenarios (only the inputs change). Tasks with fewer than 10 participants having any event are flagged: AP/AUROC for them is unstable.

| task | origins with defined label | positive origins | prevalence | participants with an event | flag |
|---|---|---|---|---|---|
| onset_sust_anx_h14 | 5744 | 188 | 0.033 | 32 |  |
| onset_sust_anx_h21 | 5192 | 255 | 0.049 | 32 |  |
| onset_sust_anx_h7 | 6152 | 103 | 0.017 | 32 |  |
| onset_sust_dep_h14 | 6043 | 188 | 0.031 | 32 |  |
| onset_sust_dep_h21 | 5467 | 259 | 0.047 | 31 |  |
| onset_sust_dep_h7 | 6475 | 103 | 0.016 | 31 |  |
| status_anx_h14 | 8000 | 624 | 0.078 | 63 |  |
| status_anx_h21 | 7200 | 556 | 0.077 | 57 |  |
| status_anx_h7 | 8600 | 669 | 0.078 | 66 |  |
| status_anx_t0 | 9400 | 744 | 0.079 | 70 |  |
| status_dep_h14 | 8000 | 572 | 0.071 | 56 |  |
| status_dep_h21 | 7200 | 501 | 0.070 | 57 |  |
| status_dep_h7 | 8600 | 610 | 0.071 | 64 |  |
| status_dep_t0 | 9400 | 688 | 0.073 | 64 |  |

Bootstrap resamples in which an AP/AUROC value was undefined (no positives or no negatives): 0 values in total.

