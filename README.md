# Do Kolmogorov–Arnold Networks Help Longitudinal Digital Phenotyping?

**A controlled synthetic evaluation of prediction, lag recovery and uncertainty.**

Undergraduate semester research project: *Longitudinal, Causally-Informed Multitask Digital Phenotyping with Uncertainty-Aware KAN Models*.

| Author | Registration |
|---|---|
| Hatim Kutarwadliwala | U24AI103 |
| Sakina Dhilawala | U24AI104 |
| Krisha Chaudhari | U24AI115 |
| Chris Reji George | U24AI010 |
| Sruthi Jayesh | U24AI004 |

**Supervisor:** Dr. Sudhakar Mishra.
**Affiliation:** Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India.

> **All data and results in this repository are SYNTHETIC.** They come from our own simulator (one simulator family). Nothing here is evidence about real patients, clinical usefulness or causal effects in humans.

---

## What this project does

Kolmogorov–Arnold Networks (KANs) replace fixed activations with learnable spline functions and are often presented as more accurate and more interpretable than MLPs. We tested whether KAN components actually help **longitudinal, multitask** prediction of depression and anxiety from daily wearable-style signals.

We built a simulator with known ground truth:
- latent depression and anxiety states;
- 8 noisy wearable channels;
- 4 planted lagged effects;
- a hidden confounder;
- informative missingness.

On this simulator we compared:
- gradient-boosted trees (GBT);
- a GRU;
- static and regularised KANs;
- a multitask KAN latent-transition model (M1);
- MLPs matched in **parameter count** to the KAN models;
- KAN-GRU, KAN-ODE, GRU-D, Transformer and CNN-LSTM.

The strongest evidence comes from **hash-locked plans and sealed test sets evaluated once**, on fresh replicate datasets.

## Main findings

| Finding | Evidence tier |
|---|---|
| No reproducible KAN advantage. On sealed tests, M1's parameter-matched MLP was consistently better in 30 comparisons and M1 in 3; KAN-reg vs its matched MLP was 4 vs 7. | A |
| No KAN-based temporal model (M1, KAN-GRU, KAN-ODE) beat the GRU. | A (M1); B (others) |
| No reproducible small-data KAN advantage at 50 or 100 training participants. | A |
| The GRU is the strongest or near-strongest model, but not uniformly the strongest (GBT matched or beat it on some families). | A / B |
| Planted effect directions were recovered, but timing mostly was not: the regression recovered 1 of 4 lags, and model attribution recovered no lags. | B (regression); A (attribution) |
| Severity conformal intervals reached nominal coverage (0.900 / 0.952). Sets for rare onset events missed positives or were uninformative. | B |
| Early warning was weak and mostly short-lead. | B |

**Evidence tiers** (defined in `final_submission/Final_Synthesis_Report.md` §3):
- **A:** sealed, pre-registered and replicated (P2-2, P3).
- **B:** the v1 dataset, whose test split was inspected early.
- **C:** calibration-only or descriptive results.

## Repository structure

| Path | Contents |
|---|---|
| `src/` | Simulator, preprocessing, models (KAN layer, static KAN, M1, GRU), baselines, causal analysis, conformal prediction, lead-time evaluation |
| `extension_*/` | Code for the later stages: Stages 7–10, P1-1 to P1-6, P2-1 (KAN-ODE), P2-2 (replication and shifts), P3 (small data) |
| `configs/` | Frozen simulator configuration (`frozen_config_v1.json`) and the replicate/shift configurations (`p22/`) |
| `tests/` | Unit and integrity tests (20 modules) |
| `data/` | `synthetic_v1.zip` (main dataset, 1,000 participants × 120 days), `pilot.zip`, and the metadata and checksums of every dataset |
| `results/` | Summary outputs of every stage: metrics tables, JSON conclusions, plots, analysis plans and hash-lock manifests |
| `final_submission/` | Final research report (`.md`/`.docx`), stage addenda and **`Final_Synthesis_Report.md`** (the authoritative summary) |
| `final_deliverables/` | IEEE TAI paper (LaTeX source and Overleaf ZIP), collective explanation PDF, speaking and viva guide PDF, QC report |
| `paper_prep/` | Literature audit, reference verification, research-gap analysis, evidence map |
| `docs/project_log/` | Project plan, decisions, status and the full experiment log |
| `references/` | Original proposal presentation and proposal guide |
| `TAI_LaTex_Template/` | IEEE TAI author template used for the paper |

## Quick start

Python 3.11 was used, on CPU only.

```bash
pip install -r requirements.txt
```

**Use the main dataset.** Unzip it in place:
```bash
cd data && unzip synthetic_v1.zip && cd ..
python src/preprocess.py --raw data/synthetic_v1 --out data/processed_v1
```

**Regenerate the replicate and shift datasets** from the frozen configurations (deterministic seeds):
```bash
python extension_p22/gen_p22.py generate --ds R1      # likewise R2, R3, S-EFF, S-AR, S-MISS, S-HET, S-NOISE
python extension_p3/gen_p3.py R4 R5
```

**Rebuild the paper figures** from the saved summary outputs:
```bash
python final_deliverables/IEEE_TAI_Paper/scripts/make_figures.py
```

**Tests.** `python -m unittest tests/test_simulator.py` runs on its own. Most other test modules check saved datasets or outputs, so they need the data regenerated first. Some need the model checkpoints, which are not included (see below).

**Training and evaluation.** Each stage's commands are documented at the top of its script, for example:
- `python src/baselines.py --models gbt kan gru --seeds 0 1 2 --out results/stage3`
- `python extension_p3/run_p3.py --set main`

The order and settings used are recorded in `docs/project_log/EXPERIMENT_LOG.md`.

## What is not included, and why

| Excluded | Size | Reason |
|---|---|---|
| Model checkpoints (`.pt`), fitted GBT models (`.joblib`), prediction arrays (`.npz`, `.npy`) and per-origin prediction CSVs larger than 5 MB | about 2.1 GB | Too large for Git. Regenerable by re-running the stage scripts. |
| Uncompressed and processed datasets, and the replicate and shift datasets | about 850 MB | Regenerable from `configs/` with the commands above. `synthetic_v1.zip` is included. |
| Third-party paper PDFs | — | Copyright. They are cited in the paper and in `paper_prep/REFERENCE_VERIFICATION.md`. |
| A teammate's separate real-data (SSAQS) exploration | — | Audited and kept separate; it is not part of this study's evidence. |

**Integrity note.** The hash-lock manifests in `results/` (for example `results/p3_small_data/test_lock.json`) list SHA-256 hashes of files that include the excluded checkpoints and data. Those locks can be re-verified only against a full local copy of the project, not against this repository alone.

## Paper and evaluation documents

- **IEEE TAI paper.** `final_deliverables/IEEE_TAI_Paper/`; upload `IEEE_TAI_Paper_Overleaf.zip` to Overleaf and compile `main.tex` with pdfLaTeX.
- **Plain-English project explanation:** `final_deliverables/Collective_Project_Explanation.pdf`.
- **Presentation and viva guide:** `final_deliverables/Individual_Speaking_and_Viva_Guide.pdf`.

## Limitations

- **Data:** synthetic, from one simulator family, with largely linear planted effects.
- **Models:** small, untuned re-implementations.
- **Replication:** few seeds and few replicate datasets.
- **Uncertainty:** no time-series or adaptive conformal methods.
- **No real-world validation.**

The full list is in `final_submission/Final_Synthesis_Report.md` §8.

## Citation

See `CITATION.cff`.

## License

**No license has been chosen yet.** Until one is added, default copyright applies, and others may not reuse the code. Add a `LICENSE` file (for example MIT for code, CC BY 4.0 for documents) once the team agrees.
