# Changelog

## v3.1.0 — 2026-10-05 — ships AstroNet-Vetting-v2 ⚠️ BIG UPDATE by Pablo

**New vetting model, AstroNet-Vetting-v2, plus the cadence-aware, scatter-weighted
preprocessing it needs (`cad_scat_v5`).**

- **Model:** AstroNet-Vetting-v2, a **10-model ensemble** (config `pablomer_final`, trained
  Mar-2026).
  - Shared copy with `MODEL_CARD.md`: `/pdo/astronet-data/models/vetting/astronet-vetting-v2/`.
  - QLP's default model dir `/pdo/astronet-data/models/vetting/production/` gets the same 10
    members when `/sw/astronet` moves to this release. QLP's `estools astronet --vetting` is step
    7 of the pipeline.
- **Preprocessing `cad_scat_v5`:** cadence-aware binning + scatter weighting, in
  `astronet/preprocess/generate_input_records.py`, the only generator. That file defines
  `PREPROCESSING_VERSION = "cad_scat_v5"` and logs it.
  - Verified to reproduce the model's training records (`dec2025_cad_scat_v5_aug`).
  - **Not compatible** with models trained on ≤ 3.0.1 preprocessing.
- **Ensemble scoring:** average `disp_p` over the 10 members. With QLP's filter, use
  `astronet-filter --score-calc avg`; its default `max` is wrong for this ensemble.
- **Script mode fixed:** removed a leftover debug filter that wrote a single TCE, and replaced
  `np.int`, which crashed on NumPy ≥ 1.24.
- **Known jitter:** `sample_segments` breaks ties between equal-sized folds at random (unseeded,
  as in 3.0.1). The ensemble-mean `disp_p` can move by up to ~0.07 between runs; 99% of TCEs
  move less than 0.02.
- **Other changes:**
  - Architecture refactor in `astronet/astro_cnn_model/astro_cnn_model.py`
    (`backbone()` / `head()` / `get_embeddings()`).
  - Preprocessing changes in `astronet/preprocess/{preprocess,generate_input_records}.py` and
    `light_curve_util/median_filter2.py`.
- **API:** the QLP↔astronet public API is unchanged:
  `predict.batch_predict(models_dir, data_files)` and
  `preprocess.generate_input_records.create(..., mode=)`.

> ### ⏪ If something breaks, THIS is the reference point.
> **Roll back:** point `/sw/astronet` back to `astronet-3.0.1` (tag **`v3.0.1`**) and move
> the previous model `AstroCNNModelVetting_cshallue_20250429_181612` from
> `/pdo/astronet-data/models/vetting/legacy/` back into `production/`.
>
> **Running the previous model without rolling back:**
> `PYTHONPATH=/sw/astronet-versions/astronet-3.0.1:$PYTHONPATH qlp estools astronet --vetting --model-dir /pdo/astronet-data/models/vetting/legacy ...`

This release was landed as a **code-only** curated merge from `pablomer-dev-training`:
notebooks, data files, and scratch/experiment scripts were intentionally excluded
(see `.gitignore`). `astronet/models.py` was kept from `main`, which drops the
experimental `ablation_studies` imports that are not part of production.

See `UPDATING.md` for how to put this version on your `PYTHONPATH`.

---

## v3.0.1 and earlier

Previous production vetting model: `AstroCNNModelVetting_cshallue_20250429_181612`.
See git history for details.
