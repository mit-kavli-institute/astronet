# AstroNet-Vetting-v2: from labels to the deployed ensemble

How the 10-model vetting ensemble that QLP runs today (and that the 2026 paper reports) was
built: tables, light curves, TFRecords, training. Written for anyone who wants to reproduce it
or start fine-tuning / continual retraining from it. Everything below was re-checked on disk on
2026-10-15. Maintainer: Pablo (pablomer@mit.edu).

## 1. Where everything is

| What | Where |
|---|---|
| Code that scored the paper and runs in QLP | git tag `v3.1.0` = `5531bb2` (branch `curated-merge-2026.06`). `/sw/astronet` -> `/sw/astronet-versions/astronet-3.1.0`, byte-identical to that tag. |
| Driver scripts used for the paper (verbatim copies) | `scripts/vetting_v2_paper/` in this repo. Originals: branch `pablomer-dev-training` @ `6bec68d`, `astronet/*.sh`. |
| Trained ensemble (10 members + `all_preds.csv`) | `/pdo/astronet-data/models/vetting/production/` (what QLP runs). Same bytes in `../astronet-vetting-v2/` (with `MODEL_CARD.md`) and `../experimental/pablomer/march2026/20260305/pablomer_final-final-3k-ensemble/` (where it was trained). |
| Triage checkpoint used as warm start | `/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651/` |
| TFRecords (train / val / test) | `/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1/tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025-{train,val,test}/` |
| TCE tables (one row per labelled TCE) | `/pdo/users/pablomer/mnt/tess/astronet/tces-vetting-v01-tois-triageJs-nocentroid-dec2025-{all,train,val,test}.csv` |
| Light curves (FITS, one file per TIC, all sectors stitched) | `/pdo/astronet-data/data/fits/oct2025_dataset_all_fits_files/` (19,395 files, 19 GB). `/pdo/users/pablomer/mnt/tess/april2025_dataset_fits_files/` is the same set. |
| Label sources | `/pdo/users/pablomer/mnt/tess/labels/vetting-v01.csv` (group vetting labels + TOIs, Jan 2025), `vetting-new-events-withfilenames.csv` (TOIs added Feb 2025), `junk_data_2025dataset.csv` (6,007 B/N/J rows drawn from the triage set); triage table `/pdo/users/pablomer/mnt/tess/astronet/tces-v14-all.csv`. |
| Python environment | `/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python` (Python 3.9.16, TensorFlow 2.12.0, NumPy 1.22.3, Astropy 5.1). Used for both record generation and training. |

Dataset name used throughout: `vetting-v01-tois-triageJs-nocentroid-dec2025`
("vetting-v01 labels + TOIs + triage junk, no centroid features, December 2025 tables").

## 2. Step 1: TCE tables

Notebook: `preprocessing/Preprocess-vetting-cross-validation.ipynb` on branch
`pablomer-dev-training` (run from the repo root; paths are relative, `mnt/tess/...`).

What it does:
1. Reads `vetting-v01.csv` and appends `vetting-new-events-withfilenames.csv` (durations there
   are in hours, divided by 24).
2. Appends the 6,007 junk rows from `junk_data_2025dataset.csv` as class `jj` (`disp_j = 1`).
   Those rows are all 784 B (contact binaries) and 155 N from the triage v14 consensus labels
   plus 5,061 J sampled with `random_state=42`
   (`preprocessing/Preprocess-adding-junk-examples.ipynb`).
3. Drops rows without a P/E/J-type label and rows whose light-curve file is missing or too small.
4. Writes `tces-...-all.csv`, then a stratified split on `Final` with
   `train_test_split(test_size=0.1, random_state=42)` and again `test_size=1/9` for val.

Result: 15,471 TCEs (P 5,823 / E 3,390 / J 6,258); train 12,376 / val 1,547 / test 1,548.
Label columns the model reads: `disp_p, disp_e, disp_n, disp_j` (`disp_n` has no examples).
The split is by row, not by TIC.

## 3. Step 2: light curves

Each table row has a `File` column naming one FITS file in the FITS directory above: a
multi-sector QLP light curve for that TIC with SAP flux and the small/mid/large apertures
(used by the `local_aperture_s/m/l` views). The TOI and EB light curves came from the group's
shared Drive archives (2024); later additions (new TOIs Feb 2025, junk rows Apr 2025) were
extracted from the QLP HDF5 files with `preprocessing/h5_to_fits_frommichelle.py`
(`preprocessing/gen_fits.sh` shows the call; `-s 85` = stitch through sector 85), both on
`pablomer-dev-training`.

## 4. Step 3: TFRecords (`cad_scat_v5` preprocessing)

Generator: `astronet/preprocess/generate_input_records.py` at `v3.1.0`. On
`pablomer-dev-training` the same code is the file `generate_input_records_3.py` that the scripts
call; `v3.1.0` renamed it, removed a leftover debug filter, replaced `np.int`, added
`PREPROCESSING_VERSION = "cad_scat_v5"` and the triage dispatch. The vetting algorithm is the
same. This was verified on 2026-10-05: the `v3.1.0` code regenerates training shards 0 and 31
and the whole test set from the FITS files with every feature identical except the
`sample_segments_*` views, which draw an unseeded tie-break between equal-size folds (notes:
`Astronet-Triage/qlp_deployment/preprocessing_check/NOTES.md` on Pablo's disk).

Two scripts, in `scripts/vetting_v2_paper/`:

1. `gen_tfrecords_vetting_2025tois_3.sh`: base records, 50 shards each for train, val, test
   (`--mode=vetting --num_shards=50`). It wrote them to
   `/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_duration24/`; the val/test shards and
   the 50 un-augmented train shards in `dec2025_cad_scat_v5_aug/10x_0p1/` are copies of those
   (shard `00000` checked byte-identical). Written 2025-12-16.
2. `gen_tfrecords_vetting_2025tois_aug.sh`: augmented training copies. Each run adds
   `--remove_random_points` (drops 10 % of the detrended points at random before folding) and
   writes 50 shards with suffix `_aug_repK`. It was run in batches until `K = 1..10`; the committed
   script shows the last batch (`OFFSET=7`, `AUGMENT_TIMES=3`, reps 8-10). Written 2025-12-18/19.

Training set = 50 base + 10 x 50 augmented shards = 550 shards, 6.3 GB. Val and test are not
augmented. With `v3.1.0` the equivalent command is

```bash
python astronet/preprocess/generate_input_records.py --mode=vetting \
  --input_tce_csv_file=<tces-...-train.csv> --tess_data_dir=<FITS dir> \
  --output_dir=<out> --num_shards=50 [--remove_random_points --file_suffix=_repK]
```

## 5. Step 4: training

Script: `scripts/vetting_v2_paper/ensemble_train_vetting_2025_final.sh`. It runs
`astronet/train.py` ten times with

```
--model=AstroCNNModelVetting --config_name=pablomer_final --config_overrides=""
--pretrain_model_dir=<triage checkpoint above>
--train_files=<10x_0p1>/...-train/*  --eval_files=val:.../-val/*  --eval_files=test:.../-test/*
--dump_block_weights=false
```

and then `astronet/combine_model_results.py --base_path=<output dir>` to write `all_preds.csv`
(test-set predictions of all members). Each member's `train_flags.json` in `production/`
matches these flags (`shuffle_buffer_size` 25000, no `astro_ids_file`).

Config `pablomer_final` (`astronet/astro_cnn_model/configurations_vetting.py`, identical on
`v3.1.0` and `pablomer-dev-training`): 3,000 steps, batch 512, Adam, constant learning rate
1e-5, weight decay 0.021, 4 x Dense(512) head with dropout 0.016, 4 exclusive labels with
planet as primary class and non-primary classes down-weighted x2, random time reversal,
warm start from the triage checkpoint (not frozen). Views and scalars: see the model card.

Ten members trained 2026-03-05 15:41-17:23 on a GPU node, about 11 min each. Ensemble score =
mean `disp_p` over the members; at the paper threshold 0.427 the 1,548-TCE test set gives
precision 95.1 % / recall 97.1 % (QLP uses `astronet-filter --score-calc avg`).

## 6. Provenance checks (2026-10-15)

- `pablomer_final()` from `v3.1.0` rebuilds the saved `config.json` of all 10 members with
  zero differences (compared key by key).
- `train.py`, `training.py`, `evaluation.py`, `predict.py`, `astro_cnn_model/*` and `util/*`
  are identical in `c96a474` (last commit before training), `6bec68d` (`pablomer-dev-training`
  head) and `v3.1.0`. The only training-side files that differ are `configurations_vetting.py`
  (`pablomer_final` was committed after the run; the saved `config.json` pins it) and
  `models.py` (the dev branch registers ablation models).
- The training script itself was committed three weeks after the run (`edd8a1f`); its flags and
  output path match the members' `train_flags.json` and directory name.
- Preprocessing: golden-record check above, plus end-to-end rescoring of the regenerated test set
  with the `v3.1.0` `predict.batch_predict`: 95.3 % / 97.1 %.
- Not provable: whether the 2026-03-05 working tree carried uncommitted edits to the files listed
  above. Nothing found suggests it.

## 7. Reusing this for fine-tuning or continual retraining

- Generate new records with the `v3.1.0` generator, `--mode=vetting`. For unlabelled sectors add
  `--not-training`. `scripts/vetting_v2_paper/gen_tfrecords_vetting_2025_sectors.sh` is the loop
  used for the offline live-sector records (inputs
  `/pdo/astronet-data/data/properties/tces-sector<N>-with-filenames.csv` and
  `/pdo/astronet-data/data/fits/sector-<N>-p2026/`, outputs
  `/pdo/astronet-data/data/tfrecords/sector-<N>-scatter/`).
- Start from any member directory (`config.json`, `model.weights.h5`, `train_flags.json`).
- Known differences between these offline records and what QLP feeds the model in production:
  QLP rebins extended-mission cadences to 30 min before calling `create()` (the `*_std` channels
  come out 2-3x smaller than in training), sets `SRadEst = NaN` for every TCE, maps apertures as
  best / best-1 / best+1 instead of the fixed 2.5 / 3.0 / 3.5 px apertures of the training FITS,
  and masks the transits of lower planet numbers. Evaluate retrained models on records made the
  same way as the ones they will see.
- `sample_segments_*` is unseeded, so two generations of the same TCE can differ; the ensemble
  mean moves by up to ~0.07 (99 % of TCEs < 0.02).
- Other branches: `pablomer-dev-training` (all experiments: cross-validation, Optuna tuning,
  ablations, notebooks; same training code), `pablomer-binary-p-vs-notp` (binary head
  experiment). A no-warm-start control ensemble with the same test metrics is in
  `/pdo/astronet-data/models/vetting/experimental/pablomer/leakage_control/20260918/nopretrain-10/`.
