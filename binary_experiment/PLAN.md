# P vs not-P sub-experiment — implementation plan

**Branch:** `pablomer-binary-p-vs-notp` (off `pablomer-dev-training` @ `6bec68d`)
**Question:** does collapsing the 4-class vetting head `[disp_p, disp_e, disp_n, disp_j]`
into a single binary "planet vs not-planet" head change planet-class precision/recall —
on the held-out test split and on real sector-93 vetting labels?

---

## 1. Design: what differs between the arms, and what provably doesn't

| | **4class (baseline)** | **binary** |
|---|---|---|
| config | `pablomer_final()` | `pablomer_final_binary()` (new) |
| label_columns | `[disp_p, disp_e, disp_n, disp_j]` | `[disp_p]` |
| head | Dense(4) + softmax | Dense(1) + sigmoid |
| loss | CategoricalCrossentropy | BinaryCrossentropy |
| everything else | identical | identical |

`pablomer_final_binary()` **calls `pablomer_final()` and overrides only 3 label keys**
(`configurations_vetting.py`), so features, hparams, augmentation, and train_steps
cannot drift between arms.

Facts verified before implementation (all checked directly against the data/code):

- **Labels are strictly one-hot** in the dec2025 tfrecords (train/val/test all
  contain only `(1,0,0,0)`, `(0,1,0,0)`, `(0,0,0,1)`; `disp_n` never fires).
  So the binary positive set (`min(disp_p,1)` in `input_ds._extract_labels`)
  is *exactly* the `argmax==0` set of the 4-class arm. Same examples, same split,
  same positives. P fraction ≈ 0.38 in all three splits.
- **Sample weights are bit-identical across arms.** With one label column,
  `input_ds` sets `primary_class=0`, and the `non_primary_downweight_factor=2.0`
  down-weighting triggers on the same condition (`disp_p < 1`) in both arms.
- **Pretrained init is unaffected.** `train.py` copies only `ts_blocks` (conv
  towers) from the triage checkpoint — never the dense head — and
  `validate_pretrain_config` checks only CNN blocks. 4→1 output units is safe.
- **Both arms are retrained fresh in this run** (the baseline is *not* the March
  production ensemble), so code/branch state is identical across arms.

## 2. Code changes (this branch only)

1. `astronet/astro_cnn_model/configurations_vetting.py` — added
   `pablomer_final_binary()` (derives from `pablomer_final()`).
2. `astronet/evaluation.py` — `export_dash_file()` previously hardcoded
   `predictions[:, 1..3]` and would crash for a 1-column model; added an
   early-return branch for single-output predictions. **4-class path untouched.**
3. `binary_experiment/` (new directory) — drivers + evaluation, nothing outside
   it except the two edits above.

No other repo code is touched; `main`/`pablomer-dev-training` are unaffected.

## 3. Training protocol

- **1 seed per arm** (per Pablo). Caveat: any gap smaller than seed-to-seed noise
  won't be resolvable; if results are close we add seeds later (the eval script
  already averages over however many member dirs it finds per arm).
- Same driver settings as production `ensemble_train_vetting_2025_final.sh`:
  same `DATA_DIR` (`dec2025_cad_scat_v5_aug/10x_0p1`), same tfrecord prefix,
  same triage pretrained checkpoint, 3000 steps, batch 512.
- Output: `/pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp/<date>/{4class,binary}/`

## 4. Metrics protocol (the part worth reviewing)

**Do not compare the arms at a shared fixed threshold** (e.g. the production
0.215): softmax and sigmoid scores live on different scales, so a shared
threshold measures calibration, not discrimination. Instead:

1. **Headline: threshold-free** — Average Precision (PR-AUC) and ROC-AUC of the
   planet score (`pred[:,0]` in both arms).
2. **Operating points:** each arm's threshold = its **F1-optimum on the val
   split**, frozen *before* touching test or S93. P/R/F1 reported at that
   frozen threshold.
3. **Matched-operating-point tables:** precision at recall ≥ {0.6,0.7,0.8,0.9}
   and recall at precision ≥ {0.6,0.7,0.8,0.9}, read off the PR curves.
4. 4-class @ 0.215 is also reported, as a familiar reference row only.

## 5. Sector-93 external evaluation

- **Scores:** all 5,956 unique astro_ids in
  `/pdo/astronet-data/data/tfrecords/sector-93-scatter/` (the same records the
  v2 flux/radius ensemble scored), inference via the same
  `load_model_from_checkpoint` + `build_eval_dataset` path as
  `rp_filter/predict_live_sectors_flux_radius_ensemble_mean_sector93.py`.
- **Labels:** Aug-5 sheets in `sector93_results/data/full_results/`
  (planet=132, eb=188, junk=118, iv-fail=35; uid = TIC·100+planetno = astro_id).
  374 labelled TCEs; 370 have tfrecords; 283 are *unanimous* (in exactly one
  sheet), of which 95 planets.
- **Leakage check (done):** 0 of the 374 S93 uids appear in any of
  train (12,375) / val (1,547) / test (1,548). Genuinely held out.
- **Three label policies**, reported side by side:
  - `unanimous` (**primary**): only TCEs on exactly one sheet (N≈279 scored;
    positives = planet sheet, 34% prevalence — close to the training 38%).
  - `any_planet`: all labelled; positive = on the planet sheet at all (132 pos).
  - `strict_planet`: all labelled; disputed planets count as negatives.
- Same frozen val thresholds as §4; AP/ROC-AUC as headline.

**Known caveats** (they cap absolute numbers but don't bias the arm comparison,
since both arms see identical inputs and labels):
- The labelled set = astronet top-322 ∪ operator picks — a preselected,
  hard-example-enriched slice; numbers are *comparative*, not sector-wide.
- The scatter-weighting train/inference mismatch (see memory notes) applies
  equally to both arms.
- Pre-existing repo quirk, not fixed here: `evaluate_model` casts astro_ids to
  int32, which overflows for large TICs. Irrelevant to this experiment — the
  comparison joins on int64 ids produced by our own inference loop, and the
  in-distribution metrics never touch astro_ids.

## 6. Execution — what Pablo runs on the GPU machine

```bash
cd /pdo/users/pablomer/Astronet-Triage/binary_experiment

# 1) train both arms, 1 seed each (~2 x 11 min on the CPU box; less on GPU)
./run_training.sh

# 2) evaluate + compare (prints the run root at the end of step 1)
./run_evaluation.sh /pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp/<date>
```

Knobs (env vars): `PYTHON_BIN` (if the GPU machine uses a different env),
`OUT_ROOT`, `TRAIN_STEPS=20` for a smoke run, `CODE_DIR`.
`run_training.sh 4class|binary` trains a single arm.
Step 2 can run on the CPU box too; S93 member predictions are cached under
`comparison/s93_preds/` so re-runs are instant.

## 7. Deliverables

`<RUN_ROOT>/comparison/`:
- `results.md` — all tables (test split + 3 S93 policies), incl. **planet-class
  precision and recall for each arm**
- `results.json` — same, machine-readable
- `pr_test.png`, `pr_s93_unanimous.png` — PR curves, operating points marked
- `s93_preds/` — cached per-member and per-arm-mean S93 scores (CSV)

## 8. Extension: 5-seed no-pretrain run (added 2026-08-06)

Result of run 1 (pretrained, 1 seed): statistical tie on S93; on the test split
the binary arm (AP 0.9911) fell below the March production ensemble's 10-member
seed band (0.9931 ± 0.0003), suggesting a small real in-distribution cost.
But both arms inherit conv towers pretrained on **5-class triage labels**, so
run 1 cannot say whether multi-class supervision shapes the *representation* —
the class structure was already baked in at init.

`run_nopretrain_5seeds.sh` addresses both limitations at once:
- new configs `pablomer_final_nopretrain()` / `pablomer_final_binary_nopretrain()`
  (`init_from_pretrained_model=False`, everything else identical);
- **5 seeds per arm** → per-seed spread measured within the run itself;
- evaluation + all plots run automatically at the end, including
  `ap_seed_spread.png` (per-seed AP dot plot, test + S93 panels) and per-seed
  tables in `results.md`.

Output root: `.../binary_pvsnotp_nopretrain/<date>/`. Caveat to keep in mind
when reading results: 3000 steps @ lr 1e-5 was tuned for fine-tuning; from
scratch both arms may be undertrained. That affects both arms equally (the
comparison stays internally valid) but absolute numbers will sit below run 1.

## 9. Status

- [x] Config `pablomer_final_binary()` added
- [x] `export_dash_file` guarded for 1-column output
- [x] `run_training.sh`, `evaluate_comparison.py`, `run_evaluation.sh`
- [x] Smoke test (tiny `TRAIN_STEPS`, both arms, full eval path)
- [x] Run 1: pretrained, 1 seed/arm (2026-08-05, pdogpu3) — results copied to
      `binary_experiment/results_20260805/`; verdict: tie on S93, small
      in-distribution edge for 4-class (outside the production seed band)
- [x] No-pretrain configs + `run_nopretrain_5seeds.sh` + per-seed eval/plots
- [ ] Run 2: no-pretrain, 5 seeds/arm (Pablo, GPU machine)
- [ ] Results review
