# Branch and input-feature inventory — 2026-10-13

For the current production/main distinction, see [repository status](repository-status.md).
The inventory below records the original read-only inspection of
`/pdo/users/dimond/astronet_secondary/astronet` on pdo6.
Remote-tracking references below are the server checkout's cached refs, not a fresh
GitHub fetch. No existing branch was deleted, merged, reset, or checked out.

## Where the work lives

| Work | Location / observed revision | Treatment |
|---|---|---|
| David's continuous-learning experiments | `dimond-hard-mining`, `f69012f` (2026-05-28), plus five modified tracked files and untracked `hard_mining/` scripts | Source of this isolated cleanup |
| Release preprocessing | `origin/curated-merge-2026.06`, `5531bb2` (2026-10-05) | Production integration target; do not replace with this older experimental checkout |
| Pablo's newer training code | `origin/pablomer-dev-training`, `6bec68d` (2026-06-25) | Distinct from stale local `pablomer-dev-training` at `8f8c157` (2025-11-15) |
| Existing binary-head experiment | `origin/pablomer-binary-p-vs-notp`, `1d9b26e` (2026-10-05) | Inspect before designing another binary conversion |
| Older Teff/SNR definitions | `origin/sector-analysis` contains `183519f`; `dimond-hard-mining` adds definitions in `f69012f` | Definitions alone are not active model inputs |
| ExoDash evaluation | `/pdo/users/dimond/astronet`, with its own uncommitted changes | Previously imported by launchers from a second checkout; not required by the cleaned training runner |
| Local `main` and cached `origin/main` | `76309bf` (2025-09-02) | Not evidence of the current release; do not merge everything into it blindly |

The five modified files were `configurations_vetting.py`, `input_ds.py`,
`evaluation.py`, `train.py`, and `training.py`. Relevant support fixes are ported onto main, including integer-ID handling,
exclusion filtering and optional sample weights. The personal data source injected
by the old experimental trainer is not present in main or this branch. Unrelated
experimental ablation configurations are preserved in the original checkout rather
than being added to the handoff.

The historical replay launcher uses full-weight restoration, with a fresh optimizer
per cycle. Its v2 conversion approximates the multiclass output with a new binary
head; this is not score-preserving. The new runner requires an explicit binary
checkpoint rather than silently doing that conversion.

## Radius and feature availability

These are counts of nonblank entries in saved sector CSVs, not proof that the values
are scientifically valid or measurements of every live sector. Files are under
`/pdo/astronet-data/data/properties/tces-sectorN.csv` and `tces-sectorN-qlp.csv`.

| Sector | Main rows | `SRad` present | `SRadEst` present |
|---|---:|---:|---:|
| 86 | 3,324 | 2,865 | 0 |
| 88 | 5,733 | 4,848 | 0 |
| 89 | 3,865 | 3,272 | 0 |
| 93 | 5,962 | 4,437 | 0 |
| 94 | 2,267 | 2,107 | 0 |

`SRad` becomes `star_rad` and `star_rad_present`. `SRadEst` becomes the separate
`star_rad_est` and `star_rad_est_present` fields. Missing fields are serialized as
zero plus a zero presence flag. Both pairs appear in the production member's active
auxiliary inputs. The documented shortcut concerns the latter pair, not a universal
absence of stellar radius and not a planet-radius field.

In sector 93's matching 5,962-row QLP table, `planet_radius` is populated for 4,522,
`star_t_eff` for 4,666, and `snr`/`snr_bls` for all 5,962. The sector-86 QLP table
contains only 149 rows, so its metadata coverage cannot be extrapolated to all 3,324
main rows. Join coverage and signal identities need checking in any feature build.

The inspected production configuration consumes neither Teff, SNR, nor explicit
planet radius. `configurations.py:pablomer` defines Teff/SNR (and additional fields),
but its `aux_inputs` omits them and this checkout's standard record generator does
not serialize them. The actual augmentation work was found in
`origin/sector-analysis:append_data_to_tfrecords.ipynb`: it joins by `astro_id` and
adds `star_t_eff`, `snr`, `t12_t14`, and `numcont`, each with a presence flag. Its
saved code targets the validation split under `10x_0p1_with_new_features/val/01`;
this does not prove all splits were regenerated. It keeps the first duplicate ID,
and uses fixed mean defaults. Audit duplicate joins, split coverage and the origin
of normalization statistics before reusing it. Planet radius is not among the four
fields in that notebook. The presence of a column, feature specification, or saved
TFRecord does not demonstrate that the active network reads it.

The production configuration already consumes `secondary_view`, a doubled-period
global view, and local odd/even channels. That is not proof it receives the same
occultation data, significance, or normalization as the QLP report. Explicit
secondary depth/error/phase and odd-even significance deserve an input-by-input
comparison with QLP. The secondary phase is serialized by this checkout's generator
but absent from the inspected production auxiliary-input list.

No production scoring or feature-generation behavior was changed in this cleanup.
The radius ablation should be a separate, controlled training change. New Teff,
SNR, and planet-radius features require consistent units, provenance, missingness
handling, joins, regenerated records, and corresponding active model inputs.

## Follow-on cleanup

1. Review and integrate the portable continual-learning module and prerequisite
   fixes into the agreed release/training branch with checkpoint compatibility tests.
2. Develop radius ablation separately from new-feature changes, preserving an
   interpretable control. Do not replace missing values only at inference.
3. Put Teff/SNR/planet-radius generation and QLP diagnostic extraction behind one
   documented feature schema shared by training and inference.
4. Archive redundant experiment launchers only after their replacement is accepted.
   Delete branches only after owners confirm their unique work has been retained.
