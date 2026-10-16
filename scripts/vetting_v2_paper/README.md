# Driver scripts behind AstroNet-Vetting-v2 (verbatim)

Byte-for-byte copies of the shell scripts that produced the paper / production ensemble, taken
from branch `pablomer-dev-training` (`astronet/*.sh`, commit `6bec68d`; the sectors script was
only on disk). They are kept unchanged on purpose: absolute paths point at Pablo's environment and
at the shared `/pdo/astronet-data` tree. The full recipe, with what each step produced, is in
`docs/astronet-vetting-v2-training-recipe.md`.

| Script | Step | Original commit |
|---|---|---|
| `gen_tfrecords_vetting_2025tois_3.sh` | base train/val/test TFRecords (50 shards each) | `pablomer-dev-training` |
| `gen_tfrecords_vetting_2025tois_aug.sh` | augmented train replicas (`_aug_repK`, 10 % random point removal) | `6db0833` (2025-12-19) |
| `ensemble_train_vetting_2025_final.sh` | 10 x `train.py --config_name=pablomer_final` + `combine_model_results.py` | `edd8a1f` (2026-03-26) |
| `gen_tfrecords_vetting_2025_sectors.sh` | offline live-sector records (`--not-training`) | not committed before this copy |

On this branch the generator they call, `astronet/preprocess/generate_input_records_3.py`, is
`astronet/preprocess/generate_input_records.py` (same `cad_scat_v5` algorithm; see the recipe
document, section 4). Replace the path when running from `v3.1.0` or later.
