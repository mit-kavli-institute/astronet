# Astronet repository status

Verified on 2026-10-13 using GitHub refs and read-only inspection of pdo6.
This page distinguishes installed release code, development branches and weights.

| Purpose | Reference | Evidence |
|---|---|---|
| Astronet installed for QLP | `v3.1.0` → `5531bb2df7be54ff40d214346f569237573e7b6f` | `/sw/astronet` resolves to `/pdo/app/astronet-versions/astronet-3.1.0`; all 39 tracked Python files under `astronet/` match the GitHub tag byte-for-byte |
| Release branch | `curated-merge-2026.06` → `5531bb2` | GitHub `ls-remote` agrees with the release tag |
| GitHub default development base | `main` → `76309bf2f1bd2f06c9c0265ed96816f2321b64f7` | Verified from GitHub; older than the deployed release |
| Portable continuous retraining | `continuous-retraining`, based directly on that `main` | Replay runner and necessary support ported independently; experimental training branch |
| Original retraining source | `dimond-hard-mining` → `f69012f`, plus uncommitted files | Original PDO checkout retained; relevant code extracted into the portable runner |
| Pablo's training development | `pablomer-dev-training` → `6bec68d` | GitHub ref; its name does not identify the production release |
| Separate binary-head experiment | `pablomer-binary-p-vs-notp` → `1d9b26e` | GitHub ref; not silently incorporated into this runner |
| Production weights | `/pdo/astronet-data/models/vetting/production/` | Ten `AstroCNNModelVetting_pablomer_final_20260305_*` members observed |

The installed tree has no Git metadata, so the release match was established by
comparing source hashes rather than assuming a version-directory name proves it.
This verifies the Astronet source under `astronet/`; it is not an audit of every
QLP job, command-line override, dependency, or historical sector run.

Some October 5 model cards and deployment notes still say the release is pending.
Those passages are superseded by the observed installation and published tag.
QLP wrapper code lives in FFITools; this Astronet branch does not change it.

## Which code and data trained the installed ensemble?

All ten installed vetting members are named
`AstroCNNModelVetting_pablomer_final_20260305_*`. Their individual
`train_flags.json` files agree on:

- Configuration: `pablomer_final`, four output classes.
- Training data: `/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1/`
  followed by `tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025-train/*`.
- Validation and test data: the corresponding splits under that same root.
- Triage initialization:
  `/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651`.

The model card at
`/pdo/astronet-data/models/vetting/astronet-vetting-v2/MODEL_CARD.md` records
March 5, 2026 training on `pablomer-dev-training`, with
`c96a474` as the last committed code before training. It explicitly says the
**working-tree state was not recorded**; whether that checkout was clean is unknown.
That is a provenance gap: the commit is a useful starting point, not an exact
snapshot of the training source. The October `v3.1.0` tag identifies the installed
inference implementation, not the March training checkout.

The release generator explicitly declares `cad_scat_v5`: cadence-aware binning
and scatter weighting computed on detrended light curves. This changes the values
inside model input views; compatible tensor shapes alone do not establish matching
preprocessing. Triage intentionally retains separate `legacy_v301` preprocessing.
These scatter weights are not the per-example loss weights exposed by the
retraining branch's `--weight_table_file` option.

Newer experiments exist. The August 5 binary/four-class comparison, August 6
no-pretraining comparison, and sampled September 18 leakage-control checkpoints
also point to the same `10x_0p1` training root. They are separate from the installed
March ensemble. This was a scoped inspection, not a claim that every later
experimental run has been inventoried.

A sibling `10x_0p1_with_new_features` data directory exists. It is **not** the
training root named in the production members' saved flags. Its existence is not
evidence that the new scalar features are deployed.

For future training, save the exact code commit, any working-tree patch, resolved
configuration, preprocessing identity, file hashes, split membership, and starting
checkpoint together. The continual runner records code status, configuration,
input hashes and checkpoint lineage; run it from a committed clean checkout to
make the source reproducible.

## Which reference to use

- To inspect or reproduce the installed Astronet implementation, start at `v3.1.0`.
- To share and review the replay-based retraining workflow, use
  `continuous-retraining` and its [run guide](continuous-learning.md).
- Do not generate new vetting records with the older preprocessing on this branch
  and label them `cad_scat_v5`. Use the release generator, then train from those
  prepared records with a compatible checkpoint.
- Before consolidating the repository, review the release branch against `main`.
  Merging it into `main` is a separate change from sharing the retraining code.

For new releases, record the Git tag/commit, installed import path, preprocessing
version, model/checkpoint identity, and ensemble aggregation together. Update this
page after deployment; a recent branch name alone is not a deployment record.
