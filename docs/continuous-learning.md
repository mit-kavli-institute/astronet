# Continuous learning: teammate handoff

This workflow fine-tunes an existing binary vetting checkpoint on a fixed replay
buffer plus new labeled examples, then initializes the next cycle from all the
previous cycle's model weights. It is supervised continual learning, not an RL
algorithm. The portable recipe uses a constant learning rate. The optimizer is reset each
cycle, matching the original experiments;
this is not an exact optimizer-state resume.

The entry point is `python -m astronet.continual`. No personal checkout, Python
executable, experiment date, output location, or evaluation dashboard is embedded
in the runner. Run commands from this repository's root.

## Environment

Use Python 3.11 and `python -m pip install -r requirements-continual.txt`, or an
existing environment with these packages. The pins are the versions observed in
the working PDO experiment environment. This environment supports training from
existing TFRecords; light-curve regeneration and ExoDash have separate dependencies.

## Data and checkpoint requirements

Copy `experiments/continual.example.json` and edit its paths. Paths are relative to
the manifest, not the shell's current directory; environment variables are also
supported. The example uses `ASTRONET_EXPERIMENT_DATA` as a convenience. The model
and observational data are intentionally not distributed with the source.

Provide:

- A binary vetting checkpoint directory containing `config.json` and
  `model.weights.h5`, with `label_columns` equal to `["disp_p"]`.
- A replay TFRecord and one new-data TFRecord (or glob list) per cycle, produced
  with the declared preprocessing version. Existing experimental files can be used.
- An optional headerless, one-column CSV of integer Astro IDs to exclude. Set
  `exclude_astro_ids` to null when there are none. These exclusions apply to both
  replay and new data; check their ID namespace and counts carefully.
- A new output directory. Existing output directories are never overwritten.

The historical sequence names n1/n3/n5/n8/n10/n13/n15 indicate the cumulative number
of training sectors, not the cycle index. Each `new_files` list must contain the
delta only, not all earlier sectors again. The example assumes the existing prepared
artifacts; reconstructing adjudicated labels or the original replay selection is
outside this runner. For comparison to old results, use the same input file hashes.

For a triage-backbone start, set `initialization` to `triage`, provide the triage
checkpoint, and set `config_name` to `dimond_2label`. Only the first cycle transfers
the compatible convolution blocks. Later cycles restore the entire vetting model.
For a vetting start, the architecture and hyperparameters come from the checkpoint;
`config_name` is unused. There is no implicit four-class-to-binary conversion.
The historical mean-logit conversion changes scores and requires its own control;
it is deliberately not hidden inside this training command.

## Validate and run

```bash
export ASTRONET_EXPERIMENT_DATA=/path/to/prepared/artifacts
python -m astronet.continual experiments/my-experiment.json --check
python -m astronet.continual experiments/my-experiment.json
```

`--check` reads all inputs, verifies every TFRecord against the checkpoint schema,
checks for nonfinite features and empty pools after exclusion, and prints file
hashes and label counts. It does not train or create output files. Missing globs
fail individually, rather than silently omitting a data source. The preprocessing
version is an explicit provenance declaration: tensor shape validation cannot prove
which algorithm generated a record. If checkpoint metadata exists, it must agree.

Each run writes `run.json` with resolved inputs, SHA-256 hashes, exclusion-file and
initial-checkpoint hashes, seed, code revision/dirty status, TensorFlow version,
status, and completed checkpoints. Each cycle saves the model, config, preprocessing
declaration, training history, and checkpoint lineage. On failure, completed
checkpoints remain available and `run.json` records the error. To continue from a
completed cycle, explicitly make a new manifest with that checkpoint, the remaining
cycles, and a new output directory; adjust the seed to preserve the cycle sequence.
Seeds improve repeatability, but do not promise bitwise reproducibility on all hardware.

The replay/new examples retain their natural sampling frequency; class weighting
comes from the checkpoint's `non_primary_downweight_factor`. “Natural sampling”
does not mean equal positive/negative loss weights.

The standard `astronet.train` entry point also accepts `--astro_ids_file` and
`--weight_table_file` (CSV columns `astro_id,sample_weight`). Unspecified weights
default to one and multiply the existing class weights. The continual runner
preserves the original replay recipe and does not apply an additional weight table.

## Validation and evaluation

```bash
python -m unittest astronet.continual.test_workflow \
  astronet.astro_cnn_model.input_ds_test astronet.util.configdict_test
```

The tests use synthetic records and a tiny real AstroCNNModel: they check exclusion
of large IDs, finite termination when all examples are excluded, checkpoint chaining,
schema failures, output protection, and a two-cycle CPU training/save/load roundtrip.

Validation on 2026-10-13 in the pinned PDO environment: all 28 tests passed
across the workflow, input-dataset, and config-dictionary suites. A read-only
preflight validated the real 1,500-record replay buffer and 168-record first delta;
67 replay records were excluded. The existing `20261008/v2_member0_binary`
checkpoint loaded and produced finite predictions for four real examples using
59 input features. This checks execution and compatibility, not scientific quality.
Repository-wide Pylint still reports pre-existing findings from `main`; comparison
against that baseline found no new diagnostics. The new workflow and modified
training/data-loader modules pass their focused lint check.

Training is intentionally separate from scientific evaluation. The old shell
scripts imported ExoDash from a second personal checkout and some used stale default
TFRecords. Use a fixed benchmark, correct preprocessing for each model, and identical
candidate rows. Do not interpret a decreasing training loss as a production gain.
The existing Astro-ID exclusion list is not proof of TIC-disjoint evaluation; audit
TIC overlap across sectors, aliases, and the original supervised corpus separately.

## Scope of this cleanup

Branch base: GitHub `main` at `76309bf2f1bd2f06c9c0265ed96816f2321b64f7`.
The runner was extracted from David's `dimond-hard-mining` working tree at
`f69012f8fbb377be6c1c5b89d293c0b3ad474d10`, then ported with the required data-loader,
optimizer, exclusion, and weighting support. The original `dimond_2label` recipe
is stored in `astronet/continual/configs/dimond_2label.json`; the corresponding
configuration builder returns a fresh copy. This preserves the experiment without
importing unrelated configurations, notebooks, or preprocessing changes.

GitHub `main` is older than the deployed release. See
[repository status](repository-status.md): production Astronet matches `v3.1.0`
at `5531bb2`. Use that release to generate production-compatible `cad_scat_v5`
records. This branch trains from prepared records and does not replace QLP's
preprocessing or model installation.

The full repository still contains historical notebooks and legacy configurations.
The shareable path is now isolated and documented; repository-wide consolidation
should follow reviewed, small changes against the current release branch.
