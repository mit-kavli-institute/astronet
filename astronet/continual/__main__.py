"""Run an explicit, reproducible sequence of replay fine-tuning cycles.

Usage: python -m astronet.continual experiment.json [--check]
"""

# Keep CLI help available without importing the ML environment.
# pylint: disable=import-outside-toplevel,invalid-name

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from astronet.util.training_data import read_ids


def _path(value, base):
  expanded = os.path.expandvars(os.path.expanduser(value))
  if "$" in expanded:
    raise ValueError(f"Unresolved environment variable in {value!r}")
  path = Path(expanded)
  return str(path if path.is_absolute() else base / path)


def _files(patterns, base):
  if not isinstance(patterns, list) or not patterns:
    raise ValueError("File patterns must be a nonempty JSON list")
  result = []
  for pattern in patterns:
    matches = sorted(glob.glob(_path(pattern, base)))
    if not matches or any(not Path(p).is_file() for p in matches):
      raise ValueError(f"Pattern must match regular files: {pattern}")
    for path in matches:
      if "," in path:
        raise ValueError(
            "AstroNet's file-pattern API does not support commas in paths")
      if path not in result:
        result.append(path)
  return result


def load_experiment(path):
  """Resolve input paths relative to the manifest."""
  path = Path(path).resolve()
  spec = json.loads(path.read_text(encoding="utf-8"))
  allowed = {
      "schema_version", "preprocessing_version", "initial_checkpoint",
      "initialization", "config_name", "output_dir", "seed", "train_steps",
      "shuffle_buffer_size", "exclude_astro_ids", "replay_files", "cycles"
  }
  unknown = set(spec) - allowed
  if unknown:
    raise ValueError(f"Unknown experiment options: {sorted(unknown)}")
  if spec.get("schema_version") != 1:
    raise ValueError("Expected schema_version=1")
  if not spec.get("preprocessing_version"):
    raise ValueError(
        "Declare the preprocessing_version used to create the records")
  if spec.get("initialization", "vetting") not in ("vetting", "triage"):
    raise ValueError("initialization must be vetting or triage")
  spec.setdefault("initialization", "vetting")
  spec.setdefault("config_name", "dimond_2label")
  spec.setdefault("seed", 42)
  spec.setdefault("shuffle_buffer_size", 25000)
  for key in ("train_steps", "shuffle_buffer_size"):
    if (not isinstance(spec.get(key), int) or isinstance(spec[key], bool) or
        spec[key] <= 0):
      raise ValueError(f"{key} must be a positive integer")
  if (not isinstance(spec["seed"], int) or isinstance(spec["seed"], bool) or
      spec["seed"] < 0):
    raise ValueError("seed must be a nonnegative integer")
  for key in ("initial_checkpoint", "output_dir"):
    spec[key] = _path(spec[key], path.parent)
  for name in ("config.json", "model.weights.h5"):
    if not (Path(spec["initial_checkpoint"]) / name).is_file():
      raise ValueError(f"Initial checkpoint is missing {name}")
  metadata = Path(spec["initial_checkpoint"]) / "preprocessing.json"
  if metadata.exists() and spec["initialization"] == "vetting":
    version = json.loads(
        metadata.read_text(encoding="utf-8")).get("preprocessing_version")
    if version != spec["preprocessing_version"]:
      raise ValueError(
          "Checkpoint and declared record preprocessing versions differ")
  spec["exclude_astro_ids"] = (
      _path(spec["exclude_astro_ids"], path.parent)
      if spec.get("exclude_astro_ids") else None)
  read_ids(spec["exclude_astro_ids"])
  spec["replay_files"] = _files(spec["replay_files"], path.parent)
  if not isinstance(spec.get("cycles"), list) or not spec["cycles"]:
    raise ValueError("Provide at least one cycle")
  names = set()
  for cycle in spec["cycles"]:
    if set(cycle) != {"name", "new_files"}:
      raise ValueError("Each cycle requires exactly name and new_files")
    name = cycle["name"]
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
      raise ValueError(f"Invalid cycle name: {name!r}")
    if name in names:
      raise ValueError(f"Duplicate cycle name: {name}")
    names.add(name)
    cycle["new_files"] = _files(cycle["new_files"], path.parent)
    if set(cycle["new_files"]) & set(spec["replay_files"]):
      raise ValueError("A file cannot be both replay and new data")
  return spec


def fingerprint(path):
  digest = hashlib.sha256()
  with open(path, "rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return {
      "path": str(path),
      "bytes": Path(path).stat().st_size,
      "sha256": digest.hexdigest()
  }


def check_records(spec):
  """Validate every record against the model schema and count usable labels."""
  import tensorflow as tf

  from astronet import models
  from astronet.astro_cnn_model.input_ds import ExampleParser
  from astronet.util import config_util

  config = (
      models.get_model_config("AstroCNNModelVetting", spec["config_name"])
      if spec["initialization"] == "triage" else config_util.load_config(
          spec["initial_checkpoint"]))
  if list(config.inputs.label_columns) != ["disp_p"]:
    raise ValueError(
        "This workflow requires a binary disp_p checkpoint/config. "
        "A four-class model needs an explicit, separately evaluated conversion."
    )
  config_util.validate(config)
  if spec["initialization"] == "triage":
    config_util.validate_pretrain_config(
        config, config_util.load_config(spec["initial_checkpoint"]))
  parser = ExampleParser(
      config.inputs, include_labels=True, include_identifiers=True)
  excluded = read_ids(spec["exclude_astro_ids"])
  files = dict.fromkeys(spec["replay_files"] +
                        [f for c in spec["cycles"] for f in c["new_files"]])
  for path in files:
    counts = {"records": 0, "excluded": 0, "positive": 0, "negative": 0}
    ids = set()
    for raw in tf.data.TFRecordDataset(path):
      features, label, _, identifier = parser(raw)
      for name, value in features.items():
        tf.debugging.assert_all_finite(
            value, f"Nonfinite model feature {name} in {path}")
      counts["records"] += 1
      identifier = int(identifier.numpy())
      if identifier in excluded:
        counts["excluded"] += 1
        continue
      label = float(label.numpy())
      if label not in (0.0, 1.0):
        raise ValueError(f"Nonbinary disp_p in {path}")
      counts["positive" if label else "negative"] += 1
      ids.add(identifier)
    counts["unique_astro_ids"] = len(ids)
    files[path] = {**fingerprint(path), **counts}
  for cycle in spec["cycles"]:
    for role, paths in (("replay", spec["replay_files"]), ("new",
                                                           cycle["new_files"])):
      if not sum(files[p]["positive"] + files[p]["negative"] for p in paths):
        name = cycle["name"]
        raise ValueError(
            f"Cycle {name} has no usable {role} records after exclusion")
  return list(files.values())


def _write_json(path, value):
  Path(path).write_text(
      json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _revision():
  repo = Path(__file__).resolve().parents[2]
  try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=repo,
                                     stderr=subprocess.DEVNULL,
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                    cwd=repo,
                                    stderr=subprocess.DEVNULL,
                                    text=True).strip()
    return {"commit": commit, "dirty": bool(dirty)}
  except (OSError, subprocess.CalledProcessError):
    manifest = repo / "SOURCE_MANIFEST.json"
    return {
        "commit": None,
        "dirty": None,
        "source_manifest": fingerprint(manifest) if manifest.exists() else None
    }


def run(spec, inventory):
  """Continue all model weights; initialize a fresh optimizer for each cycle."""
  import tensorflow as tf

  from astronet import models, training
  from astronet.util import config_util

  output = Path(spec["output_dir"])
  output.mkdir(parents=True, exist_ok=False)
  manifest = {
      "experiment": spec,
      "code": _revision(),
      "tensorflow": tf.__version__,
      "optimizer_state": "reset_each_cycle",
      "records": inventory,
      "initial_checkpoint": [
          fingerprint(Path(spec["initial_checkpoint"]) / n)
          for n in ("config.json", "model.weights.h5")
      ],
      "exclusions": (fingerprint(spec["exclude_astro_ids"])
                     if spec["exclude_astro_ids"] else None),
      "status": "running",
      "completed_cycles": []
  }
  _write_json(output / "run.json", manifest)
  previous = spec["initial_checkpoint"]
  excluded = read_ids(spec["exclude_astro_ids"])
  try:
    for index, cycle in enumerate(spec["cycles"]):
      tf.keras.backend.clear_session()
      tf.keras.utils.set_random_seed(spec["seed"] + index)
      if index == 0 and spec["initialization"] == "triage":
        config = models.get_model_config("AstroCNNModelVetting",
                                         spec["config_name"])
        model = models.get_model_class("AstroCNNModelVetting")(config)
        pretrain = models.load_model("AstroCNNModel", previous)
        for name, block in model.ts_blocks.items():
          if name in pretrain.ts_blocks:
            block.set_weights(pretrain.ts_blocks[name].get_weights())
            if config.get("freeze_pretrained_params", False):
              block.trainable = False
      else:
        model = models.load_model("AstroCNNModelVetting", previous)
        config = model.config
      config["train_steps"] = spec["train_steps"]
      files = spec["replay_files"] + cycle["new_files"]
      history = training.train(
          model,
          config,
          train_files=",".join(files),
          shuffle_buffer_size=spec["shuffle_buffer_size"],
          exclude_astro_ids=excluded or None,
          weight_table=None)
      destination = output / cycle["name"]
      destination.mkdir()
      config_util.save_config(config, str(destination))
      models.save_model(model, str(destination), "h5")
      _write_json(destination / "preprocessing.json",
                  {"preprocessing_version": spec["preprocessing_version"]})
      _write_json(
          destination / "train_flags.json", {
              "model": "AstroCNNModelVetting",
              "train_files": files,
              "previous_checkpoint": previous,
              "seed": spec["seed"] + index
          })
      _write_json(destination / "history.json", {
          k: [float(x) for x in v] for k, v in history.history.items()
      })
      manifest["completed_cycles"].append({
          "name": cycle["name"],
          "checkpoint": str(destination),
          "weights": fingerprint(destination / "model.weights.h5")
      })
      _write_json(output / "run.json", manifest)
      previous = str(destination)
    manifest["status"] = "complete"
  except BaseException as error:
    manifest["status"] = "failed"
    manifest["error"] = f"{type(error).__name__}: {error}"
    raise
  finally:
    _write_json(output / "run.json", manifest)
  return previous


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("experiment", help="Path to a JSON experiment manifest")
  parser.add_argument(
      "--check",
      action="store_true",
      help="Validate paths and every record without training or writing outputs"
  )
  args = parser.parse_args(argv)
  spec = load_experiment(args.experiment)
  if not args.check and Path(spec["output_dir"]).exists():
    raise FileExistsError("Choose a new output_dir: " + spec["output_dir"])
  inventory = check_records(spec)
  if args.check:
    print(json.dumps({"experiment": spec, "records": inventory}, indent=2))
  else:
    print(f"SAVED_CHECKPOINT_DIR={run(spec, inventory)}")


if __name__ == "__main__":
  main()
