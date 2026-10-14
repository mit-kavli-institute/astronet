"""Integration checks for continual-learning runs and data exclusions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from astronet import models, training
from astronet.astro_cnn_model import input_ds
from astronet.continual import __main__ as workflow
from astronet.util import config_util, configdict, training_data


def tiny_config():
  return configdict.ConfigDict({
      "train_steps": 1,
      "inputs": {
          "features": {
              "x": {
                  "shape": [1],
                  "is_time_series": False
              }
          },
          "label_columns": ["disp_p"],
          "label_scheme": "binary",
          "primary_class": 0,
          "exclusive_labels": False,
          "random_reverse_time_series": True,
          "non_primary_downweight_factor": 1.0,
      },
      "hparams": {
          "time_series_hidden": {},
          "aux_inputs": ["x"],
          "num_pre_logits_hidden_layers": 1,
          "pre_logits_hidden_layer_size": 4,
          "use_batch_norm": False,
          "pre_logits_dropout_rate": 0.0,
          "batch_size": 2,
          "learning_rate_schedule": "constant",
          "learning_rate_warmup_frac": 0.0,
          "learning_rate": 0.01,
          "optimizer": "adam",
          "one_minus_adam_beta_1": 0.1,
          "one_minus_adam_beta_2": 0.001,
          "adam_epsilon": 1e-7,
      },
  })


def write_records(path, rows):
  with tf.io.TFRecordWriter(str(path)) as writer:
    for identifier, value, label in rows:
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  "astro_id":
                      tf.train.Feature(
                          int64_list=tf.train.Int64List(value=[identifier])),
                  "disp_p":
                      tf.train.Feature(
                          int64_list=tf.train.Int64List(value=[label])),
                  "x":
                      tf.train.Feature(
                          float_list=tf.train.FloatList(value=[value])),
              }))
      writer.write(example.SerializeToString())


class WorkflowTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.checkpoint = self.root / "initial"
    self.checkpoint.mkdir()
    self.config = tiny_config()
    model = models.get_model_class("AstroCNNModelVetting")(self.config)
    config_util.save_config(self.config, str(self.checkpoint))
    models.save_model(model, str(self.checkpoint), "h5")
    (self.checkpoint / "preprocessing.json").write_text(
        json.dumps({"preprocessing_version": "synthetic-test-v1"}))
    write_records(self.root / "replay.tfrecord", [(90000000001, 0.9, 1),
                                                  (2, 0.1, 0)])
    write_records(self.root / "new.tfrecord", [(3, 0.8, 1), (4, 0.2, 0)])
    self.manifest = self.root / "experiment.json"
    self.spec = {
        "schema_version": 1,
        "preprocessing_version": "synthetic-test-v1",
        "initial_checkpoint": "initial",
        "output_dir": "run",
        "train_steps": 1,
        "shuffle_buffer_size": 4,
        "replay_files": ["replay.tfrecord"],
        "cycles": [{
            "name": "one",
            "new_files": ["new.tfrecord"]
        }]
    }

  def load(self):
    self.manifest.write_text(json.dumps(self.spec))
    return workflow.load_experiment(self.manifest)

  def test_relative_paths_and_no_validation_outputs(self):
    spec = self.load()
    counts = workflow.check_records(spec)
    self.assertEqual(sum(x["records"] for x in counts), 4)
    self.assertEqual(counts[0]["positive"], 1)
    self.assertEqual(spec["replay_files"], [str(self.root / "replay.tfrecord")])
    self.assertFalse((self.root / "run").exists())

  def test_each_glob_must_match(self):
    self.spec["replay_files"].append("missing-*.tfrecord")
    with self.assertRaisesRegex(ValueError, "Pattern must match"):
      self.load()

  def test_duplicate_names_and_output_traversal_rejected(self):
    original = copy.deepcopy(self.spec)
    self.spec["cycles"].append(copy.deepcopy(self.spec["cycles"][0]))
    with self.assertRaisesRegex(ValueError, "Duplicate cycle"):
      self.load()
    self.spec = original
    self.spec["cycles"][0]["name"] = "../escape"
    with self.assertRaisesRegex(ValueError, "Invalid cycle"):
      self.load()

  def test_preprocessing_mismatch(self):
    self.spec["preprocessing_version"] = "different"
    with self.assertRaisesRegex(ValueError, "preprocessing versions differ"):
      self.load()

  def test_unknown_options_rejected(self):
    self.spec["train_step"] = 4
    with self.assertRaisesRegex(ValueError, "Unknown experiment"):
      self.load()

  def test_nonfinite_features_rejected(self):
    write_records(self.root / "new.tfrecord", [(3, float("nan"), 1)])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      workflow.check_records(self.load())

  def test_empty_delta_after_exclusion_rejected(self):
    (self.root / "exclude.csv").write_text("3\n4\n")
    self.spec["exclude_astro_ids"] = "exclude.csv"
    with self.assertRaisesRegex(ValueError, "no usable new records"):
      workflow.check_records(self.load())

  def test_large_id_excluded_before_augmentation(self):
    dataset = input_ds.build_train_dataset(
        str(self.root / "replay.tfrecord"),
        self.config.inputs,
        batch_size=2,
        shuffle_values_buffer=2,
        exclude_astro_ids={90000000001})
    features, labels, _ = next(iter(dataset))
    self.assertEqual(set(features), {"x"})
    np.testing.assert_allclose(features["x"], [[0.1], [0.1]])
    np.testing.assert_array_equal(labels, [0, 0])

  def test_all_excluded_terminates(self):
    dataset = input_ds.build_train_dataset(
        str(self.root / "replay.tfrecord"),
        self.config.inputs,
        batch_size=2,
        shuffle_values_buffer=2,
        exclude_astro_ids={90000000001, 2})
    self.assertIsNone(next(iter(dataset), None))

  def test_weights_preserve_ids_and_apply_with_exclusions(self):
    weights_path = self.root / "weights.csv"
    weights_path.write_text("astro_id,sample_weight\n2,2.5\n90000000001,3\n")
    weights = training_data.make_weight_table(weights_path)
    dataset = input_ds.build_train_dataset(
        str(self.root / "replay.tfrecord"),
        self.config.inputs,
        batch_size=2,
        shuffle_values_buffer=2,
        exclude_astro_ids={90000000001},
        weight_table=weights)
    _, labels, loss_weights = next(iter(dataset))
    np.testing.assert_array_equal(labels, [0, 0])
    np.testing.assert_allclose(loss_weights, [2.5, 2.5])

  def test_weight_only_dataset_keeps_existing_batch_structure(self):
    weights_path = self.root / "weights.csv"
    weights_path.write_text("astro_id,sample_weight\n90000000001,3\n")
    dataset = input_ds.build_dataset(
        str(self.root / "replay.tfrecord"),
        self.config.inputs,
        batch_size=2,
        weight_table=training_data.make_weight_table(weights_path))
    features, labels, loss_weights = next(iter(dataset))
    self.assertEqual(set(features), {"x"})
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_allclose(loss_weights, [3, 1])

  def test_bad_weight_rows_rejected(self):
    weights_path = self.root / "weights.csv"
    for rows in ["1,nan\n", "1,-1\n", "1,1\n1,2\n"]:
      weights_path.write_text("astro_id,sample_weight\n" + rows)
      with self.assertRaises(ValueError):
        training_data.read_weights(weights_path)

  def test_main_configs_without_schedule_still_compile(self):
    config = tiny_config()
    del config.hparams["learning_rate_schedule"]
    del config.hparams["learning_rate_warmup_frac"]
    model = models.get_model_class("AstroCNNModelVetting")(config)
    training.compile_model(model, config)
    self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.01)

  def test_binary_recipe_loads_without_experimental_branch(self):
    config = models.get_model_config("AstroCNNModelVetting", "dimond_2label")
    self.assertEqual(list(config.inputs.label_columns), ["disp_p"])
    self.assertEqual(config.hparams.learning_rate, 1e-5)
    config.inputs.label_columns.append("changed")
    fresh = models.get_model_config("AstroCNNModelVetting", "dimond_2label")
    self.assertEqual(list(fresh.inputs.label_columns), ["disp_p"])

  def test_two_real_training_cycles_and_reload(self):
    self.spec["cycles"].append({"name": "two", "new_files": ["new.tfrecord"]})
    spec = self.load()
    initial = models.load_model("AstroCNNModelVetting", str(self.checkpoint))
    before = [w.copy() for w in initial.get_weights()]
    final = workflow.run(spec, workflow.check_records(spec))
    self.assertEqual(final, str(self.root / "run" / "two"))
    manifest = json.loads((self.root / "run" / "run.json").read_text())
    self.assertEqual(manifest["status"], "complete")
    self.assertEqual(len(manifest["completed_cycles"]), 2)
    flags = json.loads((Path(final) / "train_flags.json").read_text())
    self.assertEqual(flags["previous_checkpoint"],
                     str(self.root / "run" / "one"))
    restored = models.load_model("AstroCNNModelVetting", final)
    self.assertTrue(
        any(not np.array_equal(a, b)
            for a, b in zip(before, restored.get_weights())))
    prediction = restored({"x": tf.constant([[0.5]])}, training=False).numpy()
    self.assertTrue(np.isfinite(prediction).all())
    with self.assertRaises(FileExistsError):
      workflow.run(spec, [])


if __name__ == "__main__":
  unittest.main()
