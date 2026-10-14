"""Functions used for training an AstroNet model."""

import tensorflow as tf
from absl import logging

from astronet.astro_cnn_model import input_ds


class ThresholdPrecision(tf.keras.metrics.Metric):
  """Precision at an explicit prediction threshold."""

  def __init__(self, threshold=0.3, name="precision_thresh", **kwargs):
    super().__init__(name=name, **kwargs)
    self.threshold = threshold
    self.precision = tf.keras.metrics.Precision()

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred_thresh = tf.cast(y_pred > self.threshold, tf.float32)
    self.precision.update_state(y_true, y_pred_thresh, sample_weight)

  def result(self):
    return self.precision.result()

  def reset_state(self):
    self.precision.reset_state()


class ThresholdRecall(tf.keras.metrics.Metric):
  """Recall at an explicit prediction threshold."""

  def __init__(self, threshold=0.3, name="recall_thresh", **kwargs):
    super().__init__(name=name, **kwargs)
    self.threshold = threshold
    self.recall = tf.keras.metrics.Recall()

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred_thresh = tf.cast(y_pred > self.threshold, tf.float32)
    self.recall.update_state(y_true, y_pred_thresh, sample_weight)

  def result(self):
    return self.recall.result()

  def reset_state(self):
    self.recall.reset_state()


def compile_model(model, config):
  """Compiles a model for training."""
  # Set up the learning rate schedule.
  if config.hparams.get("learning_rate_schedule", "constant") == "constant":
    if config.hparams.get("learning_rate_warmup_frac", 0):
      raise ValueError(
          "Learning rate warmup is not supported with constant schedule")
    learning_rate = config.hparams.learning_rate
  else:
    raise ValueError("This training workflow supports a constant learning rate")

  # Set up the optimizer.
  opt_hparams = dict(
      learning_rate=learning_rate,
      weight_decay=config.hparams.get("weight_decay"))
  if config.hparams.optimizer == "sgd":
    opt_hparams.update(momentum=1.0 - config.hparams.one_minus_momentum,)
    optimizer = tf.keras.optimizers.SGD(**opt_hparams)
  elif config.hparams.optimizer == "adam":
    opt_hparams.update(
        beta_1=1.0 - config.hparams.one_minus_adam_beta_1,
        beta_2=1.0 - config.hparams.one_minus_adam_beta_2,
        epsilon=config.hparams.adam_epsilon,
    )
    optimizer = tf.keras.optimizers.Adam(**opt_hparams)
  else:
    raise ValueError(config.hparams.optimizer)
  logging.info(
      f"Using '{optimizer.name}' optimizer with parameters {opt_hparams}")

  n_labels = len(config.inputs.label_columns)
  if n_labels > 1 and config.inputs.get("exclusive_labels", False):
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=config.hparams.get("label_smoothing", 0.0))
  else:
    loss = tf.keras.losses.BinaryCrossentropy()
  logging.info(f"Using '{loss.name}' loss")

  model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=[
          tf.keras.metrics.Precision(name="precision"),
          tf.keras.metrics.Recall(name="recall"),
          tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
          ThresholdPrecision(threshold=0.3),
          ThresholdRecall(threshold=0.3)
      ])


def train(model,
          config,
          train_files,
          shuffle_buffer_size=2500,
          exclude_astro_ids=None,
          weight_table=None):
  """Train on explicit file patterns with optional exclusions and loss weights.

  Multiple patterns are sampled at their natural per-record frequency. Optional
  sample weights multiply the checkpoint's existing class weights. Exclusions
  apply before repeating the finite source, including injected examples.
  """
  ds = input_ds.build_train_dataset(
      file_pattern=train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=shuffle_buffer_size,
      exclude_astro_ids=exclude_astro_ids,
      weight_table=weight_table,
  )

  compile_model(model, config)
  history = model.fit(ds, steps_per_epoch=config["train_steps"])
  return history
