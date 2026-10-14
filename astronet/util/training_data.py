"""Readers for explicit training exclusions and per-example loss weights."""

import csv
import math


def read_ids(path):
  """Read a one-column integer Astro-ID CSV without a header or rounding."""
  if path is None:
    return set()
  result = set()
  with open(path, newline="", encoding="utf-8") as stream:
    for row in csv.reader(stream):
      if not row or not any(cell.strip() for cell in row):
        continue
      if len(row) != 1:
        raise ValueError(f"Expected one integer Astro ID per row in {path}")
      result.add(int(row[0]))
  return result


def read_weights(path):
  """Read astro_id,sample_weight columns, validating IDs and weights."""
  weights = {}
  with open(path, newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    if not {"astro_id", "sample_weight"}.issubset(reader.fieldnames or []):
      raise ValueError("Weight CSV requires astro_id and sample_weight columns")
    for row in reader:
      identifier = int(row["astro_id"])
      weight = float(row["sample_weight"])
      if identifier in weights:
        raise ValueError(f"Duplicate weight for Astro ID {identifier}")
      if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"Invalid sample weight for Astro ID {identifier}")
      weights[identifier] = weight
  if not weights:
    raise ValueError("Weight CSV contains no examples")
  return weights


def make_weight_table(path):
  """Build a TensorFlow lookup; IDs not in the CSV retain their base weight."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  weights = read_weights(path)
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          list(weights),
          list(weights.values()),
          key_dtype=tf.int64,
          value_dtype=tf.float32),
      default_value=1.0)
