# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end tests for PyGrain, ArrayRecord, and TensorFlow training pipelines.

Verifies that PyGrain's ArrayRecordDataSource, MapDataset, and DataLoader
(both in-process and multiprocess) work seamlessly alongside TensorFlow's
protobuf serialization/parsing (tf.train.Example / tf.io.parse_single_example)
and @tf.function training loops.
"""

import os
import subprocess
import sys
import textwrap
from typing import Any

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import grain
import numpy as np
import tensorflow as tf

from array_record.python import array_record_module

_FEATURE_SPEC = {
    "index": tf.io.FixedLenFeature([], tf.int64),
    "x": tf.io.FixedLenFeature([2], tf.float32),
    "y": tf.io.FixedLenFeature([1], tf.float32),
}


def _serialize_tf_example(index: int, x0: float, x1: float, y: float) -> bytes:
  """Serializes a synthetic training example as a tf.train.Example proto."""
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              "index": tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[index])
              ),
              "x": tf.train.Feature(
                  float_list=tf.train.FloatList(value=[x0, x1])
              ),
              "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
          }
      )
  )
  return example.SerializeToString()


def _parse_tf_example(raw_bytes: bytes) -> dict[str, np.ndarray]:
  """Parses a serialized tf.train.Example using both Python and TF C++ APIs."""
  py_example = tf.train.Example.FromString(raw_bytes)
  py_index = py_example.features.feature["index"].int64_list.value[0]
  parsed = tf.io.parse_single_example(raw_bytes, _FEATURE_SPEC)
  tf_index = int(parsed["index"].numpy())
  if py_index != tf_index:
    raise ValueError(f"Index mismatch: {py_index} != {tf_index}")
  return {
      "index": np.asarray(tf_index, dtype=np.int64),
      "x": parsed["x"].numpy().astype(np.float32),
      "y": parsed["y"].numpy().astype(np.float32),
  }


class _ParseTfExampleTransform(grain.transforms.Map):
  """PyGrain Map transform that parses serialized tf.train.Example records."""

  def map(self, element: bytes) -> dict[str, Any]:
    return _parse_tf_example(element)


class ArrayRecordGrainTensorFlowPipelineTest(absltest.TestCase):
  """Tests for PyGrain, ArrayRecord, and TensorFlow pipeline compatibility."""

  def setUp(self):
    super().setUp()
    if "grain_use_fast_array_record_reader" in flags.FLAGS:
      self.enter_context(
          flagsaver.flagsaver(grain_use_fast_array_record_reader=False)
      )

  def _write_tf_example_shards(
      self, temp_dir: str, num_shards: int = 2, records_per_shard: int = 16
  ) -> list[str]:
    """Writes synthetic tf.train.Example records across ArrayRecord shards."""
    rng = np.random.default_rng(123)
    shard_paths = []
    record_idx = 0
    for shard_idx in range(num_shards):
      shard_path = os.path.join(
          temp_dir,
          f"train.array_record-{shard_idx:05d}-of-{num_shards:05d}",
      )
      writer = array_record_module.ArrayRecordWriter(shard_path, "group_size:1")
      for _ in range(records_per_shard):
        x = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        y = float(2.0 * x[0] - 3.0 * x[1] + 0.5)
        writer.write(
            _serialize_tf_example(
                index=record_idx,
                x0=float(x[0]),
                x1=float(x[1]),
                y=y,
            )
        )
        record_idx += 1
      writer.close()
      shard_paths.append(shard_path)
    return shard_paths

  def _run_tf_training_on_batches(self, batches) -> tuple[float, float]:
    """Runs a compiled TensorFlow training loop and returns (first, last) loss."""
    tf.keras.utils.set_random_seed(42)
    layer = tf.keras.layers.Dense(
        units=1,
        kernel_initializer="zeros",
        bias_initializer="zeros",
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    @tf.function
    def train_step(batch_x: tf.Tensor, batch_y: tf.Tensor) -> tf.Tensor:
      with tf.GradientTape() as tape:
        preds = layer(batch_x, training=True)
        loss = tf.reduce_mean(tf.square(preds - batch_y))
      grads = tape.gradient(loss, layer.trainable_variables)
      optimizer.apply_gradients(zip(grads, layer.trainable_variables))
      return loss

    losses = []
    for batch in batches:
      batch_x = tf.convert_to_tensor(batch["x"], dtype=tf.float32)
      batch_y = tf.convert_to_tensor(batch["y"], dtype=tf.float32)
      loss = float(train_step(batch_x, batch_y).numpy())
      losses.append(loss)

    self.assertNotEmpty(losses)
    self.assertTrue(np.all(np.isfinite(losses)))
    return losses[0], losses[-1]

  def test_grain_map_dataset_and_tf_training_pipeline(self):
    """Verifies PyGrain MapDataset over ArrayRecord feeding TF training."""
    temp_dir = self.create_tempdir().full_path
    shard_paths = self._write_tf_example_shards(
        temp_dir, num_shards=2, records_per_shard=16
    )

    source = grain.sources.ArrayRecordDataSource(shard_paths)
    self.assertLen(source, 32)

    dataset = (
        grain.MapDataset.source(source)
        .seed(42)
        .shuffle()
        .map(_parse_tf_example)
        .repeat(15)
        .batch(batch_size=8, drop_remainder=True)
    )

    initial_loss, final_loss = self._run_tf_training_on_batches(dataset)
    self.assertLess(final_loss, initial_loss * 0.05)

  def test_grain_data_loader_and_tf_training_pipeline(self):
    """Verifies PyGrain DataLoader (in-process and multiprocess) with TF training."""
    temp_dir = self.create_tempdir().full_path
    shard_paths = self._write_tf_example_shards(
        temp_dir, num_shards=2, records_per_shard=16
    )
    source = grain.sources.ArrayRecordDataSource(shard_paths)

    for worker_count in (0, 2):
      with self.subTest(worker_count=worker_count):
        sampler = grain.samplers.IndexSampler(
            num_records=len(source),
            shard_options=grain.sharding.NoSharding(),
            shuffle=True,
            num_epochs=15,
            seed=42,
        )
        loader = grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                _ParseTfExampleTransform(),
                grain.transforms.Batch(batch_size=8, drop_remainder=True),
            ],
            worker_count=worker_count,
        )

        # Verify iterator checkpointing round-trip.
        iterator = iter(loader)
        first_batch = next(iterator)
        self.assertEqual(first_batch["x"].shape, (8, 2))
        saved_state = iterator.get_state()
        expected_second_batch = next(iterator)
        iterator.set_state(saved_state)
        restored_second_batch = next(iterator)
        np.testing.assert_array_equal(
            expected_second_batch["index"], restored_second_batch["index"]
        )
        del first_batch, expected_second_batch, restored_second_batch, iterator

        initial_loss, final_loss = self._run_tf_training_on_batches(loader)
        self.assertLess(final_loss, initial_loss * 0.05)

  def test_reverse_import_order_in_subprocess(self):
    """Verifies both import orders of Grain/ArrayRecord and TF in subprocesses."""
    temp_dir = self.create_tempdir().full_path
    record_path = os.path.join(temp_dir, "reverse_grain.array_record")

    module_name = array_record_module.__name__
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    for grain_first in (True, False):
      with self.subTest(grain_first=grain_first):
        if grain_first:
          imports = (
              "import importlib\n"
              "import grain\n"
              f"ar_mod = importlib.import_module({module_name!r})\n"
              "import tensorflow as tf"
          )
        else:
          imports = (
              "import importlib\n"
              "import tensorflow as tf\n"
              "import grain\n"
              f"ar_mod = importlib.import_module({module_name!r})"
          )
        script = imports + "\n" + textwrap.dedent(f"""
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={{
                            "val": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[77])
                            ),
                        }}
                    )
                )
                writer = ar_mod.ArrayRecordWriter({record_path!r}, "group_size:1")
                writer.write(example.SerializeToString())
                writer.close()

                source = grain.sources.ArrayRecordDataSource({record_path!r})
                assert len(source) == 1
                raw = source[0]
                parsed = tf.io.parse_single_example(
                    raw, {{"val": tf.io.FixedLenFeature([], tf.int64)}}
                )
                assert int(parsed["val"].numpy()) == 77
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                f"Subprocess failed (grain_first={grain_first}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            ),
        )


if __name__ == "__main__":
  absltest.main()
