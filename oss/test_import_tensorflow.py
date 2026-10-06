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

"""End-to-end tests for ArrayRecord and TensorFlow compatibility.

Verifies that ArrayRecord's C++ extension (which statically links Protobuf and
Abseil) and TensorFlow can coexist in the same process across both import
orders, serialize/deserialize tf.train.Example protos, and execute a TensorFlow
training pipeline without Protobuf descriptor or symbol collisions.
"""

import os
import subprocess
import sys
import textwrap

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from array_record.python import array_record_module


def _serialize_tf_example(
    index: int, features: list[float], label: float, text: bytes
) -> bytes:
  """Creates and serializes a tf.train.Example protobuf message."""
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              "index": tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[index])
              ),
              "x": tf.train.Feature(
                  float_list=tf.train.FloatList(value=features)
              ),
              "y": tf.train.Feature(
                  float_list=tf.train.FloatList(value=[label])
              ),
              "text": tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[text])
              ),
          }
      )
  )
  return example.SerializeToString()


_FEATURE_SPEC = {
    "index": tf.io.FixedLenFeature([], tf.int64),
    "x": tf.io.FixedLenFeature([2], tf.float32),
    "y": tf.io.FixedLenFeature([1], tf.float32),
    "text": tf.io.FixedLenFeature([], tf.string),
}


class ArrayRecordTensorFlowCompatibilityTest(absltest.TestCase):
  """Tests for ArrayRecord and TensorFlow runtime compatibility."""

  def test_write_and_read_tf_examples(self):
    """Verifies writing and reading serialized tf.train.Example protos."""
    temp_dir = self.create_tempdir().full_path
    record_path = os.path.join(temp_dir, "examples.array_record")
    num_records = 16

    expected_serialized = []
    writer = array_record_module.ArrayRecordWriter(record_path, "group_size:4")
    self.assertTrue(writer.ok())
    self.assertTrue(writer.is_open())
    for i in range(num_records):
      x0 = float(i) * 0.25
      x1 = float(num_records - i) * 0.1
      y = 2.0 * x0 - 3.0 * x1 + 0.5
      raw = _serialize_tf_example(
          index=i,
          features=[x0, x1],
          label=y,
          text=f"record_{i}".encode("utf-8"),
      )
      expected_serialized.append(raw)
      writer.write(raw)
    writer.close()
    self.assertFalse(writer.is_open())

    reader = array_record_module.ArrayRecordReader(record_path)
    self.assertTrue(reader.ok())
    self.assertTrue(reader.is_open())
    self.assertEqual(reader.num_records(), num_records)
    self.assertIn("group_size:4", reader.writer_options_string())
    self.assertEqual(reader.record_index(), 0)

    # Verify batch random access reading and range reading.
    subset_indices = [0, 5, 11, 15]
    subset_records = reader.read(subset_indices)
    self.assertEqual(
        subset_records, [expected_serialized[i] for i in subset_indices]
    )
    self.assertEqual(reader.read(2, 6), expected_serialized[2:6])

    # Verify full read and both Python and C++ TensorFlow Protobuf parsing.
    all_records = reader.read_all()
    self.assertLen(all_records, num_records)
    for i, raw in enumerate(all_records):
      # Python protobuf deserialization via tf.train.Example.FromString.
      py_example = tf.train.Example.FromString(raw)
      self.assertEqual(
          py_example.features.feature["index"].int64_list.value, [i]
      )
      self.assertEqual(
          py_example.features.feature["text"].bytes_list.value,
          [f"record_{i}".encode("utf-8")],
      )

      # TensorFlow C++ protobuf deserialization via tf.io.parse_single_example.
      parsed = tf.io.parse_single_example(raw, _FEATURE_SPEC)
      self.assertEqual(int(parsed["index"].numpy()), i)
      self.assertEqual(parsed["text"].numpy(), f"record_{i}".encode("utf-8"))
      expected_x0 = float(i) * 0.25
      expected_x1 = float(num_records - i) * 0.1
      expected_y = 2.0 * expected_x0 - 3.0 * expected_x1 + 0.5
      np.testing.assert_allclose(
          parsed["x"].numpy(), [expected_x0, expected_x1], rtol=1e-5
      )
      np.testing.assert_allclose(parsed["y"].numpy(), [expected_y], rtol=1e-5)

    reader.close()
    self.assertFalse(reader.is_open())

  def test_tf_data_and_training_pipeline(self):
    """Verifies a tf.data and @tf.function training pipeline over ArrayRecord."""
    temp_dir = self.create_tempdir().full_path
    record_path = os.path.join(temp_dir, "train.array_record")
    num_records = 32

    rng = np.random.default_rng(42)
    writer = array_record_module.ArrayRecordWriter(record_path, "group_size:1")
    for i in range(num_records):
      x = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
      y = float(2.0 * x[0] - 3.0 * x[1] + 0.5)
      writer.write(
          _serialize_tf_example(
              index=i,
              features=[float(x[0]), float(x[1])],
              label=y,
              text=f"train_{i}".encode("utf-8"),
          )
      )
    writer.close()

    def record_generator():
      reader = array_record_module.ArrayRecordReader(record_path)
      try:
        for raw in reader.read_all():
          yield raw
      finally:
        reader.close()

    def parse_record(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
      parsed = tf.io.parse_single_example(serialized, _FEATURE_SPEC)
      return parsed["x"], parsed["y"]

    dataset = (
        tf.data.Dataset.from_generator(
            record_generator,
            output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
        )
        .repeat(15)
        .map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(8, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

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

    losses = [
        float(train_step(batch_x, batch_y).numpy())
        for batch_x, batch_y in dataset
    ]
    self.assertNotEmpty(losses)
    self.assertTrue(np.all(np.isfinite(losses)))
    self.assertLess(losses[-1], losses[0] * 0.05)

  def test_reverse_import_order_in_subprocess(self):
    """Verifies both import orders of ArrayRecord and TensorFlow in subprocesses."""
    temp_dir = self.create_tempdir().full_path
    record_path = os.path.join(temp_dir, "reverse_order.array_record")

    module_name = array_record_module.__name__
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    for ar_first in (True, False):
      with self.subTest(ar_first=ar_first):
        if ar_first:
          imports = (
              "import importlib\n"
              f"ar_mod = importlib.import_module({module_name!r})\n"
              "import tensorflow as tf"
          )
        else:
          imports = (
              "import importlib\n"
              "import tensorflow as tf\n"
              f"ar_mod = importlib.import_module({module_name!r})"
          )
        script = imports + "\n" + textwrap.dedent(f"""
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={{
                            "val": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[123])
                            ),
                        }}
                    )
                )
                serialized = example.SerializeToString()

                writer = ar_mod.ArrayRecordWriter({record_path!r}, "group_size:1")
                writer.write(serialized)
                writer.close()

                reader = ar_mod.ArrayRecordReader({record_path!r})
                assert reader.num_records() == 1
                raw = reader.read([0])[0]
                reader.close()

                parsed = tf.io.parse_single_example(
                    raw, {{"val": tf.io.FixedLenFeature([], tf.int64)}}
                )
                assert int(parsed["val"].numpy()) == 123
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
                f"Subprocess failed (ar_first={ar_first}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            ),
        )


if __name__ == "__main__":
  absltest.main()
