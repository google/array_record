# Copyright 2022 Google LLC
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

"""Tests for ArrayRecord data sources."""

from concurrent import futures
import dataclasses
import pathlib
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from array_record.python import array_record_data_source


FLAGS = flags.FLAGS


@dataclasses.dataclass
class DummyFileInstruction:
  filename: str
  skip: int
  take: int
  examples_in_shard: int


class ArrayRecordDataSourcesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = pathlib.Path(FLAGS.test_srcdir)

  def test_array_record_data_source_len(self):
    ar = array_record_data_source.ArrayRecordDataSource([
        self.testdata_dir / "digits.array_record-00000-of-00002",
        self.testdata_dir / "digits.array_record-00001-of-00002",
    ])
    self.assertLen(ar, 10)

  def assert_underlying_readers_closed(self, array_record_data_source_instance):
    for reader in array_record_data_source_instance._readers:
      # some readers might be None if we didn't read records from them.
      if reader:
        self.assertFalse(reader.is_open())

  def test_array_record_data_source_single_path(self):
    # Use a single path instead of a list of paths/file_instructions.
    ar = array_record_data_source.ArrayRecordDataSource(
        self.testdata_dir / "digits.array_record-00000-of-00002"
    )
    indices_to_read = [0, 1, 2, 3, 4]
    expected_data = [b"0", b"1", b"2", b"3", b"4"]
    actual_data = ar[indices_to_read]
    ar.close()
    self.assertEqual(expected_data, actual_data)
    self.assert_underlying_readers_closed(ar)

  def test_array_record_data_source_reverse_order(self):
    ar = array_record_data_source.ArrayRecordDataSource([
        self.testdata_dir / "digits.array_record-00000-of-00002",
        self.testdata_dir / "digits.array_record-00001-of-00002",
    ])
    indices_to_read = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    expected_data = [b"9", b"8", b"7", b"6", b"5", b"4", b"3", b"2", b"1", b"0"]
    actual_data = ar[indices_to_read]
    ar.close()
    self.assertEqual(expected_data, actual_data)
    self.assert_underlying_readers_closed(ar)

  def test_array_record_data_source_random_order(self):
    ar = array_record_data_source.ArrayRecordDataSource([
        self.testdata_dir / "digits.array_record-00000-of-00002",
        self.testdata_dir / "digits.array_record-00001-of-00002",
    ])
    # some random permutation
    indices_to_read = [3, 0, 5, 9, 2, 1, 4, 7, 8, 6]
    expected_data = [b"3", b"0", b"5", b"9", b"2", b"1", b"4", b"7", b"8", b"6"]
    actual_data = ar[indices_to_read]
    ar.close()
    self.assertEqual(expected_data, actual_data)
    self.assert_underlying_readers_closed(ar)

  def test_array_record_data_source_file_instructions(self):
    file_instruction_one = DummyFileInstruction(
        filename=self.testdata_dir / "digits.array_record-00000-of-00002",
        skip=2,
        take=1,
        examples_in_shard=3,
    )

    file_instruction_two = DummyFileInstruction(
        filename=self.testdata_dir / "digits.array_record-00001-of-00002",
        skip=2,
        take=2,
        examples_in_shard=99,
    )

    ar = array_record_data_source.ArrayRecordDataSource(
        [file_instruction_one, file_instruction_two]
    )
    self.assertLen(ar, 3)
    indices_to_read = [0, 1, 2]
    expected_data = [b"2", b"7", b"8"]
    actual_data = ar[indices_to_read]
    ar.close()
    self.assertEqual(expected_data, actual_data)
    self.assert_underlying_readers_closed(ar)

  def test_array_record_source_reader_idx_and_position(self):
    file_instructions = [
        # 2 records
        DummyFileInstruction(
            filename="file_1", skip=0, take=2, examples_in_shard=2
        ),
        # 3 records
        DummyFileInstruction(
            filename="file_2", skip=2, take=3, examples_in_shard=99
        ),
        # 1 record
        DummyFileInstruction(
            filename="file_3", skip=10, take=1, examples_in_shard=99
        ),
    ]

    ar = array_record_data_source.ArrayRecordDataSource(file_instructions)
    expected_indices_and_positions = [
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 10),
    ]
    self.assertLen(ar, 6)
    for record_key in range(len(ar)):
      self.assertEqual(
          expected_indices_and_positions[record_key],
          ar._reader_idx_and_position(record_key),
      )

  def test_array_record_source_reader_idx_and_position_negative_idx(self):
    ar = array_record_data_source.ArrayRecordDataSource([
        self.testdata_dir / "digits.array_record-00000-of-00002",
        self.testdata_dir / "digits.array_record-00001-of-00002",
    ])
    with self.assertRaises(ValueError):
      ar._reader_idx_and_position(-1)

    with self.assertRaises(ValueError):
      ar._reader_idx_and_position(len(ar))

  def test_array_record_source_invalid_type(self):
    with self.assertRaises(ValueError):
      array_record_data_source.ArrayRecordDataSource(
          {
              "data": self.testdata_dir / "digits.array_record-00000-of-00001",
          }
      )

  def test_array_record_source_empty_sequence(self):
    with self.assertRaises(ValueError):
      array_record_data_source.ArrayRecordDataSource([])


class RunInParallelTest(parameterized.TestCase):

  def test_the_function_is_executed_with_kwargs(self):
    function = mock.Mock(return_value="return value")
    list_of_kwargs_to_function = [
        {"foo": 1},
        {"bar": 2},
    ]
    result = array_record_data_source._run_in_parallel(
        function=function,
        list_of_kwargs_to_function=list_of_kwargs_to_function,
        num_workers=1,
    )
    self.assertEqual(result, ["return value", "return value"])
    self.assertEqual(function.call_count, 2)
    function.assert_has_calls([mock.call(foo=1), mock.call(bar=2)])

  def test_exception_is_re_raised(self):
    function = mock.Mock()
    side_effect = ["return value", ValueError("Raised!")]
    function.side_effect = side_effect
    list_of_kwargs_to_function = [
        {"foo": 1},
        {"bar": 2},
    ]
    self.assertEqual(len(side_effect), len(list_of_kwargs_to_function))
    with self.assertRaisesRegex(ValueError, "Raised!"):
      array_record_data_source._run_in_parallel(
          function=function,
          list_of_kwargs_to_function=list_of_kwargs_to_function,
          num_workers=1,
      )

  @parameterized.parameters([(-2,), (-1,), (0,)])
  def test_num_workers_cannot_be_null_or_negative(self, num_workers):
    function = mock.Mock(return_value="return value")
    list_of_kwargs_to_function = [
        {"foo": 1},
        {"bar": 2},
    ]
    with self.assertRaisesRegex(
        ValueError, "num_workers must be >=1 for parallelism."
    ):
      array_record_data_source._run_in_parallel(
          function=function,
          list_of_kwargs_to_function=list_of_kwargs_to_function,
          num_workers=num_workers,
      )

  def test_num_workers_is_passed_to_thread_executor(self):
    function = mock.Mock(return_value="return value")
    list_of_kwargs_to_function = [
        {"foo": 1},
        {"bar": 2},
    ]
    num_workers = 42
    with mock.patch(
        "concurrent.futures.ThreadPoolExecutor",
        wraps=futures.ThreadPoolExecutor,
    ) as executor:
      array_record_data_source._run_in_parallel(
          function=function,
          list_of_kwargs_to_function=list_of_kwargs_to_function,
          num_workers=num_workers,
      )
      executor.assert_called_with(num_workers)


if __name__ == "__main__":
  absltest.main()
