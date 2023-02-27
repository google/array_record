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

"""Tests for array_record_module."""

import os

from absl.testing import absltest

from array_record.python.array_record_module import ArrayRecordReader
from array_record.python.array_record_module import ArrayRecordWriter


class ArrayRecordModuleTest(absltest.TestCase):

  def setUp(self):
    super(ArrayRecordModuleTest, self).setUp()
    self.test_file = os.path.join(self.create_tempdir().full_path,
                                  "test.arecord")

  def test_open_and_close(self):
    writer = ArrayRecordWriter(self.test_file)
    self.assertTrue(writer.ok())
    self.assertTrue(writer.is_open())
    writer.close()
    self.assertFalse(writer.is_open())

    reader = ArrayRecordReader(self.test_file)
    self.assertTrue(reader.ok())
    self.assertTrue(reader.is_open())
    reader.close()
    self.assertFalse(reader.is_open())

  def test_bad_options(self):

    def create_writer():
      ArrayRecordWriter(self.test_file, "blah")

    def create_reader():
      ArrayRecordReader(self.test_file, "blah")

    self.assertRaises(ValueError, create_writer)
    self.assertRaises(ValueError, create_reader)

  def test_write_read(self):
    writer = ArrayRecordWriter(self.test_file)
    test_strs = [b"abc", b"def", b"ghi"]
    for s in test_strs:
      writer.write(s)
    writer.close()
    reader = ArrayRecordReader(
        self.test_file, "readahead_buffer_size:0,max_parallelism:0"
    )
    num_strs = len(test_strs)
    self.assertEqual(reader.num_records(), num_strs)
    self.assertEqual(reader.record_index(), 0)
    for gt in test_strs:
      result = reader.read()
      self.assertEqual(result, gt)
    self.assertRaises(IndexError, reader.read)
    reader.seek(0)
    self.assertEqual(reader.record_index(), 0)
    self.assertEqual(reader.read(), test_strs[0])
    self.assertEqual(reader.record_index(), 1)

  def test_write_read_non_unicode(self):
    writer = ArrayRecordWriter(self.test_file)
    b = b"F\xc3\xb8\xc3\xb6\x97\xc3\xa5r"
    writer.write(b)
    writer.close()
    reader = ArrayRecordReader(self.test_file)
    self.assertEqual(reader.read(), b)

  def test_write_read_with_file_reader_buffer_size(self):
    writer = ArrayRecordWriter(self.test_file)
    b = b"F\xc3\xb8\xc3\xb6\x97\xc3\xa5r"
    writer.write(b)
    writer.close()
    reader = ArrayRecordReader(self.test_file, file_reader_buffer_size=2**10)
    self.assertEqual(reader.read(), b)

  def test_batch_read(self):
    writer = ArrayRecordWriter(self.test_file)
    test_strs = [b"abc", b"def", b"ghi", b"kkk", b"..."]
    for s in test_strs:
      writer.write(s)
    writer.close()
    reader = ArrayRecordReader(self.test_file)
    results = reader.read_all()
    self.assertEqual(test_strs, results)
    indices = [1, 3, 0]
    expected = [test_strs[i] for i in indices]
    batch_fetch = reader.read(indices)
    self.assertEqual(expected, batch_fetch)

  def test_read_range(self):
    writer = ArrayRecordWriter(self.test_file)
    test_strs = [b"abc", b"def", b"ghi", b"kkk", b"..."]
    for s in test_strs:
      writer.write(s)
    writer.close()
    reader = ArrayRecordReader(self.test_file)

    def invalid_range1():
      reader.read(0, 0)

    self.assertRaises(IndexError, invalid_range1)

    def invalid_range2():
      reader.read(0, 100)

    self.assertRaises(IndexError, invalid_range2)

    def invalid_range3():
      reader.read(3, 2)

    self.assertRaises(IndexError, invalid_range3)

    self.assertEqual(reader.read(0, -1), test_strs[0:-1])
    self.assertEqual(reader.read(-3, -1), test_strs[-3:-1])
    self.assertEqual(reader.read(1, 3), test_strs[1:3])

  def test_writer_options(self):
    writer = ArrayRecordWriter(self.test_file, "group_size:42")
    writer.write(b"test123")
    writer.close()
    reader = ArrayRecordReader(self.test_file)
    # Includes default options.
    self.assertEqual(
        reader.writer_options_string(),
        "group_size:42,transpose:false,pad_to_block_boundary:false,zstd:3,"
        "window_log:20,max_parallelism:1")

if __name__ == "__main__":
  absltest.main()
