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

from http import server
import os
import re
import threading

from absl.testing import absltest

from python.array_record_module import ArrayRecordReader
from python.array_record_module import ArrayRecordWriter


class _MockGcsServer:
  """Minimal HTTP server emulating the GCS JSON API media download endpoint."""

  def __init__(self, payload: bytes):
    self.payload = payload
    self.requests = []
    self.tcp_connections = 0
    self.aborted_streams = 0
    self._lock = threading.Lock()
    parent = self

    class Handler(server.BaseHTTPRequestHandler):
      protocol_version = "HTTP/1.1"

      def handle(self):
        with parent._lock:
          parent.tcp_connections += 1
        try:
          super().handle()
        except ConnectionResetError:
          with parent._lock:
            parent.aborted_streams += 1

      def do_GET(self):  # pylint: disable=invalid-name
        range_header = self.headers.get("Range")
        with parent._lock:
          parent.requests.append({
              "path": self.path,
              "range": range_header,
          })
        total = len(parent.payload)
        start = 0
        end = total - 1
        status = 200
        if range_header:
          m_last = re.fullmatch(r"bytes=-(\d+)", range_header)
          m_range = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
          if m_last:
            last_n = int(m_last.group(1))
            start = max(0, total - last_n)
            status = 206
          elif m_range:
            start = int(m_range.group(1))
            if m_range.group(2):
              end = min(total - 1, int(m_range.group(2)))
            status = 206
        data = parent.payload[start : end + 1]
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
          self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
          with parent._lock:
            parent.aborted_streams += 1

      def log_message(self, fmt, *args):
        del fmt, args

    self._httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.port = self._httpd.server_address[1]
    self._thread = threading.Thread(
        target=self._httpd.serve_forever, daemon=True
    )
    self._thread.start()

  def close(self):
    self._httpd.shutdown()
    self._httpd.server_close()
    self._thread.join(timeout=5.0)


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

  def test_gcs_read_single_open_rpc(self):
    # Write a >2 MiB uncompressed ArrayRecord file so that the file size exceeds
    # the 1 MiB tail prefetch window.
    writer = ArrayRecordWriter(self.test_file, "group_size:1,uncompressed")
    records = [f"record-{i:04d}-".encode() + (b"x" * 65536) for i in range(36)]
    for r in records:
      writer.write(r)
    writer.close()
    with open(self.test_file, "rb") as f:
      payload = f.read()
    self.assertGreater(len(payload), 2 * 1024 * 1024)

    mock_gcs = _MockGcsServer(payload)
    old_endpoint = os.environ.get("CLOUD_STORAGE_EMULATOR_ENDPOINT")
    os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = (
        f"http://127.0.0.1:{mock_gcs.port}"
    )
    try:
      # Opening and closing a gs:// shard should issue exactly 1 HTTP GET
      # (with Range: bytes=-1048576) instead of 3 HTTP GETs.
      num_records = len(records)
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "readahead_buffer_size:0,max_parallelism:0",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(reader.num_records(), num_records)
      reader.close()
      self.assertLen(mock_gcs.requests, 1)
      self.assertEqual(mock_gcs.requests[0]["range"], "bytes=-1048576")
      self.assertEqual(mock_gcs.aborted_streams, 0)

      # Opening 10 more shards sequentially should issue 10 GETs while reusing
      # the single existing TCP connection (via GetSharedGcsClient()).
      mock_gcs.requests.clear()
      for i in range(10):
        r = ArrayRecordReader(f"gs://test-bucket/shard-{i:05d}")
        self.assertEqual(r.num_records(), num_records)
        r.close()
      self.assertLen(mock_gcs.requests, 10)
      self.assertEqual(mock_gcs.tcp_connections, 1)
      self.assertEqual(mock_gcs.aborted_streams, 0)

      # Verify record payload reading works over gs://.
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "readahead_buffer_size:0,max_parallelism:0",
      )
      self.assertEqual(
          reader.read([0, 17, 35]), [records[0], records[17], records[35]]
      )
      reader.close()

      # Opening a gs:// shard with index_storage_option:offloaded should also
      # issue only 1 HTTP GET by serving OffloadedChunkOffset lookups from the
      # prefetched tail buffer.
      mock_gcs.requests.clear()
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "index_storage_option:offloaded,readahead_buffer_size:0,max_parallelism:0",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(reader.num_records(), num_records)
      reader.close()
      self.assertLen(mock_gcs.requests, 1)
      self.assertEqual(mock_gcs.requests[0]["range"], "bytes=-1048576")
    finally:
      if old_endpoint is None:
        os.environ.pop("CLOUD_STORAGE_EMULATOR_ENDPOINT", None)
      else:
        os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = old_endpoint
      mock_gcs.close()


if __name__ == "__main__":
  absltest.main()
