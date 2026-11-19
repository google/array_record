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

from concurrent import futures
from http import server
import os
import re
import socket
import struct
import threading
from unittest import mock

from absl.testing import absltest

from python.array_record_module import ArrayRecordReader
from python.array_record_module import ArrayRecordWriter


def _make_crc32c_table() -> list[int]:
  table = []
  for i in range(256):
    crc = i
    for _ in range(8):
      crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
    table.append(crc)
  return table


_CRC32C_TABLE = _make_crc32c_table()


def _crc32c(data: bytes) -> int:
  crc = 0xFFFFFFFF
  for byte in data:
    crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
  return crc ^ 0xFFFFFFFF


def _encode_varint(value: int) -> bytes:
  if value < 0:
    value += 1 << 64
  out = bytearray()
  while value > 0x7F:
    out.append((value & 0x7F) | 0x80)
    value >>= 7
  out.append(value)
  return bytes(out)


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
  value = 0
  shift = 0
  while True:
    byte = buf[pos]
    pos += 1
    value |= (byte & 0x7F) << shift
    if not byte & 0x80:
      return value, pos
    shift += 7


def _bytes_field(field_num: int, payload: bytes) -> bytes:
  return (
      _encode_varint((field_num << 3) | 2)
      + _encode_varint(len(payload))
      + payload
  )


def _varint_field(field_num: int, value: int) -> bytes:
  return _encode_varint(field_num << 3) + _encode_varint(value)


def _fixed32_field(field_num: int, value: int) -> bytes:
  return _encode_varint((field_num << 3) | 5) + struct.pack("<I", value)


def _parse_varint_fields(buf: bytes) -> dict[int, int]:
  """Returns the varint fields of a serialized proto, skipping other types."""
  fields = {}
  pos = 0
  while pos < len(buf):
    tag, pos = _decode_varint(buf, pos)
    field_num, wire_type = tag >> 3, tag & 7
    if wire_type == 0:
      fields[field_num], pos = _decode_varint(buf, pos)
    elif wire_type == 1:
      pos += 8
    elif wire_type == 2:
      length, pos = _decode_varint(buf, pos)
      pos += length
    elif wire_type == 5:
      pos += 4
    else:
      break
  return fields


def _h2_frame(frame_type: int, flags: int, stream_id: int, payload: bytes):
  length = len(payload)
  return (
      bytes([(length >> 16) & 0xFF, (length >> 8) & 0xFF, length & 0xFF])
      + struct.pack("!BBI", frame_type, flags, stream_id)
      + payload
  )


class _MockGcsGrpcServer:
  """Minimal HTTP/2 server that answers google.storage.v2 ReadObject RPCs.

  It serves byte ranges of one payload and counts TCP connections, so tests
  can check how many gRPC channels the readers open.
  """

  def __init__(self, payload: bytes):
    self.payload = payload
    self.read_requests = 0
    self.tcp_connections = 0
    self._lock = threading.Lock()
    self._stop = False
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._sock.bind(("127.0.0.1", 0))
    self._sock.listen(64)
    self.port = self._sock.getsockname()[1]
    self._thread = threading.Thread(target=self._accept_loop, daemon=True)
    self._thread.start()

  def _recv_exact(self, conn: socket.socket, size: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < size:
      try:
        chunk = conn.recv(size - len(buf))
      except OSError:
        return None
      if not chunk:
        return None
      buf.extend(chunk)
    return bytes(buf)

  def _read_object(self, request: bytes) -> bytes:
    """Returns a ReadObjectResponse for a ReadObjectRequest."""
    fields = _parse_varint_fields(request)
    read_offset = fields.get(4, 0)
    if read_offset >= 1 << 63:
      read_offset -= 1 << 64
    read_limit = fields.get(5, 0)
    with self._lock:
      self.read_requests += 1
    total = len(self.payload)
    if read_offset < 0:
      start = max(0, total + read_offset)
    else:
      start = min(total, read_offset)
    end = total if read_limit <= 0 else min(total, start + read_limit)
    data = self.payload[start:end]
    checksummed_data = _bytes_field(1, data) + _fixed32_field(2, _crc32c(data))
    content_range = (
        _varint_field(1, start)
        + _varint_field(2, end)
        + _varint_field(3, total)
    )
    metadata = _bytes_field(1, b"object") + _varint_field(6, total)
    return (
        _bytes_field(1, checksummed_data)
        + _bytes_field(3, content_range)
        + _bytes_field(4, metadata)
    )

  def _respond(self, conn: socket.socket, stream_id: int, request: bytes):
    response = self._read_object(request)
    grpc_frame = b"\x00" + struct.pack("!I", len(response)) + response
    out = bytearray(
        _h2_frame(
            0x01,
            0x04,
            stream_id,
            b"\x88\x00\x0ccontent-type\x10application/grpc",
        )
    )
    for offset in range(0, len(grpc_frame), 16384):
      out.extend(_h2_frame(0x00, 0x00, stream_id, grpc_frame[offset:][:16384]))
    out.extend(_h2_frame(0x01, 0x05, stream_id, b"\x00\x0bgrpc-status\x010"))
    conn.sendall(bytes(out))

  def _handle_conn(self, conn: socket.socket):
    with self._lock:
      self.tcp_connections += 1
    with conn:
      if self._recv_exact(conn, 24) != b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n":
        return
      streams: dict[int, bytearray] = {}
      try:
        conn.sendall(
            _h2_frame(0x04, 0x00, 0, struct.pack("!HI", 0x0004, 0x00FFFFFF))
            + _h2_frame(0x08, 0x00, 0, struct.pack("!I", 0x00FF0000))
        )
        while not self._stop:
          header = self._recv_exact(conn, 9)
          if not header:
            return
          length = (header[0] << 16) | (header[1] << 8) | header[2]
          frame_type, flags, stream_id = struct.unpack("!BBI", header[3:9])
          stream_id &= 0x7FFFFFFF
          payload = self._recv_exact(conn, length) if length else b""
          if payload is None:
            return
          if frame_type == 0x04 and not flags & 0x01:  # SETTINGS
            conn.sendall(_h2_frame(0x04, 0x01, 0, b""))
          elif frame_type == 0x06 and not flags & 0x01:  # PING
            conn.sendall(_h2_frame(0x06, 0x01, 0, payload))
          elif frame_type == 0x03:  # RST_STREAM
            streams.pop(stream_id, None)
          elif frame_type == 0x01 and stream_id:  # HEADERS
            streams.setdefault(stream_id, bytearray())
          elif frame_type == 0x00 and stream_id:  # DATA
            if flags & 0x08:
              payload = payload[1 : len(payload) - payload[0]]
            streams.setdefault(stream_id, bytearray()).extend(payload)
          if stream_id and flags & 0x01 and stream_id in streams:  # END_STREAM
            raw = bytes(streams.pop(stream_id))
            message_length = struct.unpack("!I", raw[1:5])[0]
            self._respond(conn, stream_id, raw[5 : 5 + message_length])
      except OSError:
        return

  def _accept_loop(self):
    while not self._stop:
      try:
        conn, _ = self._sock.accept()
      except OSError:
        return
      threading.Thread(
          target=self._handle_conn, args=(conn,), daemon=True
      ).start()

  def close(self):
    self._stop = True
    self._sock.close()
    self._thread.join(timeout=5.0)


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
      self.assertLen(
          mock_gcs.requests,
          1,
          msg=(
              f"ranges={[r['range'] for r in mock_gcs.requests]}, "
              f"tcp_connections={mock_gcs.tcp_connections}, "
              f"aborted_streams={mock_gcs.aborted_streams}"
          ),
      )
      self.assertEqual(mock_gcs.requests[0]["range"], "bytes=-1048576")
      self.assertEqual(mock_gcs.tcp_connections, 1)
      self.assertEqual(mock_gcs.aborted_streams, 0)

      # Custom footer_prefetch_bytes:512K should issue 1 HTTP GET with
      # Range: bytes=-524288.
      mock_gcs.requests.clear()
      conns_before = mock_gcs.tcp_connections
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "footer_prefetch_bytes:512K,readahead_buffer_size:0,max_parallelism:0",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(reader.num_records(), num_records)
      reader.close()
      self.assertLen(mock_gcs.requests, 1)
      self.assertEqual(mock_gcs.requests[0]["range"], "bytes=-524288")
      self.assertEqual(mock_gcs.tcp_connections - conns_before, 1)
      self.assertEqual(mock_gcs.aborted_streams, 0)

      # Opening with index_storage_option:offloaded without
      # footer_prefetch_bytes:0 raises ValueError.
      with self.assertRaises(ValueError):
        ArrayRecordReader(
            "gs://test-bucket/shard-00000-of-00001",
            "index_storage_option:offloaded,readahead_buffer_size:0,max_parallelism:0",
            file_reader_buffer_size=32768,
        )

      # Opening when the footer exceeds footer_prefetch_bytes (e.g. 64K, which
      # only covers the postscript) falls back cleanly to a second HTTP GET for
      # the footer.
      mock_gcs.requests.clear()
      conns_before = mock_gcs.tcp_connections
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "footer_prefetch_bytes:64K,readahead_buffer_size:0,max_parallelism:0",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(reader.num_records(), num_records)
      reader.close()
      self.assertLen(mock_gcs.requests, 2)
      self.assertEqual(mock_gcs.requests[0]["range"], "bytes=-65536")
      self.assertEqual(mock_gcs.tcp_connections - conns_before, 1)
      self.assertEqual(mock_gcs.aborted_streams, 0)

      # Verify record payload reading works over gs:// and reuses the same
      # TCP connection within the ArrayRecordReader instance.
      mock_gcs.requests.clear()
      conns_before = mock_gcs.tcp_connections
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "readahead_buffer_size:16M,max_parallelism:1",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(
          reader.read([0, 17, 35]), [records[0], records[17], records[35]]
      )
      reader.close()
      self.assertEqual(mock_gcs.tcp_connections - conns_before, 1)
    finally:
      if old_endpoint is None:
        os.environ.pop("CLOUD_STORAGE_EMULATOR_ENDPOINT", None)
      else:
        os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = old_endpoint
      mock_gcs.close()

  def test_gcs_concurrent_multithreaded_reads(self):
    writer = ArrayRecordWriter(self.test_file, "group_size:1,uncompressed")
    records = [f"record-{i:04d}-".encode() + (b"x" * 65536) for i in range(36)]
    for r in records:
      writer.write(r)
    writer.close()
    with open(self.test_file, "rb") as f:
      payload = f.read()

    mock_gcs = _MockGcsServer(payload)
    old_endpoint = os.environ.get("CLOUD_STORAGE_EMULATOR_ENDPOINT")
    os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = (
        f"http://127.0.0.1:{mock_gcs.port}"
    )
    try:
      def worker(idx: int):
        reader = ArrayRecordReader(
            f"gs://test-bucket/shard-{idx:05d}",
            "readahead_buffer_size:16M,max_parallelism:1",
        )
        self.assertLen(records, reader.num_records())
        fetched = reader.read([0, idx % len(records), len(records) - 1])
        reader.close()
        return fetched

      with futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(worker, range(32)))

      for idx, fetched in enumerate(results):
        self.assertEqual(
            fetched,
            [records[0], records[idx % len(records)], records[-1]],
        )
      self.assertLen(mock_gcs.requests, 64)
    finally:
      if old_endpoint is None:
        os.environ.pop("CLOUD_STORAGE_EMULATOR_ENDPOINT", None)
      else:
        os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = old_endpoint
      mock_gcs.close()

  def _write_records(self, count: int) -> tuple[list[bytes], bytes]:
    writer = ArrayRecordWriter(self.test_file, "group_size:1,uncompressed")
    records = [
        f"record-{i:04d}-".encode() + (b"y" * 4096) for i in range(count)
    ]
    for r in records:
      writer.write(r)
    writer.close()
    with open(self.test_file, "rb") as f:
      return records, f.read()

  def _read_concurrently(self, records: list[bytes]) -> None:
    def worker(idx: int):
      reader = ArrayRecordReader(
          f"gs://test-bucket/shard-{idx:05d}",
          "readahead_buffer_size:0,max_parallelism:0",
      )
      fetched = reader.read([0, idx % len(records), len(records) - 1])
      reader.close()
      return fetched

    with futures.ThreadPoolExecutor(max_workers=16) as executor:
      results = list(executor.map(worker, range(32)))
    for idx, fetched in enumerate(results):
      self.assertEqual(
          fetched, [records[0], records[idx % len(records)], records[-1]]
      )

  def test_gcs_grpc_transport_shares_one_client(self):
    # The shared gRPC client binds to its endpoint on first use, so this is the
    # only test in this process that may read over gRPC.
    records, payload = self._write_records(24)
    mock_grpc = _MockGcsGrpcServer(payload)
    env = {
        "ARRAY_RECORD_GCS_TRANSPORT": "grpc",
        "ARRAY_RECORD_GCS_GRPC_CHANNELS": "2",
        "CLOUD_STORAGE_EMULATOR_ENDPOINT": f"http://127.0.0.1:{mock_grpc.port}",
        "CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT": (
            f"127.0.0.1:{mock_grpc.port}"
        ),
    }
    try:
      with mock.patch.dict(os.environ, env):
        os.environ.pop("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG", None)
        self._read_concurrently(records)
    finally:
      mock_grpc.close()

    self.assertGreater(mock_grpc.read_requests, 0)
    # 32 readers share one client with 2 channels. A client per reader would
    # open at least one connection per reader. gRPC may reconnect a channel, so
    # allow a few more connections than channels.
    self.assertLess(mock_grpc.tcp_connections, 8)

  def test_gcs_grpc_config_none_keeps_http(self):
    records, payload = self._write_records(24)
    mock_gcs = _MockGcsServer(payload)
    env = {
        "ARRAY_RECORD_GCS_TRANSPORT": "grpc",
        "GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG": "none",
        "CLOUD_STORAGE_EMULATOR_ENDPOINT": f"http://127.0.0.1:{mock_gcs.port}",
    }
    try:
      with mock.patch.dict(os.environ, env):
        self._read_concurrently(records)
    finally:
      mock_gcs.close()

    # Every reader opens its own HTTP client, so the reads reach the HTTP
    # server.
    self.assertNotEmpty(mock_gcs.requests)


if __name__ == "__main__":
  absltest.main()
