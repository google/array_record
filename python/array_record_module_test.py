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
import signal
import socket
import struct
import tempfile
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

  It serves byte ranges of one payload, records each request's
  (read_offset, read_limit) and counts TCP connections, so tests can check the
  requested ranges and how many connections the readers open.
  """

  def __init__(self, payload: bytes):
    self.payload = payload
    self.reads: list[tuple[int, int]] = []
    self.tcp_connections = 0
    self._lock = threading.Lock()
    self._stop = False
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._sock.bind(("127.0.0.1", 0))
    # The client connects all of its max(4, nproc) channels at once. A short
    # backlog drops some of them, and gRPC retries those after a backoff.
    self._sock.listen(socket.SOMAXCONN)
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
      self.reads.append((read_offset, read_limit))
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


# Environment variables that select or configure the GCS client.
_GCS_ENV_VARS = (
    "ARRAY_RECORD_GCS_TRANSPORT",
    "CLOUD_STORAGE_EMULATOR_ENDPOINT",
    "CLOUD_STORAGE_TESTBENCH_ENDPOINT",
    "CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT",
    "GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG",
)


def _write_large_file(path: str) -> list[bytes]:
  """Writes a > 2 MiB file of 36 records, larger than the footer prefetch."""
  writer = ArrayRecordWriter(path, "group_size:1,uncompressed")
  records = [f"record-{i:04d}-".encode() + (b"x" * 65536) for i in range(36)]
  for r in records:
    writer.write(r)
  writer.close()
  return records


class ArrayRecordModuleTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # The process-wide gRPC client binds to the endpoint of the first gs://
    # reader that uses gRPC. So all gRPC tests share this server.
    tmp_dir = tempfile.TemporaryDirectory()
    cls.addClassCleanup(tmp_dir.cleanup)
    path = os.path.join(tmp_dir.name, "grpc.arecord")
    cls.grpc_records = _write_large_file(path)
    with open(path, "rb") as f:
      cls.grpc_payload = f.read()
    cls.grpc_server = _MockGcsGrpcServer(cls.grpc_payload)
    cls.addClassCleanup(cls.grpc_server.close)

  def setUp(self):
    super(ArrayRecordModuleTest, self).setUp()
    self.test_file = os.path.join(self.create_tempdir().full_path,
                                  "test.arecord")
    # Start every test without GCS settings from the test environment (such as
    # --test_env), so that no test reaches production GCS.
    env_patcher = mock.patch.dict(os.environ)
    env_patcher.start()
    self.addCleanup(env_patcher.stop)
    for name in _GCS_ENV_VARS:
      os.environ.pop(name, None)

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

      # Explicitly setting footer_prefetch_bytes > 0 with
      # index_storage_option:offloaded raises ValueError.
      with self.assertRaises(ValueError):
        ArrayRecordReader(
            "gs://test-bucket/shard-00000-of-00001",
            "index_storage_option:offloaded,footer_prefetch_bytes:512K,"
            "readahead_buffer_size:0,max_parallelism:0",
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
      # TCP connection within the ArrayRecordReader instance. The records are
      # less than 1 MiB apart, so they are read with one range read.
      mock_gcs.requests.clear()
      conns_before = mock_gcs.tcp_connections
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "readahead_buffer_size:16M,max_parallelism:1",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(
          reader.read([0, 7, 15]), [records[0], records[7], records[15]]
      )
      reader.close()
      self.assertEqual(mock_gcs.tcp_connections - conns_before, 1)

      # Opening with index_storage_option:offloaded defaults
      # footer_prefetch_bytes to 0 and skips tail prefetching.
      mock_gcs.requests.clear()
      reader = ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "index_storage_option:offloaded,"
          "readahead_buffer_size:0,max_parallelism:0",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(reader.num_records(), num_records)
      self.assertEqual(reader.read([17]), [records[17]])
      reader.close()
      self.assertTrue(
          all(
              not (r["range"] or "").startswith("bytes=-")
              for r in mock_gcs.requests
          )
      )
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
      # Records 0-15 span less than 1 MiB, so each batch is one range read.
      def worker(idx: int):
        reader = ArrayRecordReader(
            f"gs://test-bucket/shard-{idx:05d}",
            "readahead_buffer_size:16M,max_parallelism:1",
        )
        self.assertLen(records, reader.num_records())
        fetched = reader.read([0, idx % 16, 15])
        reader.close()
        return fetched

      with futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(worker, range(32)))

      for idx, fetched in enumerate(results):
        self.assertEqual(
            fetched,
            [records[0], records[idx % 16], records[15]],
        )
      # One footer read and one range read per reader.
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

  def _read_concurrently(
      self, records: list[bytes], num_readers: int = 32
  ) -> None:
    def worker(idx: int):
      reader = ArrayRecordReader(
          f"gs://test-bucket/shard-{idx:05d}",
          "readahead_buffer_size:0,max_parallelism:0",
      )
      fetched = reader.read([0, idx % len(records), len(records) - 1])
      reader.close()
      return fetched

    with futures.ThreadPoolExecutor(max_workers=16) as executor:
      results = list(executor.map(worker, range(num_readers)))
    for idx, fetched in enumerate(results):
      self.assertEqual(
          fetched, [records[0], records[idx % len(records)], records[-1]]
      )

  def _use_grpc(self, transport: str = "grpc") -> None:
    """Selects the gRPC transport, served by the class's mock server."""
    port = self.grpc_server.port
    os.environ.update({
        "ARRAY_RECORD_GCS_TRANSPORT": transport,
        "CLOUD_STORAGE_EMULATOR_ENDPOINT": f"http://127.0.0.1:{port}",
        "CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT": (
            f"127.0.0.1:{port}"
        ),
    })

  def _read_reads(self) -> list[tuple[int, int]]:
    """Returns the gRPC reads served so far."""
    with self.grpc_server._lock:  # pylint: disable=protected-access
      return list(self.grpc_server.reads)

  def test_gcs_grpc_transport_shares_one_client(self):
    self._use_grpc()
    num_readers = 32
    reads_before = len(self._read_reads())

    self._read_concurrently(self.grpc_records, num_readers)

    self.assertGreater(len(self._read_reads()), reads_before)
    # Only the shared client connects to this server, so the count covers every
    # earlier test too. It opens one connection per channel, max(4, nproc) for
    # this non-DirectPath endpoint, and gRPC may reconnect a few. A client per
    # reader would open at least one connection per reader, and its own
    # max(4, nproc) channels, which exceeds this bound up to 16 CPUs.
    channels = max(4, os.cpu_count() or 1)
    self.assertLess(
        self.grpc_server.tcp_connections,
        2 * channels,
        msg=f"channels={channels}",
    )

  def test_gcs_grpc_footer_prefetch(self):
    self._use_grpc()
    num_records = len(self.grpc_records)
    self.assertGreater(len(self.grpc_payload), 2 * 1024 * 1024)

    # The default footer_prefetch_bytes reads the last 1 MiB in one request.
    reads_before = len(self._read_reads())
    reader = ArrayRecordReader(
        "gs://test-bucket/footer-default",
        "readahead_buffer_size:0,max_parallelism:0",
        file_reader_buffer_size=32768,
    )
    self.assertEqual(reader.num_records(), num_records)
    reader.close()
    reads = self._read_reads()[reads_before:]
    self.assertLen(reads, 1)
    self.assertEqual(reads[0][0], -1024 * 1024)

    # footer_prefetch_bytes:512K reads the last 512 KiB in one request.
    reads_before = len(self._read_reads())
    reader = ArrayRecordReader(
        "gs://test-bucket/footer-512k",
        "footer_prefetch_bytes:512K,readahead_buffer_size:0,max_parallelism:0",
        file_reader_buffer_size=32768,
    )
    self.assertEqual(reader.num_records(), num_records)
    reader.close()
    reads = self._read_reads()[reads_before:]
    self.assertLen(reads, 1)
    self.assertEqual(reads[0][0], -512 * 1024)

    # A footer larger than footer_prefetch_bytes:64K takes a second request.
    reads_before = len(self._read_reads())
    reader = ArrayRecordReader(
        "gs://test-bucket/footer-64k",
        "footer_prefetch_bytes:64K,readahead_buffer_size:0,max_parallelism:0",
        file_reader_buffer_size=32768,
    )
    self.assertEqual(reader.num_records(), num_records)
    self.assertEqual(
        reader.read([0, 17, 35]),
        [self.grpc_records[0], self.grpc_records[17], self.grpc_records[35]],
    )
    reader.close()
    reads = self._read_reads()[reads_before:]
    self.assertGreaterEqual(len(reads), 2)
    self.assertEqual(reads[0][0], -64 * 1024)

  def test_gcs_transport_is_case_insensitive(self):
    self._use_grpc(transport="GRPC")
    reads_before = len(self._read_reads())

    reader = ArrayRecordReader(
        "gs://test-bucket/upper-case",
        "readahead_buffer_size:0,max_parallelism:0",
    )
    self.assertEqual(reader.read([1]), [self.grpc_records[1]])
    reader.close()

    self.assertGreater(len(self._read_reads()), reads_before)

  def test_gcs_http_read_after_grpc_client_exists(self):
    self._use_grpc()
    reader = ArrayRecordReader(
        "gs://test-bucket/grpc-first",
        "readahead_buffer_size:0,max_parallelism:0",
    )
    self.assertEqual(reader.read([2]), [self.grpc_records[2]])
    reader.close()
    grpc_reads = len(self._read_reads())

    records, payload = self._write_records(24)
    mock_gcs = _MockGcsServer(payload)
    try:
      os.environ.pop("ARRAY_RECORD_GCS_TRANSPORT")
      os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = (
          f"http://127.0.0.1:{mock_gcs.port}"
      )
      reader = ArrayRecordReader(
          "gs://test-bucket/http-second",
          "readahead_buffer_size:0,max_parallelism:0",
      )
      self.assertEqual(reader.read([3]), [records[3]])
      reader.close()
    finally:
      mock_gcs.close()

    self.assertNotEmpty(mock_gcs.requests)
    self.assertLen(self._read_reads(), grpc_reads)

  def test_gcs_invalid_transport_raises(self):
    os.environ["ARRAY_RECORD_GCS_TRANSPORT"] = "quic"
    with self.assertRaisesRegex(ValueError, "ARRAY_RECORD_GCS_TRANSPORT"):
      ArrayRecordReader("gs://test-bucket/invalid-transport")

  def test_gcs_http_and_empty_transport_use_http(self):
    records, payload = self._write_records(24)
    grpc_reads = len(self._read_reads())
    for transport in ("http", "HTTP", ""):
      with self.subTest(transport=transport):
        mock_gcs = _MockGcsServer(payload)
        try:
          os.environ.update({
              "ARRAY_RECORD_GCS_TRANSPORT": transport,
              "CLOUD_STORAGE_EMULATOR_ENDPOINT": (
                  f"http://127.0.0.1:{mock_gcs.port}"
              ),
          })
          reader = ArrayRecordReader(
              "gs://test-bucket/http-transport",
              "readahead_buffer_size:0,max_parallelism:0",
          )
          self.assertEqual(reader.read([5]), [records[5]])
          reader.close()
        finally:
          mock_gcs.close()
        self.assertNotEmpty(mock_gcs.requests)
    self.assertLen(self._read_reads(), grpc_reads)

  def test_gcs_grpc_with_http_emulator_uses_http(self):
    # The gRPC testbench variable isn't set, so the gRPC client would read
    # production GCS. The reader uses the HTTP emulator instead.
    records, payload = self._write_records(24)
    grpc_reads = len(self._read_reads())
    mock_gcs = _MockGcsServer(payload)
    try:
      os.environ.update({
          "ARRAY_RECORD_GCS_TRANSPORT": "grpc",
          "CLOUD_STORAGE_EMULATOR_ENDPOINT": (
              f"http://127.0.0.1:{mock_gcs.port}"
          ),
      })
      reader = ArrayRecordReader(
          "gs://test-bucket/grpc-with-http-emulator",
          "readahead_buffer_size:0,max_parallelism:0",
      )
      self.assertEqual(reader.read([6]), [records[6]])
      reader.close()
    finally:
      mock_gcs.close()
    self.assertNotEmpty(mock_gcs.requests)
    self.assertLen(self._read_reads(), grpc_reads)

  def test_gcs_grpc_in_forked_child_raises(self):
    self._use_grpc()
    # Create the shared gRPC client in this process, and keep one reader open
    # across the fork.
    open_reader = ArrayRecordReader(
        "gs://test-bucket/before-fork",
        "readahead_buffer_size:0,max_parallelism:0",
    )
    self.assertEqual(open_reader.read([4]), [self.grpc_records[4]])

    pid = os.fork()
    if pid == 0:
      # Child: exit 0 only if every call raises RuntimeError. Without the
      # check, the calls on `open_reader` stop responding and SIGALRM ends the
      # child.
      signal.alarm(30)
      calls = (
          lambda: ArrayRecordReader("gs://test-bucket/after-fork"),
          lambda: open_reader.seek(0),
          open_reader.read,
          lambda: open_reader.read([1]),
          lambda: open_reader.read(0, 2),
          open_reader.read_all,
      )
      exit_code = 0
      for index, call in enumerate(calls):
        try:
          call()
          exit_code = 10 + index
        except RuntimeError as e:
          if "fork()" not in str(e):
            exit_code = 20 + index
        except BaseException:  # pylint: disable=broad-exception-caught
          exit_code = 30 + index
        if exit_code:
          break
      os._exit(exit_code)  # pylint: disable=protected-access
    _, status = os.waitpid(pid, 0)
    open_reader.close()
    self.assertTrue(os.WIFEXITED(status), msg=f"status={status}")
    self.assertEqual(os.WEXITSTATUS(status), 0)


if __name__ == "__main__":
  absltest.main()
