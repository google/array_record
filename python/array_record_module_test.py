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
import json
import os
import re
import socket
import struct
import threading
import urllib.parse

from absl.testing import absltest

from python.array_record_module import ArrayRecordReader
from python.array_record_module import ArrayRecordWriter


def _crc32c_table() -> list[int]:
  table = [0] * 256
  for i in range(256):
    crc = i
    for _ in range(8):
      crc = (crc >> 1) ^ 0x82F63B78 if (crc & 1) else (crc >> 1)
    table[i] = crc
  return table


_CRC32C_TABLE = _crc32c_table()
_CRC32C_CACHE: dict[tuple[int, bytes, bytes], int] = {}
_CRC32C_LOCK = threading.Lock()


def _crc32c(data: bytes) -> int:
  key = (len(data), data[:32], data[-32:])
  with _CRC32C_LOCK:
    cached = _CRC32C_CACHE.get(key)
    if cached is not None:
      return cached
    c = 0xFFFFFFFF
    for b in data:
      c = _CRC32C_TABLE[(c ^ b) & 0xFF] ^ (c >> 8)
    res = c ^ 0xFFFFFFFF
    _CRC32C_CACHE[key] = res
    return res


def _encode_varint(val: int) -> bytes:
  if val < 0:
    val += 1 << 64
  out = bytearray()
  while val > 0x7F:
    out.append((val & 0x7F) | 0x80)
    val >>= 7
  out.append(val & 0x7F)
  return bytes(out)


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
  val = 0
  shift = 0
  while True:
    b = buf[pos]
    pos += 1
    val |= (b & 0x7F) << shift
    if not (b & 0x80):
      return val, pos
    shift += 7


def _encode_bytes_field(field_num: int, payload: bytes) -> bytes:
  return (
      _encode_varint((field_num << 3) | 2)
      + _encode_varint(len(payload))
      + payload
  )


def _encode_varint_field(field_num: int, val: int) -> bytes:
  return _encode_varint((field_num << 3) | 0) + _encode_varint(val)


def _encode_fixed32_field(field_num: int, val: int) -> bytes:
  return _encode_varint((field_num << 3) | 5) + struct.pack("<I", val)


def _parse_proto(buf: bytes) -> list[tuple[int, int, int | bytes]]:
  fields = []
  pos = 0
  n = len(buf)
  while pos < n:
    tag, pos = _decode_varint(buf, pos)
    fnum, wtype = tag >> 3, tag & 7
    if wtype == 0:
      val, pos = _decode_varint(buf, pos)
      fields.append((fnum, wtype, val))
    elif wtype == 1:
      fields.append((fnum, wtype, buf[pos : pos + 8]))
      pos += 8
    elif wtype == 2:
      length, pos = _decode_varint(buf, pos)
      fields.append((fnum, wtype, buf[pos : pos + length]))
      pos += length
    elif wtype == 5:
      fields.append((fnum, wtype, struct.unpack("<I", buf[pos : pos + 4])[0]))
      pos += 4
    else:
      break
  return fields


def _h2_frame(ftype: int, flags: int, sid: int, payload: bytes) -> bytes:
  length = len(payload)
  return (
      bytes([(length >> 16) & 0xFF, (length >> 8) & 0xFF, length & 0xFF])
      + struct.pack("!BBI", ftype, flags, sid)
      + payload
  )


class _MockGcsGrpcServer:
  """Minimal HTTP/2 gRPC server emulating google.storage.v2.Storage."""

  def __init__(self, payload: bytes = b""):
    self.default_payload = payload
    self.objects: dict[str, bytes] = {}
    self.uploads: dict[str, bytearray] = {}
    self.upload_names: dict[str, str] = {}
    self.start_resumable_requests = 0
    self.write_object_requests = 0
    self.read_object_requests = 0
    self.read_requests: list[dict[str, int | str]] = []
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

  def _recv_exact(self, conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
      try:
        chunk = conn.recv(n - len(buf))
      except OSError:
        return None
      if not chunk:
        return None
      buf.extend(chunk)
    return bytes(buf)

  def _extract_object_name_from_spec(self, spec_bytes: bytes) -> str:
    for fnum, wtype, val in _parse_proto(spec_bytes):
      if fnum == 1 and wtype == 2 and isinstance(val, bytes):
        for rfnum, rwtype, rval in _parse_proto(val):
          if rfnum == 1 and rwtype == 2 and isinstance(rval, bytes):
            return rval.decode("utf-8", errors="replace")
    return "object"

  def _handle_rpc(self, msgs: list[bytes]) -> bytes:
    first_fields = _parse_proto(msgs[0]) if msgs else []
    f1_bytes = next(
        (
            v
            for f, w, v in first_fields
            if f == 1 and w == 2 and isinstance(v, bytes)
        ),
        None,
    )
    has_f4_bytes = any(
        any(f == 4 and w == 2 for f, w, _ in _parse_proto(m)) for m in msgs
    )
    has_f7 = any(
        any(f == 7 and w == 0 for f, w, _ in _parse_proto(m)) for m in msgs
    )

    # 1. ReadObjectRequest: field 1 is bucket ("projects/_/buckets/...")
    if (
        len(msgs) == 1
        and f1_bytes is not None
        and f1_bytes.startswith(b"projects/")
        and not has_f4_bytes
        and not has_f7
    ):
      obj_name = ""
      read_offset = 0
      read_limit = 0
      for fnum, wtype, val in first_fields:
        if fnum == 2 and wtype == 2 and isinstance(val, bytes):
          obj_name = val.decode("utf-8", errors="replace")
        elif fnum == 4 and wtype == 0 and isinstance(val, int):
          read_offset = val - (1 << 64) if val >= (1 << 63) else val
        elif fnum == 5 and wtype == 0 and isinstance(val, int):
          read_limit = val
      with self._lock:
        self.read_object_requests += 1
        self.read_requests.append({
            "object": obj_name,
            "read_offset": read_offset,
            "read_limit": read_limit,
        })
        payload = self.objects.get(
            obj_name,
            next(iter(self.objects.values()), self.default_payload),
        )
      total = len(payload)
      if read_offset < 0:
        start = max(0, total + read_offset)
      else:
        start = min(total, read_offset)
      end = total if read_limit <= 0 else min(total, start + read_limit)
      slice_bytes = payload[start:end]
      checksummed_data = _encode_bytes_field(
          1, slice_bytes
      ) + _encode_fixed32_field(2, _crc32c(slice_bytes))
      content_range = (
          _encode_varint_field(1, start)
          + _encode_varint_field(2, end)
          + _encode_varint_field(3, total)
      )
      metadata = _encode_bytes_field(
          1, obj_name.encode()
      ) + _encode_varint_field(6, total)
      return (
          _encode_bytes_field(1, checksummed_data)
          + _encode_bytes_field(3, content_range)
          + _encode_bytes_field(4, metadata)
      )

    # 2. StartResumableWriteRequest: field 1 is WriteObjectSpec
    if (
        len(msgs) == 1
        and f1_bytes is not None
        and not f1_bytes.startswith(b"upload-")
        and not has_f4_bytes
        and not has_f7
    ):
      obj_name = self._extract_object_name_from_spec(f1_bytes)
      with self._lock:
        self.start_resumable_requests += 1
        upload_id = f"upload-{len(self.uploads)}"
        self.uploads[upload_id] = bytearray()
        self.upload_names[upload_id] = obj_name
      return _encode_bytes_field(1, upload_id.encode())

    # 3. WriteObjectRequest (streaming)
    upload_id = ""
    obj_name = "object"
    finish_write = False
    chunks = bytearray()
    for m in msgs:
      for fnum, wtype, val in _parse_proto(m):
        if fnum == 1 and wtype == 2 and isinstance(val, bytes):
          upload_id = val.decode("utf-8", errors="replace")
        elif fnum == 2 and wtype == 2 and isinstance(val, bytes):
          obj_name = self._extract_object_name_from_spec(val)
        elif fnum == 4 and wtype == 2 and isinstance(val, bytes):
          for cfnum, cwtype, cval in _parse_proto(val):
            if cfnum == 1 and cwtype == 2 and isinstance(cval, bytes):
              chunks.extend(cval)
        elif fnum == 7 and wtype == 0 and isinstance(val, int):
          if val:
            finish_write = True
    with self._lock:
      self.write_object_requests += 1
      if upload_id:
        self.uploads.setdefault(upload_id, bytearray()).extend(chunks)
        data_bytes = bytes(self.uploads[upload_id])
        obj_name = self.upload_names.get(upload_id, obj_name)
      else:
        data_bytes = bytes(chunks)
      self.objects[obj_name] = data_bytes
    if finish_write:
      checksums = _encode_fixed32_field(1, _crc32c(data_bytes))
      resource = (
          _encode_bytes_field(1, obj_name.encode())
          + _encode_varint_field(6, len(data_bytes))
          + _encode_bytes_field(16, checksums)
      )
      return _encode_bytes_field(2, resource)
    return _encode_varint_field(1, len(data_bytes))

  def _handle_conn(self, conn: socket.socket):
    with self._lock:
      self.tcp_connections += 1
    with conn:
      preface = self._recv_exact(conn, 24)
      if preface != b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n":
        return
      try:
        conn.sendall(
            _h2_frame(0x04, 0x00, 0, struct.pack("!HI", 0x0004, 0x00FFFFFF))
            + _h2_frame(0x08, 0x00, 0, struct.pack("!I", 0x00FF0000))
        )
      except OSError:
        return
      stream_bufs: dict[int, bytearray] = {}
      while not self._stop:
        hdr = self._recv_exact(conn, 9)
        if not hdr:
          break
        length = (hdr[0] << 16) | (hdr[1] << 8) | hdr[2]
        ftype, flags, sid = struct.unpack("!BBI", hdr[3:9])
        sid &= 0x7FFFFFFF
        payload = self._recv_exact(conn, length) if length > 0 else b""
        if payload is None:
          break
        if ftype == 0x04 and not (flags & 0x01):
          try:
            conn.sendall(_h2_frame(0x04, 0x01, 0, b""))
          except OSError:
            break
        elif ftype == 0x06 and not (flags & 0x01):
          try:
            conn.sendall(_h2_frame(0x06, 0x01, 0, payload))
          except OSError:
            break
        elif ftype == 0x03 and sid > 0:
          stream_bufs.pop(sid, None)
          continue
        elif ftype == 0x01 and sid > 0:
          stream_bufs.setdefault(sid, bytearray())
        elif ftype == 0x00 and sid > 0:
          data_payload = payload
          if flags & 0x08:
            pad_len = payload[0]
            data_payload = payload[1 : len(payload) - pad_len]
          stream_bufs.setdefault(sid, bytearray()).extend(data_payload)
        if sid > 0 and (flags & 0x01) and sid in stream_bufs:
          raw = bytes(stream_bufs.pop(sid))
          msgs = []
          idx = 0
          while idx + 5 <= len(raw):
            mlen = struct.unpack("!I", raw[idx + 1 : idx + 5])[0]
            msgs.append(raw[idx + 5 : idx + 5 + mlen])
            idx += 5 + mlen
          resp_proto = self._handle_rpc(msgs)
          grpc_frame = b"\x00" + struct.pack("!I", len(resp_proto)) + resp_proto
          out = bytearray(
              _h2_frame(
                  0x01,
                  0x04,
                  sid,
                  b"\x88\x00\x0ccontent-type\x10application/grpc",
              )
          )
          for off in range(0, len(grpc_frame), 16384):
            out.extend(
                _h2_frame(0x00, 0x00, sid, grpc_frame[off : off + 16384])
            )
          out.extend(_h2_frame(0x01, 0x05, sid, b"\x00\x0bgrpc-status\x010"))
          try:
            conn.sendall(bytes(out))
          except OSError:
            break

  def _accept_loop(self):
    while not self._stop:
      try:
        conn, _ = self._sock.accept()
      except OSError:
        break
      threading.Thread(
          target=self._handle_conn, args=(conn,), daemon=True
      ).start()

  def reset(self, default_payload: bytes = b""):
    with self._lock:
      self.default_payload = default_payload
      self.objects.clear()
      self.uploads.clear()
      self.upload_names.clear()
      self.start_resumable_requests = 0
      self.write_object_requests = 0
      self.read_object_requests = 0
      self.read_requests.clear()

  def close(self):
    self._stop = True
    self._sock.close()
    self._thread.join(timeout=5.0)


class _MockGcsServer:
  """Minimal HTTP server emulating GCS JSON API upload & media download."""

  def __init__(self, payload: bytes = b""):
    self.payload = payload
    self.objects: dict[str, bytes] = {}
    self.uploads: dict[str, bytearray] = {}
    self.requests = []
    self.post_put_requests = []
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

      def do_POST(self):  # pylint: disable=invalid-name
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        upload_type = qs.get("uploadType", [""])[0]
        name = qs.get("name", ["object"])[0]
        content_len = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_len) if content_len > 0 else b""
        with parent._lock:
          parent.post_put_requests.append({
              "method": "POST",
              "path": self.path,
              "uploadType": upload_type,
          })
        if upload_type == "resumable":
          upload_id = f"upload-{len(parent.uploads)}"
          with parent._lock:
            parent.uploads[upload_id] = bytearray(body)
            parent.objects[name] = bytes(parent.uploads[upload_id])
          self.send_response(200)
          self.send_header(
              "Location",
              f"http://127.0.0.1:{parent.port}/upload/storage/v1/b/test-bucket/o?uploadType=resumable&name={urllib.parse.quote(name)}&upload_id={upload_id}",
          )
          self.send_header("Content-Length", "0")
          self.end_headers()
          return
        if upload_type == "multipart":
          content_type = self.headers.get("Content-Type", "")
          m_boundary = re.search(r"boundary=([^\s;]+)", content_type)
          data_bytes = body
          if m_boundary:
            boundary = m_boundary.group(1).strip('"').encode()
            parts = body.split(b"--" + boundary)
            for part in parts[2:]:
              if part.startswith(b"--"):
                break
              sep = part.find(b"\r\n\r\n")
              if sep != -1:
                data_bytes = part[sep + 4 :]
                if data_bytes.endswith(b"\r\n"):
                  data_bytes = data_bytes[:-2]
                break
          with parent._lock:
            parent.objects[name] = data_bytes
          resp = json.dumps(
              {"name": name, "size": str(len(data_bytes))}
          ).encode()
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.send_header("Content-Length", str(len(resp)))
          self.end_headers()
          self.wfile.write(resp)
          return
        with parent._lock:
          parent.objects[name] = body
        resp = json.dumps({"name": name, "size": str(len(body))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

      def do_PUT(self):  # pylint: disable=invalid-name
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        upload_id = qs.get("upload_id", [""])[0]
        name = qs.get("name", ["object"])[0]
        content_len = int(self.headers.get("Content-Length", "0"))
        chunk = self.rfile.read(content_len) if content_len > 0 else b""
        with parent._lock:
          parent.post_put_requests.append({
              "method": "PUT",
              "path": self.path,
          })
          parent.uploads.setdefault(upload_id, bytearray()).extend(chunk)
          data_bytes = bytes(parent.uploads[upload_id])
          parent.objects[name] = data_bytes
        content_range = self.headers.get("Content-Range", "")
        is_final = not content_range.endswith("/*")
        if is_final:
          resp = json.dumps(
              {"name": name, "size": str(len(data_bytes))}
          ).encode()
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.send_header("Content-Length", str(len(resp)))
          self.end_headers()
          self.wfile.write(resp)
        else:
          self.send_response(308)
          self.send_header("Range", f"bytes=0-{len(data_bytes) - 1}")
          self.send_header("Content-Length", "0")
          self.end_headers()

      def do_GET(self):  # pylint: disable=invalid-name
        m_obj = re.search(r"/b/[^/]+/o/([^?]+)", self.path)
        obj_name = urllib.parse.unquote(m_obj.group(1)) if m_obj else ""
        range_header = self.headers.get("Range")
        with parent._lock:
          parent.requests.append({
              "path": self.path,
              "range": range_header,
          })
          payload = parent.objects.get(
              obj_name,
              next(iter(parent.objects.values()), parent.payload),
          )
        total = len(payload)
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
        data = payload[start : end + 1]
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

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # GetGcsClient() in array_record_module.cc caches a single process-wide
    # static google::cloud::storage::Client without a mutex, and
    # MakeGrpcClient() binds its gRPC channels to
    # CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT only once upon first
    # construction. If each test method started its own _MockGcsGrpcServer on a
    # new ephemeral port, the static gRPC client would remain bound to the
    # first test's closed port and fail subsequent tests.
    cls.mock_grpc = _MockGcsGrpcServer()
    cls._old_emulator = os.environ.get("CLOUD_STORAGE_EMULATOR_ENDPOINT")
    cls._old_testbench = os.environ.get(
        "CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT"
    )
    cls._old_grpc_config = os.environ.get(
        "GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG"
    )
    os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = (
        f"http://127.0.0.1:{cls.mock_grpc.port}"
    )
    os.environ["CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT"] = (
        f"127.0.0.1:{cls.mock_grpc.port}"
    )
    os.environ.pop("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG", None)

  @classmethod
  def tearDownClass(cls):
    if cls._old_emulator is None:
      os.environ.pop("CLOUD_STORAGE_EMULATOR_ENDPOINT", None)
    else:
      os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = cls._old_emulator
    if cls._old_testbench is None:
      os.environ.pop(
          "CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT", None
      )
    else:
      os.environ["CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT"] = (
          cls._old_testbench
      )
    if cls._old_grpc_config is None:
      os.environ.pop("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG", None)
    else:
      os.environ["GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG"] = cls._old_grpc_config
    cls.mock_grpc.close()
    super().tearDownClass()

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

    self.mock_grpc.reset(payload)
    # Opening and closing a gs:// shard over the default gRPC client should
    # issue exactly 1 ReadObject RPC (with read_offset == -1048576 via
    # ReadLast) instead of 3 ReadObject RPCs.
    num_records = len(records)
    reader = ArrayRecordReader(
        "gs://test-bucket/shard-00000-of-00001",
        "readahead_buffer_size:0,max_parallelism:0",
        file_reader_buffer_size=32768,
    )
    self.assertEqual(reader.num_records(), num_records)
    reader.close()
    self.assertLen(self.mock_grpc.read_requests, 1)
    self.assertEqual(self.mock_grpc.read_requests[0]["read_offset"], -1048576)

    # Custom footer_prefetch_bytes:512K should issue 1 ReadObject RPC with
    # read_offset == -524288.
    self.mock_grpc.read_requests.clear()
    reader = ArrayRecordReader(
        "gs://test-bucket/shard-00000-of-00001",
        "footer_prefetch_bytes:512K,readahead_buffer_size:0,max_parallelism:0",
        file_reader_buffer_size=32768,
    )
    self.assertEqual(reader.num_records(), num_records)
    reader.close()
    self.assertLen(self.mock_grpc.read_requests, 1)
    self.assertEqual(self.mock_grpc.read_requests[0]["read_offset"], -524288)

    # Opening with index_storage_option:offloaded without
    # footer_prefetch_bytes:0 raises ValueError.
    with self.assertRaises(ValueError):
      ArrayRecordReader(
          "gs://test-bucket/shard-00000-of-00001",
          "index_storage_option:offloaded,readahead_buffer_size:0,max_parallelism:0",
          file_reader_buffer_size=32768,
      )

    # Opening when the footer exceeds footer_prefetch_bytes (e.g. 64K, which
    # only covers the postscript) falls back cleanly to a second ReadObject
    # RPC for the footer.
    self.mock_grpc.read_requests.clear()
    reader = ArrayRecordReader(
        "gs://test-bucket/shard-00000-of-00001",
        "footer_prefetch_bytes:64K,readahead_buffer_size:0,max_parallelism:0",
        file_reader_buffer_size=32768,
    )
    self.assertEqual(reader.num_records(), num_records)
    reader.close()
    self.assertLen(self.mock_grpc.read_requests, 2)
    self.assertEqual(self.mock_grpc.read_requests[0]["read_offset"], -65536)

    # Verify record payload reading works over gs://.
    self.mock_grpc.read_requests.clear()
    reader = ArrayRecordReader(
        "gs://test-bucket/shard-00000-of-00001",
        "readahead_buffer_size:16M,max_parallelism:1",
        file_reader_buffer_size=32768,
    )
    self.assertEqual(
        reader.read([0, 17, 35]), [records[0], records[17], records[35]]
    )
    reader.close()
    self.assertLen(self.mock_grpc.read_requests, 2)

  def test_gcs_concurrent_multithreaded_reads(self):
    writer = ArrayRecordWriter(self.test_file, "group_size:1,uncompressed")
    records = [f"record-{i:04d}-".encode() + (b"x" * 65536) for i in range(36)]
    for r in records:
      writer.write(r)
    writer.close()
    with open(self.test_file, "rb") as f:
      payload = f.read()

    self.mock_grpc.reset(payload)

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
      results = list(executor.map(worker, range(64)))

    for idx, fetched in enumerate(results):
      self.assertEqual(
          fetched,
          [records[0], records[idx % len(records)], records[-1]],
      )
    self.assertLen(self.mock_grpc.read_requests, 128)
    # All 64 concurrent ArrayRecordReader instances (128 RPCs) share the
    # cached gRPC client's channel pool (DefaultGrpcNumChannels =
    # max(4, hardware_concurrency)) instead of opening 128 separate channels.
    self.assertLessEqual(
        self.mock_grpc.tcp_connections, max(4, os.cpu_count() or 4)
    )

  def test_gcs_grpc_write_and_read_roundtrip(self):
    self.mock_grpc.reset()
    gcs_uri = "gs://test-bucket/shard-00000-of-00001"
    writer = ArrayRecordWriter(gcs_uri, "group_size:1,uncompressed")
    records = [f"record-{i:04d}-".encode() + (b"x" * 1024) for i in range(16)]
    for r in records:
      writer.write(r)
    writer.close()
    self.assertGreater(self.mock_grpc.write_object_requests, 0)

    reader = ArrayRecordReader(
        gcs_uri, "readahead_buffer_size:0,max_parallelism:0"
    )
    self.assertEqual(reader.num_records(), 16)
    self.assertEqual(self.mock_grpc.read_requests[0]["read_offset"], -1048576)
    self.assertEqual(
        reader.read([0, 7, 15]), [records[0], records[7], records[15]]
    )
    reader.close()
    self.assertGreater(self.mock_grpc.read_object_requests, 0)

  def test_gcs_http_fallback_write_and_read_roundtrip(self):
    mock_gcs = _MockGcsServer()
    old_endpoint = os.environ.get("CLOUD_STORAGE_EMULATOR_ENDPOINT")
    old_grpc_config = os.environ.get("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG")
    os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = (
        f"http://127.0.0.1:{mock_gcs.port}"
    )
    # Verify that setting GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG=none disables
    # gRPC and falls back to the HTTP/REST client without code changes.
    os.environ["GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG"] = "none"
    try:
      gcs_uri = "gs://test-bucket/shard-00000-of-00001"
      writer = ArrayRecordWriter(gcs_uri, "group_size:1,uncompressed")
      records = [
          f"record-{i:04d}-".encode() + (b"x" * 65536) for i in range(36)
      ]
      for r in records:
        writer.write(r)
      writer.close()
      self.assertNotEmpty(mock_gcs.post_put_requests)

      # Opening and closing over HTTP/REST fallback should issue 1 HTTP GET
      # with Range: bytes=-1048576 (via ReadLast).
      mock_gcs.requests.clear()
      conns_before = mock_gcs.tcp_connections
      reader = ArrayRecordReader(
          gcs_uri,
          "readahead_buffer_size:0,max_parallelism:0",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(reader.num_records(), 36)
      reader.close()
      self.assertLen(mock_gcs.requests, 1)
      self.assertEqual(mock_gcs.requests[0]["range"], "bytes=-1048576")
      self.assertEqual(mock_gcs.tcp_connections - conns_before, 1)
      self.assertEqual(mock_gcs.aborted_streams, 0)

      # Verify record payload reading works over HTTP/REST fallback and reuses
      # the same TCP connection within the ArrayRecordReader instance.
      mock_gcs.requests.clear()
      conns_before = mock_gcs.tcp_connections
      reader = ArrayRecordReader(
          gcs_uri,
          "readahead_buffer_size:16M,max_parallelism:1",
          file_reader_buffer_size=32768,
      )
      self.assertEqual(
          reader.read([0, 17, 35]), [records[0], records[17], records[35]]
      )
      reader.close()
      self.assertLen(mock_gcs.requests, 2)
      self.assertEqual(mock_gcs.tcp_connections - conns_before, 1)

      # Verify concurrent multithreaded HTTP/REST fallback readers each use an
      # isolated client instance without libcurl handle-pool races under TSAN.
      def http_worker(idx: int):
        r = ArrayRecordReader(
            gcs_uri,
            "readahead_buffer_size:16M,max_parallelism:1",
            file_reader_buffer_size=32768,
        )
        fetched = r.read([0, idx % len(records), len(records) - 1])
        r.close()
        return fetched

      with futures.ThreadPoolExecutor(max_workers=8) as executor:
        http_results = list(executor.map(http_worker, range(16)))
      for idx, fetched in enumerate(http_results):
        self.assertEqual(
            fetched,
            [records[0], records[idx % len(records)], records[-1]],
        )
    finally:
      if old_endpoint is None:
        os.environ.pop("CLOUD_STORAGE_EMULATOR_ENDPOINT", None)
      else:
        os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = old_endpoint
      if old_grpc_config is None:
        os.environ.pop("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG", None)
      else:
        os.environ["GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG"] = old_grpc_config
      mock_gcs.close()


if __name__ == "__main__":
  absltest.main()
