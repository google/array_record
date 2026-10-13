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

  def test_gcs_grpc_write_and_read_roundtrip(self):
    crc32c_table = [0] * 256
    for i in range(256):
      crc = i
      for _ in range(8):
        crc = (crc >> 1) ^ 0x82F63B78 if (crc & 1) else (crc >> 1)
      crc32c_table[i] = crc

    def crc32c(data: bytes) -> int:
      c = 0xFFFFFFFF
      for b in data:
        c = crc32c_table[(c ^ b) & 0xFF] ^ (c >> 8)
      return c ^ 0xFFFFFFFF

    def encode_varint(val: int) -> bytes:
      if val < 0:
        val += 1 << 64
      out = bytearray()
      while val > 0x7F:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
      out.append(val & 0x7F)
      return bytes(out)

    def decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
      val = 0
      shift = 0
      while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
          return val, pos
        shift += 7

    def encode_bytes_field(field_num: int, payload: bytes) -> bytes:
      return (
          encode_varint((field_num << 3) | 2)
          + encode_varint(len(payload))
          + payload
      )

    def encode_varint_field(field_num: int, val: int) -> bytes:
      return encode_varint((field_num << 3) | 0) + encode_varint(val)

    def encode_fixed32_field(field_num: int, val: int) -> bytes:
      return encode_varint((field_num << 3) | 5) + struct.pack("<I", val)

    def parse_proto(buf: bytes) -> list[tuple[int, int, int | bytes]]:
      fields = []
      pos = 0
      n = len(buf)
      while pos < n:
        tag, pos = decode_varint(buf, pos)
        fnum, wtype = tag >> 3, tag & 7
        if wtype == 0:
          val, pos = decode_varint(buf, pos)
          fields.append((fnum, wtype, val))
        elif wtype == 1:
          fields.append((fnum, wtype, buf[pos : pos + 8]))
          pos += 8
        elif wtype == 2:
          length, pos = decode_varint(buf, pos)
          fields.append((fnum, wtype, buf[pos : pos + length]))
          pos += length
        elif wtype == 5:
          fields.append(
              (fnum, wtype, struct.unpack("<I", buf[pos : pos + 4])[0])
          )
          pos += 4
        else:
          break
      return fields

    def h2_frame(ftype: int, flags: int, sid: int, payload: bytes) -> bytes:
      length = len(payload)
      return (
          bytes([(length >> 16) & 0xFF, (length >> 8) & 0xFF, length & 0xFF])
          + struct.pack("!BBI", ftype, flags, sid)
          + payload
      )

    class _MockGcsGrpcServer:
      """Minimal HTTP/2 gRPC server emulating google.storage.v2.Storage."""

      def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, bytearray] = {}
        self.upload_names: dict[str, str] = {}
        self.start_resumable_requests = 0
        self.write_object_requests = 0
        self.read_object_requests = 0
        self._lock = threading.Lock()
        self._stop = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

      def _recv_exact(self, conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
          chunk = conn.recv(n - len(buf))
          if not chunk:
            return None
          buf.extend(chunk)
        return bytes(buf)

      def _extract_object_name_from_spec(self, spec_bytes: bytes) -> str:
        for fnum, wtype, val in parse_proto(spec_bytes):
          if fnum == 1 and wtype == 2 and isinstance(val, bytes):
            for rfnum, rwtype, rval in parse_proto(val):
              if rfnum == 1 and rwtype == 2 and isinstance(rval, bytes):
                return rval.decode("utf-8", errors="replace")
        return "object"

      def _handle_rpc(self, msgs: list[bytes]) -> bytes:
        first_fields = parse_proto(msgs[0]) if msgs else []
        f1_bytes = next(
            (
                v
                for f, w, v in first_fields
                if f == 1 and w == 2 and isinstance(v, bytes)
            ),
            None,
        )
        has_f4_bytes = any(
            any(f == 4 and w == 2 for f, w, _ in parse_proto(m)) for m in msgs
        )
        has_f7 = any(
            any(f == 7 and w == 0 for f, w, _ in parse_proto(m)) for m in msgs
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
            payload = self.objects.get(
                obj_name, next(iter(self.objects.values()), b"")
            )
          total = len(payload)
          if read_offset < 0:
            start = max(0, total + read_offset)
          else:
            start = min(total, read_offset)
          end = total if read_limit <= 0 else min(total, start + read_limit)
          slice_bytes = payload[start:end]
          checksummed_data = encode_bytes_field(
              1, slice_bytes
          ) + encode_fixed32_field(2, crc32c(slice_bytes))
          content_range = (
              encode_varint_field(1, start)
              + encode_varint_field(2, end)
              + encode_varint_field(3, total)
          )
          metadata = encode_bytes_field(
              1, obj_name.encode()
          ) + encode_varint_field(6, total)
          return (
              encode_bytes_field(1, checksummed_data)
              + encode_bytes_field(3, content_range)
              + encode_bytes_field(4, metadata)
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
          return encode_bytes_field(1, upload_id.encode())

        # 3. WriteObjectRequest (streaming)
        upload_id = ""
        obj_name = "object"
        finish_write = False
        chunks = bytearray()
        for m in msgs:
          for fnum, wtype, val in parse_proto(m):
            if fnum == 1 and wtype == 2 and isinstance(val, bytes):
              upload_id = val.decode("utf-8", errors="replace")
            elif fnum == 2 and wtype == 2 and isinstance(val, bytes):
              obj_name = self._extract_object_name_from_spec(val)
            elif fnum == 4 and wtype == 2 and isinstance(val, bytes):
              for cfnum, cwtype, cval in parse_proto(val):
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
          checksums = encode_fixed32_field(1, crc32c(data_bytes))
          resource = (
              encode_bytes_field(1, obj_name.encode())
              + encode_varint_field(6, len(data_bytes))
              + encode_bytes_field(16, checksums)
          )
          return encode_bytes_field(2, resource)
        return encode_varint_field(1, len(data_bytes))

      def _handle_conn(self, conn: socket.socket):
        with conn:
          preface = self._recv_exact(conn, 24)
          if preface != b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n":
            return
          conn.sendall(
              h2_frame(0x04, 0x00, 0, struct.pack("!HI", 0x0004, 0x00FFFFFF))
              + h2_frame(0x08, 0x00, 0, struct.pack("!I", 0x00FF0000))
          )
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
              conn.sendall(h2_frame(0x04, 0x01, 0, b""))
            elif ftype == 0x06 and not (flags & 0x01):
              conn.sendall(h2_frame(0x06, 0x01, 0, payload))
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
              grpc_frame = (
                  b"\x00" + struct.pack("!I", len(resp_proto)) + resp_proto
              )
              out = bytearray(
                  h2_frame(
                      0x01,
                      0x04,
                      sid,
                      b"\x88\x00\x0ccontent-type\x10application/grpc",
                  )
              )
              for off in range(0, len(grpc_frame), 16384):
                out.extend(
                    h2_frame(0x00, 0x00, sid, grpc_frame[off : off + 16384])
                )
              out.extend(h2_frame(0x01, 0x05, sid, b"\x00\x0bgrpc-status\x010"))
              conn.sendall(bytes(out))

      def _accept_loop(self):
        while not self._stop:
          try:
            conn, _ = self._sock.accept()
          except OSError:
            break
          threading.Thread(
              target=self._handle_conn, args=(conn,), daemon=True
          ).start()

      def close(self):
        self._stop = True
        self._sock.close()
        self._thread.join(timeout=5.0)

    mock_grpc = _MockGcsGrpcServer()
    old_emulator = os.environ.get("CLOUD_STORAGE_EMULATOR_ENDPOINT")
    old_testbench = os.environ.get(
        "CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT"
    )
    old_grpc_config = os.environ.get("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG")
    os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = (
        f"http://127.0.0.1:{mock_grpc.port}"
    )
    os.environ["CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT"] = (
        f"127.0.0.1:{mock_grpc.port}"
    )
    os.environ.pop("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG", None)
    try:
      gcs_uri = "gs://test-bucket/shard-00000-of-00001"
      writer = ArrayRecordWriter(gcs_uri, "group_size:1,uncompressed")
      records = [
          f"record-{i:04d}-".encode() + (b"x" * 1024) for i in range(16)
      ]
      for r in records:
        writer.write(r)
      writer.close()
      self.assertGreater(mock_grpc.write_object_requests, 0)

      reader = ArrayRecordReader(
          gcs_uri, "readahead_buffer_size:0,max_parallelism:0"
      )
      self.assertEqual(reader.num_records(), 16)
      self.assertEqual(
          reader.read([0, 7, 15]), [records[0], records[7], records[15]]
      )
      reader.close()
      self.assertGreater(mock_grpc.read_object_requests, 0)
    finally:
      if old_emulator is None:
        os.environ.pop("CLOUD_STORAGE_EMULATOR_ENDPOINT", None)
      else:
        os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = old_emulator
      if old_testbench is None:
        os.environ.pop(
            "CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT", None
        )
      else:
        os.environ["CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT"] = (
            old_testbench
        )
      if old_grpc_config is None:
        os.environ.pop("GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG", None)
      else:
        os.environ["GOOGLE_CLOUD_CPP_STORAGE_GRPC_CONFIG"] = old_grpc_config
      mock_grpc.close()

  def test_gcs_http_fallback_write_and_read_roundtrip(self):

    class _MockGcsServer:
      """Minimal HTTP server emulating GCS JSON API upload & media download."""

      def __init__(self, initial_objects: dict[str, bytes] | None = None):
        self.objects = dict(initial_objects or {})
        self.uploads = {}
        self.get_requests = []
        self.post_put_requests = []
        self._lock = threading.Lock()
        parent = self

        class Handler(server.BaseHTTPRequestHandler):
          protocol_version = "HTTP/1.1"

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
              payload = body
              if m_boundary:
                boundary = m_boundary.group(1).strip('"').encode()
                parts = body.split(b"--" + boundary)
                for part in parts[2:]:
                  if part.startswith(b"--"):
                    break
                  sep = part.find(b"\r\n\r\n")
                  if sep != -1:
                    payload = part[sep + 4 :]
                    if payload.endswith(b"\r\n"):
                      payload = payload[:-2]
                    break
              with parent._lock:
                parent.objects[name] = payload
              resp = json.dumps(
                  {"name": name, "size": str(len(payload))}
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
              parent.get_requests.append({
                  "path": self.path,
                  "range": range_header,
              })
              payload = parent.objects.get(
                  obj_name, next(iter(parent.objects.values()))
              )
            total = len(payload)
            start = 0
            end = total - 1
            status = 200
            if range_header:
              m_range = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
              if m_range:
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
            self.wfile.write(data)

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

      reader = ArrayRecordReader(
          gcs_uri, "readahead_buffer_size:0,max_parallelism:0"
      )
      self.assertEqual(reader.num_records(), 36)
      self.assertEqual(
          reader.read([0, 17, 35]), [records[0], records[17], records[35]]
      )
      reader.close()
      self.assertNotEmpty(mock_gcs.get_requests)
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
