/* Copyright 2022 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "cpp/array_record_reader.h"
#include "cpp/array_record_writer.h"
#include "cpp/thread_pool.h"
#include "google/cloud/credentials.h"
#include "google/cloud/options.h"
#include "google/cloud/storage/client.h"
#include "google/cloud/storage/download_options.h"
#include "google/cloud/storage/grpc_plugin.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "riegeli/base/initializer.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/gcs/gcs_object.h"
#include "riegeli/gcs/gcs_reader.h"

namespace py = pybind11;

namespace {

// Returns a new HTTP client for one reader.
//
// Readers don't share an HTTP client: concurrent use of one client's curl
// handle pool hits a libcurl data race (Curl_peer_unlink).
//
// When a storage emulator is configured, google-cloud-cpp 3.x still uses
// Application Default Credentials, even though `Client()` documents otherwise.
// Outside of GCE they look up the metadata server and the request is retried
// until the retry policy expires, so the reader appears to hang. Use insecure
// credentials, as emulators expect.
//
// TODO(zhixiangli): Use `google::cloud::storage::Client()` again once
// https://github.com/googleapis/google-cloud-cpp/issues/16678 is fixed in the
// google-cloud-cpp release used by the OSS build.
google::cloud::storage::Client MakeGcsClient() {
  google::cloud::Options options;
  if (std::getenv("CLOUD_STORAGE_EMULATOR_ENDPOINT") != nullptr ||
      std::getenv("CLOUD_STORAGE_TESTBENCH_ENDPOINT") != nullptr) {
    options.set<google::cloud::UnifiedCredentialsOption>(
        google::cloud::MakeInsecureCredentials());
  }
  return google::cloud::storage::Client(std::move(options));
}

// Returns true if `ARRAY_RECORD_GCS_TRANSPORT` selects gRPC. The value is
// case-insensitive. Unset, empty or `http` selects HTTP. Throws
// `std::invalid_argument` (`ValueError` in Python) for any other value.
//
// With `grpc`, an HTTP emulator (CLOUD_STORAGE_EMULATOR_ENDPOINT or
// CLOUD_STORAGE_TESTBENCH_ENDPOINT) still selects HTTP, unless
// CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT is also set. Only that
// variable changes the gRPC endpoint, so otherwise the reads would go to
// production GCS instead of the emulator.
bool UseGrpcTransport() {
  const char* transport = std::getenv("ARRAY_RECORD_GCS_TRANSPORT");
  if (transport == nullptr) return false;
  const std::string value = absl::AsciiStrToLower(transport);
  if (value.empty() || value == "http") return false;
  if (value == "grpc") {
    const bool http_emulator =
        std::getenv("CLOUD_STORAGE_EMULATOR_ENDPOINT") != nullptr ||
        std::getenv("CLOUD_STORAGE_TESTBENCH_ENDPOINT") != nullptr;
    return !http_emulator ||
           std::getenv("CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT") !=
               nullptr;
  }
  throw std::invalid_argument(absl::StrFormat(
      "Invalid ARRAY_RECORD_GCS_TRANSPORT value \"%s\". Expected \"http\" or "
      "\"grpc\".",
      transport));
}

// The process-wide gRPC client, and the process that created it.
struct SharedGrpcClient {
  pid_t pid;
  google::cloud::storage::Client client;
};

// Throws because the gRPC client created in process `client_pid` is used in a
// child of it created by fork().
[[noreturn]] void ThrowGrpcClientUsedAfterFork(pid_t client_pid) {
  throw std::runtime_error(absl::StrFormat(
      "ArrayRecordReader can't use gs:// paths with "
      "ARRAY_RECORD_GCS_TRANSPORT=grpc in a process created by fork() after "
      "the gRPC client was created (in process %d). Start workers with the "
      "\"spawn\" or \"forkserver\" method, or use the HTTP transport.",
      client_pid));
}

// Returns the gRPC client that all readers share when
// ARRAY_RECORD_GCS_TRANSPORT=grpc.
//
// HTTP is the default, with a new client per reader (see `MakeGcsClient()`).
// The shared gRPC client cuts connections, file descriptors and memory, and in
// our Grain benchmarks it also shortened the time to first batch and raised
// throughput. A client per reader would duplicate each channel's local
// subchannel pool (GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL), its background threads
// and its metadata zone lookup.
//
// The client keeps google-cloud-cpp's default channel count: 1 on DirectPath,
// max(4, hardware threads) elsewhere. It turns off gRPC metrics export. Without
// that, every process opens its own monitoring.googleapis.com connection, and
// the export fails without extra IAM permissions.
//
// Limitations:
// - Reads go over DirectPath only from a Compute Engine VM with an attached
//   service account, in the bucket's region (or one of its regions), with
//   egress to the DirectPath IP ranges. Otherwise they go over CloudPath,
//   where all of the client's streams can share one channel and the server
//   allows 100 concurrent streams on it. Each open reader holds one stream
//   until it's closed, so more than about 100 open readers stop responding.
//   Use HTTP in that case.
// - The client is created on first use and reused for the life of the
//   process. gRPC doesn't work in a child created by fork() after that, so
//   using a gs:// reader in such a child throws `std::runtime_error`, whether
//   the reader was opened before or after the fork. Start workers with the
//   "spawn" or "forkserver" method instead.
const SharedGrpcClient& GetSharedGrpcClient() {
  // Never destroyed, because readers can outlive static destruction.
  static const SharedGrpcClient* const kGrpcClient = new SharedGrpcClient{
      getpid(),
      google::cloud::storage::MakeGrpcClient(
          google::cloud::Options{}
              .set<
                  google::cloud::storage_experimental::EnableGrpcMetricsOption>(
                  false))};
  if (kGrpcClient->pid != getpid()) {
    ThrowGrpcClientUsedAfterFork(kGrpcClient->pid);
  }
  return *kGrpcClient;
}

// An `ArrayRecordReader` for a `gs://` path. Remembers the process that
// created the shared gRPC client it uses, if any, so that using it in a forked
// child throws instead of stopping responding.
class GcsArrayRecordReader
    : public array_record::ArrayRecordReader<riegeli::GcsReader> {
 public:
  GcsArrayRecordReader(std::optional<pid_t> grpc_client_pid,
                       riegeli::Initializer<riegeli::GcsReader> src,
                       Options options, array_record::ARThreadPool* pool)
      : ArrayRecordReader(std::move(src), std::move(options), pool),
        grpc_client_pid_(grpc_client_pid) {}

  // Throws `std::runtime_error` if the reader uses the shared gRPC client and
  // this process was created by fork() after the client was created.
  void CheckUsableInThisProcess() const {
    if (grpc_client_pid_.has_value() && *grpc_client_pid_ != getpid()) {
      ThrowGrpcClientUsedAfterFork(*grpc_client_pid_);
    }
  }

 private:
  std::optional<pid_t> grpc_client_pid_;
};

// Throws `std::runtime_error` if `reader` can't do IO in this process. See
// `GcsArrayRecordReader::CheckUsableInThisProcess()`.
void CheckUsableInThisProcess(
    const array_record::ArrayRecordReaderBase& reader) {
  const auto* gcs_reader = dynamic_cast<const GcsArrayRecordReader*>(&reader);
  if (gcs_reader != nullptr) gcs_reader->CheckUsableInThisProcess();
}

}  // namespace

PYBIND11_MODULE(array_record_module, m) {
  using array_record::ArrayRecordReaderBase;
  using array_record::ArrayRecordWriterBase;

  py::class_<ArrayRecordWriterBase>(m, "ArrayRecordWriter")
      .def(py::init([](const std::string& path, const std::string& options,
                       const py::kwargs& kwargs) -> ArrayRecordWriterBase* {
             auto status_or_option =
                 ArrayRecordWriterBase::Options::FromString(options);
             if (!status_or_option.ok()) {
               throw py::value_error(
                   std::string(status_or_option.status().message()));
             }
             riegeli::FdWriterBase::Options file_writer_options;
             file_writer_options.set_buffer_size(size_t{16} << 20);
             // Release the GIL because IO is time consuming.
             py::gil_scoped_release scoped_release;
             return new array_record::ArrayRecordWriter(
                 riegeli::Maker<riegeli::FdWriter>(
                     path, std::move(file_writer_options)),
                 status_or_option.value());
           }),
           py::arg("path"), py::arg("options") = "")
      .def("ok", &ArrayRecordWriterBase::ok)
      .def("close",
           [](ArrayRecordWriterBase& writer) {
             if (!writer.Close()) {
               throw std::runtime_error(std::string(writer.status().message()));
             }
           })
      .def("is_open", &ArrayRecordWriterBase::is_open)
      // We accept only py::bytes (and not unicode strings) since we expect
      // most users to store binary data (e.g. serialized protocol buffers).
      // We cannot know if a users wants to write+read unicode and forcing users
      // to encode() their unicode strings avoids accidental conversions.
      .def("write", [](ArrayRecordWriterBase& writer, py::bytes record) {
        if (!writer.WriteRecord(record)) {
          throw std::runtime_error(std::string(writer.status().message()));
        }
      });
  py::class_<ArrayRecordReaderBase>(m, "ArrayRecordReader")
      .def(py::init([](const std::string& path, const std::string& options,
                       const py::kwargs& kwargs) -> ArrayRecordReaderBase* {
             auto status_or_option =
                 ArrayRecordReaderBase::Options::FromString(options);
             if (!status_or_option.ok()) {
               throw py::value_error(
                   std::string(status_or_option.status().message()));
             }
             riegeli::FdReaderBase::Options file_reader_options;
             riegeli::GcsReader::Options gcs_reader_options;
             if (kwargs.contains("file_reader_buffer_size")) {
               auto file_reader_buffer_size =
                   kwargs["file_reader_buffer_size"].cast<int64_t>();
               file_reader_options.set_buffer_size(file_reader_buffer_size);
               gcs_reader_options.set_buffer_size(file_reader_buffer_size);
             }
             // Release the GIL because IO is time consuming.
             py::gil_scoped_release scoped_release;
             if (absl::StartsWith(path, "gs://")) {
               const SharedGrpcClient* grpc_client =
                   UseGrpcTransport() ? &GetSharedGrpcClient() : nullptr;
               const auto footer_prefetch_bytes =
                   status_or_option->footer_prefetch_bytes();
               return new GcsArrayRecordReader(
                   grpc_client == nullptr
                       ? std::nullopt
                       : std::optional<pid_t>(grpc_client->pid),
                   riegeli::Maker<riegeli::GcsReader>(
                       grpc_client == nullptr ? MakeGcsClient()
                                              : grpc_client->client,
                       riegeli::GcsObject(path), std::move(gcs_reader_options),
                       footer_prefetch_bytes > 0
                           ? google::cloud::storage::ReadLast(
                                 footer_prefetch_bytes)
                           : google::cloud::storage::ReadLast()),
                   status_or_option.value(),
                   array_record::ArrayRecordGlobalPool());
             } else {
               return new array_record::ArrayRecordReader(
                   riegeli::Maker<riegeli::FdReader>(
                       path, std::move(file_reader_options)),
                   status_or_option.value(),
                   array_record::ArrayRecordGlobalPool());
             }
           }),
           py::arg("path"), py::arg("options") = "", R"(
           ArrayRecordReader for fast sequential or random access.

           Args:
               path: File path to the input file.
               options: String with options for ArrayRecord. See syntax below.
           Kwargs:
               file_reader_buffer_size: Optional size of the buffer (in bytes)
                 for the underlying file (Riegeli) reader. The default buffer
                 size is 1 MiB.
               file_options: Optional file::Options to use for the underlying
                 file (Riegeli) reader.

           options ::= option? ("," option?)*
           option ::=
             "readahead_buffer_size" ":" readahead_buffer_size |
             "max_parallelism" ":" max_parallelism |
             "index_storage_option" ":" index_storage_option |
             "footer_prefetch_bytes" ":" footer_prefetch_bytes
           readahead_buffer_size ::= non-negative integer expressed as real with
             optional suffix [BkKMGTPE]. (Default 16MB). Set to 0 optimizes
             random access performance.
           max_parallelism ::= `auto` or non-negative integer. Each parallel
             thread owns its readhaed buffer with the size
             `readahead_buffer_size`.  (Default thread pool size) Set to 0
             optimizes random access performance.
           index_storage_option ::= `in_memory` or `offloaded`. Defaults to
             `in_memory`.
           footer_prefetch_bytes ::= `auto` or non-negative integer expressed as
             real with optional suffix [BkKMGTPE]. Defaults to 1MB when
             `index_storage_option` is `in_memory`, and 0 when `offloaded`.
             Values > 0 are only supported with
             `index_storage_option:in_memory`.

           The default option is optimized for sequential access. To optimize
           the random access performance, set the options to
           "readahead_buffer_size:0,max_parallelism:0".

           Google Cloud Storage:
             `path` can be a `gs://bucket/object` path. Only ArrayRecordReader
             supports `gs://` paths, ArrayRecordWriter doesn't. The environment
             variable `ARRAY_RECORD_GCS_TRANSPORT` selects the transport:
             `http` (default) or `grpc`, case-insensitive. Unset or empty means
             `http`. Any other value raises ValueError. With `http`, each reader
             has its own client. With `grpc`, all readers in the process share
             one gRPC client, which uses fewer connections and file
             descriptors.

             Reads go over DirectPath only from a Compute Engine VM with an
             attached service account, in the bucket's region, with egress to
             the DirectPath IP ranges. Otherwise they go over CloudPath, where
             more than about 100 open readers can stop responding, because
             each holds one stream and the server allows 100 per channel. Use
             `http` in that case. With `grpc`, an HTTP emulator
             (`CLOUD_STORAGE_EMULATOR_ENDPOINT`) still selects HTTP unless
             `CLOUD_STORAGE_EXPERIMENTAL_GRPC_TESTBENCH_ENDPOINT` is also set.

             The gRPC client can't be used in a process created by fork() after
             the client was created. Opening a `gs://` reader there raises
             RuntimeError, and so do seek(), read() and read_all() on a `gs://`
             reader opened before the fork. Start workers with the "spawn" or
             "forkserver" method instead.
           )")
      .def("ok", &ArrayRecordReaderBase::ok)
      .def("close",
           [](ArrayRecordReaderBase& reader) {
             if (!reader.Close()) {
               throw std::runtime_error(std::string(reader.status().message()));
             }
           })
      .def("is_open", &ArrayRecordReaderBase::is_open)
      .def("num_records", &ArrayRecordReaderBase::NumRecords)
      .def("record_index", &ArrayRecordReaderBase::RecordIndex)
      .def("writer_options_string", &ArrayRecordReaderBase::WriterOptionsString)
      .def("seek",
           [](ArrayRecordReaderBase& reader, int64_t record_index) {
             CheckUsableInThisProcess(reader);
             if (!reader.SeekRecord(record_index)) {
               throw std::runtime_error(std::string(reader.status().message()));
             }
           })
      // See write() for why this returns py::bytes.
      .def("read",
           [](ArrayRecordReaderBase& reader) {
             CheckUsableInThisProcess(reader);
             absl::string_view string_view;
             if (!reader.ReadRecord(&string_view)) {
               if (reader.ok()) {
                 throw std::out_of_range(absl::StrFormat(
                     "Out of range of num_records: %d", reader.NumRecords()));
               }
               throw std::runtime_error(std::string(reader.status().message()));
             }
             return py::bytes(string_view);
           })
      .def("read",
           [](ArrayRecordReaderBase& reader, std::vector<uint64_t> indices) {
             CheckUsableInThisProcess(reader);
             std::vector<std::string> staging(indices.size());
             py::list output(indices.size());
             {
               py::gil_scoped_release scoped_release;
               auto status = reader.ParallelReadRecordsWithIndices(
                   indices,
                   [&](uint64_t indices_index,
                       absl::string_view record_data) -> absl::Status {
                     staging[indices_index] = record_data;
                     return absl::OkStatus();
                   });
               if (!status.ok()) {
                 throw std::runtime_error(std::string(status.message()));
               }
             }
             // TODO(fchern): Can we write the data directly to the output
             // list in our Parallel loop?
             ssize_t index = 0;
             for (const auto& record : staging) {
               auto py_record = py::bytes(record);
               PyList_SET_ITEM(output.ptr(), index++,
                               py_record.release().ptr());
             }
             return output;
           })
      .def("read",
           [](ArrayRecordReaderBase& reader, int32_t begin, int32_t end) {
             CheckUsableInThisProcess(reader);
             int32_t range_begin = begin, range_end = end;
             if (range_begin < 0) {
               range_begin = reader.NumRecords() + range_begin;
             }
             if (range_end < 0) {
               range_end = reader.NumRecords() + range_end;
             }
             if (range_begin > reader.NumRecords() || range_begin < 0 ||
                 range_end > reader.NumRecords() || range_end < 0 ||
                 range_end <= range_begin) {
               throw std::out_of_range(
                   absl::StrFormat("[%d, %d) is of range of [0, %d)", begin,
                                   end, reader.NumRecords()));
             }
             int32_t num_to_read = range_end - range_begin;
             std::vector<std::string> staging(num_to_read);
             py::list output(num_to_read);
             {
               py::gil_scoped_release scoped_release;
               auto status = reader.ParallelReadRecordsInRange(
                   range_begin, range_end,
                   [&](uint64_t index,
                       absl::string_view record_data) -> absl::Status {
                     staging[index - range_begin] = record_data;
                     return absl::OkStatus();
                   });
               if (!status.ok()) {
                 throw std::runtime_error(std::string(status.message()));
               }
             }
             // TODO(fchern): Can we write the data directly to the output
             // list in our Parallel loop?
             ssize_t index = 0;
             for (const auto& record : staging) {
               auto py_record = py::bytes(record);
               PyList_SET_ITEM(output.ptr(), index++,
                               py_record.release().ptr());
             }
             return output;
           })
      .def("read_all", [](ArrayRecordReaderBase& reader) {
        CheckUsableInThisProcess(reader);
        std::vector<std::string> staging(reader.NumRecords());
        py::list output(reader.NumRecords());
        {
          py::gil_scoped_release scoped_release;
          auto status = reader.ParallelReadRecords(
              [&](uint64_t index,
                  absl::string_view record_data) -> absl::Status {
                staging[index] = record_data;
                return absl::OkStatus();
              });
          if (!status.ok()) {
            throw std::runtime_error(std::string(status.message()));
          }
        }
        // TODO(fchern): Can we write the data directly to the output
        // list in our Parallel loop?
        ssize_t index = 0;
        for (const auto& record : staging) {
          auto py_record = py::bytes(record);
          PyList_SET_ITEM(output.ptr(), index++, py_record.release().ptr());
        }
        return output;
      });
}
