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
#include <stdlib.h>
#include <string.h>

#include <chrono>  // NOLINT(build/c++11)
#include <mutex>   // NOLINT(build/c++11)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "cpp/array_record_reader.h"
#include "cpp/array_record_writer.h"
#include "cpp/thread_pool.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/gcs/gcs_object.h"
#include "riegeli/gcs/gcs_reader.h"

namespace py = pybind11;

namespace {

void MaybeConfigureResolverOptions() {
  const char* dns_env = getenv("ARRAY_RECORD_DNS_SINGLE_REQUEST_REOPEN");
  if (dns_env == nullptr ||
      (strcmp(dns_env, "0") != 0 && strcmp(dns_env, "false") != 0)) {
    setenv("RES_OPTIONS", "single-request-reopen timeout:1 attempts:3", 0);
  }
}

google::cloud::storage::Client GetGcsClient() {
  MaybeConfigureResolverOptions();
  const char* share_env = getenv("ARRAY_RECORD_SHARE_GCS_CLIENT");
  const bool share_client =
      (share_env == nullptr ||
       (strcmp(share_env, "0") != 0 && strcmp(share_env, "false") != 0));
  const char* emulator_env = getenv("CLOUD_STORAGE_EMULATOR_ENDPOINT");
  if (!share_client || (emulator_env != nullptr && emulator_env[0] != '\0')) {
    return google::cloud::storage::Client();
  }
  static std::vector<google::cloud::storage::Client>* shared_clients = []() {
    size_t pool_size = 256;
    if (const char* pool_env = getenv("ARRAY_RECORD_GCS_POOL_SIZE")) {
      long v = strtol(pool_env, nullptr, 10);  // NOLINT
      if (v > 0) pool_size = static_cast<size_t>(v);
    }
    size_t num_clients = 4;
    if (const char* nc_env = getenv("ARRAY_RECORD_GCS_NUM_CLIENTS")) {
      long v = strtol(nc_env, nullptr, 10);  // NOLINT
      if (v > 0) num_clients = static_cast<size_t>(v);
    }
    auto* vec = new std::vector<google::cloud::storage::Client>();
    vec->reserve(num_clients);
    for (size_t i = 0; i < num_clients; ++i) {
      auto opts =
          google::cloud::Options{}
              .set<google::cloud::storage::ConnectionPoolSizeOption>(pool_size);
      const char* http_ver = getenv("ARRAY_RECORD_GCS_HTTP_VERSION");
      if (http_ver == nullptr) http_ver = "1.1";
      if (http_ver[0] != '\0') {
        opts.set<google::cloud::storage_experimental::HttpVersionOption>(
            std::string(http_ver));
      }
      long stall_s = 2;  // NOLINT
      if (const char* stall_s_env =
              getenv("ARRAY_RECORD_GCS_STALL_TIMEOUT_S")) {
        stall_s = strtol(stall_s_env, nullptr, 10);  // NOLINT
      }
      if (stall_s > 0) {
        opts.set<google::cloud::storage::TransferStallTimeoutOption>(
            std::chrono::seconds(stall_s));
        opts.set<google::cloud::storage::DownloadStallTimeoutOption>(
            std::chrono::seconds(stall_s));
      }
      long stall_rate = 1048576;  // NOLINT
      if (const char* stall_rate_env =
              getenv("ARRAY_RECORD_GCS_STALL_MIN_RATE")) {
        stall_rate = strtol(stall_rate_env, nullptr, 10);  // NOLINT
      }
      if (stall_rate > 0) {
        opts.set<google::cloud::storage::TransferStallMinimumRateOption>(
            static_cast<std::int32_t>(stall_rate));
        opts.set<google::cloud::storage::DownloadStallMinimumRateOption>(
            static_cast<std::int32_t>(stall_rate));
      }
      if (const char* dl_buf_env =
              getenv("ARRAY_RECORD_GCS_DOWNLOAD_BUFFER_BYTES")) {
        long b = strtol(dl_buf_env, nullptr, 10);  // NOLINT
        if (b > 0) {
          opts.set<google::cloud::storage::DownloadBufferSizeOption>(
              static_cast<std::size_t>(b));
        }
      }
      vec->emplace_back(std::move(opts));
    }
    return vec;
  }();
  static std::atomic<size_t> next_client_idx{0};
  size_t idx = next_client_idx.fetch_add(1, std::memory_order_relaxed) %
               shared_clients->size();
  return (*shared_clients)[idx];
}

}  // namespace

PYBIND11_MODULE(array_record_module, m) {
  MaybeConfigureResolverOptions();
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
               if (status_or_option->readahead_buffer_size() == 0) {
                 if (getenv("ARRAY_RECORD_CACHE_BLOCK_BYTES") == nullptr &&
                     options.find("cache_block_bytes") == std::string::npos) {
                   status_or_option->set_cache_block_bytes(64ULL << 20);
                 }
                 if (getenv("ARRAY_RECORD_COLD_START_READS") == nullptr &&
                     options.find("cold_start_reads") == std::string::npos) {
                   status_or_option->set_cold_start_reads(24);
                 }
                 if (getenv("ARRAY_RECORD_COLD_START_BLOCK_BYTES") == nullptr &&
                     options.find("cold_start_block_bytes") ==
                         std::string::npos) {
                   status_or_option->set_cold_start_block_bytes(8ULL << 20);
                 }
                 if (getenv("ARRAY_RECORD_MAX_PARALLEL_BLOCK_FETCHES") ==
                         nullptr &&
                     options.find("max_parallel_block_fetches") ==
                         std::string::npos) {
                   status_or_option->set_max_parallel_block_fetches(3);
                 }
                 if (getenv("ARRAY_RECORD_WAIT_FOR_BLOCK_SLOT") == nullptr &&
                     options.find("wait_for_block_slot") == std::string::npos) {
                   status_or_option->set_wait_for_block_slot(false);
                 }
                 if (getenv("ARRAY_RECORD_HOST_MAX_PARALLEL_BLOCK_FETCHES") ==
                         nullptr &&
                     options.find("host_max_parallel_block_fetches") ==
                         std::string::npos) {
                   status_or_option->set_host_max_parallel_block_fetches(16);
                 }
               }
               const auto footer_prefetch_bytes =
                   status_or_option->footer_prefetch_bytes();
               return new array_record::ArrayRecordReader(
                   riegeli::Maker<riegeli::GcsReader>(
                       GetGcsClient(), riegeli::GcsObject(path),
                       std::move(gcs_reader_options),
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
             real with optional suffix [BkKMGTPE]. Defaults to 1MB. Only
             supported with `index_storage_option:in_memory`.

           The default option is optimized for sequential access. To optimize
           the random access performance, set the options to
           "readahead_buffer_size:0,max_parallelism:0".
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
             if (!reader.SeekRecord(record_index)) {
               throw std::runtime_error(std::string(reader.status().message()));
             }
           })
      .def(
          "read_one",
          [](ArrayRecordReaderBase& reader, uint64_t record_index) {
            std::string staging;
            {
              py::gil_scoped_release scoped_release;
              uint64_t idx = record_index;
              auto status = reader.ParallelReadRecordsWithIndices(
                  absl::MakeConstSpan(&idx, 1),
                  [&](uint64_t, absl::string_view record_data) -> absl::Status {
                    staging.assign(record_data.data(), record_data.size());
                    return absl::OkStatus();
                  });
              if (!status.ok()) {
                throw std::runtime_error(std::string(status.message()));
              }
            }
            return py::bytes(staging);
          })
      // See write() for why this returns py::bytes.
      .def("read",
           [](ArrayRecordReaderBase& reader) {
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
             if (indices.size() == 1) {
               std::string staging;
               {
                 py::gil_scoped_release scoped_release;
                 auto status = reader.ParallelReadRecordsWithIndices(
                     indices,
                     [&](uint64_t,
                         absl::string_view record_data) -> absl::Status {
                       staging.assign(record_data.data(), record_data.size());
                       return absl::OkStatus();
                     });
                 if (!status.ok()) {
                   throw std::runtime_error(std::string(status.message()));
                 }
               }
               py::list output(1);
               auto py_record = py::bytes(staging);
               PyList_SET_ITEM(output.ptr(), 0, py_record.release().ptr());
               return output;
             }
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
