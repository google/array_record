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

#include <algorithm>
#include <memory>
#include <optional>
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
#include "riegeli/base/types.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/gcs/gcs_object.h"
#include "riegeli/gcs/gcs_reader.h"

namespace py = pybind11;

namespace {

// A lightweight `riegeli::Reader` for GCS objects whose size and ArrayRecord
// chunk index metadata have already been loaded in memory. Performs zero
// network I/O on construction and delegates range reads to `riegeli::GcsReader`
// via `NewReaderImpl()`.
class LazyGcsReader : public riegeli::Reader {
 public:
  LazyGcsReader(google::cloud::storage::Client client,
                riegeli::GcsObject object, riegeli::GcsReader::Options options,
                uint64_t file_size)
      : client_(std::move(client)),
        object_(std::move(object)),
        options_(std::move(options)),
        file_size_(file_size) {
    if (!object_.ok()) {
      Fail(object_.status());
    }
  }

  bool SupportsRandomAccess() override { return true; }
  bool SupportsNewReader() override { return true; }

 protected:
  bool PullSlow(size_t min_length, size_t recommended_length) override {
    return false;
  }

  bool SeekSlow(riegeli::Position new_pos) override {
    if (!ok() || new_pos > file_size_) {
      return false;
    }
    set_buffer();
    set_limit_pos(new_pos);
    return true;
  }

  std::optional<riegeli::Position> SizeImpl() override { return file_size_; }

  std::unique_ptr<riegeli::Reader> NewReaderImpl(
      riegeli::Position initial_pos) override {
    if (!ok()) {
      return nullptr;
    }
    const riegeli::Position read_from_offset =
        std::min<uint64_t>(initial_pos, file_size_ > 0 ? file_size_ - 1 : 0);
    auto reader = std::make_unique<riegeli::GcsReader>(
        client_, object_, options_,
        google::cloud::storage::ReadFromOffset(read_from_offset));
    if (initial_pos > read_from_offset) {
      if (reader->Pull()) {
        reader->move_cursor(1);
      }
    }
    return reader;
  }

 private:
  google::cloud::storage::Client client_;
  riegeli::GcsObject object_;
  riegeli::GcsReader::Options options_;
  uint64_t file_size_;
};

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
                       py::object index_metadata_arg,
                       const py::kwargs& kwargs) -> ArrayRecordReaderBase* {
             auto status_or_option =
                 ArrayRecordReaderBase::Options::FromString(options);
             if (!status_or_option.ok()) {
               throw py::value_error(
                   std::string(status_or_option.status().message()));
             }
             std::string index_metadata;
             if (!index_metadata_arg.is_none()) {
               index_metadata =
                   std::string(py::cast<py::bytes>(index_metadata_arg));
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
               if (!index_metadata.empty() &&
                   status_or_option->index_storage_option() ==
                       ArrayRecordReaderBase::Options::IndexStorageOption::
                           kInMemory) {
                 uint64_t exported_file_size = 0;
                 ArrayRecordReaderBase::ParseExportedFileSize(
                     index_metadata, &exported_file_size);
                 return new array_record::ArrayRecordReader(
                     riegeli::Maker<LazyGcsReader>(
                         google::cloud::storage::Client(),
                         riegeli::GcsObject(path),
                         std::move(gcs_reader_options), exported_file_size),
                     status_or_option.value(),
                     array_record::ArrayRecordGlobalPool(), index_metadata);
               }
               const auto footer_prefetch_bytes =
                   status_or_option->footer_prefetch_bytes();
               return new array_record::ArrayRecordReader(
                   riegeli::Maker<riegeli::GcsReader>(
                       google::cloud::storage::Client(),
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
                   array_record::ArrayRecordGlobalPool(), index_metadata);
             }
           }),
           py::arg("path"), py::arg("options") = "",
           py::arg("index_metadata") = py::none(), R"(
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
      .def("get_index_metadata",
           [](const ArrayRecordReaderBase& reader) {
             return py::bytes(reader.ExportIndexMetadata());
           })
      .def("seek",
           [](ArrayRecordReaderBase& reader, int64_t record_index) {
             if (!reader.SeekRecord(record_index)) {
               throw std::runtime_error(std::string(reader.status().message()));
             }
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
