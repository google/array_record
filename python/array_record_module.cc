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

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "cpp/array_record_reader.h"
#include "cpp/array_record_writer.h"
#include "cpp/thread_pool.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"

namespace py = pybind11;

PYBIND11_MODULE(array_record_module, m) {
  using ArrayRecordWriter =
      array_record::ArrayRecordWriter<std::unique_ptr<riegeli::Writer>>;
  using ArrayRecordReader =
      array_record::ArrayRecordReader<std::unique_ptr<riegeli::Reader>>;

  py::class_<ArrayRecordWriter>(m, "ArrayRecordWriter")
      .def(py::init([](const std::string& path, const std::string& options) {
             auto status_or_option =
                 array_record::ArrayRecordWriterBase::Options::FromString(
                     options);
             if (!status_or_option.ok()) {
               throw py::value_error(
                   std::string(status_or_option.status().message()));
             }

             std::unique_ptr<riegeli::FdWriter<>> file_writer;
             {
               py::gil_scoped_release scoped_release;
               file_writer = std::make_unique<riegeli::FdWriter<>>(path);
             }

             if (!file_writer->ok()) {
               throw std::runtime_error(
                   std::string(file_writer->status().message()));
             }
             py::gil_scoped_release scoped_release;
             return ArrayRecordWriter(std::move(file_writer),
                                      status_or_option.value());
           }),
           py::arg("path"), py::arg("options") = "")
      .def("ok", &ArrayRecordWriter::ok)
      .def("close",
           [](ArrayRecordWriter& writer) {
             if (!writer.Close()) {
               throw std::runtime_error(std::string(writer.status().message()));
             }
           })
      .def("is_open", &ArrayRecordWriter::is_open)
      // We accept only py::bytes (and not unicode strings) since we expect
      // most users to store binary data (e.g. serialized protocol buffers).
      // We cannot know if a users wants to write+read unicode and forcing users
      // to encode() their unicode strings avoids accidental conversions.
      .def("write", [](ArrayRecordWriter& writer, py::bytes record) {
        if (!writer.WriteRecord(record)) {
          throw std::runtime_error(std::string(writer.status().message()));
        }
      });
  py::class_<ArrayRecordReader>(m, "ArrayRecordReader")
        .def(py::init([](const std::string& path, const std::string& options,
                       const std::optional<int64_t> file_reader_buffer_size) {
             auto status_or_option =
                 array_record::ArrayRecordReaderBase::Options::FromString(
                     options);
             if (!status_or_option.ok()) {
               throw py::value_error(
                   std::string(status_or_option.status().message()));
             }

             std::unique_ptr<riegeli::FdReader<>> file_reader;
             riegeli::FdReaderBase::Options file_reader_options;
             {
               py::gil_scoped_release scoped_release;
               if (file_reader_buffer_size.has_value()) {
                 file_reader_options.set_buffer_size(*file_reader_buffer_size);
               }
               file_reader = std::make_unique<riegeli::FdReader<>>(
                   path, file_reader_options);
             }
             if (!file_reader->ok()) {
               throw std::runtime_error(
                   std::string(file_reader->status().message()));
             }
             py::gil_scoped_release scoped_release;
             return ArrayRecordReader(std::move(file_reader),
                                      status_or_option.value(),
                                      array_record::ArrayRecordGlobalPool());
           }),
           py::arg("path"), py::arg("options") = "",
           py::arg("file_reader_buffer_size") = std::nullopt, R"(
           ArrayRecordReader for fast sequential or random access.

           Args:
               path: File path to the input file.
               options: String with options for ArrayRecord. See syntax below.
               file_reader_buffer_size: Optional size of the buffer (in bytes)
                 for the underlying file (Riegeli) reader. The default buffer
                 size is 1 MiB.

           options ::= option? ("," option?)*
           option ::=
             "readahead_buffer_size" ":" readahead_buffer_size |
             "max_parallelism" ":" max_parallelism
           readahead_buffer_size ::= non-negative integer expressed as real with
             optional suffix [BkKMGTPE]. (Default 16MB). Set to 0 optimizes
             random access performance.
           max_parallelism ::= `auto` or non-negative integer. Each parallel
             thread owns its readhaed buffer with the size
             `readahead_buffer_size`.  (Default thread pool size) Set to 0
             optimizes random access performance.

           The default option is optimized for sequential access. To optimize
           the random access performance, set the options to
           "readahead_buffer_size:0,max_parallelism:0".
           )")
      .def("ok", &ArrayRecordReader::ok)
      .def("close",
           [](ArrayRecordReader& reader) {
             if (!reader.Close()) {
               throw std::runtime_error(std::string(reader.status().message()));
             }
           })
      .def("is_open", &ArrayRecordReader::is_open)
      .def("num_records", &ArrayRecordReader::NumRecords)
      .def("record_index", &ArrayRecordReader::RecordIndex)
      .def("writer_options_string", &ArrayRecordReader::WriterOptionsString)
      .def("seek",
           [](ArrayRecordReader& reader, int64_t record_index) {
             if (!reader.SeekRecord(record_index)) {
               throw std::runtime_error(std::string(reader.status().message()));
             }
           })
      // See write() for why this returns py::bytes.
      .def("read",
           [](ArrayRecordReader& reader) {
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
           [](ArrayRecordReader& reader, std::vector<uint64_t> indices) {
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
             ssize_t index = 0;
             for (const auto& record : staging) {
               auto py_record = py::bytes(record);
               PyList_SET_ITEM(output.ptr(), index++,
                               py_record.release().ptr());
             }
             return output;
           })
      .def("read",
           [](ArrayRecordReader& reader, int32_t begin, int32_t end) {
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
             ssize_t index = 0;
             for (const auto& record : staging) {
               auto py_record = py::bytes(record);
               PyList_SET_ITEM(output.ptr(), index++,
                               py_record.release().ptr());
             }
             return output;
           })
      .def("read_all", [](ArrayRecordReader& reader) {
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
        ssize_t index = 0;
        for (const auto& record : staging) {
          auto py_record = py::bytes(record);
          PyList_SET_ITEM(output.ptr(), index++, py_record.release().ptr());
        }
        return output;
      });
}
