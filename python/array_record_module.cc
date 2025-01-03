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

#include <stdexcept>
#include <string>
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
#include "riegeli/base/maker.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"

namespace py = pybind11;

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
             // Release the GIL because IO is time consuming.
             py::gil_scoped_release scoped_release;
             return new array_record::ArrayRecordWriter(
                 riegeli::Maker<riegeli::FdWriter>(path),
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
             if (kwargs.contains("file_reader_buffer_size")) {
               auto file_reader_buffer_size =
                   kwargs["file_reader_buffer_size"].cast<int64_t>();
               file_reader_options.set_buffer_size(file_reader_buffer_size);
             }
             // Release the GIL because IO is time consuming.
             py::gil_scoped_release scoped_release;
             return new array_record::ArrayRecordReader(
                 riegeli::Maker<riegeli::FdReader>(path, file_reader_options),
                 status_or_option.value(),
                 array_record::ArrayRecordGlobalPool());
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
