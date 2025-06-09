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

// `ArrayRecordReader` reads a riegeli file written by `ArrayRecordWriter`. It
// supports concurrent readahead, random access by record index, and parallel
// read with callbacks.
//
// The concurrency mechanism can be fiber based or thread pool based. Fiber is
// the preferred default to achieve asynchronous IO. However, user may also
// choose to use the old-school thread pool instead. In the OSS version, only
// thread pool is supported.
//
// Features to be implemented in the future:
// TODO(fchern): Supports field projection for columnar read.
//
// Low priority items. Contact us if you need any of the features below.
// TODO(fchern): Recover file/chunk corruption.
// TODO(fchern): Supports append.
//
#ifndef ARRAY_RECORD_CPP_ARRAY_RECORD_READER_H_
#define ARRAY_RECORD_CPP_ARRAY_RECORD_READER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "cpp/common.h"
#include "cpp/tri_state_ptr.h"
#include "cpp/thread_pool.h"
#include "google/protobuf/message_lite.h"
#include "riegeli/base/initializer.h"
#include "riegeli/base/object.h"
#include "riegeli/bytes/reader.h"

namespace array_record {

// template independent part of ArrayRecordReader
class ArrayRecordReaderBase : public riegeli::Object {
 public:
  ~ArrayRecordReaderBase() override;

  class Options {
   public:
    Options() {}

    enum class IndexStorageOption {
      // Keeps all the record/chunk index in memory. Trade-off memory usage for
      // speed.
      kInMemory = 0,
      // Does not keep the index in memory and reads the index from disk for
      // every access. Uses much smaller memory footprint.
      kOffloaded = 1,
    };

    // Parses options from text:
    // ```
    //   options ::= option? ("," option?)*
    //   option ::=
    //     "readahead_buffer_size" ":" readahead_buffer_size |
    //     "max_parallelism" ":" max_parallelism
    //     "index_storage_option" ":" index_storage_option
    //   readahead_buffer_size ::= non-negative integer expressed as real with
    //     optional suffix [BkKMGTPE]. (Default 16MB). Set to 0 optimizes random
    //     access performance.
    //   max_parallelism ::= `auto` or non-negative integer. Each parallel
    //     thread owns its readhaed buffer with the size
    //     `readahead_buffer_size`.  (Default thread pool size) Set to 0
    //     optimizes random access performance.
    //   index_storage_option ::= `in_memory` or `offloaded`. Default to
    //     `in_memory`.
    // ```
    static absl::StatusOr<Options> FromString(absl::string_view text);

    // Readahead speeds up sequential reads, but harms random access. When using
    // ArrayRecord for random access, user should configure the buffer size with
    // 0.
    static constexpr uint64_t kDefaultReadaheadBufferSize = 1L << 24;
    Options& set_readahead_buffer_size(uint64_t readahead_buffer_size) {
      readahead_buffer_size_ = readahead_buffer_size;
      return *this;
    }
    uint64_t readahead_buffer_size() const { return readahead_buffer_size_; }

    // Specifies max number of concurrent readaheads. Setting max_parallelism to
    // 0 disables readaheads prefetching.
    Options& set_max_parallelism(std::optional<uint32_t> max_parallelism) {
      max_parallelism_ = max_parallelism;
      return *this;
    }
    std::optional<uint32_t> max_parallelism() const { return max_parallelism_; }

    // Specifies the index storage option.
    Options& set_index_storage_option(IndexStorageOption storage_option) {
      index_storage_option_ = storage_option;
      return *this;
    }
    IndexStorageOption index_storage_option() const {
      return index_storage_option_;
    }

   private:
    std::optional<uint32_t> max_parallelism_ = std::nullopt;
    uint64_t readahead_buffer_size_ = kDefaultReadaheadBufferSize;
    IndexStorageOption index_storage_option_ = IndexStorageOption::kInMemory;
  };

  // Reads the entire file in parallel and invokes the callback function of
  // signature:
  //
  //   uint64_t record_index, absl::string_view record_data -> absl::Status
  //
  // Return values: status ok for a sucessful read.
  absl::Status ParallelReadRecords(
      absl::FunctionRef<absl::Status(/*record_index=*/uint64_t,
                                     /*record_data=*/absl::string_view)>
          callback) const;

  // Reads the entire file in parallel and invokes the callback function of
  // signature:
  //
  //   uint64_t record_index, UserProtoType user_proto -> absl::Status
  //
  // Example:
  //
  //   auto status = reader.ParallelReadRecords<GenericFeatureVector>(
  //     [&](uint64_t record_index, GenericFeatureVector result_gfv) {
  //         // do our work
  //         return absl::OkStatus();
  //     }))
  //
  // This method is a handy wrapper for processing protobufs stored in
  // ArrayRecord.
  //
  // Return values: status ok for a sucessful read.
  template <typename ProtoT, typename FunctionT,
            typename = std::enable_if_t<
                std::is_base_of_v<google::protobuf::MessageLite, ProtoT>>>
  absl::Status ParallelReadRecords(FunctionT callback) const {
    return ParallelReadRecords(
        [&](uint64_t record_idx, absl::string_view record) -> absl::Status {
          ProtoT record_proto;
          // Like ParseFromString(), but accepts messages that are missing
          // required fields.
          if (!record_proto.ParsePartialFromString(record)) {
            return InternalError("Failed to parse. record_idx: %d", record_idx);
          }
          return callback(record_idx, std::move(record_proto));
        });
  }

  // Reads the records with user supplied indices and invokes the callback
  // function of signature:
  //
  //   uint64_t indices_index, absl::string_view record_data -> absl::Status
  //
  // To obtain the record index, do `indices[indices_index]`.
  //
  // Return values: status ok for a sucessful read.
  absl::Status ParallelReadRecordsWithIndices(
      absl::Span<const uint64_t> indices,
      absl::FunctionRef<absl::Status(/*indices_index=*/uint64_t,
                                     /*record_data=*/absl::string_view)>
          callback) const;

  // Reads the records with user supplied indices and invokes the callback
  // function of signature:
  //
  //   uint64_t indices_index, UserProtoType user_proto -> absl::Status
  //
  // This method is a handy wrapper for processing protobufs stored in
  // ArrayRecord. To obtain the record index, do `indices[indices_index]`.
  //
  // Example:
  //
  //   auto status = reader.ParallelReadRecordsWithIndices<GenericFeatureVector>
  //     ({1, 2, 3},
  //     [&](uint64_t indices_index, GenericFeatureVector result_gfv) {
  //          // do our work
  //          return absl::OkStatus();
  //     });
  //
  // Return values: status ok for a sucessful read.
  template <typename ProtoT, typename FunctionT,
            typename = std::enable_if_t<
                std::is_base_of_v<google::protobuf::MessageLite, ProtoT>>>
  absl::Status ParallelReadRecordsWithIndices(
      absl::Span<const uint64_t> indices, FunctionT callback) const {
    return ParallelReadRecordsWithIndices(
        indices,
        [&](uint64_t indices_idx, absl::string_view record) -> absl::Status {
          ProtoT record_proto;
          // Like ParseFromString(), but accepts messages that are missing
          // required fields.
          if (!record_proto.ParsePartialFromString(record)) {
            return InternalError("Failed to parse. indices_idx: %d",
                                 indices_idx);
          }
          return callback(indices_idx, std::move(record_proto));
        });
  }

  // Reads the records with user supplied range and invokes the callback
  // function of signature:
  //
  //   uint64_t record_index, absl::string_view record_data -> absl::Status
  //
  // The specified `begin` and `end` must within the range of the available
  // records. `begin` is inclusive, and `end` is exclusive.
  //
  // Return values: status ok for a sucessful read.
  absl::Status ParallelReadRecordsInRange(
      uint64_t begin, uint64_t end,
      absl::FunctionRef<absl::Status(/*record_index=*/uint64_t,
                                     /*record_data=*/absl::string_view)>
          callback) const;

  // Reads the records with user supplied range and invokes the callback
  // function of signature:
  //
  //   uint64_t record_index, UserProtoType user_proto -> absl::Status
  //
  // The specified `begin` and `end` must within the range of the available
  // records. `begin` is inclusive, and `end` is exclusive.
  //
  // Return values: status ok for a sucessful read.
  template <typename ProtoT, typename FunctionT,
            typename = std::enable_if_t<
                std::is_base_of_v<google::protobuf::MessageLite, ProtoT>>>
  absl::Status ParallelReadRecordsInRange(uint64_t begin, uint64_t end,
                                          FunctionT callback) const {
    return ParallelReadRecordsInRange(
        begin, end,
        [&](uint64_t recrod_index, absl::string_view record) -> absl::Status {
          ProtoT record_proto;
          // Like ParseFromString(), but accepts messages that are missing
          // required fields.
          if (!record_proto.ParsePartialFromString(record)) {
            return InternalError("Failed to parse. record_index: %d",
                                 recrod_index);
          }
          return callback(recrod_index, std::move(record_proto));
        });
  }

  // Number of records in the opened file.
  uint64_t NumRecords() const;

  // Number of records in each compressed chunk configured at the file writing
  // stage by `ArrayRecordWriterBase::Options::set_group_size`. The acutal
  // number of records per group could be smaller equals to this number.
  uint64_t RecordGroupSize() const;

  // Index of the record to be read.
  uint64_t RecordIndex() const;

  // Seek to a particular record index for the next read. This method discards
  // read ahead buffers, so users should not rely on this function for
  // performance critical random access.
  //
  // For batch lookup, prefer `ParallelReadRecordsWithIndices` over this method.
  //
  // For low latency lookup, load all data with `ParallelReadRecords` into user
  // defined in-memory data structure then perform the lookup.
  bool SeekRecord(uint64_t record_index);

  // Reads the next record `RecordIndex()` pointed to.
  //
  // Return values:
  // `true`  (when `ok()`, `record` is set)  - success
  // `false` (when `ok()`)                   - data ends
  // `false` (when `!ok()`)                  - failure
  bool ReadRecord(google::protobuf::MessageLite* record);

  // Reads the next record `RecordIndex()` pointed to.
  //
  // Return values:
  // `true`  (when `ok()`, `record` is set)  - success
  // `false` (when `ok()`)                   - data ends
  // `false` (when `!ok()`)                  - failure
  bool ReadRecord(absl::string_view* record);

  // Returns the writer options for files produced after 2022-10-10.
  std::optional<std::string> WriterOptionsString() const;

  uint64_t ChunkStartOffset(uint64_t chunk_idx) const;
  uint64_t ChunkEndOffset(uint64_t chunk_idx) const;

 protected:
  explicit ArrayRecordReaderBase(Options options, ARThreadPool* pool);

  // Move only. Closes `other` on move.
  ArrayRecordReaderBase(ArrayRecordReaderBase&& other) noexcept;
  ArrayRecordReaderBase& operator=(ArrayRecordReaderBase&& other) noexcept;

  void Initialize();

  virtual TriStatePtr<riegeli::Reader>::SharedRef get_backing_reader()
      const = 0;

 private:
  bool ReadAheadFromBuffer(uint64_t buffer_idx);

  // Holds all the internal state in a variable to simplify the implementation
  // of the "Close after Move" semantic.
  struct ArrayRecordReaderState;
  friend class OffloadedChunkOffset;
  std::unique_ptr<ArrayRecordReaderState> state_;
};

// `ArrayRecordReader` use templated backend abstraction. To read the
// data from a string, user simply write:
//
//   std::string src = ...;
//   ArrayRecordReader reads_from_string(
//       riegeli::Maker<riegeli::StringReader>(src));
//
// Similarly, user can read the input from a cord or from a file.
//
//   absl::Cord cord = ...;
//   ArrayRecordReader reads_from_cord(
//       riegeli::Maker<riegeli::CordReader>(&cord));
//
//   ArrayRecordReader reads_from_file(
//       riegeli::Maker<riegeli::FileReader>(filename_or_file));
//
// It is necessary to call `Close()` at the end of a successful reading session.
// It is not needed to call `Close()` on early returns, assuming that contents
// of the destination do not matter after all, e.g.  because a failure is being
// reported instead; the destructor releases resources in any case.
//
// Error handling example:
//
//   // Similar to RET_CHECK and RETURN_OR_ERROR
//   if (!reader.ReadRecord(...)) return reader.status();
//   // Must close after use.
//   if (!reader.Close()) return reader.status();
//
// ArrayRecordReader is thread compatible, not thread-safe.
template <typename Src = riegeli::Reader>
class ArrayRecordReader : public ArrayRecordReaderBase {
 public:
  DECLARE_MOVE_ONLY_CLASS(ArrayRecordReader);

  // Will read from the `Reader` provided by `src`.
  explicit ArrayRecordReader(riegeli::Initializer<Src> src,
                             Options options = Options(),
                             ARThreadPool* pool = nullptr)
      : ArrayRecordReaderBase(std::move(options), pool),
        main_reader_(std::make_unique<TriStatePtr<riegeli::Reader>>(
            std::move(src))) {
    Initialize();
  }

 protected:
  TriStatePtr<riegeli::Reader>::SharedRef get_backing_reader() const override {
    return main_reader_->MakeShared();
  }

  void Done() override {
    if (main_reader_ == nullptr) return;
    auto unique = main_reader_->WaitAndMakeUnique();
    if (!unique->Close()) Fail(unique->status());
  }

 private:
  std::unique_ptr<TriStatePtr<riegeli::Reader>> main_reader_;
};

template <typename Src>
explicit ArrayRecordReader(
    Src&& src,
    ArrayRecordReaderBase::Options options = ArrayRecordReaderBase::Options(),
    ARThreadPool* pool = nullptr) -> ArrayRecordReader<riegeli::TargetT<Src>>;

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_ARRAY_RECORD_READER_H_
