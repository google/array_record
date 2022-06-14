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
#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <optional>
#include <queue>
#include <type_traits>
#include <utility>

#include "google/protobuf/message_lite.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "cpp/common.h"
#include "cpp/layout.pb.h"
#include "cpp/thread_compatible_shared_ptr.h"
#include "cpp/thread_pool.h"
#include "riegeli/base/object.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/chunk_encoding/chunk_decoder.h"
#include "riegeli/records/chunk_reader.h"
#include "riegeli/records/record_reader.h"

namespace array_record {

// template independent part of ArrayRecordReader
class ArrayRecordReaderBase : public riegeli::Object {
 public:
  class Options {
   public:
    Options() {}

    // Parses options from text:
    // ```
    //   options ::= option? ("," option?)*
    //   option ::=
    //     "readahead_buffer_size" ":" readahead_buffer_size |
    //     "max_parallelism" ":" max_parallelism
    //   readahead_buffer_size ::= positive integer expressed as real with
    //     optional suffix [BkKMGTPE]. (Default 16MB).
    //   max_parallelism ::= `auto` or positive integer. Each parallel thread
    //     owns its readhaed buffer with the size `readahead_buffer_size`.
    //     (Default thread pool size)
    // ```
    static absl::StatusOr<Options> FromString(absl::string_view text);

    static constexpr uint64_t kDefaultReadaheadBufferSize = 1L << 24;
    Options& set_readahead_buffer_size(uint64_t readahead_buffer_size) {
      readahead_buffer_size_ = readahead_buffer_size;
      return *this;
    }
    uint64_t readahead_buffer_size() const { return readahead_buffer_size_; }

    // Specifies max number of concurrent chunk encoders allowed. Default to the
    // thread pool size.
    Options& set_max_parallelism(std::optional<uint32_t> max_parallelism) {
      max_parallelism_ = max_parallelism;
      return *this;
    }
    std::optional<uint32_t> max_parallelism() const { return max_parallelism_; }

   private:
    std::optional<uint32_t> max_parallelism_ = std::nullopt;
    uint64_t readahead_buffer_size_ = kDefaultReadaheadBufferSize;
  };

  // Reads the entire file in parallel and invokes the callback function of
  // signature:
  //
  //   uint64_t record_index, absl::string_view record_data -> absl::Status
  //
  // Return values:
  // `true`  - Successfully read all the data.
  // `false` - Any of the record/chunk failure would fail the entire read.
  //           In such case the reader is no longer `heathy()` and user may
  //           query `status()` for diagnosing.
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
  //   if(!reader.ParallelReadRecords<GenericFeatureVector>(
  //     [&](uint64_t record_index, GenericFeatureVector result_gfv) {
  //         // do our work
  //         return absl::OkStatus();
  //     }))
  //   return reader.status();
  //
  //
  // This method is a handy wrapper for processing protobufs stored in
  // ArrayRecord.
  //
  // Return values:
  // `true`  - Successfully read all the data.
  // `false` - Any of the record/chunk failure would fail the entire read.
  //           In such case the reader is no longer `heathy()` and user may
  //           query `status()` for diagnosing.
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
  // Return values:
  // `true`  - Successfully read all the data.
  // `false` - Any of the record/chunk failure would fail the entire read.
  //           In such case the reader is no longer `heathy()` and user may
  //           query `status()` for diagnosing.
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
  //   if(!reader.ParallelReadRecordsWithIndices<GenericFeatureVector>(
  //     {1, 2, 3},
  //     [&](uint64_t indices_index, GenericFeatureVector result_gfv) {
  //          // do our work
  //          return absl::OkStatus();
  //     }))
  //   reader.status();
  //
  // Return values:
  // `true`  - Successfully read all the data.
  // `false` - Any of the record/chunk failure would fail the entire read.
  //           In such case the reader is no longer `heathy()` and user may
  //           query `status()` for diagnosing.
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

  // Number of records in the opened file.
  uint64_t NumRecords() const;

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

  // Reads  thenext record `RecordIndex()` pointed to.
  //
  // Return values:
  // `true`  (when `ok()`, `record` is set)  - success
  // `false` (when `ok()`)                   - data ends
  // `false` (when `!ok()`)                  - failure
  bool ReadRecord(absl::string_view* record);

 protected:
  explicit ArrayRecordReaderBase(Options options, ARThreadPool* pool);
  ~ArrayRecordReaderBase() override;

  // Move only. Closes `other` on move.
  ArrayRecordReaderBase(ArrayRecordReaderBase&& other) noexcept;
  ArrayRecordReaderBase& operator=(ArrayRecordReaderBase&& other) noexcept;

  void Initialize();

  virtual ThreadCompatibleSharedPtr<riegeli::Reader> get_backing_reader()
      const = 0;

 private:
  bool ReadAheadFromBuffer(uint64_t buffer_idx);

  // Holds all the internal state in a variable to simplify the implementation
  // of the "Close after Move" semantic.
  struct ArrayRecordReaderState;
  friend class ChunkDispatcher;
  std::unique_ptr<ArrayRecordReaderState> state_;
};

// `ArrayRecordReader` use templated backend abstraction. To read the
// data from a string, user simply write:
//
//   std::string src = ...;
//   auto reads_from_string =
//     ArrayRecordReader<riegeli::StringReader<>>(std::forward_as_tuple(src));
//
// Similarly, user can read the input from a cord or from a file.
//
//   absl::Cord cord = ...;
//   auto reads_from_cord =
//     ArrayRecordReader<riegeli::CordReader<>>(std::forward_as_tuple(cord));
//
//   File* file = ...;
//   auto writes_to_file =
//     ArrayRecordReader<riegeli::FileReader<>>(std::forward_as_tuple(file));
//
// It is necessary to call `Close()` at the end of a successful reading session.
// It is not needed to call `Close()` on early returns, assuming that contents
// of the destination do not matter after all, e.g.  because a failure is being
// reported instead; the destructor releases resources in any case.
//
// Error handling example:
//
//   // Similar to RET_CHECK and RETURN_OR_ERROR
//   if(!reader.ReadRecord(...)) return reader.status();
//   // Must close after use.
//   if(!reader.Close()) return reader.status();
//
// ArrayRecordReader is thread compatible, not thread-safe.
template <typename Src>
class ArrayRecordReader : public ArrayRecordReaderBase {
 public:
  DECLARE_MOVE_ONLY_CLASS(ArrayRecordReader);

  // Constructor that takes the ownership of the other riegeli reader.
  explicit ArrayRecordReader(Src&& src, Options options = Options(),
                             ARThreadPool* pool = nullptr)
      : ArrayRecordReaderBase(std::move(options), pool),
        main_reader_(ThreadCompatibleSharedPtr<riegeli::Reader>::Create(
            std::move(src))) {
    if (!main_reader_->SupportsNewReader()) {
      Fail(InvalidArgumentError(
          "ArrayRecordReader only work on inputs with random access support."));
      return;
    }
    Initialize();
  }

  // Constructor that forwards the argument to the underlying riegeli reader and
  // owns the reader internally till it closes.
  template <typename... SrcArgs>
  explicit ArrayRecordReader(std::tuple<SrcArgs...> src_args,
                             Options options = Options(),
                             ARThreadPool* pool = nullptr)
      : ArrayRecordReaderBase(std::move(options), pool),
        main_reader_(ThreadCompatibleSharedPtr<riegeli::Reader>::Create<Src>(
            std::move(src_args))) {
    if (!main_reader_->SupportsNewReader()) {
      Fail(absl::InvalidArgumentError(
          "ArrayRecordReader only work on inputs with random access support."));
      return;
    }
    Initialize();
  }

 protected:
  ThreadCompatibleSharedPtr<riegeli::Reader> get_backing_reader()
      const override {
    return main_reader_;
  }

  void Done() override {
    if (main_reader_.is_owning()) {
      // Blocks until all detached readahead fetches finishes.
      main_reader_->Close();
    }
  }

 private:
  ThreadCompatibleSharedPtr<riegeli::Reader> main_reader_;
};

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_ARRAY_RECORD_READER_H_
