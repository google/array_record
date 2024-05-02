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

// Low-level API for building off-memory data structures with riegeli.
//
// `SequencedChunkWriter` writes chunks of records to an abstracted destination.
// This class abstract out the generic  chunk writing logic from concrete logic
// that builds up the data structures for future access.  This class is
// thread-safe and allows users to encode each chunk concurrently while
// maintaining the sequence order of the chunks.

#ifndef ARRAY_RECORD_CPP_SEQUENCED_CHUNK_WRITER_H_
#define ARRAY_RECORD_CPP_SEQUENCED_CHUNK_WRITER_H_

#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <queue>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "cpp/common.h"
#include "riegeli/base/initializer.h"
#include "riegeli/base/object.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/chunk_encoding/chunk.h"
#include "riegeli/records/chunk_writer.h"

namespace array_record {

// Template parameter independent part of `SequencedChunkWriter`.
class SequencedChunkWriterBase : public riegeli::Object {
  SequencedChunkWriterBase(const SequencedChunkWriterBase&) = delete;
  SequencedChunkWriterBase& operator=(const SequencedChunkWriterBase&) = delete;
  SequencedChunkWriterBase(SequencedChunkWriterBase&&) = delete;
  SequencedChunkWriterBase& operator=(SequencedChunkWriterBase&&) = delete;

 public:
  // The `SequencedChunkWriter` invokes `SubmitChunkCallback` for every
  // successful chunk writing. In other words, the invocation only happens when
  // none of the errors occur (SequencedChunkWriter internal state, chunk
  // correctness, underlying writer state, etc.) The callback consists of four
  // arguments:
  //
  // chunk_seq: sequence number of the chunk in the file. Indexed from 0.
  // chunk_offset: byte offset of the chunk in the file. A reader can seek this
  //     offset and decode the chunk without other information.
  // decoded_data_size: byte size of the decoded data. Users may serialize this
  //     field for readers to allocate memory for the decoded data.
  // num_records: number of records in the chunk.
  class SubmitChunkCallback {
   public:
    virtual ~SubmitChunkCallback() {}
    virtual void operator()(uint64_t chunk_seq, uint64_t chunk_offset,
                            uint64_t decoded_data_size,
                            uint64_t num_records) = 0;
  };

  // Commits a future chunk to the `SequencedChunkWriter` before materializing
  // the chunk. Users can encode the chunk in a separated thread at the cost of
  // larger temporal memory usage. `SequencedChunkWriter` serializes the chunks
  // at the order of this function call.
  //
  // Example 1: packaged_task
  //
  //     std::packaged_task<absl::StatusOr<riegeli::Chunk>()> encoding_task(
  //       []() -> absl::StatusOr<riegeli::Chunk> {
  //         ... returns a riegeli::Chunk on success.
  //       });
  //     std::future<absl::StatusOr<riegeli::Chunk>> task_future =
  //       encoding_task.get();
  //     sequenced_chunk_writer->CommitFutureChunk(std::move(task_future));
  //
  //     // Computes the encoding task in a thread pool.
  //     pool->Schedule(std::move(encoding_task));
  //
  // Example 2: promise and future
  //
  //     std::promise<absl::StatusOr<riegeli::Chunk>> chunk_promise;
  //     RET_CHECK(sequenced_chunk_writer->CommitFutureChunk(
  //         chunk_promise.get_future())) << sequenced_chunk_writer->status();
  //     pool->Schedule([chunk_promise = std::move(chunk_promise)] {
  //       // computes chunk
  //       chunk_promise.set_value(status_or_chunk);
  //     });
  //
  // Although `SequencedChunkWriter` is thread-safe, this method should be
  // invoked from a single thread because it doesn't make sense to submit future
  // chunks without a proper order.
  bool CommitFutureChunk(
      std::future<absl::StatusOr<riegeli::Chunk>>&& future_chunk);

  // Extracts the future chunks and submits them to the underlying destination.
  // This operation may block if the argument `block` was true. This method is
  // thread-safe, and we recommend users invoke it with `block=false` in each
  // thread to reduce the temporal memory usage.
  //
  // If ok() is false before or during this operation, queue elements continue
  // to be extracted, but are immediately discarded.
  //
  // Example 1: single thread usage
  //
  //     std::promise<absl::StatusOr<riegeli::Chunk>> chunk_promise;
  //     RET_CHECK(sequenced_chunk_writer->CommitFutureChunk(
  //         chunk_promise.get_future())) << sequenced_chunk_writer->status();
  //     chunk_promise.set_value(ComputesChunk());
  //     RET_CHECK(writer->SubmitFutureChunks(true)) << writer->status();
  //
  // Example 2: concurrent access
  //
  //     auto writer = std::make_shared<SequencedChunkWriter<>(...)
  //
  //     pool->Schedule([writer,
  //                     chunk_promise = std::move(chunk_promise)]()  mutable {
  //       chunk_promise.set_value(status_or_chunk);
  //       // Should not block otherwise would enter deadlock!
  //       writer->SubmitFutureChunks(false);
  //     });
  //     // Blocking the main thread is fine.
  //     RET_CHECK(writer->SubmitFutureChunks(true)) << writer->status();
  //
  bool SubmitFutureChunks(bool block = false);

  // Pads to 64KB boundary for future chunk submission. (Default false).
  void set_pad_to_block_boundary(bool pad_to_block_boundary) {
    absl::MutexLock l(&mu_);
    pad_to_block_boundary_ = pad_to_block_boundary;
  }
  bool pad_to_block_boundary() {
    absl::MutexLock l(&mu_);
    return pad_to_block_boundary_;
  }

  // Setup a callback for each committed chunk. See CommitChunkCallback
  // comments for details.
  void set_submit_chunk_callback(SubmitChunkCallback* callback) {
    absl::MutexLock l(&mu_);
    callback_ = callback;
  }

  // Guard the status access.
  absl::Status status() const {
    absl::ReaderMutexLock l(&mu_);
    return riegeli::Object::status();
  }

 protected:
  SequencedChunkWriterBase() {}
  virtual riegeli::ChunkWriter* get_writer() = 0;

  // Initializes and validates the underlying writer states.
  void Initialize();

  // Callback for riegeli::Object::Close.
  void Done() override;

 private:
  // Attempts to submit the first chunk from the queue. Expects that the lock is
  // already held. Even if ok() is false on entry already, the queue element is
  // removed (and discarded).
  void TrySubmitFirstFutureChunk(riegeli::ChunkWriter* chunk_writer)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  bool pad_to_block_boundary_ ABSL_GUARDED_BY(mu_) = false;
  SubmitChunkCallback* callback_ ABSL_GUARDED_BY(mu_) = nullptr;

  // Records the sequence number of submitted chunks.
  uint64_t submitted_chunks_ ABSL_GUARDED_BY(mu_) = 0;

  // Queue for storing the future chunks.
  std::queue<std::future<absl::StatusOr<riegeli::Chunk>>> queue_
      ABSL_GUARDED_BY(mu_);
};

// A `SequencedChunkWriter` writes chunks (a blob of multiple and possibly
// compressed records) rather than individual records to an abstracted
// destination. `SequencedChunkWriter` allows users to encode each chunk
// concurrently while keeping the chunk sequence order as the input order.
//
// Users can also supply a `CommitChunkCallback` to collect chunk sequence
// numbers, offsets in the file, decoded data size, and the number of records in
// each chunk. Users may use the callback information to produce a lookup table
// in the footer for an efficient reader to decode multiple chunks in parallel.
//
// Example usage:
//
//   // Step 1: open the writer with file backend.
//   File* file = file::OpenOrDie(...);
//   auto writer = std::make_shared<SequencedChunkWriter<riegeli::FileWriter<>>(
//       riegeli::Maker(filename_or_file));
//
//   // Step 2: create a chunk encoding task.
//   std::packaged_task<absl::StatusOr<riegeli::Chunk>()> encoding_task(
//     []() -> absl::StatusOr<riegeli::Chunk> {
//       ... returns a riegeli::Chunk on success.
//     });
//
//   // Step 3: book a slot for writing the encoded chunk.
//   RET_CHECK(writer->CommitFutureChunk(
//       encoding_task.get_future())) << writer->status();
//
//   // Step 4: Computes the encoding task in a thread pool.
//   pool->Schedule([=,encoding_task=std::move(encoding_task)]() mutable {
//     encoding_task();  // std::promise fulfilled.
//     // shared_ptr pevents the writer to go out of scope, so it is safe to
//     // invoke the method here.
//     writer->SubmitFutureChunks(false);
//   });
//
//   // Repeats step 2 to 4.
//
//   // Finally, close the writer.
//   RET_CHECK(writer->Close()) << writer->status();
//
//
// It is necessary to call `Close()` at the end of a successful writing session,
// and it is recommended to call `Close()` at the end of a successful reading
// session. It is not needed to call `Close()` on early returns, assuming that
// contents of the destination do not matter after all, e.g.  because a failure
// is being reported instead; the destructor releases resources in any case.
//
// `SequencedChunkWriter` inherits riegeli::Object which provides useful
// abstractions for state management of IO-like operations. Instead of the
// common absl::Status/StatusOr for each method, the riegeli::Object's error
// handling mechanism uses bool and separated `status()`, `ok()`, `is_open()`
// for users to handle different types of failure states.
//
//
// `SequencedChunkWriter` use templated backend abstraction. To serialize the
// output to a string, user simply write:
//
//   std::string dest;
//   SequencedChunkWriter writes_to_string(
//       riegeli::Maker<riegeli::StringWriter>(&dest));
//
// Similarly, user can write the output to a cord or to a file.
//
//   absl::Cord cord;
//   SequencedChunkWriter writes_to_cord(
//       riegeli::Maker<riegeli::CordWriter>(&cord));
//
//   SequencedChunkWriter writes_to_file(
//       riegeli::Maker<riegeli::FileWriter>(filename_or_file));
//
// User may also use std::make_shared<...> or std::make_unique to construct the
// instance, as shown in the previous example.
template <typename Dest = riegeli::Writer*>
class SequencedChunkWriter : public SequencedChunkWriterBase {
 public:
  DECLARE_IMMOBILE_CLASS(SequencedChunkWriter);

  // Will write to the `Writer` provided by `dest`.
  explicit SequencedChunkWriter(riegeli::Initializer<Dest> dest)
      : dest_(std::move(dest)) {
    Initialize();
  }

 protected:
  riegeli::ChunkWriter* get_writer() final { return &dest_; }

 private:
  riegeli::DefaultChunkWriter<Dest> dest_;
};

template <typename Dest>
explicit SequencedChunkWriter(Dest&& dest)
    -> SequencedChunkWriter<riegeli::InitializerTargetT<Dest>>;

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_SEQUENCED_CHUNK_WRITER_H_
