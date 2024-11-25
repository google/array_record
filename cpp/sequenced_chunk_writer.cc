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

#include "cpp/sequenced_chunk_writer.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "riegeli/base/status.h"
#include "riegeli/base/types.h"
#include "riegeli/chunk_encoding/chunk.h"
#include "riegeli/chunk_encoding/constants.h"
#include "riegeli/records/chunk_writer.h"

namespace array_record {

bool SequencedChunkWriterBase::CommitFutureChunk(
    std::future<absl::StatusOr<riegeli::Chunk>>&& future_chunk) {
  absl::MutexLock l(&mu_);
  if (!ok()) {
    return false;
  }
  queue_.push(std::move(future_chunk));
  return true;
}

bool SequencedChunkWriterBase::SubmitFutureChunks(bool block) {
  // We need to use TryLock to prevent deadlock.
  //
  // std::future::get() blocks if the result wasn't ready.
  // Hence the following scenario triggers a deadlock.
  // T1:
  //   SubmitFutureChunks(true)
  //   mu_ holds
  //   Blocks on queue_.front().get();
  // T2:
  //   In charge to fulfill the future of queue_.front() on its exit.
  //   SubmitFutureChunks(false)
  //   Blocks on mu_ if we used mu_.Lock() instead of mu_.TryLock()
  //
  // NOTE: Even if ok() is false, the below loop will drain queue_, either
  // completely if block is true, or until a non-ready future is at the front of
  // the queue in the non-blocking case. If ok() is false, the front element is
  // popped from the queue and discarded.

  if (block) {
    // When blocking, we block both on mutex acquisition and on future
    // completion.
    absl::MutexLock lock(&mu_);
    riegeli::ChunkWriter* writer = get_writer();
    while (!queue_.empty()) {
      TrySubmitFirstFutureChunk(writer);
    }
    return ok();
  } else if (mu_.TryLock()) {
    // When non-blocking, we only proceed if we can lock the mutex without
    // blocking, and we only process those futures that are ready. We need
    // to unlock the mutex manually in this case, and take care to call ok()
    // under the lock.
    riegeli::ChunkWriter* writer = get_writer();
    while (!queue_.empty() &&
           queue_.front().wait_for(std::chrono::microseconds::zero()) ==
               std::future_status::ready) {
      TrySubmitFirstFutureChunk(writer);
    }
    bool result = ok();
    mu_.Unlock();
    return result;
  } else {
    return true;
  }
}

void SequencedChunkWriterBase::TrySubmitFirstFutureChunk(
    riegeli::ChunkWriter* chunk_writer) {
  auto status_or_chunk = queue_.front().get();
  queue_.pop();

  if (!ok() || !chunk_writer->ok()) {
    // Note (see above): the front of the queue is popped even if we discard it
    // now.
    return;
  }
  // Set self unhealthy for bad chunks.
  if (!status_or_chunk.ok()) {
    Fail(riegeli::Annotate(
        status_or_chunk.status(),
        absl::StrFormat("Could not submit chunk: %d", submitted_chunks_)));
    return;
  }
  riegeli::Chunk chunk = std::move(status_or_chunk.value());
  uint64_t chunk_offset = chunk_writer->pos();
  uint64_t decoded_data_size = chunk.header.decoded_data_size();
  uint64_t num_records = chunk.header.num_records();

  if (!chunk_writer->WriteChunk(std::move(chunk))) {
    Fail(riegeli::Annotate(
        chunk_writer->status(),
        absl::StrFormat("Could not submit chunk: %d", submitted_chunks_)));
    return;
  }
  if (pad_to_block_boundary_) {
    if (!chunk_writer->PadToBlockBoundary()) {
      Fail(riegeli::Annotate(
          chunk_writer->status(),
          absl::StrFormat("Could not pad boundary for chunk: %d",
                          submitted_chunks_)));
      return;
    }
  }
  if (!chunk_writer->Flush(riegeli::FlushType::kFromObject)) {
    Fail(riegeli::Annotate(
        chunk_writer->status(),
        absl::StrFormat("Could not flush chunk: %d", submitted_chunks_)));
    return;
  }
  if (callback_) {
    (*callback_)(submitted_chunks_, chunk_offset, decoded_data_size,
                 num_records);
  }
  submitted_chunks_++;
}

void SequencedChunkWriterBase::Initialize() {
  auto* chunk_writer = get_writer();
  riegeli::Chunk chunk;
  chunk.header = riegeli::ChunkHeader(chunk.data,
                                      riegeli::ChunkType::kFileSignature, 0, 0);
  if (!chunk_writer->WriteChunk(chunk)) {
    Fail(riegeli::Annotate(chunk_writer->status(),
                           "Failed to create the file header"));
  }
  if (!chunk_writer->Flush(riegeli::FlushType::kFromObject)) {
    Fail(riegeli::Annotate(chunk_writer->status(),
                           "Could not flush the file header."));
  }
}

void SequencedChunkWriterBase::Done() {
  if (!SubmitFutureChunks(true)) {
    Fail(absl::InternalError("Unable to submit pending chunks"));
    return;
  }
  auto* chunk_writer = get_writer();
  // if (!chunk_writer->Flush(riegeli::FlushType::kFromObject)) {
  //   Fail(riegeli::Annotate(chunk_writer->status(),
  //                          "Could not flush before close."));
  //   return;
  // }
  if (!chunk_writer->Close()) {
    Fail(riegeli::Annotate(chunk_writer->status(),
                           "Failed to close chunk_writer"));
  }
}

}  // namespace array_record
