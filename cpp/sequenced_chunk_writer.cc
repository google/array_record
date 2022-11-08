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

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "riegeli/base/status.h"
#include "riegeli/base/types.h"
#include "riegeli/chunk_encoding/chunk.h"
#include "riegeli/chunk_encoding/constants.h"

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
  if (block) {
    mu_.Lock();
  } else if (!mu_.TryLock()) {
    return true;
  }
  auto* chunk_writer = get_writer();
  while (!queue_.empty()) {
    if (!block) {
      if (queue_.front().wait_for(std::chrono::microseconds::zero()) !=
          std::future_status::ready) {
        break;
      }
    }
    auto status_or_chunk = queue_.front().get();
    queue_.pop();

    if (!ok() || !chunk_writer->ok()) {
      continue;
    }
    // Set self unhealthy for bad chunks.
    if (!status_or_chunk.ok()) {
      Fail(riegeli::Annotate(
          status_or_chunk.status(),
          absl::StrFormat("Could not submit chunk: %d", submitted_chunks_)));
      continue;
    }
    riegeli::Chunk chunk = std::move(status_or_chunk.value());
    uint64_t chunk_offset = chunk_writer->pos();
    uint64_t decoded_data_size = chunk.header.decoded_data_size();
    uint64_t num_records = chunk.header.num_records();

    if (!chunk_writer->WriteChunk(std::move(chunk))) {
      Fail(riegeli::Annotate(
          chunk_writer->status(),
          absl::StrFormat("Could not submit chunk: %d", submitted_chunks_)));
      continue;
    }
    if (pad_to_block_boundary_) {
      if (!chunk_writer->PadToBlockBoundary()) {
        Fail(riegeli::Annotate(
            chunk_writer->status(),
            absl::StrFormat("Could not pad boundary for chunk: %d",
                            submitted_chunks_)));
        continue;
      }
    }
    if (!chunk_writer->Flush(riegeli::FlushType::kFromObject)) {
      Fail(riegeli::Annotate(
          chunk_writer->status(),
          absl::StrFormat("Could not flush chunk: %d", submitted_chunks_)));
      continue;
    }
    if (callback_) {
      (*callback_)(submitted_chunks_, chunk_offset, decoded_data_size,
                   num_records);
    }
    submitted_chunks_++;
  }
  mu_.Unlock();
  return ok();
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
    Fail(riegeli::Annotate(chunk_writer->status(), "Could not flush"));
  }
}

void SequencedChunkWriterBase::Done() {
  SubmitFutureChunks(true);
  auto* chunk_writer = get_writer();
  if (!chunk_writer->Close()) {
    Fail(riegeli::Annotate(chunk_writer->status(),
                           "Failed to close chunk_writer"));
  }
}

}  // namespace array_record
