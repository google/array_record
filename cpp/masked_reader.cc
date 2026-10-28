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

#include "cpp/masked_reader.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cpp/common.h"
#include "cpp/tri_state_ptr.h"
#include "riegeli/base/object.h"
#include "riegeli/base/status.h"
#include "riegeli/base/types.h"

namespace array_record {

using riegeli::Annotate;
using riegeli::Position;
using riegeli::Reader;

MaskedReader::MaskedReader(
    TriStatePtr<riegeli::Reader>::SharedRef shared_src_reader, size_t length)
    : MaskedReader(&*shared_src_reader, length) {}

MaskedReader::MaskedReader(std::unique_ptr<riegeli::Reader> src_reader,
                           size_t length)
    : MaskedReader(src_reader.get(), length) {}

MaskedReader::MaskedReader(riegeli::Reader* src_reader, size_t length) {
  if (src_reader == nullptr) {
    Fail(InvalidArgumentError("Could not create reader at the given offset"));
    return;
  }
  auto pos = src_reader->pos();
  auto buffer = std::make_shared<std::string>();
  buffer->resize(length);
  if (!src_reader->Read(length, buffer->data())) {
    // A source that ends early is still ok(); Annotate(OkStatus) is OK and
    // Fail() requires a non-OK status.
    Fail(src_reader->ok()
             ? InvalidArgumentError("Could not read from the underlying reader")
             : Annotate(src_reader->status(),
                        "Could not read from the underlying reader"));
    return;
  }
  buffer_ = std::move(buffer);
  /*
   *           limit_pos
   * |---------------------------|
   *            buffer_start   buffer_limit
   * |................|----------|
   */
  set_buffer(buffer_->data(), buffer_->size());
  set_limit_pos(pos + buffer_->size());
}

MaskedReader::MaskedReader(std::shared_ptr<const std::string> block,
                           Position block_start)
    : buffer_(std::move(block)) {
  if (buffer_ == nullptr) {
    Fail(InvalidArgumentError("MaskedReader block must not be null"));
    return;
  }
  set_buffer(buffer_->data(), buffer_->size());
  set_limit_pos(block_start + buffer_->size());
}

MaskedReader::MaskedReader(MaskedReader&& other) noexcept
    : Reader(std::move(other)) {
  buffer_ = other.buffer_;        // NOLINT(bugprone-use-after-move)
  other.Reset(riegeli::kClosed);  // NOLINT(bugprone-use-after-move)
}

MaskedReader& MaskedReader::operator=(MaskedReader&& other) noexcept {
  // Move other
  Reader::operator=(static_cast<riegeli::Reader&&>(other));
  // Copy the shared buffer.
  buffer_ = other.buffer_;
  // Close `other`
  other.Reset(riegeli::kClosed);
  return *this;
}

bool MaskedReader::PullSlow(size_t min_length, size_t recommended_length) {
  Fail(FailedPreconditionError("Should not pull beyond buffer"));
  return false;
}

bool MaskedReader::SeekSlow(riegeli::Position new_pos) {
  Fail(FailedPreconditionError("Should not seek beyond buffer"));
  return false;
}

absl::optional<riegeli::Position> MaskedReader::SizeImpl() {
  return limit_pos();
}

std::unique_ptr<Reader> MaskedReader::NewReaderImpl(Position initial_pos) {
  if (!ok()) {
    return nullptr;
  }
  std::unique_ptr<Reader> reader = absl::WrapUnique(
      new MaskedReader(buffer_, limit_pos() - buffer_->size()));
  if (!reader->Seek(initial_pos)) {
    return nullptr;
  }
  return reader;
}

}  // namespace array_record
