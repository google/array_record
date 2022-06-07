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

#include "third_party/array_record/cpp/masked_reader.h"

#include <memory>
#include <string>
#include <utility>

#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"
#include "third_party/array_record/cpp/common.h"
#include "third_party/riegeli/base/base.h"
#include "third_party/riegeli/base/status.h"

namespace array_record {

using riegeli::Annotate;
using riegeli::Position;
using riegeli::Reader;

MaskedReader::MaskedReader(std::unique_ptr<riegeli::Reader> src_reader,
                           size_t length)
    : buffer_(std::make_shared<std::string>()) {
  auto pos = src_reader->pos();
  buffer_->resize(length);
  if (!src_reader->Read(length, buffer_->data())) {
    Fail(Annotate(src_reader->status(),
                  "Could not read from the underlying reader"));
    return;
  }
  /*
   *           limit_pos
   * |---------------------------|
   *            buffer_start   buffer_limit
   * |................|----------|
   */
  set_buffer(buffer_->data(), buffer_->size());
  set_limit_pos(pos + buffer_->size());
}

MaskedReader::MaskedReader(std::shared_ptr<std::string> buffer,
                           Position limit_pos)
    : buffer_(buffer) {
  /*
   *           limit_pos
   * |---------------------------|
   *            buffer_start   buffer_limit
   * |................|----------|
   */
  set_buffer(buffer_->data(), buffer_->size());
  set_limit_pos(limit_pos);
}

MaskedReader::MaskedReader(MaskedReader &&other) noexcept
    : Reader(std::move(other)) {
  buffer_ = other.buffer_;        // NOLINT(bugprone-use-after-move)
  other.Reset(riegeli::kClosed);  // NOLINT(bugprone-use-after-move)
}

MaskedReader &MaskedReader::operator=(MaskedReader &&other) noexcept {
  // Move other
  Reader::operator=(static_cast<riegeli::Reader &&>(other));
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
  std::unique_ptr<Reader> reader =
      absl::WrapUnique(new MaskedReader(buffer_, limit_pos()));
  if (!reader->Seek(initial_pos)) {
    return nullptr;
  }
  return reader;
}

}  // namespace array_record
