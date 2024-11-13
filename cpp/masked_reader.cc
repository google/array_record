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
#include <optional>
#include <utility>

#include "absl/memory/memory.h"
#include "cpp/common.h"
#include "riegeli/base/object.h"
#include "riegeli/base/shared_buffer.h"
#include "riegeli/base/status.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/reader.h"

namespace array_record {

using riegeli::Annotate;
using riegeli::Position;
using riegeli::Reader;

MaskedReader::MaskedReader(Reader& src_reader, size_t length) {
  Initialize(src_reader, length);
}

MaskedReader::MaskedReader(riegeli::SharedBuffer buffer, size_t length,
                           Position limit_pos)
    : buffer_(std::move(buffer)) {
  //           limit_pos
  // |---------------------------|
  //            buffer_start   buffer_limit
  // |................|----------|
  set_buffer(buffer_.data(), length);
  set_limit_pos(limit_pos);
}

void MaskedReader::Reset(riegeli::Closed) {
  Reader::Reset(riegeli::kClosed);
  buffer_.Reset();
}

void MaskedReader::Reset(Reader& src_reader, size_t length) {
  Reader::Reset();
  Initialize(src_reader, length);
}

void MaskedReader::Initialize(Reader& src_reader, size_t length) {
  buffer_.Reset(length);
  if (!src_reader.Read(length, buffer_.mutable_data())) {
    Fail(Annotate(src_reader.status(),
                  "Could not read from the underlying reader"));
    return;
  }
  //           limit_pos
  // |---------------------------|
  //            buffer_start   buffer_limit
  // |................|----------|
  set_buffer(buffer_.data(), length);
  set_limit_pos(src_reader.pos());
}

bool MaskedReader::PullSlow(size_t min_length, size_t recommended_length) {
  Fail(FailedPreconditionError("Should not pull beyond buffer"));
  return false;
}

bool MaskedReader::SeekSlow(Position new_pos) {
  Fail(FailedPreconditionError("Should not seek beyond buffer"));
  return false;
}

std::optional<Position> MaskedReader::SizeImpl() { return limit_pos(); }

std::unique_ptr<Reader> MaskedReader::NewReaderImpl(Position initial_pos) {
  if (!ok()) {
    return nullptr;
  }
  std::unique_ptr<Reader> reader = absl::WrapUnique(
      new MaskedReader(buffer_, start_to_limit(), limit_pos()));
  if (!reader->Seek(initial_pos)) {
    return nullptr;
  }
  return reader;
}

}  // namespace array_record
