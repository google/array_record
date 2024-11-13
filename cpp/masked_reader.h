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

#ifndef ARRAY_RECORD_CPP_MASKED_READER_H_
#define ARRAY_RECORD_CPP_MASKED_READER_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "riegeli/base/object.h"
#include "riegeli/base/shared_buffer.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/reader.h"

namespace array_record {

// `MaskedReader` is a riegeli reader constructed from the other reader with a
// cached buffer over a specified region. User can further derive new readers
// that share the same buffer.
//
// original file |----------------------------------------------|
//
// masked buffer |..............|---------------|
//               |--------------------^ Position is addressed from the
//                                      beginning of the underlying buffer.
//
// `MaskedBuffer` and the original file uses the same position base address.
// User cannot seek to the region beyond the buffer region.
//
// This class is useful for reducing the number of PReads. User may create a
// MaskedReader containing multiple chunks, then derive multiple chunk readers
// from this reader sharing the same buffer. Hence, there's only one PRead
// issued for multiple chunks.
class MaskedReader : public riegeli::Reader {
 public:
  explicit MaskedReader(riegeli::Closed) : riegeli::Reader(riegeli::kClosed) {}

  MaskedReader(riegeli::Reader& src_reader, size_t length);

  MaskedReader(MaskedReader&& other) = default;
  MaskedReader& operator=(MaskedReader&& other) = default;

  void Reset(riegeli::Closed);
  void Reset(riegeli::Reader& src_reader, size_t length);

  bool SupportsRandomAccess() override { return true; }
  bool SupportsNewReader() override { return true; }

 protected:
  bool PullSlow(size_t min_length, size_t recommended_length) override;
  bool SeekSlow(riegeli::Position new_pos) override;

  std::optional<riegeli::Position> SizeImpl() override;
  std::unique_ptr<riegeli::Reader> NewReaderImpl(
      riegeli::Position initial_pos) override;

 private:
  // Private constructor that copies itself.
  MaskedReader(riegeli::SharedBuffer buffer, size_t length,
               riegeli::Position limit_pos);

  void Initialize(riegeli::Reader& src_reader, size_t length);

  riegeli::SharedBuffer buffer_;
};

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_MASKED_READER_H_
