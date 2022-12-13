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

#include "cpp/array_record_reader.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/bind_front.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "cpp/common.h"
#include "cpp/layout.pb.h"
#include "cpp/masked_reader.h"
#include "cpp/parallel_for.h"
#include "cpp/thread_compatible_shared_ptr.h"
#include "riegeli/base/object.h"
#include "riegeli/base/options_parser.h"
#include "riegeli/base/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/chunk_encoding/chunk_decoder.h"
#include "riegeli/records/chunk_reader.h"
#include "riegeli/records/record_position.h"

namespace array_record {

// 64KB
constexpr size_t kRiegeliBlockSize = (1 << 16);

// This number should rarely change unless there's a new great layout design
// that wasn't backward compatible and justifies its performance and reliability
// worth us to implement.
constexpr uint32_t kArrayRecordV1 = 1;

// Magic number for ArrayRecord
constexpr uint64_t kMagic = 0x71930e704fdae05eULL;

using riegeli::Annotate;
using riegeli::Chunk;
using riegeli::ChunkDecoder;
using riegeli::ChunkReader;
using riegeli::OptionsParser;
using riegeli::Reader;
using riegeli::ValueParser;

template <class T>
T CeilOfRatio(T x, T d) {
  return (x + d - 1) / d;
}

absl::StatusOr<ArrayRecordReaderBase::Options>
ArrayRecordReaderBase::Options::FromString(absl::string_view text) {
  ArrayRecordReaderBase::Options options;
  OptionsParser options_parser;
  // Parallelism
  int32_t max_parallelism = -1;
  options_parser.AddOption(
      "max_parallelism",
      ValueParser::Or(ValueParser::Enum({{"auto", std::nullopt}},
                                        &options.max_parallelism_),
                      ValueParser::Int(0, INT32_MAX, &max_parallelism)));
  // ReadaheadBuffer
  options_parser.AddOption(
      "readahead_buffer_size",
      ValueParser::Or(
          ValueParser::Enum({{"auto", kDefaultReadaheadBufferSize}},
                            &options.readahead_buffer_size_),
          ValueParser::Bytes(0, std::numeric_limits<uint64_t>::max(),
                             &options.readahead_buffer_size_)));
  if (!options_parser.FromString(text)) {
    return options_parser.status();
  }
  if (max_parallelism >= 0) {
    options.set_max_parallelism(max_parallelism);
  }
  return options;
}

template <typename T>
using IndexedPair = std::pair<uint64_t, T>;

struct ArrayRecordReaderBase::ArrayRecordReaderState {
  ArrayRecordReaderState(const Options options, ARThreadPool* pool)
      : options(options), pool(pool) {}
  Options options;
  ARThreadPool* pool;
  uint64_t num_records = 0;
  uint64_t record_group_size = 0;
  uint64_t chunk_group_size = 0;
  uint64_t footer_offset = 0;
  std::vector<ArrayRecordFooter> footer;

  // Objects for managing sequential reads
  uint64_t record_idx = 0;
  uint64_t buffer_idx = UINT64_MAX;
  std::vector<ChunkDecoder> current_decoders;
  std::queue<IndexedPair<std::future<std::vector<ChunkDecoder>>>>
      future_decoders;

  // Writer options for debugging purposes.
  std::optional<std::string> writer_options = std::nullopt;

  uint64_t ChunkEndOffset(uint64_t chunk_idx) const {
    if (chunk_idx == footer.size() - 1) {
      return footer_offset;
    }
    return footer[chunk_idx + 1].chunk_offset();
  }
};

ArrayRecordReaderBase::ArrayRecordReaderBase(Options options,
                                             ARThreadPool* pool)
    : state_(std::make_unique<ArrayRecordReaderState>(options, pool)) {}

ArrayRecordReaderBase::~ArrayRecordReaderBase() = default;

ArrayRecordReaderBase::ArrayRecordReaderBase(
    ArrayRecordReaderBase&& other) noexcept
    : riegeli::Object(std::move(other)), state_(std::move(other.state_)) {
  other.Reset(riegeli::kClosed);  // NOLINT(bugprone-use-after-move)
}

ArrayRecordReaderBase& ArrayRecordReaderBase::operator=(
    ArrayRecordReaderBase&& other) noexcept {
  // Move base
  riegeli::Object::operator=(static_cast<riegeli::Object&&>(other));
  // Move self
  state_ = std::move(other.state_);
  // Close
  other.Reset(riegeli::kClosed);
  return *this;
}

// After the first access to the underlying `riegeli::Reader`, the lazily
// evaluated variables for random access are all initialized. Therefore it's
// safe to access the reader from multiple threads later on, even though the
// methods wasn't const.
ChunkDecoder ReadChunk(const ThreadCompatibleSharedPtr<Reader>& reader,
                       size_t pos, size_t len) {
  ChunkDecoder decoder;
  if (!reader->ok()) {
    decoder.Fail(reader->status());
    return decoder;
  }
  Reader* mutable_reader =
      const_cast<Reader*>(reinterpret_cast<const Reader*>(reader.get()));
  MaskedReader masked_reader(mutable_reader->NewReader(pos), len);
  if (!masked_reader.ok()) {
    decoder.Fail(masked_reader.status());
    return decoder;
  }
  auto chunk_reader = riegeli::DefaultChunkReader<>(&masked_reader);
  Chunk chunk;
  if (!chunk_reader.ReadChunk(chunk)) {
    decoder.Fail(chunk_reader.status());
    return decoder;
  }
  decoder.Decode(chunk);
  return decoder;
}

void ArrayRecordReaderBase::Initialize() {
  if (!ok()) {
    return;
  }
  uint32_t max_parallelism = 1;
  if (state_->pool) {
    max_parallelism = state_->pool->NumThreads();
    if (state_->options.max_parallelism().has_value()) {
      max_parallelism =
          std::min(max_parallelism, state_->options.max_parallelism().value());
    }
  }
  state_->options.set_max_parallelism(max_parallelism);

  AR_ENDO_TASK("Reading ArrayRecord footer");
  const auto reader = get_backing_reader();
  Reader* mutable_reader =
      const_cast<Reader*>(reinterpret_cast<const Reader*>(reader.get()));
  RiegeliFooterMetadata footer_metadata;
  ChunkDecoder footer_decoder;
  {
    AR_ENDO_SCOPE("Reading postscript and footer chunk");
    if (!mutable_reader->SupportsRandomAccess()) {
      Fail(InvalidArgumentError(
          "ArrayRecordReader only work on inputs with random access support."));
      return;
    }
    auto maybe_size = mutable_reader->Size();
    if (!maybe_size.has_value()) {
      Fail(InvalidArgumentError("Could not obtain the size of the input"));
      return;
    }
    auto size = maybe_size.value();
    if (size < kRiegeliBlockSize) {
      Fail(
          InvalidArgumentError("ArrayRecord file should be at least 64KB big"));
      return;
    }
    RiegeliPostscript postscript;
    auto postscript_decoder =
        ReadChunk(reader, size - kRiegeliBlockSize, kRiegeliBlockSize);
    if (!postscript_decoder.ReadRecord(postscript)) {
      Fail(Annotate(postscript_decoder.status(),
                    "Failed to read RiegeliPostscript"));
      return;
    }
    if (!postscript.has_footer_offset()) {
      Fail(InvalidArgumentError("Invalid postscript %s",
                                postscript.DebugString()));
      return;
    }
    if (!postscript.has_magic() || postscript.magic() != kMagic) {
      Fail(InvalidArgumentError("Invalid postscript %s",
                                postscript.DebugString()));
      return;
    }
    state_->footer_offset = postscript.footer_offset();
    footer_decoder =
        ReadChunk(reader, postscript.footer_offset(),
                  size - kRiegeliBlockSize - postscript.footer_offset());

    if (!footer_decoder.ReadRecord(footer_metadata)) {
      Fail(Annotate(footer_decoder.status(),
                    "Failed to read RiegeliFooterMetadata"));
      return;
    }
    if (!footer_metadata.has_array_record_metadata()) {
      Fail(InvalidArgumentError(
          "Could not parse footer as ArrayRecord file. Footer metadata: %s",
          footer_metadata.DebugString()));
      return;
    }
    if (footer_metadata.array_record_metadata().version() != kArrayRecordV1) {
      Fail(InvalidArgumentError(
          "Unrecognized version number. Footer metadata: %s",
          footer_metadata.DebugString()));
      return;
    }
    state_->num_records = footer_metadata.array_record_metadata().num_records();
    if (footer_metadata.array_record_metadata().has_writer_options()) {
      state_->writer_options =
          footer_metadata.array_record_metadata().writer_options();
    }
  }
  {
    AR_ENDO_SCOPE("Reading footer body");
    auto num_chunks = footer_metadata.array_record_metadata().num_chunks();
    state_->footer.resize(num_chunks);
    for (auto i : IndicesOf(state_->footer)) {
      if (!footer_decoder.ReadRecord(state_->footer[i])) {
        Fail(Annotate(footer_decoder.status(),
                      "Failed to read ArrayRecordFooter"));
        return;
      }
    }
    if (!state_->footer.empty()) {
      state_->record_group_size = state_->footer.front().num_records();
      if (!state_->footer.back().has_chunk_offset()) {
        Fail(InvalidArgumentError("Invalid footer"));
        return;
      }
      // Finds minimal chunk_group_size that is larger equals to the readahead
      // buffer. A chunk_group corresponds to a PRead call. Smaller
      // chunk_group_size is better for random access, the converse is better
      // for sequential reads.
      for (auto i : Seq(state_->footer.size())) {
        uint64_t buf_size =
            state_->ChunkEndOffset(i) - state_->footer.front().chunk_offset();
        if (buf_size >= state_->options.readahead_buffer_size()) {
          state_->chunk_group_size = i + 1;
          break;
        }
      }
      if (!state_->chunk_group_size) {
        state_->chunk_group_size = state_->footer.size();
      }
    }
  }
}

absl::Status ArrayRecordReaderBase::ParallelReadRecords(
    absl::FunctionRef<absl::Status(uint64_t, absl::string_view)> callback)
    const {
  if (!ok()) {
    return status();
  }
  if (state_->footer.empty()) {
    return absl::OkStatus();
  }
  uint64_t num_chunk_groups =
      CeilOfRatio(state_->footer.size(), state_->chunk_group_size);
  const auto reader = get_backing_reader();
  Reader* mutable_reader = const_cast<Reader*>(
      reinterpret_cast<const Reader*>(reader.get()));
  auto status = ParallelForWithStatus<1>(
      Seq(num_chunk_groups), state_->pool, [&](size_t buf_idx) -> absl::Status {
        uint64_t chunk_idx_start = buf_idx * state_->chunk_group_size;
        // inclusive index, not the conventional exclusive index.
        uint64_t last_chunk_idx =
            std::min((buf_idx + 1) * state_->chunk_group_size - 1,
                     state_->footer.size() - 1);
        uint64_t buf_len = state_->ChunkEndOffset(last_chunk_idx) -
                           state_->footer[chunk_idx_start].chunk_offset();
        AR_ENDO_JOB(
            "ArrayRecordReaderBase::ParallelReadRecords",
            absl::StrCat("buffer_idx: ", buf_idx, " buffer_len: ", buf_len));

        MaskedReader masked_reader(riegeli::kClosed);
        {
          AR_ENDO_SCOPE("MaskedReader");
          masked_reader =
              MaskedReader(mutable_reader->NewReader(
                               state_->footer[chunk_idx_start].chunk_offset()),
                           buf_len);
        }
        for (uint64_t chunk_idx = chunk_idx_start; chunk_idx <= last_chunk_idx;
             ++chunk_idx) {
          AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
          masked_reader.Seek(state_->footer[chunk_idx].chunk_offset());
          riegeli::DefaultChunkReader<> chunk_reader(&masked_reader);
          Chunk chunk;
          if (ABSL_PREDICT_FALSE(!chunk_reader.ReadChunk(chunk))) {
            return chunk_reader.status();
          }
          ChunkDecoder decoder;
          if (ABSL_PREDICT_FALSE(!decoder.Decode(chunk))) {
            return decoder.status();
          }
          uint64_t record_index_base = chunk_idx * state_->record_group_size;
          for (auto inner_record_idx : Seq(decoder.num_records())) {
            absl::string_view record;
            if (ABSL_PREDICT_FALSE(!decoder.ReadRecord(record))) {
              return decoder.status();
            }
            auto s = callback(record_index_base + inner_record_idx, record);
            if (ABSL_PREDICT_FALSE(!s.ok())) {
              return s;
            }
          }
          if (ABSL_PREDICT_FALSE(!decoder.Close())) {
            return decoder.status();
          }
          if (ABSL_PREDICT_FALSE(!chunk_reader.Close())) {
            return chunk_reader.status();
          }
        }
        return absl::OkStatus();
      });
  return status;
}

absl::Status ArrayRecordReaderBase::ParallelReadRecordsInRange(
    uint64_t begin, uint64_t end,
    absl::FunctionRef<absl::Status(uint64_t, absl::string_view)> callback)
    const {
  if (!ok()) {
    return status();
  }
  if (state_->footer.empty()) {
    return absl::OkStatus();
  }
  if (end > NumRecords() || begin >= end) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid range [%d, %d). Total records: %d", begin, end, NumRecords()));
  }
  uint64_t chunk_idx_begin = begin / state_->record_group_size;
  uint64_t chunk_idx_end = CeilOfRatio(end, state_->record_group_size);
  uint64_t num_chunks = chunk_idx_end - chunk_idx_begin;
  uint64_t num_chunk_groups = CeilOfRatio(num_chunks, state_->chunk_group_size);

  const auto reader = get_backing_reader();
  Reader* mutable_reader =
      const_cast<Reader*>(reinterpret_cast<const Reader*>(reader.get()));
  auto status = ParallelForWithStatus<1>(
      Seq(num_chunk_groups), state_->pool, [&](size_t buf_idx) -> absl::Status {
        uint64_t chunk_idx_start =
            chunk_idx_begin + buf_idx * state_->chunk_group_size;
        // inclusive index, not the conventional exclusive index.
        uint64_t last_chunk_idx = std::min(
            chunk_idx_begin + (buf_idx + 1) * state_->chunk_group_size - 1,
            chunk_idx_end - 1);
        uint64_t buf_len = state_->ChunkEndOffset(last_chunk_idx) -
                           state_->footer[chunk_idx_start].chunk_offset();
        AR_ENDO_JOB(
            "ArrayRecordReaderBase::ParallelReadRecordsWithRange",
            absl::StrCat("buffer_idx: ", buf_idx, " buffer_len: ", buf_len));

        MaskedReader masked_reader(riegeli::kClosed);
        {
          AR_ENDO_SCOPE("MaskedReader");
          masked_reader =
              MaskedReader(mutable_reader->NewReader(
                               state_->footer[chunk_idx_start].chunk_offset()),
                           buf_len);
        }
        for (uint64_t chunk_idx = chunk_idx_start; chunk_idx <= last_chunk_idx;
             ++chunk_idx) {
          AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
          masked_reader.Seek(state_->footer[chunk_idx].chunk_offset());
          riegeli::DefaultChunkReader<> chunk_reader(&masked_reader);
          Chunk chunk;
          if (ABSL_PREDICT_FALSE(!chunk_reader.ReadChunk(chunk))) {
            return chunk_reader.status();
          }
          ChunkDecoder decoder;
          if (ABSL_PREDICT_FALSE(!decoder.Decode(chunk))) {
            return decoder.status();
          }
          uint64_t record_index_base = chunk_idx * state_->record_group_size;
          uint64_t inner_record_idx_start = 0;
          if (record_index_base < begin) {
            decoder.SetIndex(begin - record_index_base);
            inner_record_idx_start = begin - record_index_base;
          }
          for (auto inner_record_idx :
               Seq(inner_record_idx_start, decoder.num_records())) {
            uint64_t record_idx = record_index_base + inner_record_idx;
            if (ABSL_PREDICT_FALSE(record_idx >= end)) {
              break;
            }
            absl::string_view record;
            if (ABSL_PREDICT_FALSE(!decoder.ReadRecord(record))) {
              return decoder.status();
            }
            auto s = callback(record_idx, record);
            if (ABSL_PREDICT_FALSE(!s.ok())) {
              return s;
            }
          }
          if (ABSL_PREDICT_FALSE(!decoder.Close())) {
            return decoder.status();
          }
          if (ABSL_PREDICT_FALSE(!chunk_reader.Close())) {
            return chunk_reader.status();
          }
        }
        return absl::OkStatus();
      });
  return status;
}

absl::Status ArrayRecordReaderBase::ParallelReadRecordsWithIndices(
    absl::Span<const uint64_t> indices,
    absl::FunctionRef<absl::Status(uint64_t, absl::string_view)> callback)
    const {
  if (!ok()) {
    return status();
  }
  if (state_->footer.empty()) {
    return absl::OkStatus();
  }

  struct IndexPair {
    IndexPair(uint64_t inner_record_index, uint64_t indices_index)
        : inner_record_index(inner_record_index),
          indices_index(indices_index) {}
    // Index of a record within a chunk.
    // Invariant: inner_record_index < group_size_ == num records per chunk.
    uint64_t inner_record_index;
    // Index to an entry in indices.
    uint64_t indices_index;
  };

  std::vector<std::vector<IndexPair>> per_chunk_indices(state_->footer.size());
  std::vector<std::vector<uint64_t>> chunk_indices_per_buffer;
  for (auto [indices_idx, record_idx] : Enumerate(indices)) {
    if (record_idx >= state_->num_records) {
      return OutOfRangeError("index %d out of bound %d", record_idx,
                             state_->num_records);
    }
    uint64_t chunk_idx = record_idx / state_->record_group_size;
    uint64_t local_idx = record_idx - chunk_idx * state_->record_group_size;
    per_chunk_indices[chunk_idx].emplace_back(local_idx, indices_idx);
  }
  bool in_buffer = false;
  for (auto i : Seq(state_->footer.size())) {
    // Find the first chunk containing indices
    if (!in_buffer) {
      if (!per_chunk_indices[i].empty()) {
        chunk_indices_per_buffer.push_back({i});
        in_buffer = true;
        continue;
      }
    }
    // Regular cases.
    if (!per_chunk_indices[i].empty()) {
      uint64_t buf_size =
          state_->ChunkEndOffset(i) - chunk_indices_per_buffer.back()[0];
      if (buf_size < state_->options.readahead_buffer_size()) {
        chunk_indices_per_buffer.back().push_back(i);
      } else {
        chunk_indices_per_buffer.push_back({i});
      }
    }
  }
  const auto reader = get_backing_reader();
  Reader* mutable_reader = const_cast<Reader*>(
      reinterpret_cast<const Reader*>(reader.get()));
  auto status = ParallelForWithStatus<1>(
      IndicesOf(chunk_indices_per_buffer), state_->pool,
      [&](size_t buf_idx) -> absl::Status {
        auto buffer_chunks =
            absl::MakeConstSpan(chunk_indices_per_buffer[buf_idx]);
        uint64_t buf_len = state_->ChunkEndOffset(buffer_chunks.back()) -
                           state_->footer[buffer_chunks[0]].chunk_offset();
        AR_ENDO_JOB(
            "ArrayRecordReaderBase::ParallelReadRecords",
            absl::StrCat("buffer_idx: ", buf_idx, " buffer_len: ", buf_len));
        MaskedReader masked_reader(riegeli::kClosed);
        {
          AR_ENDO_SCOPE("MaskedReader");
          masked_reader =
              MaskedReader(mutable_reader->NewReader(
                               state_->footer[buffer_chunks[0]].chunk_offset()),
                           buf_len);
        }
        for (auto chunk_idx : buffer_chunks) {
          AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
          masked_reader.Seek(state_->footer[chunk_idx].chunk_offset());
          riegeli::DefaultChunkReader<> chunk_reader(&masked_reader);
          Chunk chunk;
          if (ABSL_PREDICT_FALSE(!chunk_reader.ReadChunk(chunk))) {
            return chunk_reader.status();
          }
          ChunkDecoder decoder;
          if (ABSL_PREDICT_FALSE(!decoder.Decode(chunk))) {
            return decoder.status();
          }

          for (const auto& index_pair : per_chunk_indices[chunk_idx]) {
            decoder.SetIndex(index_pair.inner_record_index);
            absl::string_view record;
            if (ABSL_PREDICT_FALSE(!decoder.ReadRecord(record))) {
              return decoder.status();
            }
            auto s = callback(index_pair.indices_index, record);
            if (ABSL_PREDICT_FALSE(!s.ok())) {
              return s;
            }
          }
          if (ABSL_PREDICT_FALSE(!decoder.Close())) {
            return decoder.status();
          }
          if (ABSL_PREDICT_FALSE(!chunk_reader.Close())) {
            return chunk_reader.status();
          }
        }
        return absl::OkStatus();
      });
  return status;
}

uint64_t ArrayRecordReaderBase::NumRecords() const {
  if (!ok()) {
    return 0;
  }
  return state_->num_records;
}

uint64_t ArrayRecordReaderBase::RecordGroupSize() {
  if (!ok()) {
    return 0;
  }
  return state_->record_group_size;
}

uint64_t ArrayRecordReaderBase::RecordIndex() const {
  if (!ok()) {
    return 0;
  }
  return state_->record_idx;
}

bool ArrayRecordReaderBase::SeekRecord(uint64_t record_index) {
  if (!ok()) {
    return false;
  }
  state_->record_idx = std::min(record_index, state_->num_records);
  return true;
}

bool ArrayRecordReaderBase::ReadRecord(google::protobuf::MessageLite* record) {
  absl::string_view result_view;
  if (!ReadRecord(&result_view)) {
    return false;
  }
  return record->ParsePartialFromString(result_view.data());
}

bool ArrayRecordReaderBase::ReadRecord(absl::string_view* record) {
  if (!ok() || state_->record_idx == state_->num_records) {
    return false;
  }
  uint64_t chunk_idx = state_->record_idx / state_->record_group_size;
  uint64_t buffer_idx = chunk_idx / state_->chunk_group_size;
  uint64_t local_chunk_idx = chunk_idx - buffer_idx * state_->chunk_group_size;
  uint64_t local_record_idx =
      state_->record_idx - chunk_idx * state_->record_group_size;

  if (buffer_idx != state_->buffer_idx) {
    if (!ReadAheadFromBuffer(buffer_idx)) {
      return false;
    }
  }
  auto& decoder = state_->current_decoders[local_chunk_idx];
  if (!decoder.ok()) {
    Fail(decoder.status());
    return false;
  }
  if (decoder.index() != local_record_idx) {
    decoder.SetIndex(local_record_idx);
  }
  if (!decoder.ReadRecord(*record)) {
    Fail(decoder.status());
    return false;
  }
  state_->record_idx++;

  return true;
}

bool ArrayRecordReaderBase::ReadAheadFromBuffer(uint64_t buffer_idx) {
  uint64_t max_parallelism = state_->options.max_parallelism().value();
  if (!state_->pool || max_parallelism == 0) {
    std::vector<ChunkDecoder> decoders;
    decoders.reserve(state_->chunk_group_size);
    uint64_t chunk_start = buffer_idx * state_->chunk_group_size;
    uint64_t chunk_end = std::min(state_->footer.size(),
                                  (buffer_idx + 1) * state_->chunk_group_size);
    const auto reader = get_backing_reader();
    for (uint64_t chunk_idx = chunk_start; chunk_idx < chunk_end; ++chunk_idx) {
      uint64_t chunk_offset = state_->footer[chunk_idx].chunk_offset();
      uint64_t chunk_end_offset = state_->ChunkEndOffset(chunk_idx);
      decoders.push_back(
          ReadChunk(reader, chunk_offset, chunk_end_offset - chunk_offset));
    }
    state_->buffer_idx = buffer_idx;
    state_->current_decoders = std::move(decoders);
    return true;
  }
  // Move forward until we reach the future for buffer_idx.
  while (!state_->future_decoders.empty() &&
         buffer_idx != state_->future_decoders.front().first) {
    state_->future_decoders.pop();
  }

  // Used for running one extra task in this thread.
  std::function<void()> current_task = []{};

  while (state_->future_decoders.size() < max_parallelism) {
    uint64_t buffer_to_add = buffer_idx + state_->future_decoders.size();
    if (buffer_to_add * state_->chunk_group_size >= state_->footer.size()) {
      break;
    }
    // Although our internal ThreadPool takes absl::AnyInvocable which is
    // movable, OSS ThreadPool only takes std::function which requires all the
    // captures to be copyable. Therefore we must wrap the promise in a
    // shared_ptr to copy it over to the scheduled task.
    auto decoder_promise =
        std::make_shared<std::promise<std::vector<ChunkDecoder>>>();
    state_->future_decoders.push(
        {buffer_to_add, decoder_promise->get_future()});
    const auto reader = get_backing_reader();
    std::vector<uint64_t> chunk_offsets;
    chunk_offsets.reserve(state_->chunk_group_size);
    uint64_t chunk_start = buffer_to_add * state_->chunk_group_size;
    uint64_t chunk_end = std::min(
        state_->footer.size(), (buffer_to_add + 1) * state_->chunk_group_size);
    for (uint64_t chunk_idx = chunk_start; chunk_idx < chunk_end; ++chunk_idx) {
      chunk_offsets.push_back(state_->footer[chunk_idx].chunk_offset());
    }
    uint64_t buffer_len =
        state_->ChunkEndOffset(chunk_end - 1) - chunk_offsets[0];

    auto task = [reader, decoder_promise, chunk_offsets, buffer_to_add,
                 buffer_len] {
      AR_ENDO_JOB("ArrayRecordReaderBase::ReadAheadFromBuffer",
                  absl::StrCat("buffer_idx: ", buffer_to_add,
                               " buffer_len: ", buffer_len));
      std::vector<ChunkDecoder> decoders(chunk_offsets.size());
      if (!reader->ok()) {
        for (auto& decoder : decoders) {
          decoder.Fail(reader->status());
        }
        decoder_promise->set_value(std::move(decoders));
        return;
      }
      Reader* mutable_reader =
          const_cast<Reader*>(reinterpret_cast<const Reader*>(reader.get()));
      MaskedReader masked_reader(riegeli::kClosed);
      {
        AR_ENDO_SCOPE("MaskedReader");
        masked_reader = MaskedReader(
            mutable_reader->NewReader(chunk_offsets.front()), buffer_len);
      }
      if (!masked_reader.ok()) {
        for (auto& decoder : decoders) {
          decoder.Fail(masked_reader.status());
        }
        decoder_promise->set_value(std::move(decoders));
        return;
      }
      {
        AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
        for (auto local_chunk_idx : IndicesOf(chunk_offsets)) {
          masked_reader.Seek(chunk_offsets[local_chunk_idx]);
          auto chunk_reader = riegeli::DefaultChunkReader<>(&masked_reader);
          Chunk chunk;
          if (!chunk_reader.ReadChunk(chunk)) {
            decoders[local_chunk_idx].Fail(chunk_reader.status());
            continue;
          }
          decoders[local_chunk_idx].Decode(chunk);
        }
      }
      decoder_promise->set_value(std::move(decoders));
    };
    if (buffer_to_add == buffer_idx) {
      current_task = task;
    } else {
      state_->pool->Schedule(task);
    }
  }
  current_task();
  if (state_->future_decoders.front().first != buffer_idx) {
    Fail(InternalError(
        "state_->future_decoders.front().first %d should match buffer_idx %d",
        state_->future_decoders.front().first, buffer_idx));
    return false;
  }
  {
    AR_ENDO_JOB("ArrayRecordReaderBase::ReadAheadFromBuffer GetFuture",
                absl::StrCat("buffer_idx: ", buffer_idx));
    state_->buffer_idx = buffer_idx;
    state_->current_decoders = state_->future_decoders.front().second.get();
    state_->future_decoders.pop();
  }

  return true;
}

std::optional<std::string> ArrayRecordReaderBase::WriterOptionsString() const {
  return state_->writer_options;
}

}  // namespace array_record
