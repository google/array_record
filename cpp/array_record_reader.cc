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
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>  // NOLINT(build/c++11)
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "cpp/common.h"
#include "cpp/layout.pb.h"
#include "cpp/masked_reader.h"
#include "cpp/parallel_for.h"
#include "cpp/thread_pool.h"
#include "google/protobuf/message_lite.h"
#include "riegeli/base/object.h"
#include "riegeli/base/options_parser.h"
#include "riegeli/base/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/chunk_encoding/chunk.h"
#include "riegeli/chunk_encoding/chunk_decoder.h"
#include "riegeli/records/chunk_reader.h"

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

// Abstract class for different index storage options.
class ChunkOffset {
 public:
  virtual ~ChunkOffset() {}
  virtual uint64_t operator[](size_t idx) const = 0;
  virtual uint64_t size() const = 0;
  bool empty() const { return size() == 0; }
};

class ArrayChunkOffset : public ChunkOffset {
 public:
  static absl::StatusOr<std::unique_ptr<ChunkOffset>> Build(
      ChunkDecoder& footer_decoder, uint64_t num_chunks) {
    std::vector<uint64_t> chunk_offsets;
    chunk_offsets.reserve(num_chunks);
    footer_decoder.SetIndex(1);
    ArrayRecordFooter footer;
    for (size_t i = 0; i < num_chunks; ++i) {
      if (!footer_decoder.ReadRecord(footer)) {
        return Annotate(footer_decoder.status(),
                        absl::StrCat("Failed to read at footer record: ", i,
                                     ". Total num chunkls: ", num_chunks));
      }
      if (!footer.has_chunk_offset()) {
        return InvalidArgumentError("Invalid footer at index: %d", i);
      }
      chunk_offsets.push_back(footer.chunk_offset());
    }
    return std::unique_ptr<ChunkOffset>(
        new ArrayChunkOffset(std::move(chunk_offsets)));
  };

  uint64_t operator[](size_t idx) const override { return chunk_offsets_[idx]; }

  uint64_t size() const override { return chunk_offsets_.size(); }

 private:
  ArrayChunkOffset(std::vector<uint64_t> chunk_offsets)
      : chunk_offsets_(std::move(chunk_offsets)) {}

  std::vector<uint64_t> chunk_offsets_;
};

class OffloadedChunkOffset : public ChunkOffset {
 public:
  OffloadedChunkOffset(ArrayRecordReaderBase* base_reader,
                       uint64_t footer_offset, uint64_t num_chunks)
      : base_reader_(base_reader),
        footer_offset_(footer_offset),
        num_chunks_(num_chunks) {}

  uint64_t operator[](size_t idx) const override {
    auto backing_reader = base_reader_->get_backing_reader();
    auto reader = backing_reader->NewReader(footer_offset_);
    auto chunk_reader = riegeli::DefaultChunkReader<>(reader.get());
    Chunk chunk;
    ChunkDecoder footer_decoder;
    chunk_reader.ReadChunk(chunk);
    footer_decoder.Decode(chunk);
    // First item is the footer_metadata, so we want to skip.
    footer_decoder.SetIndex(idx + 1);
    ArrayRecordFooter footer;
    footer_decoder.ReadRecord(footer);
    return footer.chunk_offset();
  }

  uint64_t size() const override { return num_chunks_; }

  void update_base_reader(ArrayRecordReaderBase* base_reader) {
    base_reader_ = base_reader;
  }

 private:
  ArrayRecordReaderBase* base_reader_;
  uint64_t footer_offset_;
  uint64_t num_chunks_;
};

namespace {

std::optional<uint64_t> ParseEnvBytes(const char* name) {
  const char* val = std::getenv(name);
  if (val == nullptr || val[0] == '\0') return std::nullopt;
  std::string s(val);
  if (s.size() >= 2 && (s.back() == 'B' || s.back() == 'b')) {
    char prev = s[s.size() - 2];
    if (prev < '0' || prev > '9') {
      s.pop_back();
    }
  }
  uint64_t out = 0;
  OptionsParser parser;
  parser.AddOption(
      "v", ValueParser::Bytes(0, uint64_t{64} * 1024 * 1024 * 1024, &out));
  if (parser.FromString(absl::StrCat("v:", s))) {
    return out;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(val, &end, 10);  // NOLINT
  if (end != val && *end == '\0') return static_cast<uint64_t>(parsed);
  return std::nullopt;
}

std::optional<bool> ParseEnvBool(const char* name) {
  const char* val = std::getenv(name);
  if (val == nullptr || val[0] == '\0') return std::nullopt;
  if (std::strcmp(val, "0") == 0 || std::strcmp(val, "false") == 0 ||
      std::strcmp(val, "FALSE") == 0) {
    return false;
  }
  return true;
}

class OffsetBufferReader : public riegeli::Reader {
 public:
  OffsetBufferReader(absl::string_view data, riegeli::Position start_pos) {
    set_buffer(data.data(), data.size());
    set_limit_pos(start_pos + data.size());
  }

 protected:
  bool PullSlow(size_t min_length, size_t recommended_length) override {
    Fail(FailedPreconditionError("Should not pull beyond OffsetBufferReader"));
    return false;
  }
};

ABSL_CONST_INIT absl::Mutex g_host_block_mu(absl::kConstInit);
absl::CondVar& GetHostBlockCv() {
  static auto* cv = new absl::CondVar();
  return *cv;
}
std::atomic<uint64_t> g_active_host_block_fetches{0};

// Waits on `cv` until signaled or until the steady-clock `deadline` passes.
void WaitUntilSteadyDeadline(absl::CondVar& cv, absl::Mutex* mu,
                             std::chrono::steady_clock::time_point deadline)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
  const auto now = std::chrono::steady_clock::now();
  if (deadline <= now) return;
  cv.WaitWithTimeout(mu, absl::FromChrono(deadline - now));
}

}  // namespace

absl::StatusOr<ArrayRecordReaderBase::Options>
ArrayRecordReaderBase::Options::FromString(absl::string_view text) {
  ArrayRecordReaderBase::Options options;
  if (auto v = ParseEnvBytes("ARRAY_RECORD_CACHE_BLOCK_BYTES")) {
    options.set_cache_block_bytes(*v);
  }
  if (auto v = ParseEnvBytes("ARRAY_RECORD_COLD_START_READS")) {
    options.set_cold_start_reads(*v);
  }
  if (auto v = ParseEnvBytes("ARRAY_RECORD_COLD_START_BLOCK_BYTES")) {
    options.set_cold_start_block_bytes(*v);
  }
  if (auto v = ParseEnvBool("ARRAY_RECORD_ALIGN_BLOCKS")) {
    options.set_align_blocks(*v);
  }
  if (auto v = ParseEnvBytes("ARRAY_RECORD_MAX_CACHED_BYTES")) {
    options.set_max_cached_bytes(*v);
  }
  if (auto v = ParseEnvBool("ARRAY_RECORD_FAST_SINGLE_INDEX")) {
    options.set_fast_single_index(*v);
  }
  if (auto v = ParseEnvBytes("ARRAY_RECORD_MAX_PARALLEL_BLOCK_FETCHES")) {
    options.set_max_parallel_block_fetches(*v);
  }
  if (auto v = ParseEnvBool("ARRAY_RECORD_WAIT_FOR_BLOCK_SLOT")) {
    options.set_wait_for_block_slot(*v);
  }
  if (auto v = ParseEnvBytes("ARRAY_RECORD_HOST_MAX_PARALLEL_BLOCK_FETCHES")) {
    options.set_host_max_parallel_block_fetches(*v);
  }

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
      ValueParser::Or(ValueParser::Enum({{"auto", kDefaultReadaheadBufferSize}},
                                        &options.readahead_buffer_size_),
                      ValueParser::Bytes(/*min_value=*/0,
                                         /*max_value=*/1024 * 1024 * 1024,
                                         &options.readahead_buffer_size_)));
  // Index storage option
  options_parser.AddOption(
      "index_storage_option",
      ValueParser::Enum({{"", IndexStorageOption::kInMemory},
                         {"in_memory", IndexStorageOption::kInMemory},
                         {"offloaded", IndexStorageOption::kOffloaded}},
                        &options.index_storage_option_));
  // Footer prefetch bytes
  options_parser.AddOption(
      "footer_prefetch_bytes",
      ValueParser::Or(
          ValueParser::Enum({{"auto", kDefaultFooterPrefetchBytes}},
                            &options.footer_prefetch_bytes_),
          ValueParser::Bytes(/*min_value=*/0,
                             /*max_value=*/1024 * 1024 * 1024,
                             &options.footer_prefetch_bytes_)));
  // Consume-once random access block cache options
  options_parser.AddOption(
      "cache_block_bytes",
      ValueParser::Or(
          ValueParser::Enum({{"auto", kDefaultCacheBlockBytes}},
                            &options.cache_block_bytes_),
          ValueParser::Bytes(/*min_value=*/0,
                             /*max_value=*/uint64_t{16} * 1024 * 1024 * 1024,
                             &options.cache_block_bytes_)));
  options_parser.AddOption("cold_start_reads",
                           ValueParser::Bytes(uint64_t{0}, uint64_t{1000000},
                                              &options.cold_start_reads_));
  options_parser.AddOption(
      "cold_start_block_bytes",
      ValueParser::Bytes(/*min_value=*/0,
                         /*max_value=*/uint64_t{16} * 1024 * 1024 * 1024,
                         &options.cold_start_block_bytes_));
  options_parser.AddOption(
      "align_blocks",
      ValueParser::Enum(
          {{"true", true}, {"false", false}, {"1", true}, {"0", false}},
          &options.align_blocks_));
  options_parser.AddOption(
      "max_cached_bytes",
      ValueParser::Bytes(/*min_value=*/0,
                         /*max_value=*/uint64_t{64} * 1024 * 1024 * 1024,
                         &options.max_cached_bytes_));
  options_parser.AddOption(
      "fast_single_index",
      ValueParser::Enum(
          {{"true", true}, {"false", false}, {"1", true}, {"0", false}},
          &options.fast_single_index_));
  options_parser.AddOption(
      "max_parallel_block_fetches",
      ValueParser::Bytes(uint64_t{0}, uint64_t{1000000},
                         &options.max_parallel_block_fetches_));
  options_parser.AddOption(
      "wait_for_block_slot",
      ValueParser::Enum(
          {{"true", true}, {"false", false}, {"1", true}, {"0", false}},
          &options.wait_for_block_slot_));
  options_parser.AddOption(
      "host_max_parallel_block_fetches",
      ValueParser::Bytes(uint64_t{0}, uint64_t{1000000},
                         &options.host_max_parallel_block_fetches_));
  if (!options_parser.FromString(text)) {
    return options_parser.status();
  }
  if (options.index_storage_option() == IndexStorageOption::kOffloaded &&
      options.footer_prefetch_bytes() > 0) {
    return InvalidArgumentError(
        "footer_prefetch_bytes > 0 is only supported with "
        "index_storage_option:in_memory.");
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
  std::unique_ptr<ChunkOffset> chunk_offsets;

  // Objects for managing sequential reads
  uint64_t record_idx = 0;
  uint64_t buffer_idx = UINT64_MAX;
  std::vector<ChunkDecoder> current_decoders;
  std::queue<IndexedPair<std::future<std::vector<ChunkDecoder>>>>
      future_decoders;

  // Writer options for debugging purposes.
  std::optional<std::string> writer_options = std::nullopt;

  // Singleflight consume-once random access chunk cache.
  enum class ChunkCacheState : uint8_t {
    kUncached = 0,
    kFetching = 1,
    kCached = 2,
    kConsumed = 3,
  };
  struct ChunkCacheEntry {
    std::string data;
    uint32_t remaining_reads = 1;
    ChunkCacheState state = ChunkCacheState::kUncached;
  };
  mutable absl::Mutex cache_mu;
  mutable absl::CondVar cache_cv;
  mutable bool cache_initialized = false;
  mutable std::vector<ChunkCacheEntry> chunk_cache;
  mutable std::vector<std::pair<uint64_t, uint64_t>> block_ranges;
  mutable std::vector<uint32_t> chunk_to_block;
  mutable std::atomic<uint64_t> cache_misses{0};
  mutable std::atomic<int64_t> cached_bytes{0};
  mutable uint64_t active_block_fetches = 0;
  mutable uint64_t active_cold_fetches = 0;
  mutable uint64_t completed_block_fetches = 0;
  mutable uint64_t next_read_ticket = 0;
  mutable std::set<uint64_t> waiting_tickets;
  mutable uint64_t cfg_over_par_wait_ms = 0;
  mutable uint64_t cfg_over_par_block_bytes = 0;
  mutable bool cfg_align_subblocks = true;
  mutable bool cfg_fifo_slots = false;
  mutable bool cfg_lifo_slots = false;
  mutable uint64_t cfg_max_slot_wait_ms = 0;
  mutable bool cfg_ticket_cold_start = false;
  mutable bool cfg_wait_cold_complete = false;
  mutable bool cfg_reset_wait_on_progress = false;
  mutable bool cfg_rank0_only_fallback = false;
  mutable uint64_t cfg_fetching_wait_ms = 0;
  mutable uint64_t cfg_rank_wait_step_ms = 0;
  mutable uint64_t cfg_old_ticket_slot_guard_ms = 0;

  uint64_t ChunkEndOffset(uint64_t chunk_idx) const {
    if (chunk_idx == chunk_offsets->size() - 1) {
      return footer_offset;
    }
    return (*chunk_offsets)[chunk_idx + 1];
  }

  void EnsureCacheInitializedLocked() const {
    if (cache_initialized) return;
    cfg_over_par_wait_ms =
        ParseEnvBytes("ARRAY_RECORD_OVER_PAR_WAIT_MS").value_or(950);
    cfg_over_par_block_bytes =
        ParseEnvBytes("ARRAY_RECORD_OVER_PAR_BLOCK_BYTES").value_or(0);
    cfg_align_subblocks =
        ParseEnvBool("ARRAY_RECORD_ALIGN_SUBBLOCKS").value_or(true);
    cfg_fifo_slots =
        ParseEnvBool("ARRAY_RECORD_FIFO_BLOCK_SLOTS").value_or(false);
    cfg_lifo_slots =
        ParseEnvBool("ARRAY_RECORD_LIFO_BLOCK_SLOTS").value_or(true);
    cfg_max_slot_wait_ms =
        ParseEnvBytes("ARRAY_RECORD_MAX_SLOT_WAIT_MS").value_or(0);
    cfg_ticket_cold_start =
        ParseEnvBool("ARRAY_RECORD_TICKET_COLD_START").value_or(false);
    cfg_wait_cold_complete =
        ParseEnvBool("ARRAY_RECORD_WAIT_COLD_COMPLETE").value_or(false);
    cfg_reset_wait_on_progress =
        ParseEnvBool("ARRAY_RECORD_RESET_WAIT_ON_PROGRESS").value_or(false);
    cfg_rank0_only_fallback =
        ParseEnvBool("ARRAY_RECORD_RANK0_ONLY_FALLBACK").value_or(false);
    cfg_fetching_wait_ms =
        ParseEnvBytes("ARRAY_RECORD_FETCHING_WAIT_MS").value_or(950);
    cfg_rank_wait_step_ms =
        ParseEnvBytes("ARRAY_RECORD_RANK_WAIT_STEP_MS").value_or(0);
    cfg_old_ticket_slot_guard_ms =
        ParseEnvBytes("ARRAY_RECORD_OLD_TICKET_SLOT_GUARD_MS").value_or(0);
    const uint64_t num_chunks = chunk_offsets->size();
    chunk_cache.resize(num_chunks);
    const uint32_t group_sz =
        static_cast<uint32_t>(std::max<uint64_t>(1, record_group_size));
    for (uint64_t c = 0; c < num_chunks; ++c) {
      chunk_cache[c].remaining_reads = group_sz;
    }
    const uint64_t block_bytes = options.cache_block_bytes();
    if (block_bytes > 0 && options.align_blocks() && num_chunks > 0) {
      chunk_to_block.resize(num_chunks);
      uint64_t cur_start = 0;
      uint64_t cur_start_offset = (*chunk_offsets)[0];
      for (uint64_t c = 0; c < num_chunks; ++c) {
        const uint64_t c_end_offset = ChunkEndOffset(c);
        if (c > cur_start && c_end_offset - cur_start_offset > block_bytes) {
          block_ranges.emplace_back(cur_start, c);
          cur_start = c;
          cur_start_offset = (*chunk_offsets)[c];
        }
        chunk_to_block[c] = static_cast<uint32_t>(block_ranges.size());
      }
      if (cur_start < num_chunks) {
        block_ranges.emplace_back(cur_start, num_chunks);
      }
    }
    cache_initialized = true;
  }
};

ArrayRecordReaderBase::ArrayRecordReaderBase(Options options,
                                             ARThreadPool* pool)
    : state_(std::make_unique<ArrayRecordReaderState>(options, pool)) {}

ArrayRecordReaderBase::~ArrayRecordReaderBase() = default;

ArrayRecordReaderBase::ArrayRecordReaderBase(
    ArrayRecordReaderBase&& other) noexcept
    : riegeli::Object(std::move(other)), state_(std::move(other.state_)) {
  if (state_->options.index_storage_option() ==
      Options::IndexStorageOption::kOffloaded) {
    OffloadedChunkOffset* chunk_offsets_impl =
        dynamic_cast<OffloadedChunkOffset*>(state_->chunk_offsets.get());
    chunk_offsets_impl->update_base_reader(this);
  }
  other.Reset(riegeli::kClosed);  // NOLINT(bugprone-use-after-move)
}

ArrayRecordReaderBase& ArrayRecordReaderBase::operator=(
    ArrayRecordReaderBase&& other) noexcept {
  // Move base
  riegeli::Object::operator=(static_cast<riegeli::Object&&>(other));
  // Move self
  state_ = std::move(other.state_);

  if (state_->options.index_storage_option() ==
      Options::IndexStorageOption::kOffloaded) {
    OffloadedChunkOffset* chunk_offsets_impl =
        dynamic_cast<OffloadedChunkOffset*>(state_->chunk_offsets.get());
    chunk_offsets_impl->update_base_reader(this);
  }
  // Close
  other.Reset(riegeli::kClosed);
  return *this;
}

// After the first access to the underlying `riegeli::Reader`, the lazily
// evaluated variables for random access are all initialized. Therefore it's
// safe to access the reader from multiple threads later on, even though the
// methods wasn't const.
ChunkDecoder ReadChunk(Reader& reader, size_t pos, size_t len) {
  ChunkDecoder decoder;
  if (!reader.ok()) {
    decoder.Fail(reader.status());
    return decoder;
  }
  MaskedReader masked_reader(reader.NewReader(pos), len);
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
  if (state_->options.index_storage_option() ==
          Options::IndexStorageOption::kOffloaded &&
      state_->options.footer_prefetch_bytes() > 0) {
    Fail(InvalidArgumentError(
        "footer_prefetch_bytes > 0 is only supported with "
        "index_storage_option:in_memory."));
    return;
  }
  auto reader = get_backing_reader();
  if (!reader->ok()) {
    Fail(reader->status());
    return;
  }
  if (!reader->SupportsNewReader()) {
    Fail(InvalidArgumentError(
        "ArrayRecordReader only work on inputs with random access support."));
    return;
  }

  uint32_t max_parallelism = 1;
  if (state_->pool) {
    max_parallelism = state_->pool->NumThreads();
    if (state_->options.max_parallelism().has_value()) {
      max_parallelism = std::min<uint32_t>(
          max_parallelism, state_->options.max_parallelism().value());
    }
  }
  state_->options.set_max_parallelism(max_parallelism);

  AR_ENDO_TASK("Reading ArrayRecord footer");
  RiegeliFooterMetadata footer_metadata;
  ChunkDecoder footer_decoder;
  {
    AR_ENDO_SCOPE("Reading postscript and footer chunk");
    if (!reader->SupportsRandomAccess()) {
      Fail(InvalidArgumentError(
          "ArrayRecordReader only work on inputs with random access support."));
      return;
    }
    auto maybe_size = reader->Size();
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
    const uint64_t prefetch_length =
        std::min<uint64_t>(size, state_->options.footer_prefetch_bytes());
    const uint64_t prefetch_offset = size - prefetch_length;
    MaskedReader prefetched_reader(riegeli::kClosed);
    if (prefetch_length > 0) {
      if (!reader->Seek(prefetch_offset)) {
        Fail(Annotate(reader->status(),
                      "Failed to seek to footer prefetch offset"));
        return;
      }
      prefetched_reader = MaskedReader(reader, prefetch_length);
    }
    const uint64_t postscript_offset = size - kRiegeliBlockSize;
    RiegeliPostscript postscript;
    auto postscript_decoder =
        ReadChunk((prefetch_length > 0 && postscript_offset >= prefetch_offset)
                      ? prefetched_reader
                      : *reader,
                  postscript_offset, kRiegeliBlockSize);
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
    auto footer_offset = postscript.footer_offset();
    state_->footer_offset = footer_offset;
    footer_decoder =
        ReadChunk((prefetch_length > 0 && footer_offset >= prefetch_offset)
                      ? prefetched_reader
                      : *reader,
                  footer_offset, size - kRiegeliBlockSize - footer_offset);

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

    // Reads the group size from the first footer. This should match the
    // group_size in writer options but writer_options is not always available.
    if (footer_decoder.num_records() > 1) {
      ArrayRecordFooter footer;
      footer_decoder.SetIndex(1);
      if (!footer_decoder.ReadRecord(footer)) {
        Fail(Annotate(footer_decoder.status(),
                      "Failed to read ArrayRecordFooter"));
        return;
      }
      state_->record_group_size = footer.num_records();
      // Reset the index
      footer_decoder.SetIndex(1);
    }

    auto num_chunks = footer_metadata.array_record_metadata().num_chunks();
    switch (state_->options.index_storage_option()) {
      case Options::IndexStorageOption::kInMemory: {
        auto status_or_offsets =
            ArrayChunkOffset::Build(footer_decoder, num_chunks);
        if (!status_or_offsets.ok()) {
          Fail(status_or_offsets.status());
          return;
        }
        state_->chunk_offsets = *std::move(status_or_offsets);
      } break;
      case Options::IndexStorageOption::kOffloaded:
        state_->chunk_offsets = std::make_unique<OffloadedChunkOffset>(
            this, state_->footer_offset, num_chunks);
        break;
    }

    if (!state_->chunk_offsets->empty()) {
      if (state_->record_group_size <= 0) {
        Fail(InvalidArgumentError("record_group_size cannot be zero"));
        return;
      }
      // Finds minimal chunk_group_size that is larger equals to the readahead
      // buffer. A chunk_group corresponds to a PRead call. Smaller
      // chunk_group_size is better for random access, the converse is better
      // for sequential reads.
      uint64_t chunk_start_offset = (*state_->chunk_offsets)[0];
      for (auto i : Seq(state_->chunk_offsets->size())) {
        uint64_t buf_size = state_->ChunkEndOffset(i) - chunk_start_offset;
        if (buf_size >= state_->options.readahead_buffer_size()) {
          state_->chunk_group_size = i + 1;
          break;
        }
      }
      if (!state_->chunk_group_size) {
        state_->chunk_group_size = state_->chunk_offsets->size();
      }
    }
  }
}

uint64_t ArrayRecordReaderBase::ChunkStartOffset(uint64_t chunk_idx) const {
  return (*state_->chunk_offsets)[chunk_idx];
}

uint64_t ArrayRecordReaderBase::ChunkEndOffset(uint64_t chunk_idx) const {
  return state_->ChunkEndOffset(chunk_idx);
}

absl::Status ArrayRecordReaderBase::ParallelReadRecords(
    absl::FunctionRef<absl::Status(uint64_t, absl::string_view)> callback)
    const {
  if (!ok()) {
    return status();
  }
  if (state_->chunk_offsets->empty()) {
    return absl::OkStatus();
  }
  uint64_t num_chunk_groups = CeilOfRatio<uint64_t>(
      state_->chunk_offsets->size(), state_->chunk_group_size);
  auto reader = get_backing_reader();
  auto status = ParallelForWithStatus<1>(
      Seq(num_chunk_groups), state_->pool, [&](size_t buf_idx) -> absl::Status {
        uint64_t chunk_idx_start = buf_idx * state_->chunk_group_size;
        // inclusive index, not the conventional exclusive index.
        uint64_t last_chunk_idx =
            std::min<uint64_t>((buf_idx + 1) * state_->chunk_group_size - 1,
                               state_->chunk_offsets->size() - 1);
        uint64_t buf_len = state_->ChunkEndOffset(last_chunk_idx) -
                           (*state_->chunk_offsets)[chunk_idx_start];
        AR_ENDO_JOB(
            "ArrayRecordReaderBase::ParallelReadRecords",
            absl::StrCat("buffer_idx: ", buf_idx, " buffer_len: ", buf_len));

        MaskedReader masked_reader(riegeli::kClosed);
        {
          AR_ENDO_SCOPE("MaskedReader");
          masked_reader = MaskedReader(
              reader->NewReader((*state_->chunk_offsets)[chunk_idx_start]),
              buf_len);
        }
        for (uint64_t chunk_idx = chunk_idx_start; chunk_idx <= last_chunk_idx;
             ++chunk_idx) {
          AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
          masked_reader.Seek((*state_->chunk_offsets)[chunk_idx]);
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
  if (state_->chunk_offsets->empty()) {
    return absl::OkStatus();
  }
  if (end > NumRecords() || begin >= end) {
    return InvalidArgumentError("Invalid range [%d, %d). Total records: %d",
                                begin, end, NumRecords());
  }
  uint64_t chunk_idx_begin = begin / state_->record_group_size;
  uint64_t chunk_idx_end =
      CeilOfRatio<uint64_t>(end, state_->record_group_size);
  uint64_t num_chunks = chunk_idx_end - chunk_idx_begin;
  uint64_t num_chunk_groups =
      CeilOfRatio<uint64_t>(num_chunks, state_->chunk_group_size);

  auto reader = get_backing_reader();
  auto status = ParallelForWithStatus<1>(
      Seq(num_chunk_groups), state_->pool, [&](size_t buf_idx) -> absl::Status {
        uint64_t chunk_idx_start =
            chunk_idx_begin + buf_idx * state_->chunk_group_size;
        // inclusive index, not the conventional exclusive index.
        uint64_t last_chunk_idx = std::min<uint64_t>(
            chunk_idx_begin + (buf_idx + 1) * state_->chunk_group_size - 1,
            chunk_idx_end - 1);
        uint64_t buf_len = state_->ChunkEndOffset(last_chunk_idx) -
                           (*state_->chunk_offsets)[chunk_idx_start];
        AR_ENDO_JOB(
            "ArrayRecordReaderBase::ParallelReadRecordsWithRange",
            absl::StrCat("buffer_idx: ", buf_idx, " buffer_len: ", buf_len));

        MaskedReader masked_reader(riegeli::kClosed);
        {
          AR_ENDO_SCOPE("MaskedReader");
          masked_reader = MaskedReader(
              reader->NewReader((*state_->chunk_offsets)[chunk_idx_start]),
              buf_len);
        }
        for (uint64_t chunk_idx = chunk_idx_start; chunk_idx <= last_chunk_idx;
             ++chunk_idx) {
          AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
          masked_reader.Seek((*state_->chunk_offsets)[chunk_idx]);
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
  if (state_->chunk_offsets->empty() || indices.empty()) {
    return absl::OkStatus();
  }

  const uint64_t num_chunks = state_->chunk_offsets->size();
  const uint64_t cache_block_bytes = state_->options.cache_block_bytes();
  const bool use_cache = cache_block_bytes > 0;

  if (indices.size() == 1 &&
      (state_->options.fast_single_index() || use_cache)) {
    const uint64_t record_idx = indices[0];
    if (record_idx >= state_->num_records) {
      return OutOfRangeError("index %d out of bound %d", record_idx,
                             state_->num_records);
    }
    const uint64_t chunk_idx = record_idx / state_->record_group_size;
    if (chunk_idx >= num_chunks) {
      return OutOfRangeError("chunk_idx out of bounds");
    }
    const uint64_t local_idx =
        record_idx - chunk_idx * state_->record_group_size;

    auto decode_and_invoke = [&](absl::string_view raw_chunk,
                                 uint64_t c_start) -> absl::Status {
      OffsetBufferReader offset_reader(raw_chunk, c_start);
      riegeli::DefaultChunkReader<> chunk_reader(&offset_reader);
      Chunk chunk;
      if (ABSL_PREDICT_FALSE(!chunk_reader.ReadChunk(chunk))) {
        return chunk_reader.status();
      }
      ChunkDecoder decoder;
      if (ABSL_PREDICT_FALSE(!decoder.Decode(chunk))) {
        return decoder.status();
      }
      decoder.SetIndex(local_idx);
      absl::string_view record;
      if (ABSL_PREDICT_FALSE(!decoder.ReadRecord(record))) {
        return decoder.status();
      }
      auto s = callback(0, record);
      if (ABSL_PREDICT_FALSE(!s.ok())) {
        return s;
      }
      if (ABSL_PREDICT_FALSE(!decoder.Close())) {
        return decoder.status();
      }
      if (ABSL_PREDICT_FALSE(!chunk_reader.Close())) {
        return chunk_reader.status();
      }
      return absl::OkStatus();
    };

    const uint64_t c_start = (*state_->chunk_offsets)[chunk_idx];
    const uint64_t c_end = state_->ChunkEndOffset(chunk_idx);
    const uint64_t c_len = c_end - c_start;

    if (!use_cache) {
      auto reader = get_backing_reader();
      std::unique_ptr<Reader> sub_reader = reader->NewReader(c_start);
      if (sub_reader == nullptr || !sub_reader->ok()) {
        return sub_reader ? sub_reader->status()
                          : InvalidArgumentError("Could not create reader");
      }
      std::string raw_chunk(c_len, '\0');
      if (!sub_reader->Read(c_len, raw_chunk.data())) {
        return Annotate(sub_reader->status(),
                        "Could not read chunk from underlying reader");
      }
      return decode_and_invoke(raw_chunk, c_start);
    }

    using ChunkCacheState = ArrayRecordReaderState::ChunkCacheState;
    std::string raw_chunk;
    uint64_t fetch_start = chunk_idx;
    uint64_t fetch_end = chunk_idx + 1;
    bool need_fetch = false;
    bool counted_block_fetch = false;
    bool counted_cold_fetch = false;
    bool counted_host_fetch = false;
    {
      absl::MutexLock lock(state_->cache_mu);
      state_->EnsureCacheInitializedLocked();
      const uint64_t ticket = state_->next_read_ticket++;
      const uint64_t cold_reads = state_->options.cold_start_reads();
      const uint64_t max_par_blocks =
          state_->options.max_parallel_block_fetches();
      const uint64_t host_max_par =
          state_->options.host_max_parallel_block_fetches();
      const bool wait_slot = state_->options.wait_for_block_slot();
      const uint64_t over_par_wait_ms = state_->cfg_over_par_wait_ms;
      const uint64_t over_par_block_bytes = state_->cfg_over_par_block_bytes;
      const bool align_subblocks = state_->cfg_align_subblocks;
      const bool fifo_slots = state_->cfg_fifo_slots;
      const bool lifo_slots = state_->cfg_lifo_slots;
      const uint64_t max_slot_wait_ms = state_->cfg_max_slot_wait_ms;
      const auto enter_time = std::chrono::steady_clock::now();
      const auto slot_wait_deadline =
          max_slot_wait_ms > 0
              ? enter_time + std::chrono::milliseconds(max_slot_wait_ms)
              : std::chrono::steady_clock::time_point::max();
      const bool ticket_cold_start = state_->cfg_ticket_cold_start;
      const bool wait_cold_complete = state_->cfg_wait_cold_complete;
      const bool reset_wait_on_progress = state_->cfg_reset_wait_on_progress;
      const bool rank0_only_fallback = state_->cfg_rank0_only_fallback;
      const uint64_t fetching_wait_ms = state_->cfg_fetching_wait_ms;
      const uint64_t rank_wait_step_ms = state_->cfg_rank_wait_step_ms;
      const uint64_t old_ticket_slot_guard_ms =
          state_->cfg_old_ticket_slot_guard_ms;
      const auto fetching_deadline =
          fetching_wait_ms > 0
              ? enter_time + std::chrono::milliseconds(fetching_wait_ms)
              : std::chrono::steady_clock::time_point::max();
      bool deadline_started = !rank0_only_fallback;
      auto wait_deadline =
          over_par_wait_ms > 0
              ? enter_time + std::chrono::milliseconds(over_par_wait_ms)
              : std::chrono::steady_clock::time_point::min();
      uint64_t seen_completed_blocks = state_->completed_block_fetches;
      bool can_take_slot = true;
      while (true) {
        if (state_->chunk_cache[chunk_idx].state ==
            ChunkCacheState::kFetching) {
          state_->waiting_tickets.erase(ticket);
          if (fetching_wait_ms > 0) {
            if (std::chrono::steady_clock::now() >= fetching_deadline) {
              break;
            }
            WaitUntilSteadyDeadline(state_->cache_cv, &state_->cache_mu,
                                    fetching_deadline);
          } else {
            state_->cache_cv.Wait(&state_->cache_mu);
          }
          continue;
        }
        if (state_->chunk_cache[chunk_idx].state !=
            ChunkCacheState::kUncached) {
          break;
        }
        const bool is_cold_wait =
            ticket_cold_start
                ? (ticket < cold_reads)
                : (state_->cache_misses.load(std::memory_order_relaxed) <
                   cold_reads);
        if (is_cold_wait) {
          break;
        }
        if (wait_cold_complete && state_->active_cold_fetches > 0) {
          state_->waiting_tickets.insert(ticket);
          state_->cache_cv.Wait(&state_->cache_mu);
          continue;
        }
        if (reset_wait_on_progress &&
            state_->completed_block_fetches != seen_completed_blocks) {
          seen_completed_blocks = state_->completed_block_fetches;
          if (over_par_wait_ms > 0) {
            wait_deadline = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(over_par_wait_ms);
          }
        }
        const uint64_t avail_shard =
            (max_par_blocks == 0)
                ? UINT64_MAX
                : ((state_->active_block_fetches < max_par_blocks)
                       ? (max_par_blocks - state_->active_block_fetches)
                       : 0);
        const uint64_t cur_host =
            g_active_host_block_fetches.load(std::memory_order_relaxed);
        const uint64_t avail_host =
            (host_max_par == 0)
                ? UINT64_MAX
                : ((cur_host < host_max_par) ? (host_max_par - cur_host) : 0);
        const uint64_t avail_slots = std::min(avail_shard, avail_host);
        can_take_slot = (avail_slots > 0);
        const auto now_tp = std::chrono::steady_clock::now();
        if (max_slot_wait_ms > 0 && now_tp > slot_wait_deadline) {
          can_take_slot = false;
        }
        uint64_t rank = 0;
        if (fifo_slots || lifo_slots) {
          state_->waiting_tickets.insert(ticket);
          const auto it = state_->waiting_tickets.find(ticket);
          rank = static_cast<uint64_t>(
              std::distance(state_->waiting_tickets.begin(), it));
          if (fifo_slots) {
            can_take_slot = can_take_slot && (rank < avail_slots);
          } else {
            const uint64_t rank_from_back = static_cast<uint64_t>(
                std::distance(it, state_->waiting_tickets.end()) - 1);
            can_take_slot = can_take_slot && (rank_from_back < avail_slots);
          }
          if (old_ticket_slot_guard_ms > 0 && rank == 0 &&
              now_tp > enter_time + std::chrono::milliseconds(
                                        old_ticket_slot_guard_ms)) {
            can_take_slot = false;
          }
        }
        if (rank0_only_fallback && rank == 0 && !deadline_started) {
          deadline_started = true;
          if (over_par_wait_ms > 0) {
            wait_deadline =
                now_tp + std::chrono::milliseconds(over_par_wait_ms);
          }
        }
        const auto eff_wait_deadline =
            (over_par_wait_ms > 0 && rank_wait_step_ms > 0)
                ? (wait_deadline +
                   std::chrono::milliseconds(rank * rank_wait_step_ms))
                : wait_deadline;
        if (!can_take_slot) {
          if (wait_slot || (rank0_only_fallback && rank > 0)) {
            if (avail_shard > 0 && avail_host == 0) {
              state_->cache_cv.WaitWithTimeout(&state_->cache_mu,
                                               absl::Milliseconds(5));
            } else {
              state_->cache_cv.Wait(&state_->cache_mu);
            }
            continue;
          }
          if (over_par_wait_ms > 0 && now_tp < eff_wait_deadline) {
            if (avail_shard > 0 && avail_host == 0) {
              const auto step = std::min(eff_wait_deadline,
                                         now_tp + std::chrono::milliseconds(5));
              WaitUntilSteadyDeadline(state_->cache_cv, &state_->cache_mu,
                                      step);
            } else {
              WaitUntilSteadyDeadline(state_->cache_cv, &state_->cache_mu,
                                      eff_wait_deadline);
            }
            continue;
          }
        }
        break;
      }
      state_->waiting_tickets.erase(ticket);
      if (rank0_only_fallback && !state_->waiting_tickets.empty()) {
        state_->cache_cv.SignalAll();
      }
      auto& entry = state_->chunk_cache[chunk_idx];
      if (entry.state == ChunkCacheState::kCached) {
        const size_t sz = entry.data.size();
        if (entry.remaining_reads <= 1) {
          raw_chunk = std::move(entry.data);
          std::string().swap(entry.data);
          entry.remaining_reads = 0;
          entry.state = ChunkCacheState::kConsumed;
          state_->cached_bytes.fetch_sub(static_cast<int64_t>(sz),
                                         std::memory_order_relaxed);
        } else {
          raw_chunk = entry.data;
          entry.remaining_reads--;
        }
      } else if (entry.state == ChunkCacheState::kUncached) {
        need_fetch = true;
        const uint64_t miss_num =
            state_->cache_misses.fetch_add(1, std::memory_order_relaxed);
        const bool is_cold =
            ticket_cold_start ? (ticket < cold_reads) : (miss_num < cold_reads);
        const uint64_t max_cached = state_->options.max_cached_bytes();
        const bool over_cap =
            max_cached > 0 &&
            state_->cached_bytes.load(std::memory_order_relaxed) >=
                static_cast<int64_t>(max_cached);
        const bool over_par =
            !wait_slot && !is_cold &&
            (!can_take_slot ||
             (max_par_blocks > 0 &&
              state_->active_block_fetches >= max_par_blocks) ||
             (host_max_par > 0 &&
              g_active_host_block_fetches.load(std::memory_order_relaxed) >=
                  host_max_par));
        const uint64_t target_bytes =
            over_cap
                ? 0
                : (over_par
                       ? over_par_block_bytes
                       : (is_cold ? state_->options.cold_start_block_bytes()
                                  : cache_block_bytes));
        if (target_bytes == 0) {
          fetch_start = chunk_idx;
          fetch_end = chunk_idx + 1;
        } else if (!is_cold && !over_par && state_->options.align_blocks() &&
                   !state_->chunk_to_block.empty()) {
          const auto [b_start, b_end] =
              state_->block_ranges[state_->chunk_to_block[chunk_idx]];
          fetch_start = b_start;
          fetch_end = b_end;
          while (fetch_start < chunk_idx &&
                 state_->chunk_cache[fetch_start].state !=
                     ChunkCacheState::kUncached) {
            ++fetch_start;
          }
          while (fetch_end > chunk_idx + 1 &&
                 state_->chunk_cache[fetch_end - 1].state !=
                     ChunkCacheState::kUncached) {
            --fetch_end;
          }
        } else if (align_subblocks && state_->options.align_blocks() &&
                   !state_->chunk_to_block.empty()) {
          const auto [b_start, b_end] =
              state_->block_ranges[state_->chunk_to_block[chunk_idx]];
          const uint64_t b_start_off = (*state_->chunk_offsets)[b_start];
          const uint64_t sub_idx = (c_start - b_start_off) / target_bytes;
          const uint64_t sub_start_off = b_start_off + sub_idx * target_bytes;
          const uint64_t sub_end_off = sub_start_off + target_bytes;
          fetch_start = chunk_idx;
          while (fetch_start > b_start &&
                 (*state_->chunk_offsets)[fetch_start - 1] >= sub_start_off) {
            --fetch_start;
          }
          fetch_end = chunk_idx + 1;
          while (fetch_end < b_end &&
                 state_->ChunkEndOffset(fetch_end) <= sub_end_off) {
            ++fetch_end;
          }
          while (fetch_start < chunk_idx &&
                 state_->chunk_cache[fetch_start].state !=
                     ChunkCacheState::kUncached) {
            ++fetch_start;
          }
          while (fetch_end > chunk_idx + 1 &&
                 state_->chunk_cache[fetch_end - 1].state !=
                     ChunkCacheState::kUncached) {
            --fetch_end;
          }
        } else {
          fetch_start = chunk_idx;
          fetch_end = chunk_idx + 1;
          while (fetch_end < num_chunks &&
                 state_->chunk_cache[fetch_end].state ==
                     ChunkCacheState::kUncached &&
                 state_->ChunkEndOffset(fetch_end) - c_start <= target_bytes) {
            ++fetch_end;
          }
        }
        if (is_cold) {
          state_->active_cold_fetches++;
          counted_cold_fetch = true;
        } else if (fetch_end - fetch_start > 1 && !over_par) {
          state_->active_block_fetches++;
          counted_block_fetch = true;
          if (host_max_par > 0) {
            g_active_host_block_fetches.fetch_add(1, std::memory_order_relaxed);
            counted_host_fetch = true;
          }
        }
        for (uint64_t c = fetch_start; c < fetch_end; ++c) {
          if (state_->chunk_cache[c].state == ChunkCacheState::kUncached ||
              c == chunk_idx) {
            state_->chunk_cache[c].state = ChunkCacheState::kFetching;
          }
        }
        if (!state_->waiting_tickets.empty()) {
          state_->cache_cv.SignalAll();
        }
      }
    }

    if (!raw_chunk.empty()) {
      return decode_and_invoke(raw_chunk, c_start);
    }

    if (!need_fetch) {
      // Entry was already kConsumed; perform direct single-chunk read.
      auto reader = get_backing_reader();
      std::unique_ptr<Reader> sub_reader = reader->NewReader(c_start);
      if (sub_reader == nullptr || !sub_reader->ok()) {
        return sub_reader ? sub_reader->status()
                          : InvalidArgumentError("Could not create reader");
      }
      raw_chunk.resize(c_len);
      if (!sub_reader->Read(c_len, raw_chunk.data())) {
        return Annotate(sub_reader->status(),
                        "Could not read chunk from underlying reader");
      }
      return decode_and_invoke(raw_chunk, c_start);
    }

    const uint64_t span_start_offset = (*state_->chunk_offsets)[fetch_start];
    const uint64_t span_end_offset = state_->ChunkEndOffset(fetch_end - 1);
    const uint64_t span_len = span_end_offset - span_start_offset;

    auto reader = get_backing_reader();
    std::unique_ptr<Reader> sub_reader = reader->NewReader(span_start_offset);
    std::string span_buf;
    absl::Status read_status = absl::OkStatus();
    if (sub_reader == nullptr || !sub_reader->ok()) {
      read_status = sub_reader
                        ? sub_reader->status()
                        : InvalidArgumentError("Could not create reader");
    } else {
      span_buf.resize(span_len);
      if (!sub_reader->Read(span_len, span_buf.data())) {
        read_status = Annotate(sub_reader->status(),
                               "Could not read block from underlying reader");
      }
    }

    if (counted_host_fetch) {
      {
        absl::MutexLock host_lock(g_host_block_mu);
        g_active_host_block_fetches.fetch_sub(1, std::memory_order_relaxed);
      }
      GetHostBlockCv().Signal();
    }

    if (!read_status.ok()) {
      absl::MutexLock lock(state_->cache_mu);
      if (counted_cold_fetch) {
        state_->active_cold_fetches--;
      }
      if (counted_block_fetch) {
        state_->active_block_fetches--;
        state_->completed_block_fetches++;
      }
      for (uint64_t c = fetch_start; c < fetch_end; ++c) {
        if (state_->chunk_cache[c].state == ChunkCacheState::kFetching) {
          state_->chunk_cache[c].state = ChunkCacheState::kUncached;
        }
      }
      state_->cache_cv.SignalAll();
      return read_status;
    }

    if (fetch_end - fetch_start == 1 && state_->record_group_size <= 1) {
      raw_chunk = std::move(span_buf);
      {
        absl::MutexLock lock(state_->cache_mu);
        if (counted_cold_fetch) {
          state_->active_cold_fetches--;
        }
        if (counted_block_fetch) {
          state_->active_block_fetches--;
          state_->completed_block_fetches++;
        }
        auto& entry = state_->chunk_cache[chunk_idx];
        entry.remaining_reads = 0;
        entry.state = ChunkCacheState::kConsumed;
      }
      state_->cache_cv.SignalAll();
      return decode_and_invoke(raw_chunk, c_start);
    }

    {
      int64_t added_bytes = 0;
      absl::MutexLock lock(state_->cache_mu);
      if (counted_cold_fetch) {
        state_->active_cold_fetches--;
      }
      if (counted_block_fetch) {
        state_->active_block_fetches--;
        state_->completed_block_fetches++;
      }
      for (uint64_t c = fetch_start; c < fetch_end; ++c) {
        const uint64_t cur_start = (*state_->chunk_offsets)[c];
        const uint64_t cur_len = state_->ChunkEndOffset(c) - cur_start;
        const size_t rel_off =
            static_cast<size_t>(cur_start - span_start_offset);
        auto& entry = state_->chunk_cache[c];
        if (c == chunk_idx) {
          raw_chunk.assign(span_buf.data() + rel_off, cur_len);
          if (entry.remaining_reads <= 1) {
            entry.remaining_reads = 0;
            entry.state = ChunkCacheState::kConsumed;
          } else {
            entry.data = raw_chunk;
            entry.remaining_reads--;
            entry.state = ChunkCacheState::kCached;
            added_bytes += static_cast<int64_t>(cur_len);
          }
        } else if (entry.state == ChunkCacheState::kFetching) {
          entry.data.assign(span_buf.data() + rel_off, cur_len);
          entry.state = ChunkCacheState::kCached;
          added_bytes += static_cast<int64_t>(cur_len);
        }
      }
      if (added_bytes > 0) {
        state_->cached_bytes.fetch_add(added_bytes, std::memory_order_relaxed);
      }
    }
    std::string().swap(span_buf);
    state_->cache_cv.SignalAll();
    return decode_and_invoke(raw_chunk, c_start);
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

  std::vector<std::vector<IndexPair>> per_chunk_indices(
      state_->chunk_offsets->size());
  std::vector<std::vector<uint64_t>> chunk_indices_per_buffer;
  for (auto [indices_idx, record_idx] : Enumerate(indices)) {
    if (record_idx >= state_->num_records) {
      return OutOfRangeError("index %d out of bound %d", record_idx,
                             state_->num_records);
    }
    uint64_t chunk_idx = record_idx / state_->record_group_size;
    if (chunk_idx >= per_chunk_indices.size()) {
      return OutOfRangeError("chunk_idx out of bounds");
    }
    uint64_t local_idx = record_idx - chunk_idx * state_->record_group_size;
    per_chunk_indices[chunk_idx].emplace_back(local_idx, indices_idx);
  }
  bool in_buffer = false;
  for (auto i : Seq(state_->chunk_offsets->size())) {
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
          state_->ChunkEndOffset(i) -
          (*state_->chunk_offsets)[chunk_indices_per_buffer.back()[0]];
      if (buf_size < state_->options.readahead_buffer_size()) {
        chunk_indices_per_buffer.back().push_back(i);
      } else {
        chunk_indices_per_buffer.push_back({i});
      }
    }
  }
  auto reader = get_backing_reader();
  auto status = ParallelForWithStatus<1>(
      IndicesOf(chunk_indices_per_buffer), state_->pool,
      [&](size_t buf_idx) -> absl::Status {
        auto buffer_chunks =
            absl::MakeConstSpan(chunk_indices_per_buffer[buf_idx]);
        uint64_t buf_len = state_->ChunkEndOffset(buffer_chunks.back()) -
                           (*state_->chunk_offsets)[buffer_chunks[0]];
        AR_ENDO_JOB(
            "ArrayRecordReaderBase::ParallelReadRecordsWithIndices",
            absl::StrCat("buffer_idx: ", buf_idx, " buffer_len: ", buf_len));
        MaskedReader masked_reader(riegeli::kClosed);
        {
          AR_ENDO_SCOPE("MaskedReader");
          masked_reader = MaskedReader(
              reader->NewReader((*state_->chunk_offsets)[buffer_chunks[0]]),
              buf_len);
        }
        for (auto chunk_idx : buffer_chunks) {
          AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
          masked_reader.Seek((*state_->chunk_offsets)[chunk_idx]);
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

uint64_t ArrayRecordReaderBase::RecordGroupSize() const {
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
  state_->record_idx = std::min<uint64_t>(record_index, state_->num_records);
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
    uint64_t chunk_end =
        std::min<uint64_t>(state_->chunk_offsets->size(),
                           (buffer_idx + 1) * state_->chunk_group_size);
    auto reader = get_backing_reader();
    for (uint64_t chunk_idx = chunk_start; chunk_idx < chunk_end; ++chunk_idx) {
      uint64_t chunk_offset = (*state_->chunk_offsets)[chunk_idx];
      uint64_t chunk_end_offset = state_->ChunkEndOffset(chunk_idx);
      decoders.push_back(
          ReadChunk(*reader, chunk_offset, chunk_end_offset - chunk_offset));
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
    if (buffer_to_add * state_->chunk_group_size >=
        state_->chunk_offsets->size()) {
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
    auto reader = get_backing_reader();
    std::vector<uint64_t> chunk_offsets;
    chunk_offsets.reserve(state_->chunk_group_size);
    uint64_t chunk_start = buffer_to_add * state_->chunk_group_size;
    uint64_t chunk_end =
        std::min<uint64_t>(state_->chunk_offsets->size(),
                           (buffer_to_add + 1) * state_->chunk_group_size);
    for (uint64_t chunk_idx = chunk_start; chunk_idx < chunk_end; ++chunk_idx) {
      chunk_offsets.push_back((*state_->chunk_offsets)[chunk_idx]);
    }
    uint64_t buffer_len =
        state_->ChunkEndOffset(chunk_end - 1) - chunk_offsets[0];

    auto task = [reader, decoder_promise, chunk_offsets, buffer_to_add,
                 buffer_len]() mutable {
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
      MaskedReader masked_reader(riegeli::kClosed);
      {
        AR_ENDO_SCOPE("MaskedReader");
        masked_reader =
            MaskedReader(reader->NewReader(chunk_offsets.front()), buffer_len);
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
