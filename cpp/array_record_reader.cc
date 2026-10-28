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
#include <cstddef>
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "cpp/block_cache.h"
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

absl::Status ValidateBlockCacheOptions(
    const ArrayRecordReaderBase::Options& o) {
  using Options = ArrayRecordReaderBase::Options;
  if (o.block_cache_max_bytes() == 0) return absl::OkStatus();
  if (o.block_cache_block_size() < Options::kMinBlockCacheBlockSize ||
      o.block_cache_block_size() > Options::kMaxBlockCacheBlockSize) {
    return InvalidArgumentError(
        "block_cache_block_size must be in [1M, 1G], got %d",
        o.block_cache_block_size());
  }
  if (o.block_cache_max_inflight() < 1 ||
      o.block_cache_max_inflight() > Options::kMaxBlockCacheMaxInflight) {
    return InvalidArgumentError(
        "block_cache_max_inflight must be in [1, 1024], got %d",
        o.block_cache_max_inflight());
  }
  if (o.block_cache_max_bytes() < o.block_cache_block_size()) {
    return InvalidArgumentError(
        "block_cache_max_bytes (%d) must be >= block_cache_block_size (%d)",
        o.block_cache_max_bytes(), o.block_cache_block_size());
  }
  if (o.index_storage_option() == Options::IndexStorageOption::kOffloaded) {
    return InvalidArgumentError(
        "block_cache_max_bytes > 0 is only supported with "
        "index_storage_option:in_memory.");
  }
  return absl::OkStatus();
}

absl::StatusOr<BlockCache::Block> FetchBlock(Reader& src, uint64_t offset,
                                             uint64_t length) {
  std::unique_ptr<Reader> sub = src.NewReader(offset);
  if (sub == nullptr) {
    return src.ok() ? InternalError("NewReader(%d) failed", offset)
                    : src.status();
  }
  auto buf = std::make_shared<std::string>();
  if (!sub->Read(length, *buf)) {
    return sub->ok()
               ? absl::DataLossError(absl::StrFormat(
                     "Short read of block at %d, length %d", offset, length))
               : sub->status();
  }
  return BlockCache::Block(std::move(buf));
}

}  // namespace

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
  // Block cache
  options_parser.AddOption(
      "block_cache_max_bytes",
      ValueParser::Bytes(/*min_value=*/0,
                         /*max_value=*/std::numeric_limits<uint64_t>::max(),
                         &options.block_cache_max_bytes_));
  options_parser.AddOption(
      "block_cache_block_size",
      ValueParser::Bytes(kMinBlockCacheBlockSize, kMaxBlockCacheBlockSize,
                         &options.block_cache_block_size_));
  int block_cache_max_inflight = kDefaultBlockCacheMaxInflight;
  options_parser.AddOption("block_cache_max_inflight",
                           ValueParser::Int(1, kMaxBlockCacheMaxInflight,
                                            &block_cache_max_inflight));
  if (!options_parser.FromString(text)) {
    return options_parser.status();
  }
  if (options.index_storage_option() == IndexStorageOption::kOffloaded &&
      options.footer_prefetch_bytes() > 0) {
    return InvalidArgumentError(
        "footer_prefetch_bytes > 0 is only supported with "
        "index_storage_option:in_memory.");
  }
  options.set_block_cache_max_inflight(
      static_cast<uint32_t>(block_cache_max_inflight));
  if (absl::Status s = ValidateBlockCacheOptions(options); !s.ok()) {
    return s;
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

  // Block cache (enabled when options.block_cache_max_bytes() > 0).
  // 0 means disabled.
  uint64_t cache_reader_id = 0;
  // First chunk of each block; size num_blocks + 1, last entry = num_chunks.
  std::vector<uint64_t> block_first_chunk;
  // Records of each block not yet served in the current pass over it. When a
  // pass completes, the block is freed and the count re-armed.
  std::vector<std::atomic<uint32_t>> block_remaining;

  BlockCache::Config CacheConfig() const {
    BlockCache::Config config;
    config.max_bytes = options.block_cache_max_bytes();
    config.max_inflight = options.block_cache_max_inflight();
    return config;
  }

  // Packs whole chunks into blocks: the next chunk joins the current block
  // only while the block span stays <= block_cache_block_size.
  void InitBlockCache() {
    BlockCache::Instance(CacheConfig());
    cache_reader_id = BlockCache::NewReaderId();
    const uint64_t num_chunks = chunk_offsets->size();
    block_first_chunk.clear();
    if (num_chunks == 0) {
      block_first_chunk.push_back(0);
      return;
    }
    const uint64_t block_size = options.block_cache_block_size();
    uint64_t start = (*chunk_offsets)[0];
    block_first_chunk.push_back(0);
    for (uint64_t c = 1; c < num_chunks; ++c) {
      if (ChunkEndOffset(c) - start > block_size) {
        block_first_chunk.push_back(c);
        start = (*chunk_offsets)[c];
      }
    }
    block_first_chunk.push_back(num_chunks);
    const size_t num_blocks = block_first_chunk.size() - 1;
    block_remaining = std::vector<std::atomic<uint32_t>>(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
      block_remaining[b].store(BlockRecords(static_cast<uint32_t>(b)),
                               std::memory_order_relaxed);
    }
  }

  // Records of block `b`, clamped to UINT32_MAX. Both ends are clamped so a
  // corrupt footer (num_records smaller than the chunks imply) cannot
  // underflow end - begin.
  uint32_t BlockRecords(uint32_t b) const {
    const uint64_t begin =
        std::min(block_first_chunk[b] * record_group_size, num_records);
    const uint64_t end =
        std::min(block_first_chunk[b + 1] * record_group_size, num_records);
    return static_cast<uint32_t>(std::min<uint64_t>(end - begin, UINT32_MAX));
  }

  uint32_t BlockOf(uint64_t chunk) const {
    auto it = std::upper_bound(block_first_chunk.begin(),
                               block_first_chunk.end(), chunk);
    return static_cast<uint32_t>(it - block_first_chunk.begin() - 1);
  }
  uint64_t BlockOffset(uint32_t b) const {
    return (*chunk_offsets)[block_first_chunk[b]];
  }
  uint64_t BlockLen(uint32_t b) const {
    return ChunkEndOffset(block_first_chunk[b + 1] - 1) - BlockOffset(b);
  }

  // Counts `n` served records of block `b`, modulo the block's record count.
  // When a pass completes, the count is re-armed to the block's record count
  // minus the overflow, so every full pass over a block frees it again.
  // Returns true iff this call completes a pass and does not start the next
  // one (the caller then frees the block). If the overflow already served
  // records of the next pass, the block is kept for the rest of that pass.
  bool ConsumeRecords(uint32_t b, uint64_t n) {
    const uint32_t total = BlockRecords(b);
    if (total == 0 || n == 0) return false;  // Count 0: never drained.
    std::atomic<uint32_t>& counter = block_remaining[b];
    // The count is always in [1, total].
    uint32_t cur = counter.load(std::memory_order_relaxed);
    while (true) {
      const bool completed_pass = n >= cur;
      const uint32_t overflow =
          completed_pass ? static_cast<uint32_t>((n - cur) % total) : 0;
      const uint32_t next =
          completed_pass ? total - overflow : cur - static_cast<uint32_t>(n);
      if (counter.compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) {
        return completed_pass && overflow == 0;
      }
    }
  }

  void ReleaseBlockCache() {
    if (cache_reader_id == 0) return;
    if (BlockCache* cache = BlockCache::InstanceIfExists()) {
      cache->EraseReader(cache_reader_id);
    }
    cache_reader_id = 0;
  }

  ~ArrayRecordReaderState() { ReleaseBlockCache(); }

  uint64_t ChunkEndOffset(uint64_t chunk_idx) const {
    if (chunk_idx == chunk_offsets->size() - 1) {
      return footer_offset;
    }
    return (*chunk_offsets)[chunk_idx + 1];
  }
};

ArrayRecordReaderBase::ArrayRecordReaderBase(Options options,
                                             ARThreadPool* pool)
    : state_(std::make_unique<ArrayRecordReaderState>(options, pool)) {}

ArrayRecordReaderBase::~ArrayRecordReaderBase() = default;

void ArrayRecordReaderBase::ReleaseBlockCache() {
  if (state_ != nullptr) state_->ReleaseBlockCache();
}

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
  if (absl::Status s = ValidateBlockCacheOptions(state_->options); !s.ok()) {
    Fail(std::move(s));
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
  if (state_->options.block_cache_max_bytes() > 0) {
    state_->InitBlockCache();
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
  if (state_->chunk_offsets->empty()) {
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

  std::vector<std::vector<IndexPair>> per_chunk_indices(
      state_->chunk_offsets->size());
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

  // Decodes `chunk_idx` from `src` (a MaskedReader in file coordinates that
  // covers the chunk) and invokes the callback for its requested records.
  auto serve_chunk = [&](MaskedReader& src,
                         uint64_t chunk_idx) -> absl::Status {
    AR_ENDO_SCOPE("ChunkReader+ChunkDecoder");
    src.Seek((*state_->chunk_offsets)[chunk_idx]);
    riegeli::DefaultChunkReader<> chunk_reader(&src);
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
    return absl::OkStatus();
  };

  auto reader = get_backing_reader();

  if (state_->cache_reader_id != 0) {
    std::vector<uint64_t> touched_chunks;
    for (auto i : Seq(per_chunk_indices.size())) {
      if (!per_chunk_indices[i].empty()) touched_chunks.push_back(i);
    }
    // `touched_chunks` is sorted and a block is a contiguous range of chunks,
    // so the touched chunks of one block form a run. A block's served records
    // are counted once, when the last chunk of its run in this call has been
    // served, so a call never frees a block that another of its chunks still
    // needs. Duplicate indices count once: a pass serves each record once.
    struct BlockRun {
      uint64_t records = 0;
      std::atomic<uint32_t> chunks_left{0};
    };
    std::vector<uint32_t> run_of(touched_chunks.size());
    uint32_t num_runs = 0;
    for (auto t : IndicesOf(touched_chunks)) {
      if (t == 0 || state_->BlockOf(touched_chunks[t]) !=
                        state_->BlockOf(touched_chunks[t - 1])) {
        ++num_runs;
      }
      run_of[t] = num_runs - 1;
    }
    std::vector<BlockRun> runs(num_runs);
    for (auto t : IndicesOf(touched_chunks)) {
      const auto& pairs = per_chunk_indices[touched_chunks[t]];
      uint64_t distinct = pairs.size();
      if (distinct > 1) {
        std::vector<uint64_t> inner;
        inner.reserve(pairs.size());
        for (const auto& p : pairs) inner.push_back(p.inner_record_index);
        std::sort(inner.begin(), inner.end());
        distinct = std::unique(inner.begin(), inner.end()) - inner.begin();
      }
      BlockRun& run = runs[run_of[t]];
      run.records += distinct;
      run.chunks_left.fetch_add(1, std::memory_order_relaxed);
    }
    // Configured once in Initialize(), so the read path skips MergeConfig's
    // lock. Instance() is needed only when the global was reset after fork().
    BlockCache* existing = BlockCache::InstanceIfExists();
    BlockCache& cache = existing != nullptr
                            ? *existing
                            : BlockCache::Instance(state_->CacheConfig());
    return ParallelForWithStatus<1>(
        IndicesOf(touched_chunks), state_->pool, [&](size_t t) -> absl::Status {
          const uint64_t chunk_idx = touched_chunks[t];
          const uint32_t b = state_->BlockOf(chunk_idx);
          const uint64_t block_offset = state_->BlockOffset(b);
          const uint64_t block_len = state_->BlockLen(b);
          AR_ENDO_JOB("ArrayRecordReaderBase::ParallelReadRecordsWithIndices",
                      absl::StrCat("chunk_idx: ", chunk_idx, " block_idx: ", b,
                                   " block_len: ", block_len));
          const BlockCache::Key key{state_->cache_reader_id, b};
          BlockCache::Block block = cache.GetOrFetch(
              key, block_len, [&]() -> absl::StatusOr<BlockCache::Block> {
                return FetchBlock(*reader, block_offset, block_len);
              });
          absl::Status s;
          if (block != nullptr) {
            MaskedReader masked(std::move(block), block_offset);
            s = serve_chunk(masked, chunk_idx);
          } else {
            const uint64_t chunk_offset = (*state_->chunk_offsets)[chunk_idx];
            MaskedReader masked(
                reader->NewReader(chunk_offset),
                state_->ChunkEndOffset(chunk_idx) - chunk_offset);
            s = serve_chunk(masked, chunk_idx);
          }
          if (ABSL_PREDICT_FALSE(!s.ok())) return s;
          // Counted after serving, so a block is never freed while one of
          // its records is still being read.
          BlockRun& run = runs[run_of[t]];
          if (run.chunks_left.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
              state_->ConsumeRecords(b, run.records)) {
            cache.DropIfReady(key);
          }
          return absl::OkStatus();
        });
  }

  std::vector<std::vector<uint64_t>> chunk_indices_per_buffer;
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
          state_->ChunkEndOffset(i) - chunk_indices_per_buffer.back()[0];
      if (buf_size < state_->options.readahead_buffer_size()) {
        chunk_indices_per_buffer.back().push_back(i);
      } else {
        chunk_indices_per_buffer.push_back({i});
      }
    }
  }
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
          auto s = serve_chunk(masked_reader, chunk_idx);
          if (ABSL_PREDICT_FALSE(!s.ok())) return s;
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
