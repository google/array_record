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

#include "cpp/array_record_writer.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "cpp/common.h"
#include "cpp/layout.pb.h"
#include "cpp/sequenced_chunk_writer.h"
#include "cpp/thread_pool.h"
#include "cpp/tri_state_ptr.h"
#include "google/protobuf/message_lite.h"
#include "riegeli/base/object.h"
#include "riegeli/base/options_parser.h"
#include "riegeli/base/status.h"
#include "riegeli/bytes/chain_writer.h"
#include "riegeli/chunk_encoding/chunk.h"
#include "riegeli/chunk_encoding/chunk_encoder.h"
#include "riegeli/chunk_encoding/compressor_options.h"
#include "riegeli/chunk_encoding/constants.h"
#include "riegeli/chunk_encoding/deferred_encoder.h"
#include "riegeli/chunk_encoding/simple_encoder.h"
#include "riegeli/chunk_encoding/transpose_encoder.h"

namespace array_record {

// This number should rarely change unless there's a new great layout design
// that wasn't backward compatible and justifies its performance and reliability
// worth us to implement.
constexpr uint32_t kArrayRecordV1 = 1;

// Heuristic default for fast compression.
constexpr uint32_t kZstdDefaultWindowLog = 20;

// Magic number for ArrayRecord
// Generated from `echo 'ArrayRecord' | md5sum | cut -b 1-16`
constexpr uint64_t kMagic = 0x71930e704fdae05eULL;

// zstd:3 gives a good trade-off for both the compression and decompression
// speed.
constexpr char kArrayRecordDefaultCompression[] = "zstd:3";

using riegeli::Chunk;
using riegeli::ChunkType;
using riegeli::CompressorOptions;
using riegeli::OptionsParser;
using riegeli::ValueParser;

namespace {

// Helper function for dumping the encoded data to a chunk.
absl::StatusOr<Chunk> EncodeChunk(riegeli::ChunkEncoder* encoder) {
  Chunk chunk;
  ChunkType chunk_type;
  uint64_t decoded_data_size;
  uint64_t num_records;
  riegeli::ChainWriter chain_writer(&chunk.data);
  if (!encoder->EncodeAndClose(chain_writer, chunk_type, num_records,
                               decoded_data_size)) {
    return encoder->status();
  }
  if (!chain_writer.Close()) {
    return chain_writer.status();
  }
  chunk.header = riegeli::ChunkHeader(chunk.data, chunk_type, num_records,
                                      decoded_data_size);
  return chunk;
}

// Helper function for generating a chunk from a span with an optional header.
template <typename T, typename H = std::string>
absl::StatusOr<Chunk> ChunkFromSpan(CompressorOptions compression_options,
                                    absl::Span<const T> span,
                                    std::optional<H> header = std::nullopt) {
  riegeli::SimpleEncoder encoder(
      compression_options,
      riegeli::SimpleEncoder::TuningOptions().set_size_hint(
          span.size() * sizeof(typename decltype(span)::value_type)));
  if (header.has_value()) {
    encoder.AddRecord(header.value());
  }
  for (const auto& record : span) {
    encoder.AddRecord(record);
  }
  return EncodeChunk(&encoder);
}

}  // namespace

ArrayRecordWriterBase::Options::Options() {
  this->compressor_options_.FromString(kArrayRecordDefaultCompression)
      .IgnoreError();
}

// static
absl::StatusOr<ArrayRecordWriterBase::Options>
ArrayRecordWriterBase::Options::FromString(absl::string_view text) {
  ArrayRecordWriterBase::Options options;
  OptionsParser options_parser;
  options_parser.AddOption("default", ValueParser::FailIfAnySeen());
  // Group
  options_parser.AddOption(
      "group_size", ValueParser::Int(1, INT32_MAX, &options.group_size_));
  int32_t max_parallelism = 0;
  options_parser.AddOption(
      "max_parallelism",
      ValueParser::Or(ValueParser::Enum({{"auto", std::nullopt}},
                                        &options.max_parallelism_),
                      ValueParser::Int(1, INT32_MAX, &max_parallelism)));
  options_parser.AddOption(
      "saturation_delay_ms",
      ValueParser::Int(1, INT32_MAX, &options.saturation_delay_ms_));
  // Transpose
  options_parser.AddOption(
      "transpose",
      ValueParser::Enum({{"", true}, {"true", true}, {"false", false}},
                        &options.transpose_));
  options_parser.AddOption(
      "transpose_bucket_size",
      ValueParser::Or(
          ValueParser::Enum({{"auto", kDefaultTransposeBucketSize}},
                            &options.transpose_bucket_size_),
          ValueParser::Bytes(1, std::numeric_limits<uint64_t>::max(),
                             &options.transpose_bucket_size_)));
  // Compressor
  std::string compressor_text;
  options_parser.AddOption("uncompressed",
                           ValueParser::CopyTo(&compressor_text));
  options_parser.AddOption("brotli", ValueParser::CopyTo(&compressor_text));
  options_parser.AddOption("zstd", ValueParser::CopyTo(&compressor_text));
  options_parser.AddOption("snappy", ValueParser::CopyTo(&compressor_text));
  options_parser.AddOption("window_log", ValueParser::CopyTo(&compressor_text));
  // Padding
  options_parser.AddOption(
      "pad_to_block_boundary",
      ValueParser::Enum({{"", true}, {"true", true}, {"false", false}},
                        &options.pad_to_block_boundary_));
  if (!options_parser.FromString(text)) {
    return options_parser.status();
  }
  // From our benchmarks we figured zstd:3 gives the best trade-off for both the
  // compression and decomopression speed.
  if (text == "default" ||
      (!absl::StrContains(compressor_text, "uncompressed") &&
       !absl::StrContains(compressor_text, "brotli") &&
       !absl::StrContains(compressor_text, "snappy") &&
       !absl::StrContains(compressor_text, "zstd"))) {
    absl::StrAppend(&compressor_text, ",", kArrayRecordDefaultCompression);
  }
  // max_parallelism is set after options_parser.FromString()
  if (max_parallelism > 0) {
    options.set_max_parallelism(max_parallelism);
  }
  auto opt_status = options.compressor_options_.FromString(compressor_text);
  if (!opt_status.ok()) {
    return opt_status;
  }
  if (options.compressor_options().compression_type() ==
          riegeli::CompressionType::kZstd &&
      !options.compressor_options().zstd_window_log().has_value()) {
    options.compressor_options_.set_window_log(kZstdDefaultWindowLog);
  }
  return options;
}

std::string ArrayRecordWriterBase::Options::ToString() const {
  std::string option;
  absl::StrAppend(&option, "group_size:", this->group_size_,
                  ",transpose:", this->transpose_ ? "true" : "false",
                  ",pad_to_block_boundary:",
                  this->pad_to_block_boundary_ ? "true" : "false");
  if (this->transpose_) {
    absl::StrAppend(&option,
                    ",transpose_bucket_size:", this->transpose_bucket_size_);
  }
  switch (this->compressor_options().compression_type()) {
    case riegeli::CompressionType::kNone:
      absl::StrAppend(&option, ",uncompressed");
      break;
    case riegeli::CompressionType::kBrotli:
      absl::StrAppend(
          &option, ",brotli:", this->compressor_options().compression_level());
      break;
    case riegeli::CompressionType::kZstd:
      absl::StrAppend(&option,
                      ",zstd:", this->compressor_options().compression_level());
      break;
    case riegeli::CompressionType::kSnappy:
      absl::StrAppend(&option, ",snappy");
      break;
  }
  if (this->compressor_options().window_log().has_value()) {
    absl::StrAppend(&option, ",window_log:",
                    this->compressor_options().window_log().value());
  }
  if (max_parallelism_.has_value()) {
    absl::StrAppend(&option, ",max_parallelism:", max_parallelism_.value());
  }
  return option;
}

// Thread compatible callback guarded by SequencedChunkWriter's mutex.
class ArrayRecordWriterBase::SubmitChunkCallback
    : public SequencedChunkWriterBase::SubmitChunkCallback {
 public:
  explicit SubmitChunkCallback(const ArrayRecordWriterBase::Options options)
      : options_(options), max_parallelism_(options.max_parallelism().value()) {
    constexpr uint64_t kDefaultDecodedDataSize = (1 << 20);
    last_decoded_data_size_.store(kDefaultDecodedDataSize);
  }
  DECLARE_IMMOBILE_CLASS(SubmitChunkCallback);

  // Callback invoked by SequencedChunkWriter.
  void operator()(uint64_t chunk_seq, uint64_t chunk_offset,
                  uint64_t decoded_data_size, uint64_t num_records) override;

  // return false if we can't schedule the callback.
  // return true and inc num_concurrent_chunk_writers if we can add a new one.
  bool TrackConcurrentChunkWriters() {
    absl::MutexLock l(mu_);
    if (num_concurrent_chunk_writers_ >= max_parallelism_) {
      return false;
    }
    num_concurrent_chunk_writers_++;
    return true;
  }

  // riegeli::ChunkEncoder requires a "size hint" in its input. This request
  // comes from a separated thread therefore we use an atomic for thread safety.
  uint64_t get_last_decoded_data_size() const {
    return last_decoded_data_size_.load();
  }

  // Aggregate the offsets information and write it to the file.
  void WriteFooterAndPostscript(
      TriStatePtr<SequencedChunkWriterBase>::SharedRef writer);

 private:
  const Options options_;

  absl::Mutex mu_;
  const int32_t max_parallelism_;
  int32_t num_concurrent_chunk_writers_ ABSL_GUARDED_BY(mu_) = 0;
  friend class absl::Condition;

  // Helper method for creating the footer chunk.
  absl::StatusOr<Chunk> CreateFooterChunk();

  bool encoding_postscript_ = false;
  std::atomic<uint64_t> last_decoded_data_size_;
  std::vector<RiegeliPostscript> postscript_;
  std::vector<ArrayRecordFooter> array_footer_;
};

ArrayRecordWriterBase::ArrayRecordWriterBase(Options options,
                                             ARThreadPool* pool)
    : options_(std::move(options)), pool_(pool) {}
ArrayRecordWriterBase::~ArrayRecordWriterBase() = default;

ArrayRecordWriterBase::ArrayRecordWriterBase(
    ArrayRecordWriterBase&& other) noexcept
    : riegeli::Object(std::move(other)),
      options_(std::move(other.options_)),
      pool_(other.pool_),
      chunk_encoder_(std::move(other.chunk_encoder_)),
      submit_chunk_callback_(std::move(other.submit_chunk_callback_)) {
  other.pool_ = nullptr;
  other.Reset(riegeli::kClosed);
}

namespace {
template <typename T>
void SilenceMoveAfterUse(T&) {}
}  // namespace

ArrayRecordWriterBase& ArrayRecordWriterBase::operator=(
    ArrayRecordWriterBase&& other) noexcept {
  riegeli::Object::operator=(std::move(other));
  // Using `other` after it was moved is correct because only the base class
  // fields were moved. Clang tidy recommended the following template to silence
  // the warning.
  // https://clang.llvm.org/extra/clang-tidy/checks/bugprone-use-after-move.html
  SilenceMoveAfterUse(other);
  options_ = std::move(other.options_);
  pool_ = other.pool_;
  other.pool_ = nullptr;
  chunk_encoder_ = std::move(other.chunk_encoder_);
  submit_chunk_callback_ = std::move(other.submit_chunk_callback_);
  other.Reset(riegeli::kClosed);
  return *this;
}

void ArrayRecordWriterBase::Initialize() {
  uint32_t max_parallelism = 1;
  if (pool_) {
    max_parallelism = pool_->NumThreads();
    if (options_.max_parallelism().has_value()) {
      max_parallelism =
          std::min(max_parallelism, options_.max_parallelism().value());
    }
  }
  options_.set_max_parallelism(max_parallelism);

  submit_chunk_callback_ = std::make_unique<SubmitChunkCallback>(options_);
  chunk_encoder_ = CreateEncoder();
  auto writer = get_writer();
  writer->set_pad_to_block_boundary(options_.pad_to_block_boundary());
  if (options_.metadata().has_value()) {
    riegeli::TransposeEncoder encoder(options_.compressor_options());
    Chunk chunk;
    ChunkType chunk_type;
    uint64_t decoded_data_size;
    uint64_t num_records;
    riegeli::ChainWriter chain_writer(&chunk.data);
    if (!encoder.AddRecord(options_.metadata().value())) {
      Fail(encoder.status());
    }
    if (!encoder.EncodeAndClose(chain_writer, chunk_type, num_records,
                                decoded_data_size)) {
      Fail(encoder.status());
    }
    if (!chain_writer.Close()) {
      Fail(chain_writer.status());
    }
    chunk.header = riegeli::ChunkHeader(chunk.data, ChunkType::kFileMetadata, 0,
                                        decoded_data_size);
    std::promise<absl::StatusOr<Chunk>> chunk_promise;
    writer->CommitFutureChunk(chunk_promise.get_future());
    chunk_promise.set_value(chunk);
    if (!writer->SubmitFutureChunks(true)) {
      Fail(writer->status());
    }
  }
  // Add callback only after we serialize header and metadata.
  writer->set_submit_chunk_callback(submit_chunk_callback_.get());
}

void ArrayRecordWriterBase::Done() {
  if (!ok()) {
    return;
  }
  auto writer = get_writer();
  if (!writer->ok()) {
    Fail(riegeli::Annotate(writer->status(), "SequencedChunkWriter failure"));
    return;
  }
  if (chunk_encoder_ == nullptr) {
    Fail(absl::InternalError("chunk_encoder_ should not be a nullptr."));
    return;
  }
  if (chunk_encoder_->num_records() > 0) {
    std::promise<absl::StatusOr<Chunk>> chunk_promise;
    if (!writer->CommitFutureChunk(chunk_promise.get_future())) {
      Fail(riegeli::Annotate(writer->status(), "Cannot commit future chunk"));
      return;
    }
    chunk_promise.set_value(EncodeChunk(chunk_encoder_.get()));
  }
  submit_chunk_callback_->WriteFooterAndPostscript(std::move(writer));
}

std::unique_ptr<riegeli::ChunkEncoder> ArrayRecordWriterBase::CreateEncoder() {
  std::unique_ptr<riegeli::ChunkEncoder> encoder;
  if (options_.transpose()) {
    encoder = std::make_unique<riegeli::TransposeEncoder>(
        options_.compressor_options(),
        riegeli::TransposeEncoder::TuningOptions().set_bucket_size(
            options_.transpose_bucket_size()));
  } else {
    encoder = std::make_unique<riegeli::SimpleEncoder>(
        options_.compressor_options(),
        riegeli::SimpleEncoder::TuningOptions().set_size_hint(
            submit_chunk_callback_->get_last_decoded_data_size()));
  }
  if (pool_) {
    return std::make_unique<riegeli::DeferredEncoder>(std::move(encoder));
  }
  return encoder;
}

bool ArrayRecordWriterBase::WriteRecord(const google::protobuf::MessageLite& record) {
  return WriteRecordImpl(record);
}

bool ArrayRecordWriterBase::WriteRecord(absl::string_view record) {
  return WriteRecordImpl(std::move(record));
}

bool ArrayRecordWriterBase::WriteRecord(const absl::Cord& record) {
  if (auto flat = record.TryFlat(); flat.has_value()) {
    return WriteRecord(*flat);
  }

  std::string cord_string;
  absl::AppendCordToString(record, &cord_string);
  return WriteRecord(cord_string);
}

bool ArrayRecordWriterBase::WriteRecord(const void* data, size_t num_bytes) {
  auto view = absl::string_view(reinterpret_cast<const char*>(data), num_bytes);
  return WriteRecordImpl(std::move(view));
}

template <typename Record>
bool ArrayRecordWriterBase::WriteRecordImpl(Record&& record) {
  if (!ok()) {
    return false;
  }
  if (!chunk_encoder_->AddRecord(std::forward<Record>(record))) {
    Fail(chunk_encoder_->status());
    return false;
  }
  if (chunk_encoder_->num_records() >= options_.group_size()) {
    auto writer = get_writer();
    auto encoder = std::move(chunk_encoder_);
    auto chunk_promise =
        std::make_shared<std::promise<absl::StatusOr<Chunk>>>();
    if (!writer->CommitFutureChunk(chunk_promise->get_future())) {
      Fail(writer->status());
      return false;
    }
    chunk_encoder_ = CreateEncoder();
    if (pool_ && options_.max_parallelism().value() > 1) {
      std::shared_ptr<riegeli::ChunkEncoder> shared_encoder =
          std::move(encoder);
      submit_chunk_callback_->TrackConcurrentChunkWriters();
      pool_->Schedule([writer, shared_encoder, chunk_promise]() mutable {
        AR_ENDO_TASK("Encode riegeli chunk");
        chunk_promise->set_value(EncodeChunk(shared_encoder.get()));
        writer->SubmitFutureChunks(false);
      });
      return true;
    }
    chunk_promise->set_value(EncodeChunk(encoder.get()));
    if (!writer->SubmitFutureChunks(true)) {
      Fail(writer->status());
      return false;
    }
  }
  return true;
}

void ArrayRecordWriterBase::SubmitChunkCallback::operator()(
    uint64_t chunk_seq, uint64_t chunk_offset, uint64_t decoded_data_size,
    uint64_t num_records) {
  if (encoding_postscript_) {
    // Happens when we submit a footer chunk. We need to record the chunk
    // offset of the footer chunk in the postscript to locate where the footer
    // starts.
    RiegeliPostscript postscript;
    postscript.set_footer_offset(chunk_offset);
    postscript.set_magic(kMagic);
    for (auto _ : Seq(3)) {
      postscript_.push_back(postscript);
    }
    return;
  }
  // Collects chunk offsets for each user supplied chunks.
  ArrayRecordFooter footer;
  footer.set_chunk_offset(chunk_offset);
  footer.set_decoded_data_size(decoded_data_size);
  footer.set_num_records(num_records);
  array_footer_.push_back(std::move(footer));

  absl::MutexLock l(mu_);
  num_concurrent_chunk_writers_--;
}

void ArrayRecordWriterBase::SubmitChunkCallback::WriteFooterAndPostscript(
    TriStatePtr<SequencedChunkWriterBase>::SharedRef writer) {
  // Flushes prior chunks
  writer->SubmitFutureChunks(true);
  // Footer and postscript must pad to block boundary
  writer->set_pad_to_block_boundary(true);

  // Writes footer
  {
    AR_ENDO_TASK("Encode ArrayRecord footer chunk");
    std::promise<absl::StatusOr<Chunk>> footer_promise;
    writer->CommitFutureChunk(footer_promise.get_future());
    footer_promise.set_value(CreateFooterChunk());
    writer->SubmitFutureChunks(true);
  }

  if (!writer->ok()) {
    return;
  }

  // Writes postscript
  std::promise<absl::StatusOr<Chunk>> postscript_promise;
  writer->CommitFutureChunk(postscript_promise.get_future());
  postscript_promise.set_value(
      ChunkFromSpan(CompressorOptions().set_uncompressed(),
                    absl::MakeConstSpan(postscript_)));
  writer->SubmitFutureChunks(true);
}

absl::StatusOr<Chunk>
ArrayRecordWriterBase::SubmitChunkCallback::CreateFooterChunk() {
  if (encoding_postscript_) {
    return absl::FailedPreconditionError(
        "CreateFooterChunk should only be called once");
  }
  encoding_postscript_ = true;
  RiegeliFooterMetadata footer_metadata;
  uint64_t num_records = 0;
  for (const auto& footer : array_footer_) {
    num_records += footer.num_records();
  }
  footer_metadata.mutable_array_record_metadata()->set_version(kArrayRecordV1);
  footer_metadata.mutable_array_record_metadata()->set_num_chunks(
      array_footer_.size());
  footer_metadata.mutable_array_record_metadata()->set_num_records(num_records);
  footer_metadata.mutable_array_record_metadata()->set_writer_options(
      options_.ToString());
  // Perhaps we can compress the footer
  return ChunkFromSpan(options_.compressor_options(),
                       absl::MakeConstSpan(array_footer_),
                       std::optional(footer_metadata));
}

}  // namespace array_record
