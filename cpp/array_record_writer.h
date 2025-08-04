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

// High-level API for building an off-memory array-like data structure with
// riegeli. `ArrayRecordWriter` creates a valid riegeli file with all
// compression options standard riegeli writer supports, along with a footer and
// a postscript for indexing the records in the file.
//
// Riegeli files are composed in data chunks. Each data chunk contains multiple
// records, and a record can be a serialized proto (with a size limit of 2GB) or
// arbitrary bytes without a size limit.
//
// Each Riegeli data chunk is encoded/compressed separately. The chunks are the
// entry points for decoding, which allows us to read the chunks in parallel if
// we know where these entry points are.
//
// `ArrayRecordWriter` encodes additional chunks of records that index the
// chunks at the end of the file, as shown in the illustration below.
//
//  +-----------------+
//  |    User Data    |
//  |  Riegeli Chunk  |
//  +-----------------+
//  |    User Data    |
//  |  Riegeli Chunk  |
//  +-----------------+
// /\/\/\/\/\/\/\/\/\/\/
// /\/\/\/\/\/\/\/\/\/\/       _+-----------------------+
//  +-----------------+      _/ |  ArrayRecordMetadata  |
//  |  Last User Data |   __/   +-----------------------+
//  |      Chunk      | _/      |   Chunk Offset Proto  |
//  +-----------------_/        +-----------------------+
//  |                 |         |   Chunk Offset Proto  |
//  |  Footer Chunk   |         +-----------------------+
//  |                 |         |   Chunk Offset Proto  |
//  +-----------------+---------+-----------------------+
//  |   Postscript    | <--- Aligns and fits in 64KB.
//  +-----------------+
//
// While the output is a valid riegeli file, users should not use the standard
// `riegeli::RecordReader` to read the file; instead, users should use
// `ArrayRecordReader,` which parses the footer and provides parallel reading
// and random access features that the standard reader doesn't offer.
//
#ifndef ARRAY_RECORD_CPP_ARRAY_RECORD_WRITER_H_
#define ARRAY_RECORD_CPP_ARRAY_RECORD_WRITER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "cpp/common.h"
#include "cpp/sequenced_chunk_writer.h"
#include "cpp/thread_pool.h"
#include "cpp/tri_state_ptr.h"
#include "riegeli/base/initializer.h"
#include "riegeli/base/object.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/chunk_encoding/chunk_encoder.h"
#include "riegeli/chunk_encoding/compressor_options.h"
#include "riegeli/chunk_encoding/constants.h"
#include "riegeli/records/records_metadata.pb.h"

namespace array_record {

// Template parameter independent part of `ArrayRecordWriter`.
class ArrayRecordWriterBase : public riegeli::Object {
 public:
  ~ArrayRecordWriterBase() override;

  class Options {
   public:
    Options();

    // Parses options from text:
    // ```
    //   options ::= option? ("," option?)*
    //   option ::=
    //     "group_size" ":" group_size |
    //     "max_parallelism" ":" max_parallelism |
    //     "saturation_delay_ms" : saturation_delay_ms |
    //     "uncompressed" |
    //     "brotli" (":" brotli_level)? |
    //     "zstd" (":" zstd_level)? |
    //     "snappy" |
    //     "transpose" (":" ("true" | "false"))? |
    //     "transpose_bucket_size" ":" transpose_bucket_size |
    //     "window_log" : window_log |
    //     "pad_to_block_boundary" (":" ("true" | "false"))?
    //   group_size ::= positive integer which specifies number of records to be
    //     grouped into a chunk before compression. (default 65536)
    //   saturation_delay_ms ::= positive integer which specifies a delay in
    //     milliseconds when the parallel writing queue is saturated.
    //   max_parallelism ::= `auto` or positive integers which specifies
    //     max number of concurrent writers allowed.
    //   brotli_level ::= integer in the range [0..11] (default 6)
    //   zstd_level ::= integer in the range [-131072..22] (default 3)
    //   transpose_bucket_size ::= `auto` or a positive integer expressed as
    //     real with optional suffix [BkKMGTPE]. (default 256)
    //   window_log ::= "auto" or integer in the range [10..31]
    // ```
    static absl::StatusOr<Options> FromString(absl::string_view text);

    // Set the number of records to be grouped into a chunk in reiegeli. Each
    // chunk is compressed separately, and is the entry for random accessing.
    //
    // The larger the value, the denser the file, at the cost of more expansive
    // random accessing.
    static constexpr uint32_t kDefaultGroupSize = 65536;
    Options& set_group_size(uint32_t group_size) {
      group_size_ = group_size;
      return *this;
    }
    uint32_t group_size() const { return group_size_; }

    // Specifies max number of concurrent chunk encoders allowed. Default to the
    // thread pool size.
    Options& set_max_parallelism(std::optional<uint32_t> max_parallelism) {
      max_parallelism_ = max_parallelism;
      return *this;
    }
    std::optional<uint32_t> max_parallelism() const { return max_parallelism_; }

    static constexpr uint32_t kDefaultSaturationDelayMs = 10;

    // Specifies a delay in milliseconds when the parallel writing queue is
    // saturated.
    Options& set_saturation_delay_ms(uint32_t delay_ms) {
      saturation_delay_ms_ = delay_ms;
      return *this;
    }
    uint32_t saturation_delay_ms() const { return saturation_delay_ms_; }

    // Changes compression algorithm to Uncompressed (turns compression off).
    Options& set_uncompressed() {
      compressor_options_.set_uncompressed();
      return *this;
    }

    // Changes compression algorithm to Brotli. Sets compression level which
    // tunes the tradeoff between compression density and compression speed
    // (higher = better density but slower).
    //
    // `compression_level` must be between `kMinBrotli` (0) and
    // `kMaxBrotli` (11). Default: `kDefaultBrotli` (6).
    //
    // This is the default compression algorithm.
    static constexpr int kMinBrotli = riegeli::CompressorOptions::kMinBrotli;
    static constexpr int kMaxBrotli = riegeli::CompressorOptions::kMaxBrotli;
    static constexpr int kDefaultBrotli =
        riegeli::CompressorOptions::kDefaultBrotli;
    Options& set_brotli(int compression_level = kDefaultBrotli) {
      compressor_options_.set_brotli(compression_level);
      return *this;
    }

    // Changes compression algorithm to Zstd. Sets compression level which tunes
    // the tradeoff between compression density and compression speed (higher =
    // better density but slower).
    //
    // `compression_level` must be between `kMinZstd` (-131072) and
    // `kMaxZstd` (22). Level 0 is currently equivalent to 3.
    // Default: `kDefaultZstd` (3).
    static constexpr int kMinZstd = riegeli::CompressorOptions::kMinZstd;
    static constexpr int kMaxZstd = riegeli::CompressorOptions::kMaxZstd;
    static constexpr int kDefaultZstd =
        riegeli::CompressorOptions::kDefaultZstd;
    Options& set_zstd(int compression_level = kDefaultZstd) {
      compressor_options_.set_zstd(compression_level);
      return *this;
    }

    // Changes compression algorithm to Snappy.
    //
    // There are no Snappy compression levels to tune.
    Options& set_snappy() {
      compressor_options_.set_snappy();
      return *this;
    }

    // Logarithm of the LZ77 sliding window size. This tunes the tradeoff
    // between compression density and memory usage (higher = better density but
    // more memory).
    //
    // Special value `absl::nullopt` means to keep the default (Brotli: 22,
    // Zstd: 20)
    //
    // For Uncompressed and Snappy, `window_log` must be `absl::nullopt`.
    //
    // For Brotli, `window_log` must be `absl::nullopt` or between
    // `BrotliWriterBase::Options::kMinWindowLog` (10) and
    // `BrotliWriterBase::Options::kMaxWindowLog` (30).
    //
    // For Zstd, `window_log` must be `absl::nullopt` or between
    // `ZstdWriterBase::Options::kMinWindowLog` (10) and
    // `ZstdWriterBase::Options::kMaxWindowLog` (30 in 32-bit build,
    // 31 in 64-bit build).
    //
    // Default: `absl::nullopt`.
    Options& set_window_log(absl::optional<int> window_log) {
      compressor_options_.set_window_log(window_log);
      return *this;
    }
    absl::optional<int> window_log() const {
      return compressor_options_.window_log();
    }

    // Pads to 64KB boundary for every chunk. Padding to block boundaries allows
    // the standard Riegeli record reader to skip records by finding the
    // subsequent block offsets if some chunks were corrupted.
    //
    // It is not as useful in ArrayRecordReader because we get all the chunk
    // offsets from the footer.
    //
    // Default: `false`
    Options& set_pad_to_block_boundary(bool pad_to_block_boundary) {
      pad_to_block_boundary_ = pad_to_block_boundary;
      return *this;
    }
    bool pad_to_block_boundary() const { return pad_to_block_boundary_; }

    // If `true`, records should be serialized proto messages (but nothing will
    // break if they are not). A chunk of records will be processed in a way
    // which allows for better compression.
    //
    // If `false`, a chunk of records will be stored in a simpler format,
    // directly or with compression.
    //
    // Default: `false`.
    Options& set_transpose(bool transpose) {
      transpose_ = transpose;
      return *this;
    }
    bool transpose() const { return transpose_; }

    // Sets the column size for non-proto records.
    static constexpr uint64_t kDefaultTransposeBucketSize = 256;
    Options& set_transpose_bucket_size(uint64_t bucket_size) {
      transpose_bucket_size_ = bucket_size;
      return *this;
    }
    uint64_t transpose_bucket_size() const { return transpose_bucket_size_; }

    // Returns the compression type
    riegeli::CompressionType compression_type() const {
      return compressor_options_.compression_type();
    }

    // Returns the compression level
    int compression_level() const {
      return compressor_options_.compression_level();
    }

    const riegeli::CompressorOptions& compressor_options() const {
      return compressor_options_;
    }

    // Sets file metadata to be written at the beginning (unless
    // `absl::nullopt`).
    //
    // Default: no fields set.
    Options& set_metadata(
        const std::optional<riegeli::RecordsMetadata>& metadata) {
      metadata_ = metadata;
      return *this;
    }
    std::optional<riegeli::RecordsMetadata> metadata() const {
      return metadata_;
    }

    // Serialize the options to a string.
    std::string ToString() const;

   private:
    int32_t group_size_ = kDefaultGroupSize;
    riegeli::CompressorOptions compressor_options_;
    std::optional<riegeli::RecordsMetadata> metadata_;
    bool pad_to_block_boundary_ = false;
    bool transpose_ = false;
    uint64_t transpose_bucket_size_ = kDefaultTransposeBucketSize;
    std::optional<uint32_t> max_parallelism_ = std::nullopt;
    int32_t saturation_delay_ms_ = kDefaultSaturationDelayMs;
  };

  // Write records of various types.
  bool WriteRecord(const google::protobuf::MessageLite& record);
  bool WriteRecord(absl::string_view record);
  bool WriteRecord(const absl::Cord& record);
  bool WriteRecord(const void* data, size_t num_bytes);
  template <typename T>
  bool WriteRecord(absl::Span<const T> record) {
    return WriteRecord(record.data(),
                       record.size() * sizeof(decltype(record)::value_type));
  }

 protected:
  ArrayRecordWriterBase(Options options, ARThreadPool* pool);

  // Move only, but we need to override the default for closing the rvalues.
  ArrayRecordWriterBase(ArrayRecordWriterBase&& other) noexcept;
  ArrayRecordWriterBase& operator=(ArrayRecordWriterBase&& other) noexcept;

  virtual TriStatePtr<SequencedChunkWriterBase>::SharedRef get_writer() = 0;

  // Initializes and validates the underlying writer states.
  void Initialize();

  // Callback for riegeli::Object::Close.
  void Done() override;

 private:
  std::unique_ptr<riegeli::ChunkEncoder> CreateEncoder();
  template <typename Record>
  bool WriteRecordImpl(Record&& record);

  // Pimpl (pointer to implementation)
  // http://bitboom.github.io/pimpl-idiom
  class SubmitChunkCallback;

  Options options_;
  ARThreadPool* pool_;
  std::unique_ptr<riegeli::ChunkEncoder> chunk_encoder_;
  std::unique_ptr<SubmitChunkCallback> submit_chunk_callback_;
};

// `ArrayRecordWriter` use templated backend abstraction. To serialize the
// output to a string, user simply write:
//
//   std::string dest;
//   ArrayRecordWriter writes_to_string(
//       riegeli::Maker<riegeli::StringWriter>(&dest));
//
// Similarly, user can write the output to a cord or to a file.
//
//   absl::Cord cord;
//   ArrayRecordWriter writes_to_cord(
//       riegeli::Maker<riegeli::CordWriter>(&cord));
//
//   ArrayRecordWriter writes_to_file(
//       riegeli::Maker<riegeli::FileWriter>(filename_or_file));
//
// It is necessary to call `Close()` at the end of a successful writing session,
// and it is recommended to call `Close()` at the end of a successful reading
// session. It is not needed to call `Close()` on early returns, assuming that
// contents of the destination do not matter after all, e.g.  because a failure
// is being reported instead; the destructor releases resources in any case.
//
// Error handling example:
//
//   // Just like RET_CHECK and RETURN_OR_ERROR
//   if (!writer.WriteRecord(...)) return writer.status();
//   // writer doesn't close on destruction, user must call `Close()` and check
//   // the status.
//   if (!writer.Close()) return writer.status();
//
template <typename Dest = riegeli::Writer*>
class ArrayRecordWriter : public ArrayRecordWriterBase {
 public:
  DECLARE_MOVE_ONLY_CLASS(ArrayRecordWriter);

  // Will write to the `Writer` provided by `dest`.
  explicit ArrayRecordWriter(riegeli::Initializer<Dest> dest,
                             Options options = Options(),
                             ARThreadPool* pool = nullptr)
      : ArrayRecordWriterBase(std::move(options), pool),
        main_writer_(std::make_unique<TriStatePtr<SequencedChunkWriterBase>>(
            std::make_unique<SequencedChunkWriter<Dest>>(std::move(dest)))) {
    auto writer = get_writer();
    if (!writer->ok()) {
      Fail(writer->status());
      return;
    }
    Initialize();
  }

 protected:
  TriStatePtr<SequencedChunkWriterBase>::SharedRef get_writer() final {
    return main_writer_->MakeShared();
  }

  void Done() override {
    if (main_writer_ == nullptr) return;
    ArrayRecordWriterBase::Done();
    if (!ok()) {
      return;
    }
    // Ensures all pending tasks are finished.
    auto unique = main_writer_->WaitAndMakeUnique();
    if (!unique->Close()) Fail(unique->status());
  }

 private:
  std::unique_ptr<TriStatePtr<SequencedChunkWriterBase>> main_writer_;
};

template <typename Dest>
explicit ArrayRecordWriter(
    Dest&& dest,
    ArrayRecordWriterBase::Options options = ArrayRecordWriterBase::Options(),
    ARThreadPool* pool = nullptr) -> ArrayRecordWriter<riegeli::TargetT<Dest>>;

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_ARRAY_RECORD_WRITER_H_
