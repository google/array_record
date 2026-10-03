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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "cpp/array_record_writer.h"
#include "cpp/common.h"
#include "cpp/layout.pb.h"
#include "cpp/test_utils.h"
#include "cpp/thread_pool.h"
#include "riegeli/base/maker.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/chain_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/chunk_encoding/chunk.h"
#include "riegeli/chunk_encoding/chunk_decoder.h"
#include "riegeli/chunk_encoding/compressor_options.h"
#include "riegeli/chunk_encoding/constants.h"
#include "riegeli/chunk_encoding/simple_encoder.h"
#include "riegeli/records/chunk_reader.h"
#include "riegeli/records/chunk_writer.h"

constexpr uint32_t kDatasetSize = 3210;

namespace array_record {
namespace {

enum class CompressionType { kUncompressed, kBrotli, kZstd, kSnappy };

using IndexStorageOption = ArrayRecordReaderBase::Options::IndexStorageOption;

// Tuple params
//   CompressionType
//   transpose
//   use_thread_pool
//   optimize_for_random_access
//   index_storage_option
class ArrayRecordReaderTest
    : public testing::TestWithParam<
          std::tuple<CompressionType, bool, bool, bool, IndexStorageOption>> {
 public:
  ARThreadPool* get_pool() { return ArrayRecordGlobalPool(); }
  ArrayRecordWriterBase::Options GetWriterOptions() {
    auto options = ArrayRecordWriterBase::Options();
    switch (std::get<0>(GetParam())) {
      case CompressionType::kUncompressed:
        options.set_uncompressed();
        break;
      case CompressionType::kBrotli:
        options.set_brotli();
        break;
      case CompressionType::kZstd:
        options.set_zstd();
        break;
      case CompressionType::kSnappy:
        options.set_snappy();
        break;
    }
    options.set_transpose(transpose());
    return options;
  }

  bool transpose() { return std::get<1>(GetParam()); }
  bool use_thread_pool() { return std::get<2>(GetParam()); }
  bool optimize_for_random_access() { return std::get<3>(GetParam()); }
  IndexStorageOption index_storage_option() { return std::get<4>(GetParam()); }
};

TEST_P(ArrayRecordReaderTest, MoveTest) {
  std::string encoded;
  auto writer_options = GetWriterOptions().set_group_size(2);
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);

  // Empty string should not crash the writer or the reader.
  std::vector<std::string> test_str{"aaa", "", "ccc", "dd", "e"};
  for (const auto& s : test_str) {
    EXPECT_TRUE(writer.WriteRecord(s));
  }
  ASSERT_TRUE(writer.Close());

  auto reader_opt = ArrayRecordReaderBase::Options();
  reader_opt.set_index_storage_option(index_storage_option());
  if (optimize_for_random_access()) {
    reader_opt.set_max_parallelism(0);
    reader_opt.set_readahead_buffer_size(0);
  }

  auto reader_before_move =
      ArrayRecordReader(riegeli::Maker<riegeli::StringReader>(encoded),
                        reader_opt, use_thread_pool() ? get_pool() : nullptr);
  ASSERT_TRUE(reader_before_move.status().ok());

  ASSERT_TRUE(
      reader_before_move
          .ParallelReadRecords([&](uint64_t record_index,
                                   absl::string_view record) -> absl::Status {
            EXPECT_EQ(record, test_str[record_index]);
            return absl::OkStatus();
          })
          .ok());

  EXPECT_EQ(reader_before_move.RecordGroupSize(), 2);

  ArrayRecordReader reader = std::move(reader_before_move);
  // Once a reader is moved, it is closed.
  ASSERT_FALSE(reader_before_move.is_open());  // NOLINT

  auto recorded_writer_options = ArrayRecordWriterBase::Options::FromString(
                                     reader.WriterOptionsString().value())
                                     .value();
  EXPECT_EQ(writer_options.compression_type(),
            recorded_writer_options.compression_type());
  EXPECT_EQ(writer_options.compression_level(),
            recorded_writer_options.compression_level());
  EXPECT_EQ(writer_options.transpose(), recorded_writer_options.transpose());

  std::vector<uint64_t> indices = {1, 2, 4};
  ASSERT_TRUE(reader
                  .ParallelReadRecordsWithIndices(
                      indices,
                      [&](uint64_t indices_idx,
                          absl::string_view record) -> absl::Status {
                        EXPECT_EQ(record, test_str[indices[indices_idx]]);
                        return absl::OkStatus();
                      })
                  .ok());

  absl::string_view record_view;
  for (auto i : IndicesOf(test_str)) {
    EXPECT_TRUE(reader.ReadRecord(&record_view));
    EXPECT_EQ(record_view, test_str[i]);
  }
  // Cannot read once we are at the end of the file.
  EXPECT_FALSE(reader.ReadRecord(&record_view));
  // But the reader should still be healthy.
  EXPECT_TRUE(reader.ok());

  // Seek to a particular record works.
  EXPECT_TRUE(reader.SeekRecord(2));
  EXPECT_TRUE(reader.ReadRecord(&record_view));
  EXPECT_EQ(record_view, test_str[2]);

  // Seek out of bound would not fail.
  EXPECT_TRUE(reader.SeekRecord(10));
  EXPECT_FALSE(reader.ReadRecord(&record_view));
  EXPECT_TRUE(reader.ok());

  EXPECT_EQ(reader.RecordGroupSize(), 2);

  ASSERT_TRUE(reader.Close());
}

TEST_P(ArrayRecordReaderTest, RandomDatasetTest) {
  std::mt19937 bitgen;
  std::vector<std::string> records(kDatasetSize);
  std::uniform_int_distribution<> dist(0, 123);
  for (auto i : Seq(kDatasetSize)) {
    size_t len = dist(bitgen);
    records[i] = MTRandomBytes(bitgen, len);
  }

  std::string encoded;
  auto writer =
      ArrayRecordWriter(riegeli::Maker<riegeli::StringWriter>(&encoded),
                        GetWriterOptions(), get_pool());
  for (auto i : Seq(kDatasetSize)) {
    EXPECT_TRUE(writer.WriteRecord(records[i]));
  }
  ASSERT_TRUE(writer.Close());

  auto reader_opt = ArrayRecordReaderBase::Options();
  reader_opt.set_index_storage_option(index_storage_option());
  if (optimize_for_random_access()) {
    reader_opt.set_max_parallelism(0);
    reader_opt.set_readahead_buffer_size(0);
  }

  auto reader =
      ArrayRecordReader(riegeli::Maker<riegeli::StringReader>(encoded),
                        reader_opt, use_thread_pool() ? get_pool() : nullptr);
  ASSERT_TRUE(reader.status().ok());
  EXPECT_EQ(reader.NumRecords(), kDatasetSize);
  uint64_t group_size =
      std::min(ArrayRecordWriterBase::Options::kDefaultGroupSize, kDatasetSize);
  EXPECT_EQ(reader.RecordGroupSize(), group_size);

  // vector bool is casted as bit string, which is not thread safe to write.
  std::vector<int> read_all_records(kDatasetSize, 0);
  ASSERT_TRUE(reader
                  .ParallelReadRecords(
                      [&](uint64_t record_index,
                          absl::string_view result_view) -> absl::Status {
                        EXPECT_EQ(result_view, records[record_index]);
                        EXPECT_FALSE(read_all_records[record_index]);
                        read_all_records[record_index] = 1;
                        return absl::OkStatus();
                      })
                  .ok());
  for (auto record_was_read : read_all_records) {
    EXPECT_TRUE(record_was_read);
  }

  std::vector<uint64_t> indices = {0, 3, 5, 7, 101, 2000};
  // vector bool is casted as bit string, which is not thread safe to write.
  std::vector<int> read_indexed_records(indices.size(), 0);
  ASSERT_TRUE(reader
                  .ParallelReadRecordsWithIndices(
                      indices,
                      [&](uint64_t indices_idx,
                          absl::string_view result_view) -> absl::Status {
                        EXPECT_EQ(result_view, records[indices[indices_idx]]);
                        EXPECT_FALSE(read_indexed_records[indices_idx]);
                        read_indexed_records[indices_idx] = 1;
                        return absl::OkStatus();
                      })
                  .ok());
  for (auto record_was_read : read_indexed_records) {
    EXPECT_TRUE(record_was_read);
  }

  uint64_t begin = 10, end = 101;
  // vector bool is casted as bit string, which is not thread safe to write.
  std::vector<int> read_range_records(end - begin, 0);
  ASSERT_TRUE(reader
                  .ParallelReadRecordsInRange(
                      begin, end,
                      [&](uint64_t record_index,
                          absl::string_view result_view) -> absl::Status {
                        EXPECT_EQ(result_view, records[record_index]);
                        EXPECT_FALSE(read_range_records[record_index - begin]);
                        read_range_records[record_index - begin] = 1;
                        return absl::OkStatus();
                      })
                  .ok());
  for (auto record_was_read : read_range_records) {
    EXPECT_TRUE(record_was_read);
  }

  // Test sequential read
  absl::string_view result_view;
  for (auto record_index : Seq(kDatasetSize)) {
    ASSERT_TRUE(reader.ReadRecord(&result_view));
    EXPECT_EQ(result_view, records[record_index]);
  }
  // Reached to the end.
  EXPECT_FALSE(reader.ReadRecord(&result_view));
  EXPECT_TRUE(reader.ok());
  // We can still seek back.
  EXPECT_TRUE(reader.SeekRecord(5));
  EXPECT_TRUE(reader.ReadRecord(&result_view));
  EXPECT_EQ(result_view, records[5]);

  ASSERT_TRUE(reader.Close());
}

INSTANTIATE_TEST_SUITE_P(
    ParamTest, ArrayRecordReaderTest,
    testing::Combine(testing::Values(CompressionType::kUncompressed,
                                     CompressionType::kBrotli,
                                     CompressionType::kZstd,
                                     CompressionType::kSnappy),
                     testing::Bool(), testing::Bool(), testing::Bool(),
                     testing::Values(IndexStorageOption::kInMemory,
                                     IndexStorageOption::kOffloaded)));

TEST(ArrayRecordReaderOptionTest, ParserTest) {
  {
    auto option = ArrayRecordReaderBase::Options::FromString("").value();
    EXPECT_EQ(option.max_parallelism(), std::nullopt);
    EXPECT_EQ(option.readahead_buffer_size(),
              ArrayRecordReaderBase::Options::kDefaultReadaheadBufferSize);
  }
  {
    auto option =
        ArrayRecordReaderBase::Options::FromString("max_parallelism:16")
            .value();
    EXPECT_EQ(option.max_parallelism(), 16);
    EXPECT_EQ(option.readahead_buffer_size(),
              ArrayRecordReaderBase::Options::kDefaultReadaheadBufferSize);
  }
  {
    auto option = ArrayRecordReaderBase::Options::FromString(
                      "max_parallelism:16,readahead_buffer_size:16384")
                      .value();
    EXPECT_EQ(option.max_parallelism(), 16);
    EXPECT_EQ(option.readahead_buffer_size(), 16384);
  }
  {
    auto option = ArrayRecordReaderBase::Options::FromString(
                      "max_parallelism:0,readahead_buffer_size:0")
                      .value();
    EXPECT_EQ(option.max_parallelism(), 0);
    EXPECT_EQ(option.readahead_buffer_size(), 0);
  }
}

std::string CorruptFooterGroupSize(absl::string_view valid_encoded,
                                   int new_group_size) {
  std::string corrupted_encoded;
  riegeli::StringReader<> string_reader(valid_encoded);
  riegeli::StringWriter<> string_writer(&corrupted_encoded);
  riegeli::DefaultChunkReader<> chunk_reader(&string_reader);
  riegeli::DefaultChunkWriter<> chunk_writer(&string_writer);

  bool past_footer = false;
  riegeli::Chunk chunk;
  while (chunk_reader.ReadChunk(chunk)) {
    if (past_footer) {
      chunk_writer.WriteChunk(chunk);
      continue;
    }

    riegeli::ChunkDecoder decoder;
    if (!decoder.Decode(chunk)) {
      chunk_writer.WriteChunk(chunk);
      continue;
    }

    decoder.SetIndex(0);
    absl::string_view first_record;
    if (!decoder.ReadRecord(first_record)) {
      chunk_writer.WriteChunk(chunk);
      continue;
    }

    RiegeliFooterMetadata metadata;
    if (metadata.ParsePartialFromString(first_record) &&
        metadata.has_array_record_metadata()) {
      riegeli::SimpleEncoder footer_encoder(
          riegeli::CompressorOptions().set_uncompressed());
      footer_encoder.AddRecord(first_record);

      absl::string_view footer_record;
      while (decoder.ReadRecord(footer_record)) {
        ArrayRecordFooter footer;
        if (footer.ParsePartialFromString(footer_record) &&
            footer.has_chunk_offset()) {
          footer.set_num_records(new_group_size);
          footer_encoder.AddRecord(footer.SerializeAsString());
        } else {
          footer_encoder.AddRecord(footer_record);
        }
      }
      riegeli::Chunk corrupted_chunk;
      riegeli::ChunkType chunk_type;
      uint64_t num_records;
      uint64_t decoded_data_size;
      riegeli::ChainWriter<> chain_writer(&corrupted_chunk.data);
      footer_encoder.EncodeAndClose(chain_writer, chunk_type, num_records,
                                    decoded_data_size);
      chain_writer.Close();
      corrupted_chunk.header = riegeli::ChunkHeader(
          corrupted_chunk.data, chunk_type, num_records, decoded_data_size);
      chunk_writer.WriteChunk(corrupted_chunk);
      past_footer = true;
    } else {
      chunk_writer.WriteChunk(chunk);
    }
  }
  chunk_writer.Close();
  string_writer.Close();

  if (corrupted_encoded.size() < 64 * 1024) {
    corrupted_encoded.resize(64 * 1024, '\0');
  }
  return corrupted_encoded;
}

TEST(ArrayRecordReaderTest, GroupSizeZero) {
  std::string encoded;
  auto writer_options =
      ArrayRecordWriterBase::Options().set_group_size(10).set_uncompressed();
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(writer.WriteRecord("test"));
  }
  ASSERT_TRUE(writer.Close());

  std::string corrupted = CorruptFooterGroupSize(encoded, 0);

  auto reader_opt = ArrayRecordReaderBase::Options();
  auto reader = ArrayRecordReader(
      riegeli::Maker<riegeli::StringReader>(corrupted), reader_opt, nullptr);
  EXPECT_FALSE(reader.status().ok()) << reader.status().message();
  EXPECT_EQ(reader.status().code(), absl::StatusCode::kInvalidArgument)
      << reader.status().message();
}

TEST(ArrayRecordReaderTest, OutOfBoundsChunkIdx) {
  std::string encoded;
  auto writer_options =
      ArrayRecordWriterBase::Options().set_group_size(10).set_uncompressed();
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(writer.WriteRecord("test"));
  }
  ASSERT_TRUE(writer.Close());

  // Valid file has 1 chunk. group_size in footer is 5.
  // We corrupt it to group_size = 1.
  // The metadata says num_records = 5, num_chunks = 1.
  // Reader will think record_group_size = 1.
  // So chunk_idx = record_idx / 1.
  // If we read index 4: chunk_idx = 4.
  // But per_chunk_indices size = num_chunks = 1.
  // 4 >= 1, triggering OutOfRangeError!
  std::string corrupted = CorruptFooterGroupSize(encoded, 1);

  auto reader_opt = ArrayRecordReaderBase::Options();
  auto reader = ArrayRecordReader(
      riegeli::Maker<riegeli::StringReader>(corrupted), reader_opt, nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();

  std::vector<uint64_t> indices = {4};
  auto status = reader.ParallelReadRecordsWithIndices(
      indices, [&](uint64_t, absl::string_view) -> absl::Status {
        return absl::OkStatus();
      });

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
}

// Simulates a Reader (such as riegeli::GcsReader opened with ReadLast) that
// starts at `max(0, size - tail_prefetch_size)` with an initially empty buffer
// that is populated on Pull().
class TailPrefetchTestReader : public riegeli::Reader {
 public:
  TailPrefetchTestReader(absl::string_view data, size_t tail_prefetch_size,
                         int* pull_count, int* new_reader_count)
      : data_(data),
        initial_pos_(data.size() > tail_prefetch_size
                         ? data.size() - tail_prefetch_size
                         : 0),
        pull_count_(pull_count),
        new_reader_count_(new_reader_count) {
    set_limit_pos(initial_pos_);
  }

  bool SupportsRandomAccess() override { return true; }
  bool SupportsNewReader() override { return true; }

 protected:
  void Done() override {
    // Ensure available() == 0 on Close() so BufferedReader::Done() does not
    // issue a spurious SeekBehindBuffer() RPC.
    EXPECT_EQ(available(), 0);
    riegeli::Reader::Done();
  }

  bool PullSlow(size_t min_length, size_t recommended_length) override {
    if (pull_count_ != nullptr) {
      ++(*pull_count_);
    }
    if (min_length > data_.size() - initial_pos_) {
      return false;
    }
    set_buffer(data_.data() + initial_pos_, min_length);
    set_limit_pos(initial_pos_ + min_length);
    return true;
  }

  std::optional<riegeli::Position> SizeImpl() override { return data_.size(); }

  std::unique_ptr<riegeli::Reader> NewReaderImpl(
      riegeli::Position initial_pos) override {
    if (new_reader_count_ != nullptr) {
      ++(*new_reader_count_);
    }
    constexpr size_t kBlockSize = 1 << 16;
    const size_t end_pos = initial_pos < data_.size() - kBlockSize
                               ? data_.size() - kBlockSize
                               : data_.size();
    auto reader =
        std::make_unique<ExactSliceReader>(data_.substr(0, end_pos));
    reader->Seek(initial_pos);
    return reader;
  }

 private:
  class ExactSliceReader : public riegeli::StringReader<> {
   public:
    explicit ExactSliceReader(absl::string_view data)
        : riegeli::StringReader<>(data) {}

    ~ExactSliceReader() override { EXPECT_EQ(pos(), limit_pos()); }
  };

  absl::string_view data_;
  size_t initial_pos_;
  int* pull_count_;
  int* new_reader_count_;
};

TEST(ArrayRecordReaderTest, TailPrefetchReadsPostscriptAndFooterFromBuffer) {
  std::string encoded;
  auto writer_options =
      ArrayRecordWriterBase::Options().set_group_size(2).set_uncompressed();
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);
  std::vector<std::string> test_str{std::string(1 << 20, 'a'), "bbb", "ccc",
                                    "ddd", "eee"};
  for (const auto& s : test_str) {
    EXPECT_TRUE(writer.WriteRecord(s));
  }
  ASSERT_TRUE(writer.Close());
  ASSERT_GT(encoded.size(), 1 << 20);

  int pull_count = 0;
  int new_reader_count = 0;
  constexpr size_t kTailPrefetch = 1 << 20;  // 1 MiB (exact boundary)
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(encoded, kTailPrefetch,
                                             &pull_count, &new_reader_count),
      ArrayRecordReaderBase::Options(), nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();
  // Both postscript and footer should be read directly from the pulled tail
  // buffer without any NewReader() calls during Initialize().
  EXPECT_EQ(pull_count, 1);
  EXPECT_EQ(new_reader_count, 0);
  EXPECT_EQ(reader.NumRecords(), test_str.size());
  ASSERT_TRUE(reader.Close());
}

TEST(ArrayRecordReaderTest,
     TailPrefetchDoesNotPullWhenRemainingExceedsMaxTailPrefetch) {
  std::string encoded;
  auto writer_options =
      ArrayRecordWriterBase::Options().set_group_size(2).set_uncompressed();
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);
  std::vector<std::string> test_str{std::string(1 << 20, 'a'), "bbb"};
  for (const auto& s : test_str) {
    EXPECT_TRUE(writer.WriteRecord(s));
  }
  ASSERT_TRUE(writer.Close());
  ASSERT_GT(encoded.size(), 1 << 20);

  int pull_count = 0;
  int new_reader_count = 0;
  // Start at offset 0 on an input larger than kTailPrefetchSize (1 MiB).
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(encoded, encoded.size(),
                                             &pull_count, &new_reader_count),
      ArrayRecordReaderBase::Options(), nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();
  // Remaining bytes exceed 1 MiB, so Initialize() must not Pull() the entire
  // file and instead reads postscript and footer via NewReader().
  EXPECT_EQ(pull_count, 0);
  EXPECT_EQ(new_reader_count, 2);
  EXPECT_EQ(reader.NumRecords(), test_str.size());
  ASSERT_TRUE(reader.Close());
}

TEST(ArrayRecordReaderTest,
     TailPrefetchFallsBackToNewReaderWhenFooterBeforeStartPos) {
  std::string encoded;
  auto writer_options =
      ArrayRecordWriterBase::Options().set_group_size(2).set_uncompressed();
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);
  std::vector<std::string> test_str{"aaa", "bbb", "ccc", "ddd", "eee"};
  for (const auto& s : test_str) {
    EXPECT_TRUE(writer.WriteRecord(s));
  }
  ASSERT_TRUE(writer.Close());

  int pull_count = 0;
  int new_reader_count = 0;
  // Only prefetch the 64 KiB postscript block so footer_offset < start_pos().
  constexpr size_t kPostscriptOnlyPrefetch = 1 << 16;  // 64 KiB
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(encoded, kPostscriptOnlyPrefetch,
                                             &pull_count, &new_reader_count),
      ArrayRecordReaderBase::Options(), nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();
  // Postscript is read from the buffered tail (0 NewReader calls), while footer
  // falls back to NewReader() (1 NewReader call).
  EXPECT_EQ(pull_count, 1);
  EXPECT_EQ(new_reader_count, 1);
  EXPECT_EQ(reader.NumRecords(), test_str.size());
  ASSERT_TRUE(reader.Close());
}

TEST(ArrayRecordReaderTest, TailPrefetchReadsOffloadedIndexFromBuffer) {
  std::string encoded;
  auto writer_options =
      ArrayRecordWriterBase::Options().set_group_size(2).set_uncompressed();
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);
  std::vector<std::string> test_str{"aaa", "bbb", "ccc", "ddd", "eee"};
  for (const auto& s : test_str) {
    EXPECT_TRUE(writer.WriteRecord(s));
  }
  ASSERT_TRUE(writer.Close());

  int pull_count = 0;
  int new_reader_count = 0;
  constexpr size_t kTailPrefetch = 1 << 20;  // 1 MiB
  auto reader_options =
      ArrayRecordReaderBase::Options()
          .set_index_storage_option(IndexStorageOption::kOffloaded)
          .set_max_parallelism(0);
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(encoded, kTailPrefetch,
                                             &pull_count, &new_reader_count),
      reader_options, nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();
  // Postscript, footer, and OffloadedChunkOffset lookups during Initialize()
  // should all be served from the pulled tail buffer without NewReader() calls.
  EXPECT_EQ(pull_count, 1);
  EXPECT_EQ(new_reader_count, 0);
  EXPECT_EQ(reader.NumRecords(), test_str.size());
  ASSERT_TRUE(reader.Close());
}

TEST(ArrayRecordReaderTest, TailPrefetchFailsOnCorruptedChunkInBuffer) {
  std::string corrupted(1 << 16, 'x');
  int pull_count = 0;
  int new_reader_count = 0;
  constexpr size_t kTailPrefetch = 1 << 20;
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(corrupted, kTailPrefetch,
                                             &pull_count, &new_reader_count),
      ArrayRecordReaderBase::Options(), nullptr);
  EXPECT_FALSE(reader.status().ok());
  EXPECT_EQ(pull_count, 1);
  EXPECT_EQ(new_reader_count, 0);
}

}  // namespace
}  // namespace array_record
