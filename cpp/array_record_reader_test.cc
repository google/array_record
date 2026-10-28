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
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "cpp/array_record_writer.h"
#include "cpp/block_cache.h"
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
  if (index_storage_option() == IndexStorageOption::kOffloaded) {
    reader_opt.set_footer_prefetch_bytes(0);
  }
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
  if (index_storage_option() == IndexStorageOption::kOffloaded) {
    reader_opt.set_footer_prefetch_bytes(0);
  }
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
    EXPECT_EQ(option.footer_prefetch_bytes(),
              ArrayRecordReaderBase::Options::kDefaultFooterPrefetchBytes);
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
  {
    auto option =
        ArrayRecordReaderBase::Options::FromString("footer_prefetch_bytes:512K")
            .value();
    EXPECT_EQ(option.footer_prefetch_bytes(), 512 * 1024);
  }
  {
    auto option =
        ArrayRecordReaderBase::Options::FromString("footer_prefetch_bytes:0")
            .value();
    EXPECT_EQ(option.footer_prefetch_bytes(), 0);
  }
  {
    auto option =
        ArrayRecordReaderBase::Options::FromString("footer_prefetch_bytes:auto")
            .value();
    EXPECT_EQ(option.footer_prefetch_bytes(),
              ArrayRecordReaderBase::Options::kDefaultFooterPrefetchBytes);
  }
  {
    auto status_or_option = ArrayRecordReaderBase::Options::FromString(
        "index_storage_option:offloaded");
    EXPECT_FALSE(status_or_option.ok());
    EXPECT_EQ(status_or_option.status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  {
    auto option = ArrayRecordReaderBase::Options::FromString(
                      "index_storage_option:offloaded,footer_prefetch_bytes:0")
                      .value();
    EXPECT_EQ(option.index_storage_option(), IndexStorageOption::kOffloaded);
    EXPECT_EQ(option.footer_prefetch_bytes(), 0);
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
                         int* pull_count, int* new_reader_count,
                         int* seek_count = nullptr)
      : data_(data),
        initial_pos_(data.size() > tail_prefetch_size
                         ? data.size() - tail_prefetch_size
                         : 0),
        pull_count_(pull_count),
        new_reader_count_(new_reader_count),
        seek_count_(seek_count) {
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
    const size_t remaining = data_.size() - limit_pos();
    if (min_length > remaining) {
      return false;
    }
    set_buffer(data_.data() + limit_pos(), remaining);
    set_limit_pos(data_.size());
    return true;
  }

  bool SeekSlow(riegeli::Position new_pos) override {
    if (seek_count_ != nullptr) {
      ++(*seek_count_);
    }
    if (new_pos > data_.size()) {
      return false;
    }
    set_buffer();
    set_limit_pos(new_pos);
    return true;
  }

  std::optional<riegeli::Position> SizeImpl() override { return data_.size(); }

  std::unique_ptr<riegeli::Reader> NewReaderImpl(
      riegeli::Position initial_pos) override {
    if (new_reader_count_ != nullptr) {
      ++(*new_reader_count_);
    }
    auto reader = std::make_unique<riegeli::StringReader<>>(data_);
    reader->Seek(initial_pos);
    return reader;
  }

 private:
  absl::string_view data_;
  size_t initial_pos_;
  int* pull_count_;
  int* new_reader_count_;
  int* seek_count_;
};

TEST(ArrayRecordReaderTest,
     FooterPrefetchReadsPostscriptAndFooterFromMaskedReader) {
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
  int seek_count = 0;
  constexpr size_t kTailPrefetch = 1 << 20;  // 1 MiB
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(
          encoded, kTailPrefetch, &pull_count, &new_reader_count, &seek_count),
      ArrayRecordReaderBase::Options(), nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();
  // Both postscript and footer should be read directly from the MaskedReader
  // constructed over the main reader without any SeekSlow() or NewReader()
  // calls on the backing reader during Initialize().
  EXPECT_EQ(seek_count, 0);
  EXPECT_EQ(pull_count, 1);
  EXPECT_EQ(new_reader_count, 0);
  EXPECT_EQ(reader.NumRecords(), test_str.size());
  ASSERT_TRUE(reader.Close());
}

TEST(ArrayRecordReaderTest,
     FooterPrefetchSeeksAndReadsWithoutNewReaderWhenStartingAtZero) {
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
  int seek_count = 0;
  // Start at offset 0 on an input larger than footer_prefetch_bytes (1 MiB).
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(
          encoded, encoded.size(), &pull_count, &new_reader_count, &seek_count),
      ArrayRecordReaderBase::Options(), nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();
  // Initialize() seeks the main reader to prefetch_offset and reads both
  // postscript and footer from the MaskedReader without any NewReader() calls.
  EXPECT_EQ(seek_count, 1);
  EXPECT_EQ(pull_count, 1);
  EXPECT_EQ(new_reader_count, 0);
  EXPECT_EQ(reader.NumRecords(), test_str.size());
  ASSERT_TRUE(reader.Close());
}

TEST(ArrayRecordReaderTest,
     FooterPrefetchFallsBackToNewReaderWhenFooterExceedsPrefetch) {
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
  // Only prefetch the 64 KiB postscript block so footer_offset <
  // prefetch_offset.
  constexpr size_t kPostscriptOnlyPrefetch = 1 << 16;  // 64 KiB
  auto reader_options = ArrayRecordReaderBase::Options()
                            .set_footer_prefetch_bytes(kPostscriptOnlyPrefetch);
  auto reader = ArrayRecordReader(
      riegeli::Maker<TailPrefetchTestReader>(encoded, kPostscriptOnlyPrefetch,
                                             &pull_count, &new_reader_count),
      reader_options, nullptr);
  ASSERT_TRUE(reader.status().ok()) << reader.status().message();
  // Postscript is read from prefetched_reader (1 Pull on main reader), while
  // footer falls back to NewReader(footer_offset) (1 NewReader call).
  EXPECT_EQ(pull_count, 1);
  EXPECT_EQ(new_reader_count, 1);
  EXPECT_EQ(reader.NumRecords(), test_str.size());
  ASSERT_TRUE(reader.Close());
}

TEST(ArrayRecordReaderTest, OffloadedIndexWithPositiveFooterPrefetchFails) {
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

  auto reader_options = ArrayRecordReaderBase::Options()
                            .set_index_storage_option(
                                IndexStorageOption::kOffloaded);
  auto reader = ArrayRecordReader(
      riegeli::Maker<riegeli::StringReader>(encoded), reader_options, nullptr);
  EXPECT_FALSE(reader.status().ok());
  EXPECT_EQ(reader.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ArrayRecordReaderTest, FooterPrefetchFailsOnCorruptedChunk) {
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

TEST(ArrayRecordReaderTest, FooterEndsAtExactBlockBoundaryWithoutPadding) {
  constexpr size_t kBlockSize = 1 << 16;        // 64 KiB
  constexpr size_t kProbeRecordSize = 1 << 15;  // 32 KiB
  auto writer_options =
      ArrayRecordWriterBase::Options().set_group_size(1).set_uncompressed();

  std::string probe_encoded;
  {
    auto writer = ArrayRecordWriter(
        riegeli::Maker<riegeli::StringWriter>(&probe_encoded), writer_options,
        nullptr);
    ASSERT_TRUE(writer.WriteRecord(std::string(kProbeRecordSize, 'a')));
    ASSERT_TRUE(writer.Close());
  }

  // Locate the end of the footer chunk (chunk 2: after signature and 1 data
  // chunk) in the probe file to compute the exact record size that makes the
  // footer chunk end on the 64 KiB block boundary without trailing padding.
  riegeli::StringReader<> probe_string_reader(probe_encoded);
  riegeli::DefaultChunkReader<> probe_chunk_reader(&probe_string_reader);
  riegeli::Chunk chunk;
  ASSERT_TRUE(probe_chunk_reader.ReadChunk(chunk));  // Signature chunk
  ASSERT_TRUE(probe_chunk_reader.ReadChunk(chunk));  // Data chunk
  ASSERT_TRUE(probe_chunk_reader.ReadChunk(chunk));  // Footer chunk
  const size_t probe_footer_end = probe_chunk_reader.pos();
  ASSERT_LT(probe_footer_end, kBlockSize);

  const size_t target_record_size =
      kProbeRecordSize + (kBlockSize - probe_footer_end);
  const std::string expected_record(target_record_size, 'b');

  std::string encoded;
  {
    auto writer = ArrayRecordWriter(
        riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options,
        nullptr);
    ASSERT_TRUE(writer.WriteRecord(expected_record));
    ASSERT_TRUE(writer.Close());
  }
  ASSERT_EQ(encoded.size(), 2 * kBlockSize);

  // Verify the footer chunk ends directly at the postscript boundary
  // (encoded.size() - kBlockSize) with zero padding chunks in between.
  riegeli::StringReader<> verify_string_reader(encoded);
  riegeli::DefaultChunkReader<> verify_chunk_reader(&verify_string_reader);
  ASSERT_TRUE(verify_chunk_reader.ReadChunk(chunk));  // Signature chunk
  ASSERT_TRUE(verify_chunk_reader.ReadChunk(chunk));  // Data chunk
  ASSERT_TRUE(verify_chunk_reader.ReadChunk(chunk));  // Footer chunk
  ASSERT_EQ(verify_chunk_reader.pos(), encoded.size() - kBlockSize);

  // 1. Read with footer prefetch enabled (prefetched_reader branch).
  {
    int pull_count = 0;
    int new_reader_count = 0;
    constexpr size_t kTailPrefetch = 1 << 20;
    auto reader = ArrayRecordReader(
        riegeli::Maker<TailPrefetchTestReader>(encoded, kTailPrefetch,
                                               &pull_count, &new_reader_count),
        ArrayRecordReaderBase::Options(), nullptr);
    ASSERT_TRUE(reader.status().ok()) << reader.status().message();
    EXPECT_EQ(pull_count, 1);
    EXPECT_EQ(new_reader_count, 0);
    EXPECT_EQ(reader.NumRecords(), 1);
    absl::string_view record_view;
    ASSERT_TRUE(reader.ReadRecord(&record_view));
    EXPECT_EQ(record_view, expected_record);
    ASSERT_TRUE(reader.Close());
  }

  // 2. Read with footer prefetch disabled (*reader branch).
  {
    auto reader_options =
        ArrayRecordReaderBase::Options().set_footer_prefetch_bytes(0);
    auto reader = ArrayRecordReader(
        riegeli::Maker<riegeli::StringReader>(encoded), reader_options,
        nullptr);
    ASSERT_TRUE(reader.status().ok()) << reader.status().message();
    EXPECT_EQ(reader.NumRecords(), 1);
    absl::string_view record_view;
    ASSERT_TRUE(reader.ReadRecord(&record_view));
    EXPECT_EQ(record_view, expected_record);
    ASSERT_TRUE(reader.Close());
  }
}

TEST(BlockCacheOptionsTest, ParsesKeys) {
  auto options = ArrayRecordReaderBase::Options::FromString(
      "block_cache_max_bytes:32G,block_cache_block_size:64M,"
      "block_cache_max_inflight:24");
  ASSERT_TRUE(options.ok()) << options.status();
  EXPECT_EQ(options->block_cache_max_bytes(), uint64_t{32} << 30);
  EXPECT_EQ(options->block_cache_block_size(), uint64_t{64} << 20);
  EXPECT_EQ(options->block_cache_max_inflight(), 24u);
}

TEST(BlockCacheOptionsTest, DefaultsAreOff) {
  auto options = ArrayRecordReaderBase::Options::FromString("");
  ASSERT_TRUE(options.ok());
  EXPECT_EQ(options->block_cache_max_bytes(), 0u);
  EXPECT_EQ(options->block_cache_block_size(), uint64_t{16} << 20);
  EXPECT_EQ(options->block_cache_max_inflight(), 16u);
}

TEST(BlockCacheOptionsTest, RejectsInvalidCombinations) {
  for (absl::string_view text : {
           "block_cache_max_bytes:1M,block_cache_block_size:16M",
           "block_cache_max_bytes:32M,index_storage_option:offloaded,"
           "footer_prefetch_bytes:0",
           "block_cache_max_bytes:32M,block_cache_block_size:512K",
           "block_cache_max_bytes:32M,block_cache_block_size:2G",
           "block_cache_max_bytes:32M,block_cache_max_inflight:0",
           "block_cache_max_bytes:32M,block_cache_max_inflight:2k",
       }) {
    EXPECT_FALSE(ArrayRecordReaderBase::Options::FromString(text).ok()) << text;
  }
}

TEST(BlockCacheOptionsTest, InitializeRejectsInvalidSetterConfig) {
  std::string encoded;
  {
    auto writer =
        ArrayRecordWriter(riegeli::Maker<riegeli::StringWriter>(&encoded),
                          ArrayRecordWriterBase::Options(), nullptr);
    ASSERT_TRUE(writer.WriteRecord("a"));
    ASSERT_TRUE(writer.Close());
  }
  ArrayRecordReaderBase::Options options;
  options.set_block_cache_max_bytes(1 << 20);
  options.set_block_cache_block_size(16 << 20);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           options, nullptr);
  EXPECT_FALSE(reader.ok());
  EXPECT_EQ(reader.status().code(), absl::StatusCode::kInvalidArgument);
}

// ---------------------------------------------------------------------------
// Block cache.

constexpr uint64_t kMiB = uint64_t{1} << 20;

// Writes uncompressed records of 1-3 KiB (so chunks straddle 64 KiB Riegeli
// block boundaries). `big_record_at`, if set, makes that record 3 MiB.
std::string WriteBlockCacheTestFile(
    uint32_t num_records, uint32_t group_size,
    std::vector<std::string>* records,
    std::optional<uint32_t> big_record_at = std::nullopt) {
  std::string encoded;
  auto writer_options = ArrayRecordWriterBase::Options();
  writer_options.set_uncompressed().set_group_size(group_size);
  auto writer = ArrayRecordWriter(
      riegeli::Maker<riegeli::StringWriter>(&encoded), writer_options, nullptr);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> size_dist(1024, 3071);
  records->clear();
  for (uint32_t i = 0; i < num_records; ++i) {
    std::string r = absl::StrCat("record-", i, "-");
    size_t size = (big_record_at == i) ? 3 * kMiB : size_dist(rng);
    r.resize(size, static_cast<char>('a' + i % 26));
    records->push_back(r);
    EXPECT_TRUE(writer.WriteRecord(r));
  }
  EXPECT_TRUE(writer.Close());
  return encoded;
}

ArrayRecordReaderBase::Options BlockCacheOptions(uint64_t max_bytes,
                                                 uint64_t block_size) {
  ArrayRecordReaderBase::Options options;
  options.set_block_cache_max_bytes(max_bytes).set_block_cache_block_size(
      block_size);
  return options;
}

std::vector<uint64_t> Permutation(uint64_t n, uint32_t seed) {
  std::vector<uint64_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);
  return order;
}

// Reads each index with its own ParallelReadRecordsWithIndices call, the way
// Grain's ArrayRecordDataSource does, and checks the record.
void ReadSinglyAndCheck(const ArrayRecordReaderBase& reader,
                        absl::Span<const uint64_t> order,
                        const std::vector<std::string>& records) {
  for (uint64_t idx : order) {
    uint64_t one = idx;
    ASSERT_TRUE(reader
                    .ParallelReadRecordsWithIndices(
                        absl::MakeConstSpan(&one, 1),
                        [&](uint64_t, absl::string_view record) {
                          EXPECT_EQ(record, records[idx]) << idx;
                          return absl::OkStatus();
                        })
                    .ok());
  }
}

// Number of blocks the reader packs, using the rule in the spec: add the next
// chunk only while the block span stays <= block_size.
uint64_t ExpectedNumBlocks(const ArrayRecordReaderBase& reader,
                           uint64_t num_chunks, uint64_t block_size) {
  uint64_t blocks = 1;
  uint64_t start = reader.ChunkStartOffset(0);
  for (uint64_t c = 1; c < num_chunks; ++c) {
    if (reader.ChunkEndOffset(c) - start > block_size) {
      ++blocks;
      start = reader.ChunkStartOffset(c);
    }
  }
  return blocks;
}

class BlockCacheReaderTest : public testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override { BlockCache::ResetGlobalForTesting(); }
  void TearDown() override { BlockCache::ResetGlobalForTesting(); }
  uint32_t group_size() const { return GetParam(); }
};

TEST_P(BlockCacheReaderTest, MatchesUncachedReadsSingleAndBatch) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, group_size(), &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(64 * kMiB, kMiB),
                           ArrayRecordGlobalPool());
  ASSERT_TRUE(reader.ok()) << reader.status();
  ReadSinglyAndCheck(reader, Permutation(records.size(), 1), records);

  // Batches with duplicates, spread over many blocks.
  std::mt19937 rng(7);
  std::uniform_int_distribution<uint64_t> dist(0, records.size() - 1);
  for (int batch = 0; batch < 20; ++batch) {
    std::vector<uint64_t> indices;
    for (int i = 0; i < 64; ++i) indices.push_back(dist(rng));
    indices.push_back(indices.front());  // duplicate
    ASSERT_TRUE(reader
                    .ParallelReadRecordsWithIndices(
                        indices,
                        [&](uint64_t i, absl::string_view record) {
                          EXPECT_EQ(record, records[indices[i]]);
                          return absl::OkStatus();
                        })
                    .ok());
  }
  ASSERT_TRUE(reader.Close());
}

INSTANTIATE_TEST_SUITE_P(GroupSizes, BlockCacheReaderTest,
                         testing::Values(1u, 3u));

class BlockCacheReaderSingleTest : public testing::Test {
 protected:
  void SetUp() override { BlockCache::ResetGlobalForTesting(); }
  void TearDown() override { BlockCache::ResetGlobalForTesting(); }
};

TEST_F(BlockCacheReaderSingleTest, FetchesEachBlockOnceAndDrains) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(64 * kMiB, kMiB),
                           /*pool=*/nullptr);
  ASSERT_TRUE(reader.ok()) << reader.status();
  const uint64_t blocks = ExpectedNumBlocks(reader, records.size(), kMiB);
  ASSERT_GT(blocks, 4u);
  ReadSinglyAndCheck(reader, Permutation(records.size(), 2), records);
  BlockCache::Stats s = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(s.misses_fetched, blocks);
  EXPECT_EQ(s.hits, records.size() - blocks);
  EXPECT_EQ(s.drained_frees, blocks);
  EXPECT_EQ(s.resident_bytes, 0u);
  EXPECT_EQ(s.fallbacks_inflight + s.fallbacks_budget + s.fallbacks_error, 0u);
}

TEST_F(BlockCacheReaderSingleTest, SmallBudgetStaysCorrectAndBounded) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(2 * kMiB, kMiB),
                           ArrayRecordGlobalPool());
  ASSERT_TRUE(reader.ok());
  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      auto order = Permutation(records.size(), 10 + t);
      order.resize(1000);
      ReadSinglyAndCheck(reader, order, records);
    });
  }
  for (auto& th : threads) th.join();
  // The budget fills up: later misses fall back to direct reads.
  BlockCache::Stats s = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_GT(s.fallbacks_budget, 0u);
  EXPECT_LE(s.peak_resident_bytes, 2 * kMiB);
}

TEST_F(BlockCacheReaderSingleTest, BudgetEqualToBlockSizeUnderConcurrency) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(kMiB, kMiB),
                           ArrayRecordGlobalPool());
  ASSERT_TRUE(reader.ok());
  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      auto order = Permutation(records.size(), 20 + t);
      order.resize(500);
      ReadSinglyAndCheck(reader, order, records);
    });
  }
  for (auto& th : threads) th.join();
  BlockCache::Stats s = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_LE(s.peak_resident_bytes, kMiB);
  EXPECT_EQ(s.pending_bytes, 0u);
}

TEST_F(BlockCacheReaderSingleTest, SecondEpochIsCorrectAndBounded) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(2000, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(2 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok());
  ReadSinglyAndCheck(reader, Permutation(records.size(), 3), records);
  // ~4 MiB of records, a 2 MiB budget and 1 MiB blocks: in epoch 1 some
  // misses fall back to direct reads, and blocks whose records are all served
  // are freed.
  BlockCache::Stats epoch1 = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_GT(epoch1.fallbacks_budget, 0u);
  EXPECT_GT(epoch1.drained_frees, 0u);
  // Each epoch completes one pass over every block, so nothing stays resident.
  EXPECT_EQ(epoch1.resident_bytes, 0u);
  ReadSinglyAndCheck(reader, Permutation(records.size(), 4), records);
  BlockCache::Stats epoch2 = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_LE(epoch2.peak_resident_bytes, 2 * kMiB);
  EXPECT_GT(epoch2.drained_frees, epoch1.drained_frees);
  EXPECT_EQ(epoch2.resident_bytes, 0u);
}

TEST_F(BlockCacheReaderSingleTest, EveryFullPassDrainsEachBlockAgain) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok()) << reader.status();
  const uint64_t blocks = ExpectedNumBlocks(reader, records.size(), kMiB);
  ASSERT_GT(blocks, 4u);
  ReadSinglyAndCheck(reader, Permutation(records.size(), 11), records);
  BlockCache::Stats epoch1 = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(epoch1.drained_frees, blocks);
  EXPECT_EQ(epoch1.resident_bytes, 0u);
  // The drain counters re-arm, so the second pass frees every block again.
  ReadSinglyAndCheck(reader, Permutation(records.size(), 12), records);
  BlockCache::Stats epoch2 = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(epoch2.misses_fetched, 2 * blocks);
  EXPECT_EQ(epoch2.drained_frees, 2 * blocks);
  EXPECT_EQ(epoch2.resident_bytes, 0u);
  EXPECT_EQ(epoch2.fallbacks_budget, 0u);
}

TEST_F(BlockCacheReaderSingleTest, OutOfRangeIndexStillFails) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(100, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(16 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok());
  uint64_t bad = records.size();
  absl::Status s = reader.ParallelReadRecordsWithIndices(
      absl::MakeConstSpan(&bad, 1),
      [](uint64_t, absl::string_view) { return absl::OkStatus(); });
  EXPECT_EQ(s.code(), absl::StatusCode::kOutOfRange);
}

TEST_F(BlockCacheReaderSingleTest, OversizedChunkFallsBackWithoutEvicting) {
  std::vector<std::string> records;
  std::string encoded =
      WriteBlockCacheTestFile(1000, 1, &records, /*big_record_at=*/500);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(2 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok());
  ReadSinglyAndCheck(reader, std::vector<uint64_t>{0, 1, 2}, records);
  BlockCache::Stats before = BlockCache::InstanceIfExists()->GetStats();
  ReadSinglyAndCheck(reader, std::vector<uint64_t>{500}, records);
  BlockCache::Stats after = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(after.fallbacks_budget, before.fallbacks_budget + 1);
  EXPECT_EQ(after.drained_frees, before.drained_frees);
  EXPECT_EQ(after.closed_frees, before.closed_frees);
  EXPECT_EQ(after.resident_bytes, before.resident_bytes);
}

TEST_F(BlockCacheReaderSingleTest, UncachedReaderInSameProcessIsUnaffected) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(1000, 1, &records);
  ArrayRecordReader cached(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(16 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(cached.ok());
  ReadSinglyAndCheck(cached, std::vector<uint64_t>{0, 999}, records);
  BlockCache::Stats before = BlockCache::InstanceIfExists()->GetStats();
  ArrayRecordReader uncached(riegeli::Maker<riegeli::StringReader>(encoded),
                             ArrayRecordReaderBase::Options(), nullptr);
  ASSERT_TRUE(uncached.ok());
  ReadSinglyAndCheck(uncached, Permutation(records.size(), 5), records);
  BlockCache::Stats after = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(after.hits, before.hits);
  EXPECT_EQ(after.misses_fetched, before.misses_fetched);
}

TEST_F(BlockCacheReaderSingleTest, EmptyFileBehavesLikeUncached) {
  std::string encoded;
  {
    auto writer =
        ArrayRecordWriter(riegeli::Maker<riegeli::StringWriter>(&encoded),
                          ArrayRecordWriterBase::Options(), nullptr);
    ASSERT_TRUE(writer.Close());
  }
  ArrayRecordReader uncached(riegeli::Maker<riegeli::StringReader>(encoded),
                             ArrayRecordReaderBase::Options(), nullptr);
  ArrayRecordReader cached(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(16 * kMiB, kMiB), nullptr);
  EXPECT_EQ(cached.ok(), uncached.ok());
  if (cached.ok()) {
    EXPECT_EQ(cached.NumRecords(), 0u);
    EXPECT_TRUE(cached
                    .ParallelReadRecordsWithIndices(
                        {}, [](uint64_t,
                               absl::string_view) { return absl::OkStatus(); })
                    .ok());
  }
}

TEST_F(BlockCacheReaderSingleTest, OneChunkFile) {
  std::vector<std::string> records;
  // group_size > num_records: all 10 records land in a single chunk.
  std::string encoded =
      WriteBlockCacheTestFile(10, /*group_size=*/16, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(16 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok()) << reader.status();
  ReadSinglyAndCheck(reader, Permutation(records.size(), 8), records);
  BlockCache::Stats s = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(s.misses_fetched, 1u);
  EXPECT_EQ(s.hits, records.size() - 1);
  EXPECT_EQ(s.drained_frees, 1u);
  EXPECT_EQ(s.resident_bytes, 0u);
}

TEST_F(BlockCacheReaderSingleTest, CountingReaderSeesOneNewReaderPerBlock) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  int pull_count = 0;
  int new_reader_count = 0;
  // tail_prefetch_size == size: the reader starts at 0, and Initialize()
  // reads the footer without NewReader() (see
  // FooterPrefetchSeeksAndReadsWithoutNewReaderWhenStartingAtZero).
  ArrayRecordReader reader(
      riegeli::Maker<TailPrefetchTestReader>(encoded, encoded.size(),
                                             &pull_count, &new_reader_count),
      BlockCacheOptions(64 * kMiB, kMiB), /*pool=*/nullptr);
  ASSERT_TRUE(reader.ok()) << reader.status();
  const uint64_t blocks = ExpectedNumBlocks(reader, records.size(), kMiB);
  const int new_readers_before = new_reader_count;
  ReadSinglyAndCheck(reader, Permutation(records.size(), 9), records);
  // No pressure: each block is fetched with exactly one NewReader().
  EXPECT_EQ(static_cast<uint64_t>(new_reader_count - new_readers_before),
            blocks);
  EXPECT_EQ(BlockCache::InstanceIfExists()->GetStats().misses_fetched, blocks);
  ASSERT_TRUE(reader.Close());
}

// Serves the footer from the full data (so Initialize() succeeds), but every
// NewReader() reads a truncated copy, as if the object shrank: block fetches
// short-read, and so does the direct-read fallback.
class TruncatedNewReaderTestReader : public TailPrefetchTestReader {
 public:
  TruncatedNewReaderTestReader(absl::string_view data, size_t keep_bytes)
      : TailPrefetchTestReader(data, data.size(), /*pull_count=*/nullptr,
                               /*new_reader_count=*/nullptr),
        truncated_(data.substr(0, keep_bytes)) {}

 protected:
  std::unique_ptr<riegeli::Reader> NewReaderImpl(
      riegeli::Position initial_pos) override {
    auto reader = std::make_unique<riegeli::StringReader<>>(truncated_);
    reader->Seek(initial_pos);  // Past the end: stays ok() at EOF.
    return reader;
  }

 private:
  std::string truncated_;
};

TEST_F(BlockCacheReaderSingleTest, ShortReadFallsBackWithUncachedError) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  const size_t keep_bytes = encoded.size() / 2;
  uint64_t last = records.size() - 1;  // In the truncated half.
  auto read_last = [&](const ArrayRecordReaderBase& reader) {
    return reader.ParallelReadRecordsWithIndices(
        absl::MakeConstSpan(&last, 1),
        [](uint64_t, absl::string_view) { return absl::OkStatus(); });
  };

  ArrayRecordReader uncached(
      riegeli::Maker<TruncatedNewReaderTestReader>(encoded, keep_bytes),
      ArrayRecordReaderBase::Options(), nullptr);
  ASSERT_TRUE(uncached.ok()) << uncached.status();
  const absl::Status uncached_status = read_last(uncached);
  ASSERT_FALSE(uncached_status.ok());

  ArrayRecordReader cached(
      riegeli::Maker<TruncatedNewReaderTestReader>(encoded, keep_bytes),
      BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(cached.ok()) << cached.status();
  const absl::Status cached_status = read_last(cached);
  EXPECT_FALSE(cached_status.ok());
  EXPECT_EQ(cached_status.code(), uncached_status.code()) << cached_status;
  BlockCache::Stats s = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_GT(s.fallbacks_error, 0u);
  EXPECT_EQ(s.resident_bytes, 0u);
}

uint64_t ResidentBytes() {
  BlockCache* cache = BlockCache::InstanceIfExists();
  return cache == nullptr ? 0 : cache->GetStats().resident_bytes;
}

uint64_t ClosedFrees() {
  BlockCache* cache = BlockCache::InstanceIfExists();
  return cache == nullptr ? 0 : cache->GetStats().closed_frees;
}

// Number of resident blocks: nothing is evicted, so every fetched block is
// either resident or freed by a drain or a close. Reads the process-wide
// stats, so it is meaningful only on a fresh cache with one live reader.
uint64_t ResidentBlocks() {
  BlockCache* cache = BlockCache::InstanceIfExists();
  if (cache == nullptr) return 0;
  BlockCache::Stats s = cache->GetStats();
  return s.misses_fetched - s.drained_frees - s.closed_frees;
}

TEST_F(BlockCacheReaderSingleTest, CloseReleasesBlocks) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok());
  auto order = Permutation(records.size(), 6);
  order.resize(200);
  ReadSinglyAndCheck(reader, order, records);
  ASSERT_GT(ResidentBytes(), 0u);
  const uint64_t blocks = ResidentBlocks();
  ASSERT_TRUE(reader.Close());
  EXPECT_EQ(ResidentBytes(), 0u);
  EXPECT_EQ(ClosedFrees(), blocks);
}

TEST_F(BlockCacheReaderSingleTest, DestructionReleasesBlocks) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  uint64_t blocks = 0;
  {
    ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                             BlockCacheOptions(64 * kMiB, kMiB), nullptr);
    ASSERT_TRUE(reader.ok());
    ReadSinglyAndCheck(reader, std::vector<uint64_t>{0, 3999}, records);
    ASSERT_GT(ResidentBytes(), 0u);
    blocks = ResidentBlocks();
  }
  EXPECT_EQ(ResidentBytes(), 0u);
  EXPECT_EQ(ClosedFrees(), blocks);
}

TEST_F(BlockCacheReaderSingleTest, MoveAssignmentReleasesOldReaderBlocks) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader a(riegeli::Maker<riegeli::StringReader>(encoded),
                      BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ArrayRecordReader b(riegeli::Maker<riegeli::StringReader>(encoded),
                      BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  ReadSinglyAndCheck(a, std::vector<uint64_t>{0, 1}, records);
  const uint64_t a_bytes = ResidentBytes();
  const uint64_t a_blocks = ResidentBlocks();
  // A mid-file index: its block holds many records, so one read does not
  // drain it (the last block may hold only the last record).
  ReadSinglyAndCheck(b, std::vector<uint64_t>{2000}, records);
  const uint64_t b_bytes = ResidentBytes() - a_bytes;
  ASSERT_GT(a_bytes, 0u);
  ASSERT_GT(b_bytes, 0u);
  a = std::move(b);  // Old `a` state is destroyed.
  EXPECT_EQ(ResidentBytes(), b_bytes);
  EXPECT_EQ(ClosedFrees(), a_blocks);
  ReadSinglyAndCheck(a, std::vector<uint64_t>{2001}, records);
}

TEST_F(BlockCacheReaderSingleTest, TwoReadersOfSameFileAreIndependent) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader r1(riegeli::Maker<riegeli::StringReader>(encoded),
                       BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ArrayRecordReader r2(riegeli::Maker<riegeli::StringReader>(encoded),
                       BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(r1.ok());
  ASSERT_TRUE(r2.ok());
  ReadSinglyAndCheck(r1, std::vector<uint64_t>{0}, records);
  const uint64_t r1_bytes = ResidentBytes();
  ReadSinglyAndCheck(r2, std::vector<uint64_t>{0}, records);
  // Separate reader ids: the same block is cached once per reader.
  EXPECT_EQ(ResidentBytes(), 2 * r1_bytes);
  ASSERT_TRUE(r1.Close());
  EXPECT_EQ(ResidentBytes(), r1_bytes);  // r2's block survives
  EXPECT_EQ(ClosedFrees(), 1u);
  ReadSinglyAndCheck(r2, std::vector<uint64_t>{1}, records);
}

// Reads `indices` with one ParallelReadRecordsWithIndices call and checks the
// records.
void ReadBatchAndCheck(const ArrayRecordReaderBase& reader,
                       const std::vector<uint64_t>& indices,
                       const std::vector<std::string>& records) {
  ASSERT_TRUE(reader
                  .ParallelReadRecordsWithIndices(
                      indices,
                      [&](uint64_t i, absl::string_view record) {
                        EXPECT_EQ(record, records[indices[i]]) << indices[i];
                        return absl::OkStatus();
                      })
                  .ok());
}

TEST_F(BlockCacheReaderSingleTest, BatchWithDuplicatesFetchesBlockOnce) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(10, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(16 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok()) << reader.status();
  ASSERT_EQ(ExpectedNumBlocks(reader, records.size(), kMiB), 1u);
  // Every record of the only block, plus a duplicate of record 0.
  ReadBatchAndCheck(reader, {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, records);
  BlockCache::Stats s = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(s.misses_fetched, 1u);
  EXPECT_EQ(s.drained_frees, 1u);
  EXPECT_EQ(s.resident_bytes, 0u);
}

TEST_F(BlockCacheReaderSingleTest, BatchAcrossPassBoundaryKeepsBlock) {
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(10, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(16 * kMiB, kMiB),
                           ArrayRecordGlobalPool());
  ASSERT_TRUE(reader.ok()) << reader.status();
  ASSERT_EQ(ExpectedNumBlocks(reader, records.size(), kMiB), 1u);
  ReadSinglyAndCheck(reader, std::vector<uint64_t>{0, 1}, records);
  // Ends pass 1 and starts pass 2 in one call: the block stays for pass 2.
  ReadBatchAndCheck(reader, {2, 3, 4, 5, 6, 7, 8, 9, 0, 1}, records);
  BlockCache::Stats mid = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(mid.misses_fetched, 1u);
  EXPECT_EQ(mid.drained_frees, 0u);
  EXPECT_GT(mid.resident_bytes, 0u);
  // The rest of pass 2 hits the block and frees it at the end.
  ReadSinglyAndCheck(reader, std::vector<uint64_t>{2, 3, 4, 5, 6, 7, 8, 9},
                     records);
  BlockCache::Stats end = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(end.misses_fetched, 1u);
  EXPECT_EQ(end.drained_frees, 1u);
  EXPECT_EQ(end.resident_bytes, 0u);
}

TEST_F(BlockCacheReaderSingleTest, StridedPassDoesNotDrainUntilRestIsRead) {
  // A reader that serves every second record (as with strided sharding across
  // processes) does not finish a pass over any block, so nothing is freed
  // until the other records are read too.
  std::vector<std::string> records;
  std::string encoded = WriteBlockCacheTestFile(4000, 1, &records);
  ArrayRecordReader reader(riegeli::Maker<riegeli::StringReader>(encoded),
                           BlockCacheOptions(64 * kMiB, kMiB), nullptr);
  ASSERT_TRUE(reader.ok()) << reader.status();
  std::vector<uint64_t> even, odd;
  for (uint64_t i = 0; i < records.size(); ++i) {
    (i % 2 == 0 ? even : odd).push_back(i);
  }
  ReadSinglyAndCheck(reader, even, records);
  BlockCache::Stats half = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(half.drained_frees, 0u);
  EXPECT_GT(half.resident_bytes, 0u);
  ReadSinglyAndCheck(reader, odd, records);
  BlockCache::Stats full = BlockCache::InstanceIfExists()->GetStats();
  EXPECT_EQ(full.drained_frees, full.misses_fetched);
  EXPECT_EQ(full.resident_bytes, 0u);
}

}  // namespace
}  // namespace array_record
