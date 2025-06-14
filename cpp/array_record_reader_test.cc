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
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"

constexpr uint32_t kDatasetSize = 10050;

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
  auto writer_options = GetWriterOptions();
  int32_t group_size;
  if (optimize_for_random_access()) {
    group_size = 1;
    writer_options.set_groups_awaiting_flush(256);
  } else {
    group_size = 3;
    writer_options.set_groups_awaiting_flush(1);
  }
  writer_options.set_group_size(group_size);
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

  EXPECT_EQ(reader_before_move.RecordGroupSize(), group_size);

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

  EXPECT_EQ(reader.RecordGroupSize(), group_size);

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
  auto writer_options = GetWriterOptions();
  int32_t group_size;
  if (optimize_for_random_access()) {
    group_size = 1;
    writer_options.set_groups_awaiting_flush(1024);
  } else {
    group_size = 1024;
    writer_options.set_groups_awaiting_flush(1);
  }
  writer_options.set_group_size(group_size);

  std::string encoded;
  auto writer =
      ArrayRecordWriter(riegeli::Maker<riegeli::StringWriter>(&encoded),
                        writer_options, get_pool());
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
  EXPECT_EQ(reader.RecordGroupSize(), group_size);

  std::vector<int32_t> read_all_records(kDatasetSize, 0);
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
  uint32_t records_read = 0;
  for (auto record_was_read : read_all_records) {
    if (record_was_read) {
      records_read++;
    }
  }
  EXPECT_EQ(records_read, kDatasetSize);
  EXPECT_TRUE(reader.SeekRecord(0));

  std::vector<uint64_t> indices = {0, 3, 5, 7, 101, 2000};
  std::vector<int32_t> read_indexed_records(indices.size(), 0);
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
  std::vector<int32_t> read_range_records(end - begin, 0);
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
  records_read = 0;
  for (auto record_was_read : read_range_records) {
    if (record_was_read) {
      records_read++;
    }
  }
  EXPECT_EQ(records_read, end - begin);

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

}  // namespace
}  // namespace array_record
