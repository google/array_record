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

#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "cpp/array_record_writer.h"
#include "cpp/common.h"
#include "cpp/layout.pb.h"
#include "cpp/test_utils.h"
#include "cpp/thread_pool.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/chunk_encoding/chunk_decoder.h"
#include "riegeli/records/chunk_reader.h"

constexpr uint32_t kDatasetSize = 10000;

namespace array_record {
namespace {

enum class CompressionType { kUncompressed, kBrotli, kZstd, kSnappy };

// Tuple params
//   CompressionType
//   transpose
//   use_thread_pool
class ArrayRecordReaderTest
    : public testing::TestWithParam<std::tuple<CompressionType, bool, bool>> {
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
};

TEST_P(ArrayRecordReaderTest, MoveTest) {
  std::string encoded;
  auto writer_options = GetWriterOptions().set_group_size(2);
  auto writer = ArrayRecordWriter<riegeli::StringWriter<>>(
      std::forward_as_tuple(&encoded), writer_options, nullptr);

  // Empty string should not crash the writer or the reader.
  std::vector<std::string> test_str{"aaa", "", "ccc", "dd", "e"};
  for (const auto& s : test_str) {
    EXPECT_TRUE(writer.WriteRecord(s));
  }
  ASSERT_TRUE(writer.Close());

  auto reader_before_move = ArrayRecordReader<riegeli::StringReader<>>(
      std::forward_as_tuple(encoded), ArrayRecordReaderBase::Options(),
      use_thread_pool() ? get_pool() : nullptr);
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

  ArrayRecordReader<riegeli::StringReader<>> reader =
      std::move(reader_before_move);
  // Once a reader is moved, it is closed.
  ASSERT_FALSE(reader_before_move.is_open());  // NOLINT

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
  auto writer = ArrayRecordWriter<riegeli::StringWriter<>>(
      std::forward_as_tuple(&encoded), GetWriterOptions(), get_pool());
  for (auto i : Seq(kDatasetSize)) {
    EXPECT_TRUE(writer.WriteRecord(records[i]));
  }
  ASSERT_TRUE(writer.Close());

  auto reader = ArrayRecordReader<riegeli::StringReader<>>(
      std::forward_as_tuple(encoded),
      ArrayRecordReaderBase::Options().set_readahead_buffer_size(2048),
      use_thread_pool() ? get_pool() : nullptr);
  ASSERT_TRUE(reader.status().ok());
  EXPECT_EQ(reader.NumRecords(), kDatasetSize);
  uint64_t group_size =
      std::min(ArrayRecordWriterBase::Options::kDefaultGroupSize, kDatasetSize);
  EXPECT_EQ(reader.RecordGroupSize(), group_size);

  ASSERT_TRUE(reader
                  .ParallelReadRecords(
                      [&](uint64_t record_index,
                          absl::string_view result_view) -> absl::Status {
                        EXPECT_EQ(result_view, records[record_index]);
                        return absl::OkStatus();
                      })
                  .ok());

  std::vector<uint64_t> indices = {0, 3, 5, 7, 101, 2000};
  ASSERT_TRUE(reader
                  .ParallelReadRecordsWithIndices(
                      indices,
                      [&](uint64_t indices_idx,
                          absl::string_view result_view) -> absl::Status {
                        EXPECT_EQ(result_view, records[indices[indices_idx]]);
                        return absl::OkStatus();
                      })
                  .ok());

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
                     testing::Bool(), testing::Bool()));

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
}

}  // namespace
}  // namespace array_record
