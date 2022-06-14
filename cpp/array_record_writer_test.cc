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

#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "cpp/common.h"
#include "cpp/layout.pb.h"
#include "cpp/test_utils.h"
#include "cpp/thread_pool.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/chunk_encoding/constants.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "riegeli/records/records_metadata.pb.h"

namespace array_record {

namespace {

enum class CompressionType { kUncompressed, kBrotli, kZstd, kSnappy };

// Tuple params
//   CompressionType
//   padding
//   transpose
//   use ThreadPool
class ArrayRecordWriterTest
    : public testing::TestWithParam<
          std::tuple<CompressionType, bool, bool, bool>> {
 public:
  ArrayRecordWriterBase::Options GetOptions() {
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
    options.set_pad_to_block_boundary(std::get<1>(GetParam()));
    options.set_transpose(std::get<2>(GetParam()));
    return options;
  }
};

template <typename T>
void SilenceMoveAfterUseForTest(T&) {}

TEST_P(ArrayRecordWriterTest, MoveTest) {
  std::string encoded;
  ARThreadPool* pool = nullptr;
  if (std::get<3>(GetParam())) {
    pool = ArrayRecordGlobalPool();
  }
  auto options = GetOptions();
  options.set_group_size(2);
  auto writer = ArrayRecordWriter<riegeli::StringWriter<>>(
      std::forward_as_tuple(&encoded), options, pool);

  // Empty string should not crash the writer/reader.
  std::vector<std::string> test_str{"aaa", "", "ccc", "dd", "e"};
  for (auto i : Seq(3)) {
    EXPECT_TRUE(writer.WriteRecord(test_str[i]));
  }

  auto moved_writer = std::move(writer);
  SilenceMoveAfterUseForTest(writer);
  // Once moved, writer is closed.
  ASSERT_FALSE(writer.is_open());
  ASSERT_TRUE(moved_writer.is_open());
  // Once moved we can no longer write records.
  EXPECT_FALSE(writer.WriteRecord(test_str[3]));

  ASSERT_TRUE(moved_writer.status().ok());
  EXPECT_TRUE(moved_writer.WriteRecord(test_str[3]));
  EXPECT_TRUE(moved_writer.WriteRecord(test_str[4]));
  ASSERT_TRUE(moved_writer.Close());

  auto reader = riegeli::RecordReader<riegeli::StringReader<>>(
      std::forward_as_tuple(encoded));
  for (const auto& expected : test_str) {
    std::string result;
    reader.ReadRecord(result);
    EXPECT_EQ(result, expected);
  }
}

TEST_P(ArrayRecordWriterTest, RandomDatasetTest) {
  std::mt19937 bitgen;
  constexpr uint32_t kGroupSize = 100;
  constexpr uint32_t num_records = 1357;
  std::vector<std::string> records(num_records);
  std::uniform_int_distribution<> dist(0, 123);
  for (auto i : Seq(num_records)) {
    size_t len = dist(bitgen);
    records[i] = MTRandomBytes(bitgen, len);
  }
  // results are stored in encoded
  std::string encoded;

  ARThreadPool* pool = nullptr;
  if (std::get<3>(GetParam())) {
    pool = ArrayRecordGlobalPool();
  }
  auto options = GetOptions();
  options.set_group_size(kGroupSize);

  auto writer = ArrayRecordWriter<riegeli::StringWriter<>>(
      std::forward_as_tuple(&encoded), options, pool);

  for (auto i : Seq(num_records)) {
    EXPECT_TRUE(writer.WriteRecord(records[i]));
  }
  ASSERT_TRUE(writer.Close());

  auto reader = riegeli::RecordReader<riegeli::StringReader<>>(
      std::forward_as_tuple(encoded));

  // Verify metadata
  ASSERT_TRUE(reader.CheckFileFormat());

  // Verify each record
  for (auto i : Seq(num_records)) {
    absl::string_view result_view;
    ASSERT_TRUE(reader.ReadRecord(result_view));
    EXPECT_EQ(result_view, records[i]);
  }

  // Verify postcript
  ASSERT_TRUE(reader.Seek(reader.Size().value() - (1 << 16)));
  RiegeliPostscript postscript;
  ASSERT_TRUE(reader.ReadRecord(postscript));
  ASSERT_EQ(postscript.magic(), 0x71930e704fdae05eULL);

  // Verify Footer
  ASSERT_TRUE(reader.Seek(postscript.footer_offset()));
  RiegeliFooterMetadata footer_metadata;
  ASSERT_TRUE(reader.ReadRecord(footer_metadata));
  ASSERT_EQ(footer_metadata.array_record_metadata().version(), 1);
  auto num_chunks = footer_metadata.array_record_metadata().num_chunks();
  std::vector<ArrayRecordFooter> footers(num_chunks);
  for (auto i : Seq(num_chunks)) {
    ASSERT_TRUE(reader.ReadRecord(footers[i]));
  }

  // Verify we can access the file randomly by chunk_offset recorded in the
  // footer
  for (auto i = 0UL; i < num_chunks; ++i) {
    ASSERT_TRUE(reader.Seek(footers[i].chunk_offset()));
    absl::string_view result_view;
    ASSERT_TRUE(reader.ReadRecord(result_view)) << reader.status();
    EXPECT_EQ(result_view, records[i * kGroupSize]);
  }
  ASSERT_TRUE(reader.Close());
}

INSTANTIATE_TEST_SUITE_P(
    ParamTest, ArrayRecordWriterTest,
    testing::Combine(testing::Values(CompressionType::kUncompressed,
                                     CompressionType::kBrotli,
                                     CompressionType::kZstd,
                                     CompressionType::kSnappy),
                     testing::Bool(), testing::Bool(), testing::Bool()));

TEST(ArrayRecordWriterOptionsTest, ParsingTest) {
  {
    auto option = ArrayRecordWriterBase::Options::FromString("").value();
    EXPECT_EQ(option.group_size(),
              ArrayRecordWriterBase::Options::kDefaultGroupSize);
    EXPECT_FALSE(option.transpose());
    EXPECT_EQ(option.max_parallelism(), std::nullopt);
    EXPECT_EQ(option.compressor_options().compression_type(),
              riegeli::CompressionType::kBrotli);
    EXPECT_FALSE(option.pad_to_block_boundary());
  }
  {
    auto option = ArrayRecordWriterBase::Options::FromString(
                      "group_size:32,transpose,window_log:20")
                      .value();
    EXPECT_EQ(option.group_size(), 32);
    EXPECT_TRUE(option.transpose());
    EXPECT_EQ(option.max_parallelism(), std::nullopt);
    EXPECT_EQ(option.compressor_options().compression_type(),
              riegeli::CompressionType::kBrotli);
    EXPECT_EQ(option.compressor_options().brotli_window_log(), 20);
    EXPECT_FALSE(option.pad_to_block_boundary());
  }
  {
    auto option = ArrayRecordWriterBase::Options::FromString(
                      "group_size:32,transpose,zstd:5")
                      .value();
    EXPECT_EQ(option.group_size(), 32);
    EXPECT_TRUE(option.transpose());
    EXPECT_EQ(option.max_parallelism(), std::nullopt);
    EXPECT_EQ(option.compressor_options().compression_type(),
              riegeli::CompressionType::kZstd);
    EXPECT_EQ(option.compressor_options().zstd_window_log(), 20);
    EXPECT_EQ(option.compressor_options().compression_level(), 5);
    EXPECT_FALSE(option.pad_to_block_boundary());
  }
  {
    auto option = ArrayRecordWriterBase::Options::FromString(
                      "uncompressed,pad_to_block_boundary:true")
                      .value();
    EXPECT_EQ(option.group_size(),
              ArrayRecordWriterBase::Options::kDefaultGroupSize);
    EXPECT_FALSE(option.transpose());
    EXPECT_EQ(option.max_parallelism(), std::nullopt);
    EXPECT_EQ(option.compressor_options().compression_type(),
              riegeli::CompressionType::kNone);
    EXPECT_TRUE(option.pad_to_block_boundary());
  }
}

}  // namespace
}  // namespace array_record
