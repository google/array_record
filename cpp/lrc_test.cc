/* Copyright 2025 Google LLC. All Rights Reserved.

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

#include "cpp/lrc.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "absl/log/log.h"

namespace array_record {
namespace {

template <typename T>
void EvalLRCDecoder(const LRCDecoder<T>& decoder,
                    const std::vector<T>& expected) {
  EXPECT_EQ(decoder.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(decoder[i], expected[i]) << "i: " << i;
  }
}

TEST(LRCTest, ZeroItemTest) {
  LRCEncoder encoder;
  auto u32_lrc_data = encoder.Encode<uint32_t>(nullptr, 0);
  auto u64_lrc_data = encoder.Encode<uint64_t>(nullptr, 0);
  auto u32_decoder = LRCDecoder(u32_lrc_data);
  auto u64_decoder = LRCDecoder(u64_lrc_data);
  EXPECT_EQ(u32_lrc_data.num_elements, 0);
  EXPECT_EQ(u32_lrc_data.num_cls, 0);
  EXPECT_EQ(u64_lrc_data.num_elements, 0);
  EXPECT_EQ(u64_lrc_data.num_cls, 0);
  EXPECT_EQ(u32_decoder.size(), 0);
  EXPECT_EQ(u64_decoder.size(), 0);
}

TEST(LRCTest, U32ExtremeValueTest) {
  const uint32_t maxv = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> arr0{maxv};
  std::vector<uint32_t> arr1{0, maxv};
  std::vector<uint32_t> arr2{maxv, 0};

  LRCEncoder encoder;
  auto lrc0 = encoder.Encode(arr0.data(), arr0.size());
  auto lrc1 = encoder.Encode(arr1.data(), arr1.size());
  auto lrc2 = encoder.Encode(arr2.data(), arr2.size());

  auto lrc0_decoder = LRCDecoder(lrc0);
  auto lrc1_decoder = LRCDecoder(lrc1);
  auto lrc2_decoder = LRCDecoder(lrc2);
  EvalLRCDecoder(lrc0_decoder, arr0);
  EvalLRCDecoder(lrc1_decoder, arr1);
  EvalLRCDecoder(lrc2_decoder, arr2);
}

TEST(LRCTest, U64ExtremeValueTest) {
  const uint64_t max64 = std::numeric_limits<int64_t>::max();
  const uint64_t max32 = std::numeric_limits<uint32_t>::max();
  std::vector<uint64_t> arr0{max64, max32};
  std::vector<uint64_t> arr1{0, max32, max32 * 2};

  LRCEncoder encoder;
  auto lrc0 = encoder.Encode(arr0.data(), arr0.size());
  auto lrc1 = encoder.Encode(arr1.data(), arr1.size());

  EXPECT_EQ(lrc0.num_cls, 17);
  EXPECT_EQ(lrc1.num_cls, 2);

  auto lrc0_decoder = LRCDecoder(lrc0);
  auto lrc1_decoder = LRCDecoder(lrc1);
  EvalLRCDecoder(lrc0_decoder, arr0);
  EvalLRCDecoder(lrc1_decoder, arr1);
}

using PLRCTest = testing::TestWithParam<std::tuple<size_t, double>>;

TEST_P(PLRCTest, U32LRCTest) {
  size_t len = std::get<0>(GetParam());
  std::mt19937 gen(42);
  std::poisson_distribution<uint32_t> dist(std::get<1>(GetParam()));
  std::vector<uint32_t> arr(len);
  uint32_t acc = 0;
  for (size_t i = 0; i < len; ++i) {
    acc += dist(gen);
    arr[i] = acc;
  }
  LRCEncoder encoder;
  auto lrc = encoder.Encode(arr.data(), len);
  auto lrc_decoder = LRCDecoder(lrc);
  EvalLRCDecoder(lrc_decoder, arr);
  LOG(INFO) << "Bit rate: " << lrc_decoder.bit_rate();
}

TEST_P(PLRCTest, U64LRCTest) {
  size_t len = std::get<0>(GetParam());
  std::mt19937 gen(42);
  double poisson_lambda = std::get<1>(GetParam());
  std::poisson_distribution<uint64_t> dist(poisson_lambda);
  std::vector<uint64_t> arr(len);
  uint32_t acc = 0;
  for (size_t i = 0; i < len; ++i) {
    acc += dist(gen);
    arr[i] = acc;
  }
  LRCEncoder encoder;
  auto lrc = encoder.Encode(arr.data(), len);
  auto lrc_decoder = LRCDecoder(lrc);
  EvalLRCDecoder(lrc_decoder, arr);
  LOG(INFO) << "Poisson lambda: " << poisson_lambda
            << ", Bit rate: " << lrc_decoder.bit_rate();
}

// TODO(fchern): Small poisson_lambda didn't produce small bitrate.
// We might be able to fix this by using minimax regression instead of ordinary
// least squares.
INSTANTIATE_TEST_SUITE_P(PLRCTestSuite, PLRCTest,
                         testing::Combine(testing::Values(17, 300, 5000, 20000),
                                          testing::Values(5.0, 100.0,
                                                          1000000.0)));

}  // namespace
}  // namespace array_record
