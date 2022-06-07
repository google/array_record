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

#include "third_party/array_record/cpp/test_utils.h"

#include <random>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/array_record/cpp/common.h"

namespace array_record {
namespace {

TEST(MTRandomBytesTest, ZeroLen) {
  std::mt19937 bitgen;
  auto result = MTRandomBytes(bitgen, 0);
  ASSERT_EQ(result.size(), 0);
}

TEST(MTRandomBytesTest, OneByte) {
  std::mt19937 bitgen, bitgen2;
  auto result = MTRandomBytes(bitgen, 1);
  ASSERT_EQ(result.size(), 1);
  ASSERT_NE(result[0], '\0');

  auto val = bitgen2();
  char char_val = *reinterpret_cast<char*>(&val);
  ASSERT_EQ(result[0], char_val);
}

TEST(MTRandomBytesTest, LargeVals) {
  constexpr size_t len = 123;
  std::mt19937 bitgen, bitgen2;

  auto result1 = MTRandomBytes(bitgen, len);
  auto result2 = MTRandomBytes(bitgen2, len);
  ASSERT_EQ(result1.size(), len);
  ASSERT_EQ(result2.size(), len);
  ASSERT_EQ(result1, result2);
}

}  // namespace

}  // namespace array_record
