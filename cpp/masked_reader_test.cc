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

#include "third_party/array_record/cpp/masked_reader.h"

#include <string>
#include <tuple>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/riegeli/bytes/string_reader.h"

namespace array_record {
namespace {

using riegeli::StringReader;

TEST(MaskedReaderTest, SanityTest) {
  auto data = std::string("0123456789abcdef");
  auto base_reader = StringReader<>(std::forward_as_tuple(data));
  // 56789abc
  auto masked_reader1 = MaskedReader(base_reader.NewReader(5), 8);
  // Matches where we offset the reader.
  EXPECT_EQ(masked_reader1.pos(), 5);
  // Matches offset + mask length
  EXPECT_EQ(masked_reader1.Size(), 8 + 5);
  {
    std::string result;
    masked_reader1.Read(4, result);
    EXPECT_EQ(result, "5678");
    EXPECT_EQ(masked_reader1.pos(), 5 + 4);
    masked_reader1.Read(4, result);
    EXPECT_EQ(result, "9abc");
    EXPECT_EQ(masked_reader1.pos(), 5 + 8);
  }

  auto masked_reader2 = masked_reader1.NewReader(7);
  // Size does not change
  EXPECT_EQ(masked_reader2->Size(), 8 + 5);
  // pos is the new position we set from NewReader
  EXPECT_EQ(masked_reader2->pos(), 7);
  {
    std::string result;
    masked_reader2->Read(4, result);
    EXPECT_EQ(result, "789a");
  }

  // Reaching position that is out of bound does not fail the base reader.
  // It simply returns a nullptr.
  EXPECT_EQ(masked_reader1.NewReader(0), nullptr);
  EXPECT_EQ(masked_reader1.NewReader(20), nullptr);
  EXPECT_TRUE(masked_reader1.ok());

  // Support seek
  masked_reader1.Seek(6);
  {
    std::string result;
    masked_reader1.Read(4, result);
    EXPECT_EQ(result, "6789");
  }

  // Seek beyond buffer is a failure.
  EXPECT_FALSE(masked_reader1.Seek(20));
}

}  // namespace

}  // namespace array_record
