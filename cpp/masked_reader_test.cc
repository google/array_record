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

#include "cpp/masked_reader.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "cpp/tri_state_ptr.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/string_reader.h"

namespace array_record {
namespace {

using riegeli::StringReader;

TEST(MaskedReaderTest, SanityTest) {
  auto data = std::string("0123456789abcdef");
  auto base_reader = StringReader(data);
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

TEST(MaskedReaderTest, SharedRefAndNullptrTest) {
  std::string data = "0123456789abcdef";
  TriStatePtr<riegeli::Reader> main_reader(riegeli::Maker<StringReader>(data));
  auto shared_reader = main_reader.MakeShared();
  ASSERT_TRUE(shared_reader->Seek(4));

  MaskedReader masked_reader(shared_reader, 6);
  ASSERT_TRUE(masked_reader.ok());
  EXPECT_EQ(masked_reader.pos(), 4);
  EXPECT_EQ(masked_reader.Size(), 10);
  std::string result;
  EXPECT_TRUE(masked_reader.Read(6, result));
  EXPECT_EQ(result, "456789");

  MaskedReader null_reader(std::unique_ptr<riegeli::Reader>(nullptr), 6);
  EXPECT_FALSE(null_reader.ok());
}

TEST(MaskedReaderTest, BlockConstructorKeepsFileCoordinates) {
  auto block = std::make_shared<const std::string>("0123456789");
  MaskedReader reader(block, /*block_start=*/1000);
  ASSERT_TRUE(reader.ok());
  EXPECT_EQ(reader.pos(), 1000);
  EXPECT_EQ(reader.Size(), 1010);

  ASSERT_TRUE(reader.Seek(1004));
  std::string result;
  ASSERT_TRUE(reader.Read(3, result));
  EXPECT_EQ(result, "456");
  EXPECT_EQ(reader.pos(), 1007);

  // Seeking before the block or past its end fails.
  EXPECT_FALSE(reader.Seek(999));

  MaskedReader reader2(block, /*block_start=*/1000);
  EXPECT_FALSE(reader2.Seek(1011));

  // NewReader shares the block and keeps file coordinates.
  MaskedReader reader3(block, /*block_start=*/1000);
  auto child = reader3.NewReader(1008);
  ASSERT_NE(child, nullptr);
  ASSERT_TRUE(child->Read(2, result));
  EXPECT_EQ(result, "89");
  EXPECT_EQ(reader3.NewReader(999), nullptr);

  // A null block gives a failed reader instead of a null dereference.
  MaskedReader null_block(std::shared_ptr<const std::string>(nullptr),
                          /*block_start=*/1000);
  EXPECT_FALSE(null_block.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(null_block.status()));
}

TEST(MaskedReaderTest, PrematureEndOfSourceFailsWithInvalidArgument) {
  std::string data = "0123456789";  // 10 bytes.
  auto base_reader = StringReader(data);
  // Asks for 20 bytes: the source hits EOF while still ok().
  MaskedReader reader(base_reader.NewReader(0), 20);
  EXPECT_FALSE(reader.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(reader.status()));
}

}  // namespace

}  // namespace array_record
