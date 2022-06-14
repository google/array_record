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

#include "cpp/thread_compatible_shared_ptr.h"

#include <memory>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cpp/common.h"
#include "cpp/thread_pool.h"

namespace array_record {
namespace {

class FooBase {
 public:
  virtual ~FooBase() = default;
  virtual int value() const = 0;
  virtual void add_value(int v) = 0;
  virtual void mul_value(int v) = 0;
};

class Foo : public FooBase {
 public:
  explicit Foo(int v) : value_(v) {}
  DECLARE_MOVE_ONLY_CLASS(Foo);

  int value() const override { return value_; };
  void add_value(int v) override { value_ += v; }
  void mul_value(int v) override { value_ *= v; }

 private:
  int value_;
};

class ThreadCompatibleSharedPtrTest : public testing::Test {
 public:
  ThreadCompatibleSharedPtrTest() : pool_(ArrayRecordGlobalPool()) {}

 protected:
  ARThreadPool* pool_;
};

TEST_F(ThreadCompatibleSharedPtrTest, SanityTest) {
  auto owned = ThreadCompatibleSharedPtr<FooBase>::Create(Foo(1));
  EXPECT_TRUE(owned.is_owning());
  EXPECT_EQ(owned.refcnt(), 0);

  auto new_owned = std::move(owned);
  EXPECT_TRUE(new_owned.is_owning());
  // Not owning after move
  EXPECT_FALSE(owned.is_owning());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(new_owned.refcnt(), 0);

  // Valid const access
  pool_->Schedule([refobj = new_owned]() {
    absl::SleepFor(absl::Milliseconds(10));
    EXPECT_EQ(refobj->value(), 1);
    EXPECT_EQ(refobj.refcnt(), 1);
    const auto second_ref = refobj;
    EXPECT_EQ(second_ref.refcnt(), 2);
    FooBase* foo =
        const_cast<FooBase*>(reinterpret_cast<const FooBase*>(refobj.get()));
    foo->add_value(1);
  });
  EXPECT_EQ(new_owned.refcnt(), 1);
  // Blocks until pool add value 1 by 1, value is now 2
  new_owned->mul_value(3);
  EXPECT_EQ(new_owned->value(), 6);
  // Destruction blocks until thread is executed
}

TEST_F(ThreadCompatibleSharedPtrTest, SanityTestWithTupleArg) {
  auto owned =
      ThreadCompatibleSharedPtr<FooBase>::Create<Foo>(std::forward_as_tuple(1));
  EXPECT_TRUE(owned.is_owning());
  EXPECT_EQ(owned.refcnt(), 0);

  auto new_owned = std::move(owned);
  EXPECT_TRUE(new_owned.is_owning());
  // Not owning after move
  EXPECT_FALSE(owned.is_owning());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(new_owned.refcnt(), 0);

  // Valid const access
  pool_->Schedule([refobj = new_owned]() {
    absl::SleepFor(absl::Milliseconds(10));
    EXPECT_EQ(refobj->value(), 1);
    EXPECT_EQ(refobj.refcnt(), 1);
    FooBase* foo =
        const_cast<FooBase*>(reinterpret_cast<const FooBase*>(refobj.get()));
    foo->add_value(1);
  });
  EXPECT_EQ(new_owned.refcnt(), 1);
  // Blocks until pool add value 1 by 1, value is now 2
  new_owned->mul_value(3);
  EXPECT_EQ(new_owned->value(), 6);
  // Destruction blocks until thread is executed
}

TEST_F(ThreadCompatibleSharedPtrTest, SanityTestWithUnique) {
  auto owned =
      ThreadCompatibleSharedPtr<FooBase>::Create(std::make_unique<Foo>(1));
  EXPECT_TRUE(owned.is_owning());
  EXPECT_EQ(owned.refcnt(), 0);

  auto new_owned = std::move(owned);
  EXPECT_TRUE(new_owned.is_owning());
  // Not owning after move
  EXPECT_FALSE(owned.is_owning());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(new_owned.refcnt(), 0);

  // Valid const access
  pool_->Schedule([refobj = new_owned]() {
    absl::SleepFor(absl::Milliseconds(10));
    EXPECT_EQ(refobj->value(), 1);
    EXPECT_EQ(refobj.refcnt(), 1);
    FooBase* foo =
        const_cast<FooBase*>(reinterpret_cast<const FooBase*>(refobj.get()));
    foo->add_value(1);
  });
  EXPECT_EQ(new_owned.refcnt(), 1);
  // Blocks until pool add value 1 by 1, value is now 2
  new_owned->mul_value(3);
  EXPECT_EQ(new_owned->value(), 6);
  // Destruction blocks until thread is executed
}

}  // namespace
}  // namespace array_record
