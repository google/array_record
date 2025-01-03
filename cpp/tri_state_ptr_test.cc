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

#include "cpp/tri_state_ptr.h"
#include <utility>

#include "gtest/gtest.h"
#include "absl/synchronization/notification.h"
#include "cpp/common.h"
#include "cpp/thread_pool.h"
#include "riegeli/base/maker.h"

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

class TriStatePtrTest : public testing::Test {
 public:
  TriStatePtrTest() : pool_(ArrayRecordGlobalPool()) {}

 protected:
  ARThreadPool* pool_;
};

TEST_F(TriStatePtrTest, SanityTest) {
  TriStatePtr<FooBase> foo_main(std::move(riegeli::Maker<Foo>(1)));
  EXPECT_EQ(foo_main.state(), TriStatePtr<FooBase>::State::kNoRef);
  absl::Notification notification;
  {
    pool_->Schedule(
        [foo_shared = foo_main.MakeShared(), &notification]() mutable {
          notification.WaitForNotification();
          EXPECT_EQ(foo_shared->value(), 1);
          const auto second_foo_shared = foo_shared;
          foo_shared->add_value(1);
          EXPECT_EQ(second_foo_shared->value(), 2);
        });
  }
  EXPECT_EQ(foo_main.state(), TriStatePtr<FooBase>::State::kSharing);
  notification.Notify();
  auto foo_unique = foo_main.WaitAndMakeUnique();
  foo_unique->mul_value(3);
  EXPECT_EQ(foo_unique->value(), 6);
  EXPECT_EQ(foo_main.state(), TriStatePtr<FooBase>::State::kUnique);
}

}  // namespace
}  // namespace array_record
