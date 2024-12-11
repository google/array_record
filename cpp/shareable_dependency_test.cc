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

#include "cpp/shareable_dependency.h"

#include <memory>
#include <optional>
#include <utility>

#include "gtest/gtest.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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

class ShareableDependencyTest : public testing::Test {
 public:
  ShareableDependencyTest() : pool_(ArrayRecordGlobalPool()) {}

 protected:
  ARThreadPool* pool_;
};

TEST_F(ShareableDependencyTest, SanityTest) {
  ShareableDependency<FooBase*, Foo> main(riegeli::Maker(1));
  EXPECT_TRUE(main.IsUnique());

  auto new_main = std::move(main);
  EXPECT_TRUE(new_main.IsUnique());
  // Not owning after move
  EXPECT_FALSE(main.IsUnique());  // NOLINT(bugprone-use-after-move)

  main = std::move(new_main);
  EXPECT_TRUE(main.IsUnique());
  // Not owning after move
  EXPECT_FALSE(new_main.IsUnique());  // NOLINT(bugprone-use-after-move)

  absl::Notification notification;
  pool_->Schedule(
      [refobj = std::make_shared<DependencyShare<FooBase*>>(main.Share()),
       &notification] {
        notification.WaitForNotification();
        absl::SleepFor(absl::Milliseconds(10));
        EXPECT_EQ(refobj->get()->value(), 1);
        refobj->get()->add_value(1);
      });
  EXPECT_FALSE(main.IsUnique());
  notification.Notify();
  auto& unique = main.WaitUntilUnique();
  // Value is now 2
  unique->mul_value(3);
  EXPECT_EQ(unique->value(), 6);
  // Destruction blocks until thread is executed
}

TEST_F(ShareableDependencyTest, SanityTestWithReset) {
  ShareableDependency<FooBase*, std::optional<Foo>> main;
  EXPECT_FALSE(main.IsUnique());

  main.Reset(riegeli::Maker(1));
  EXPECT_TRUE(main.IsUnique());

  absl::Notification notification;
  pool_->Schedule(
      [refobj = std::make_shared<DependencyShare<FooBase*>>(main.Share()),
       &notification] {
        notification.WaitForNotification();
        absl::SleepFor(absl::Milliseconds(10));
        EXPECT_EQ(refobj->get()->value(), 1);
        refobj->get()->add_value(1);
      });
  EXPECT_FALSE(main.IsUnique());
  notification.Notify();
  auto& unique = main.WaitUntilUnique();
  // Value is now 2
  unique->mul_value(3);
  EXPECT_EQ(unique->value(), 6);
  // Destruction blocks until thread is executed
}

}  // namespace
}  // namespace array_record
