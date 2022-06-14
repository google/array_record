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

// Tests for parallel_for.h.
#include "cpp/parallel_for.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "cpp/thread_pool.h"

namespace array_record {

class ParallelForTest : public testing::Test {
 protected:
  void SetUp() final { pool_ = ArrayRecordGlobalPool(); }

 public:
  static constexpr int32_t kNumElements = 1000000;
  ARThreadPool* pool_;
};

TEST_F(ParallelForTest, ImplicitStepImplicitBlock) {
  std::vector<double> result(kNumElements);
  auto l = [&result](size_t j) { result[j] += sqrt(j); };

  ParallelFor(Seq(kNumElements), pool_, l);
  for (size_t j : Seq(kNumElements)) {
    EXPECT_EQ(sqrt(j), result[j]);
  }

  result.clear();
  result.resize(kNumElements);
  ParallelFor(Seq(kNumElements), nullptr, l);
  for (size_t j : Seq(kNumElements)) {
    EXPECT_EQ(sqrt(j), result[j]);
  }
}

TEST_F(ParallelForTest, ImplicitStepExplicitBlock) {
  std::vector<double> result(kNumElements);
  auto l = [&result](size_t j) { result[j] += sqrt(j); };
  ParallelFor<10>(Seq(kNumElements), pool_, l);

  for (size_t j : Seq(kNumElements)) {
    EXPECT_EQ(sqrt(j), result[j]);
  }

  result.clear();
  result.resize(kNumElements);
  ParallelFor<10>(Seq(kNumElements), nullptr, l);

  for (size_t j : Seq(kNumElements)) {
    EXPECT_EQ(sqrt(j), result[j]);
  }
}

TEST_F(ParallelForTest, ExplicitStepExplicitBlock) {
  std::vector<double> result(kNumElements);
  auto l = [&result](size_t j) { result[j] += sqrt(j); };
  ParallelFor<10>(SeqWithStride<2>(kNumElements), pool_, l);

  for (size_t j : Seq(kNumElements)) {
    // We only did the even numbered elements, so the odd ones should be zero.
    if (j & 1) {
      EXPECT_EQ(result[j], 0.0);
    } else {
      EXPECT_EQ(sqrt(j), result[j]);
    }
  }

  result.clear();
  result.resize(kNumElements);
  ParallelFor<10>(SeqWithStride<2>(kNumElements), nullptr, l);

  for (size_t j : Seq(kNumElements)) {
    // We only did the even numbered elements, so the odd ones should be zero.
    if (j & 1) {
      EXPECT_EQ(result[j], 0.0);
    } else {
      EXPECT_EQ(sqrt(j), result[j]);
    }
  }
}

TEST_F(ParallelForTest, ExplicitStepImplicitBlock) {
  std::vector<double> result(kNumElements);
  auto l = [&result](size_t j) { result[j] += sqrt(j); };
  ParallelFor(SeqWithStride<2>(kNumElements), pool_, l);

  for (size_t j : Seq(kNumElements)) {
    // We only did the even numbered elements, so the odd ones should be zero.
    if (j & 1) {
      EXPECT_EQ(result[j], 0.0);
    } else {
      EXPECT_EQ(sqrt(j), result[j]);
    }
  }

  result.clear();
  result.resize(kNumElements);
  ParallelFor(SeqWithStride<2>(kNumElements), nullptr, l);

  for (size_t j : Seq(kNumElements)) {
    // We only did the even numbered elements, so the odd ones should be zero.
    if (j & 1) {
      EXPECT_EQ(result[j], 0.0);
    } else {
      EXPECT_EQ(sqrt(j), result[j]);
    }
  }
}

TEST_F(ParallelForTest, ExampleCompiles) {
  // Once c-style approves lambdas, usage of this library will become very clean
  // as illustrated below.  Compute the square root of every number from 0 to
  // 1000000 in parallel.
  std::vector<double> sqrts(1000000);
  auto pool = ArrayRecordGlobalPool();

  ParallelFor(Seq(sqrts.size()), pool,
              [&sqrts](size_t j) { sqrts[j] = sqrt(j); });

  // Only compute the square roots of even numbers by using an explicit step.
  ParallelFor(SeqWithStride<2>(sqrts.size()), pool,
              [&sqrts](size_t j) { sqrts[j] = j; });

  // The block_size parameter can be adjusted to control the granularity of
  // parallelism.  This parameter represents the number of iterations of the
  // loop that will be done in a single thread before communicating with any
  // other threads.  Smaller block sizes lead to better load balancing between
  // threads.  Larger block sizes lead to less communication overhead and less
  // risk of false sharing (http://en.wikipedia.org/wiki/False_sharing)
  // when writing to adjacent array elements from different threads based
  // on the loop index.  The default block size creates approximately 4
  // blocks per thread.
  //
  // Use an explicit block size of 10.
  ParallelFor<10>(SeqWithStride<2>(sqrts.size()), pool,
                  [&sqrts](size_t j) { sqrts[j] = j; });
}

TEST_F(ParallelForTest, ParallelForWithStatusTest) {
  std::atomic_int counter = 0;
  auto status =
      ParallelForWithStatus<1>(Seq(kNumElements), pool_, [&](size_t i) {
        counter.fetch_add(1, std::memory_order_release);
        return absl::OkStatus();
      });
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(counter.load(std::memory_order_acquire), kNumElements);
}

TEST_F(ParallelForTest, ParallelForWithStatusTestShortCircuit) {
  std::atomic_int counter = 0;
  auto status =
      ParallelForWithStatus<1>(Seq(kNumElements), pool_, [&](size_t i) {
        counter.fetch_add(1, std::memory_order_release);
        return absl::UnknownError("Intended error");
      });
  EXPECT_EQ(status.code(), absl::StatusCode::kUnknown);
  EXPECT_LT(counter.load(std::memory_order_acquire), kNumElements);
}

}  // namespace array_record
