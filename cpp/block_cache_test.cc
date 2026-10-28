/* Copyright 2026 Google LLC. All Rights Reserved.

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

#include "cpp/block_cache.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace array_record {
namespace {

using Block = BlockCache::Block;
using Key = BlockCache::Key;

BlockCache::Config Cfg(uint64_t max_bytes, uint32_t max_inflight) {
  BlockCache::Config config;
  config.max_bytes = max_bytes;
  config.max_inflight = max_inflight;
  return config;
}

Block MakeBlock(uint64_t size, char fill) {
  return std::make_shared<const std::string>(size, fill);
}

void WaitForSingleFlightWaiters(const BlockCache& cache, uint64_t n) {
  while (cache.GetStats().single_flight_waits < n) {
    absl::SleepFor(absl::Milliseconds(1));
  }
}

TEST(BlockCacheTest, MissThenHit) {
  BlockCache cache(Cfg(100, 4));
  int calls = 0;
  auto fetch = [&]() -> absl::StatusOr<Block> {
    ++calls;
    return MakeBlock(10, 'a');
  };
  Block b1 = cache.GetOrFetch(Key{1, 0}, 10, fetch);
  ASSERT_NE(b1, nullptr);
  EXPECT_EQ(*b1, std::string(10, 'a'));
  Block b2 = cache.GetOrFetch(Key{1, 0}, 10, fetch);
  EXPECT_EQ(b2, b1);
  EXPECT_EQ(calls, 1);
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.misses_fetched, 1u);
  EXPECT_EQ(s.hits, 1u);
  EXPECT_EQ(s.resident_bytes, 10u);
  EXPECT_EQ(s.pending_bytes, 0u);
  EXPECT_EQ(s.bytes_fetched, 10u);
  EXPECT_EQ(s.peak_resident_bytes, 10u);
}

TEST(BlockCacheTest, FullBudgetFallsBackAndFreesNothing) {
  BlockCache cache(Cfg(30, 4));
  int calls = 0;
  auto fetch = [&]() -> absl::StatusOr<Block> {
    ++calls;
    return MakeBlock(10, 'x');
  };
  cache.GetOrFetch(Key{1, 0}, 10, fetch);  // A
  cache.GetOrFetch(Key{1, 1}, 10, fetch);  // B
  cache.GetOrFetch(Key{1, 2}, 10, fetch);  // C
  ASSERT_EQ(calls, 3);
  // The budget is full: D falls back and nothing is freed.
  EXPECT_EQ(cache.GetOrFetch(Key{1, 3}, 10, fetch), nullptr);
  EXPECT_EQ(calls, 3);
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.fallbacks_budget, 1u);
  EXPECT_EQ(s.resident_bytes, 30u);
  EXPECT_EQ(s.pending_bytes, 0u);
  EXPECT_EQ(s.closed_frees, 0u);
  EXPECT_EQ(s.drained_frees, 0u);
  // A, B and C are all still cached.
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_NE(cache.GetOrFetch(Key{1, i}, 10, fetch), nullptr);
  }
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(cache.GetStats().hits, 3u);
  // D keeps falling back while the budget stays full.
  EXPECT_EQ(cache.GetOrFetch(Key{1, 3}, 10, fetch), nullptr);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(cache.GetStats().fallbacks_budget, 2u);
  EXPECT_EQ(cache.GetStats().resident_bytes, 30u);
}

TEST(BlockCacheTest, DropIfReadyMakesRoom) {
  BlockCache cache(Cfg(30, 4));
  int calls = 0;
  auto fetch = [&]() -> absl::StatusOr<Block> {
    ++calls;
    return MakeBlock(10, 'x');
  };
  for (uint32_t i = 0; i < 3; ++i) cache.GetOrFetch(Key{1, i}, 10, fetch);
  ASSERT_EQ(cache.GetOrFetch(Key{1, 3}, 10, fetch), nullptr);
  cache.DropIfReady(Key{1, 1});
  EXPECT_EQ(cache.GetStats().drained_frees, 1u);
  EXPECT_EQ(cache.GetStats().resident_bytes, 20u);
  // The freed space admits the next miss.
  EXPECT_NE(cache.GetOrFetch(Key{1, 3}, 10, fetch), nullptr);
  EXPECT_EQ(calls, 4);
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.misses_fetched, 4u);
  EXPECT_EQ(s.fallbacks_budget, 1u);
  EXPECT_EQ(s.resident_bytes, 30u);
  EXPECT_EQ(s.closed_frees, 0u);
}

TEST(BlockCacheTest, EraseReaderCountsClosedFreesAndMakesRoom) {
  BlockCache cache(Cfg(30, 4));
  int calls = 0;
  auto fetch = [&]() -> absl::StatusOr<Block> {
    ++calls;
    return MakeBlock(10, 'x');
  };
  cache.GetOrFetch(Key{1, 0}, 10, fetch);
  cache.GetOrFetch(Key{1, 1}, 10, fetch);
  cache.GetOrFetch(Key{2, 0}, 10, fetch);
  ASSERT_EQ(cache.GetOrFetch(Key{2, 1}, 10, fetch), nullptr);
  cache.EraseReader(1);
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.closed_frees, 2u);
  EXPECT_EQ(s.drained_frees, 0u);
  EXPECT_EQ(s.resident_bytes, 10u);
  // The freed space admits the next miss, and reader 2's block still hits.
  EXPECT_NE(cache.GetOrFetch(Key{2, 1}, 10, fetch), nullptr);
  EXPECT_NE(cache.GetOrFetch(Key{2, 0}, 10, fetch), nullptr);
  EXPECT_EQ(calls, 4);
  EXPECT_EQ(cache.GetStats().resident_bytes, 20u);
  // Erasing a reader without blocks frees nothing.
  cache.EraseReader(3);
  EXPECT_EQ(cache.GetStats().closed_frees, 2u);
}

TEST(BlockCacheTest, OversizedBlockFallsBackWithoutEvicting) {
  BlockCache cache(Cfg(30, 4));
  auto fetch10 = []() -> absl::StatusOr<Block> { return MakeBlock(10, 'a'); };
  cache.GetOrFetch(Key{1, 0}, 10, fetch10);
  cache.GetOrFetch(Key{1, 1}, 10, fetch10);
  int big_calls = 0;
  Block big = cache.GetOrFetch(Key{1, 2}, 40, [&]() -> absl::StatusOr<Block> {
    ++big_calls;
    return MakeBlock(40, 'b');
  });
  EXPECT_EQ(big, nullptr);
  EXPECT_EQ(big_calls, 0);
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.fallbacks_budget, 1u);
  EXPECT_EQ(s.closed_frees, 0u);
  EXPECT_EQ(s.drained_frees, 0u);
  EXPECT_EQ(s.resident_bytes, 20u);
}

TEST(BlockCacheTest, PendingReservationBlocksWithoutEvicting) {
  BlockCache cache(Cfg(30, 4));
  absl::Notification fetch_started, release;
  std::thread pending([&] {
    cache.GetOrFetch(Key{1, 9}, 20, [&]() -> absl::StatusOr<Block> {
      fetch_started.Notify();
      release.WaitForNotification();
      return MakeBlock(20, 'p');
    });
  });
  fetch_started.WaitForNotification();
  // 20 pending + 10 fits.
  ASSERT_NE(cache.GetOrFetch(
                Key{1, 0}, 10,
                []() -> absl::StatusOr<Block> { return MakeBlock(10, 'a'); }),
            nullptr);
  // 20 pending + 10 resident + 20 does not fit: falls back, frees nothing.
  EXPECT_EQ(cache.GetOrFetch(
                Key{1, 1}, 20,
                []() -> absl::StatusOr<Block> { return MakeBlock(20, 'c'); }),
            nullptr);
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.fallbacks_budget, 1u);
  EXPECT_EQ(s.closed_frees, 0u);
  EXPECT_EQ(s.resident_bytes, 10u);
  EXPECT_EQ(s.pending_bytes, 20u);
  release.Notify();
  pending.join();
  EXPECT_EQ(cache.GetStats().pending_bytes, 0u);
}

TEST(BlockCacheTest, InflightCapFallsBack) {
  BlockCache cache(Cfg(100, 1));
  absl::Notification fetch_started, release;
  std::thread pending([&] {
    cache.GetOrFetch(Key{1, 0}, 10, [&]() -> absl::StatusOr<Block> {
      fetch_started.Notify();
      release.WaitForNotification();
      return MakeBlock(10, 'a');
    });
  });
  fetch_started.WaitForNotification();
  EXPECT_EQ(cache.GetOrFetch(
                Key{1, 1}, 10,
                []() -> absl::StatusOr<Block> { return MakeBlock(10, 'b'); }),
            nullptr);
  EXPECT_EQ(cache.GetStats().fallbacks_inflight, 1u);
  release.Notify();
  pending.join();
}

TEST(BlockCacheTest, SingleFlightFetchesOnce) {
  BlockCache cache(Cfg(100, 4));
  constexpr int kThreads = 8;
  int calls = 0;
  std::vector<Block> results(kThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      results[i] =
          cache.GetOrFetch(Key{7, 3}, 10, [&]() -> absl::StatusOr<Block> {
            ++calls;  // Only the single fetcher runs this.
            WaitForSingleFlightWaiters(cache, kThreads - 1);
            return MakeBlock(10, 's');
          });
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_EQ(calls, 1);
  for (const auto& r : results) {
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r, results[0]);
  }
  EXPECT_EQ(cache.GetStats().single_flight_waits, uint64_t{kThreads} - 1);
}

TEST(BlockCacheTest, FetchErrorIsNotCachedAndWaitersFallBack) {
  BlockCache cache(Cfg(100, 4));
  Block waiter_result = MakeBlock(1, 'z');
  std::thread fetcher([&] {
    Block b = cache.GetOrFetch(Key{1, 0}, 10, [&]() -> absl::StatusOr<Block> {
      WaitForSingleFlightWaiters(cache, 1);
      return absl::UnavailableError("boom");
    });
    EXPECT_EQ(b, nullptr);
  });
  // Waits for the fetcher to own the pending entry before joining it.
  while (cache.GetStats().inflight == 0) absl::SleepFor(absl::Milliseconds(1));
  waiter_result =
      cache.GetOrFetch(Key{1, 0}, 10, []() -> absl::StatusOr<Block> {
        ADD_FAILURE() << "waiter must not fetch";
        return MakeBlock(10, 'w');
      });
  fetcher.join();
  EXPECT_EQ(waiter_result, nullptr);
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.fallbacks_error, 2u);  // fetcher + waiter
  EXPECT_EQ(s.resident_bytes, 0u);
  EXPECT_EQ(s.pending_bytes, 0u);
  // Not cached: the next call fetches again and succeeds.
  int calls = 0;
  ASSERT_NE(cache.GetOrFetch(Key{1, 0}, 10,
                             [&]() -> absl::StatusOr<Block> {
                               ++calls;
                               return MakeBlock(10, 'a');
                             }),
            nullptr);
  EXPECT_EQ(calls, 1);
}

TEST(BlockCacheTest, WrongSizedBlockIsAnError) {
  BlockCache cache(Cfg(100, 4));
  EXPECT_EQ(cache.GetOrFetch(
                Key{1, 0}, 10,
                []() -> absl::StatusOr<Block> { return MakeBlock(5, 'a'); }),
            nullptr);
  EXPECT_EQ(cache.GetStats().fallbacks_error, 1u);
  EXPECT_EQ(cache.GetStats().resident_bytes, 0u);
}

TEST(BlockCacheTest, DropIfReadyFreesOnlyReadyEntries) {
  BlockCache cache(Cfg(100, 4));
  cache.GetOrFetch(Key{1, 0}, 10, []() -> absl::StatusOr<Block> {
    return MakeBlock(10, 'a');
  });
  cache.DropIfReady(Key{1, 0});
  cache.DropIfReady(Key{1, 5});  // absent: no-op
  EXPECT_EQ(cache.GetStats().resident_bytes, 0u);
  EXPECT_EQ(cache.GetStats().drained_frees, 1u);

  absl::Notification fetch_started, release;
  std::thread pending([&] {
    cache.GetOrFetch(Key{1, 1}, 10, [&]() -> absl::StatusOr<Block> {
      fetch_started.Notify();
      release.WaitForNotification();
      return MakeBlock(10, 'p');
    });
  });
  fetch_started.WaitForNotification();
  cache.DropIfReady(Key{1, 1});  // pending: no-op
  release.Notify();
  pending.join();
  EXPECT_EQ(cache.GetStats().resident_bytes, 10u);
  EXPECT_EQ(cache.GetStats().drained_frees, 1u);
}

TEST(BlockCacheTest, EraseReaderFreesOnlyThatReader) {
  BlockCache cache(Cfg(100, 4));
  auto fetch = []() -> absl::StatusOr<Block> { return MakeBlock(10, 'a'); };
  cache.GetOrFetch(Key{1, 0}, 10, fetch);
  cache.GetOrFetch(Key{1, 1}, 10, fetch);
  cache.GetOrFetch(Key{2, 0}, 10, fetch);
  cache.EraseReader(1);
  EXPECT_EQ(cache.GetStats().resident_bytes, 10u);
  int calls = 0;
  cache.GetOrFetch(Key{2, 0}, 10, [&]() -> absl::StatusOr<Block> {
    ++calls;
    return MakeBlock(10, 'a');
  });
  EXPECT_EQ(calls, 0);
}

TEST(BlockCacheTest, ReturnedBlockOutlivesFree) {
  BlockCache cache(Cfg(20, 4));
  Block drained = cache.GetOrFetch(
      Key{1, 0}, 10,
      []() -> absl::StatusOr<Block> { return MakeBlock(10, 'h'); });
  Block closed = cache.GetOrFetch(Key{2, 0}, 10, []() -> absl::StatusOr<Block> {
    return MakeBlock(10, 'i');
  });
  cache.DropIfReady(Key{1, 0});
  cache.EraseReader(2);
  EXPECT_EQ(cache.GetStats().resident_bytes, 0u);
  EXPECT_EQ(*drained, std::string(10, 'h'));
  EXPECT_EQ(*closed, std::string(10, 'i'));
}

TEST(BlockCacheTest, ConcurrentRandomOpsKeepInvariants) {
  constexpr uint64_t kMaxBytes = 100;
  constexpr uint32_t kMaxInflight = 4;
  BlockCache cache(Cfg(kMaxBytes, kMaxInflight));
  auto size_of = [](uint32_t k) -> uint64_t { return 10 + k % 31; };
  auto fill_of = [](uint32_t k) -> char { return 'a' + k % 26; };
  std::vector<std::thread> threads;
  for (int t = 0; t < 16; ++t) {
    threads.emplace_back([&, t] {
      std::mt19937 rng(t);
      std::uniform_int_distribution<uint32_t> key_dist(0, 63);
      // Reader 1 is shared by all threads; reader `own` is only used (and
      // erased) by this thread, so `EraseReader` never races a fetch of it.
      const uint64_t own = 100 + t;
      for (int op = 0; op < 2000; ++op) {
        const uint32_t k = key_dist(rng);
        const uint64_t reader = op % 3 == 0 ? own : 1;
        if (op % 7 == 0) {
          cache.DropIfReady(Key{reader, k});
          continue;
        }
        if (op % 97 == 0) {
          cache.EraseReader(own);
          continue;
        }
        Block b = cache.GetOrFetch(Key{reader, k}, size_of(k),
                                   [&]() -> absl::StatusOr<Block> {
                                     return MakeBlock(size_of(k), fill_of(k));
                                   });
        if (b != nullptr) {
          ASSERT_EQ(*b, std::string(size_of(k), fill_of(k)));
        }
        if (op % 50 == 0) {
          BlockCache::Stats s = cache.GetStats();
          ASSERT_LE(s.resident_bytes + s.pending_bytes, kMaxBytes);
          ASSERT_LE(s.inflight, kMaxInflight);
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  BlockCache::Stats s = cache.GetStats();
  EXPECT_EQ(s.inflight, 0u);
  EXPECT_EQ(s.pending_bytes, 0u);
  EXPECT_LE(s.peak_resident_bytes, kMaxBytes);
  EXPECT_GT(s.fallbacks_budget, 0u);
  // Nothing is evicted: once every reader is erased, each fetched block has
  // left by exactly one drain or close.
  cache.EraseReader(1);
  for (int t = 0; t < 16; ++t) cache.EraseReader(100 + t);
  s = cache.GetStats();
  EXPECT_EQ(s.resident_bytes, 0u);
  EXPECT_EQ(s.misses_fetched, s.drained_frees + s.closed_frees);
}

TEST(BlockCacheInstanceTest, GrowOnlyConfigAndReset) {
  BlockCache::ResetGlobalForTesting();
  EXPECT_EQ(BlockCache::InstanceIfExists(), nullptr);
  BlockCache& cache = BlockCache::Instance(Cfg(100, 4));
  EXPECT_EQ(BlockCache::InstanceIfExists(), &cache);
  EXPECT_EQ(cache.GetStats().max_bytes, 100u);
  EXPECT_EQ(cache.GetStats().max_inflight, 4u);
  EXPECT_EQ(&BlockCache::Instance(Cfg(50, 8)), &cache);
  EXPECT_EQ(cache.GetStats().max_bytes, 100u);  // never shrinks
  EXPECT_EQ(cache.GetStats().max_inflight, 8u);
  BlockCache::Instance(Cfg(200, 2));
  EXPECT_EQ(cache.GetStats().max_bytes, 200u);
  EXPECT_EQ(cache.GetStats().max_inflight, 8u);
  BlockCache::ResetGlobalForTesting();
  EXPECT_EQ(BlockCache::InstanceIfExists(), nullptr);
}

TEST(BlockCacheInstanceTest, NewReaderIdsAreUniqueAndNonZero) {
  uint64_t a = BlockCache::NewReaderId();
  uint64_t b = BlockCache::NewReaderId();
  EXPECT_NE(a, 0u);
  EXPECT_NE(b, 0u);
  EXPECT_NE(a, b);
}

TEST(BlockCacheInstanceTest, ForkedChildStartsWithoutInstance) {
  BlockCache::ResetGlobalForTesting();
  BlockCache::Instance(Cfg(100, 4));
  pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    _exit(BlockCache::InstanceIfExists() == nullptr ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_NE(BlockCache::InstanceIfExists(), nullptr);  // parent unaffected
  BlockCache::ResetGlobalForTesting();
}

}  // namespace
}  // namespace array_record
