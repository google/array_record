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

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"

namespace array_record {

struct BlockCache::Entry {
  bool ready = false;
  bool failed = false;
  uint64_t bytes = 0;
  Block data;
  absl::Notification done;
};

namespace {

std::atomic<BlockCache*> g_instance{nullptr};
std::atomic<uint64_t> g_next_reader_id{1};
// The instance inherited from the parent by a forked child. Never cleared:
// keeping the pointer in a global keeps the leaked instance reachable, so
// LeakSanitizer in the child does not report it as a leak.
std::atomic<BlockCache*> g_forked_parent_instance{nullptr};

// Runs in the child after fork(): drop (leak) the parent's instance without
// touching its mutex, which another parent thread may have held. Must stay
// lock-free and allocation-free, hence relaxed atomics only.
void ResetInstanceInForkedChild() {
  BlockCache* parent = g_instance.exchange(nullptr, std::memory_order_relaxed);
  if (parent != nullptr) {
    g_forked_parent_instance.store(parent, std::memory_order_relaxed);
  }
}

}  // namespace

BlockCache& BlockCache::Instance(const Config& config) {
  static const int atfork_registered = [] {
    const int err =
        pthread_atfork(nullptr, nullptr, &ResetInstanceInForkedChild);
    if (err != 0) {
      LOG(ERROR) << "ArrayRecord block cache: pthread_atfork failed: " << err;
    }
    return err;
  }();
  (void)atfork_registered;
  BlockCache* cache = g_instance.load(std::memory_order_acquire);
  if (cache == nullptr) {
    auto* fresh = new BlockCache(config);  // Intentionally leaked.
    if (g_instance.compare_exchange_strong(cache, fresh,
                                           std::memory_order_acq_rel)) {
      LOG(INFO) << "ArrayRecord block cache enabled: max_bytes="
                << config.max_bytes << " max_inflight=" << config.max_inflight;
      return *fresh;
    }
    delete fresh;  // Lost the race; `cache` now holds the winner.
  }
  cache->MergeConfig(config);
  return *cache;
}

BlockCache* BlockCache::InstanceIfExists() {
  return g_instance.load(std::memory_order_acquire);
}

uint64_t BlockCache::NewReaderId() {
  return g_next_reader_id.fetch_add(1, std::memory_order_relaxed);
}

void BlockCache::ResetGlobalForTesting() {
  delete g_instance.exchange(nullptr, std::memory_order_acq_rel);
}

void BlockCache::MergeConfig(const Config& config) {
  DCHECK_GT(config.max_bytes, 0);
  DCHECK_GE(config.max_inflight, 1);
  absl::MutexLock lock(mu_);
  bool changed = false;
  if (config.max_bytes > config_.max_bytes) {
    config_.max_bytes = config.max_bytes;
    changed = true;
  }
  if (config.max_inflight > config_.max_inflight) {
    config_.max_inflight = config.max_inflight;
    changed = true;
  }
  if (changed) {
    stats_.max_bytes = config_.max_bytes;
    stats_.max_inflight = config_.max_inflight;
    LOG(INFO) << "ArrayRecord block cache grown: max_bytes="
              << config_.max_bytes << " max_inflight=" << config_.max_inflight;
  }
}

BlockCache::BlockCache(Config config) : config_(config) {
  DCHECK_GT(config.max_bytes, 0);
  DCHECK_GE(config.max_inflight, 1);
  stats_.max_bytes = config.max_bytes;
  stats_.max_inflight = config.max_inflight;
}

void BlockCache::EraseReadyLocked(EntryMap::iterator it) {
  Entry& e = *it->second;
  stats_.resident_bytes -= e.bytes;
  entries_.erase(it);
}

BlockCache::Block BlockCache::GetOrFetch(Key key, uint64_t block_bytes,
                                         FetchFn fetch) {
  std::shared_ptr<Entry> entry;
  bool is_fetcher = false;
  {
    absl::MutexLock lock(mu_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      entry = it->second;
      // A block index always maps to the same byte range of the same reader.
      DCHECK_EQ(entry->bytes, block_bytes);
      if (entry->ready) {
        ++stats_.hits;
        return entry->data;
      }
      ++stats_.single_flight_waits;
    } else {
      if (stats_.inflight >= config_.max_inflight) {
        ++stats_.fallbacks_inflight;
        return nullptr;
      }
      // Full: fall back to the direct read. Live blocks are never evicted;
      // space is freed only by `DropIfReady` and `EraseReader`.
      if (block_bytes > config_.max_bytes ||
          stats_.resident_bytes + stats_.pending_bytes >
              config_.max_bytes - block_bytes) {
        ++stats_.fallbacks_budget;
        return nullptr;
      }
      entry = std::make_shared<Entry>();
      entry->bytes = block_bytes;
      entries_.emplace(key, entry);
      stats_.pending_bytes += block_bytes;
      ++stats_.inflight;
      is_fetcher = true;
    }
  }

  if (!is_fetcher) {
    // Single-flight waiter. The Notification orders the fetcher's writes.
    entry->done.WaitForNotification();
    if (entry->failed) {
      absl::MutexLock lock(mu_);
      ++stats_.fallbacks_error;
      return nullptr;
    }
    return entry->data;
  }

  absl::StatusOr<Block> result = fetch();

  absl::MutexLock lock(mu_);
  --stats_.inflight;
  stats_.pending_bytes -= block_bytes;
  auto it = entries_.find(key);
  DCHECK(it != entries_.end() && it->second == entry);
  if (!result.ok() || *result == nullptr || (*result)->size() != block_bytes) {
    entries_.erase(it);
    entry->failed = true;
    entry->done.Notify();
    ++stats_.fallbacks_error;
    return nullptr;
  }
  entry->ready = true;
  entry->data = *std::move(result);
  stats_.resident_bytes += block_bytes;
  ++stats_.misses_fetched;
  stats_.bytes_fetched += block_bytes;
  stats_.peak_resident_bytes =
      std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
  entry->done.Notify();
  return entry->data;
}

void BlockCache::DropIfReady(Key key) {
  absl::MutexLock lock(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end() || !it->second->ready) return;
  EraseReadyLocked(it);
  ++stats_.drained_frees;
}

void BlockCache::EraseReader(uint64_t reader_id) {
  absl::MutexLock lock(mu_);
  std::vector<Key> ready_keys;
  for (const auto& [key, entry] : entries_) {
    if (key.reader_id != reader_id) continue;
    DCHECK(entry->ready) << "EraseReader called with a fetch in flight";
    if (entry->ready) ready_keys.push_back(key);
  }
  for (const Key& key : ready_keys) EraseReadyLocked(entries_.find(key));
  stats_.closed_frees += ready_keys.size();
}

BlockCache::Stats BlockCache::GetStats() const {
  absl::MutexLock lock(mu_);
  return stats_;
}

}  // namespace array_record
