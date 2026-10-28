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

// `BlockCache` holds immutable byte blocks of ArrayRecord files so that random
// reads cost one fetch per block instead of one per record. It has a hard
// byte budget (in-flight fetches reserve their bytes), a cap on in-flight
// fetches, and single-flight de-duplication. It never evicts a live block and
// never waits for capacity: when a block cannot be admitted (budget full or
// in-flight cap reached), `GetOrFetch` returns nullptr and the caller uses its
// direct read path. Space is freed only when the caller drops a block whose
// records have all been served (`DropIfReady`) or closes its reader
// (`EraseReader`).
//
// `ArrayRecordReader` drops a block once that reader has served every record
// of the block in a pass. This fits readers that each read whole files, e.g.
// one data source per shard under Grain's `InterleaveIterDataset`. With
// strided sharding across processes (`ds[i::n]`, or Grain's default
// `mp_prefetch` worker slicing over a `MapDataset`), each process serves only
// part of each block per pass, so blocks stay until the reader closes and each
// process fills its own budget.
#ifndef ARRAY_RECORD_CPP_BLOCK_CACHE_H_
#define ARRAY_RECORD_CPP_BLOCK_CACHE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace array_record {

class BlockCache {
 public:
  struct Config {
    uint64_t max_bytes = 0;      // Must be > 0.
    uint32_t max_inflight = 16;  // Must be >= 1.
  };

  struct Key {
    uint64_t reader_id = 0;
    uint32_t block_index = 0;

    friend bool operator==(const Key& a, const Key& b) {
      return a.reader_id == b.reader_id && a.block_index == b.block_index;
    }
    template <typename H>
    friend H AbslHashValue(H h, const Key& k) {
      return H::combine(std::move(h), k.reader_id, k.block_index);
    }
  };

  using Block = std::shared_ptr<const std::string>;
  // Returns exactly `block_bytes` bytes. Called without the cache lock held.
  using FetchFn = absl::FunctionRef<absl::StatusOr<Block>()>;

  struct Stats {
    uint64_t hits = 0;
    uint64_t misses_fetched = 0;
    uint64_t single_flight_waits = 0;
    uint64_t fallbacks_inflight = 0;
    uint64_t fallbacks_budget = 0;
    uint64_t fallbacks_error = 0;
    uint64_t closed_frees = 0;  // Blocks freed by `EraseReader`.
    uint64_t drained_frees = 0;
    uint64_t bytes_fetched = 0;
    uint64_t resident_bytes = 0;  // Ready entries.
    uint64_t pending_bytes = 0;   // Reserved by in-flight fetches.
    uint64_t peak_resident_bytes = 0;
    uint64_t inflight = 0;
    uint64_t max_bytes = 0;
    uint64_t max_inflight = 0;
  };

  explicit BlockCache(Config config);
  BlockCache(const BlockCache&) = delete;
  BlockCache& operator=(const BlockCache&) = delete;

  // Process-wide instance used by readers. Created on first use and
  // intentionally leaked. `config` is merged grow-only: `max_bytes` and
  // `max_inflight` only increase (each change is logged). After `fork()`, the
  // child starts without an instance and never touches the parent's lock.
  static BlockCache& Instance(const Config& config);
  // Returns the process-wide instance, or nullptr if never created.
  static BlockCache* InstanceIfExists();
  // Returns a process-unique reader id (>= 1).
  static uint64_t NewReaderId();
  // Destroys the process-wide instance. Tests only; no concurrent users.
  static void ResetGlobalForTesting();

  // Returns the block for `key`, fetching it with `fetch` on a miss. Returns
  // nullptr when the block cannot be admitted (in-flight cap, or the budget
  // is full: nothing is evicted to make room) or the fetch fails; the caller
  // must then use its direct read path. Never returns an error.
  Block GetOrFetch(Key key, uint64_t block_bytes, FetchFn fetch);

  // Frees `key` if it is ready (counted in `drained_frees`). No-op if absent
  // or still being fetched.
  void DropIfReady(Key key);

  // Frees all ready blocks of `reader_id` (each counted in `closed_frees`).
  // Precondition: no `GetOrFetch` for `reader_id` is in flight.
  void EraseReader(uint64_t reader_id);

  Stats GetStats() const;

 private:
  struct Entry;
  using EntryMap = absl::flat_hash_map<Key, std::shared_ptr<Entry>>;

  void MergeConfig(const Config& config);

  void EraseReadyLocked(EntryMap::iterator it)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  Config config_ ABSL_GUARDED_BY(mu_);
  EntryMap entries_ ABSL_GUARDED_BY(mu_);
  Stats stats_ ABSL_GUARDED_BY(mu_);
};

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_BLOCK_CACHE_H_
