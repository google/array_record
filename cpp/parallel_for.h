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

#ifndef ARRAY_RECORD_CPP_PARALLEL_FOR_H_
#define ARRAY_RECORD_CPP_PARALLEL_FOR_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "third_party/absl/base/thread_annotations.h"
#include "third_party/absl/functional/function_ref.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/array_record/cpp/common.h"
#include "third_party/array_record/cpp/thread_pool.h"

namespace array_record {

// kDynamicBatchSize - when a batch size isn't specified, ParallelFor defaults
// to dividing the work into (4 * num_threads) batches, enabling decently good
// parallelism, while minimizing coordination overhead.
enum : size_t {
  kDynamicBatchSize = std::numeric_limits<size_t>::max(),
};

// Options for ParallelFor.  The defaults are sufficient for most users.
struct ParallelForOptions {
  // It may be desirable to limit parallelism in some cases if e.g.:
  //   1.  A portion of the loop body requires synchronization and Amdahl's Law
  //       prevents scaling past a small number of threads.
  //   2.  You're running on a NUMA system and don't want this loop to execute
  //       across NUMA nodes.
  size_t max_parallelism = std::numeric_limits<size_t>::max();
};

// ParallelFor - execute a for-loop in parallel, using both the calling thread,
// plus the threads available in a ARThreadPool argument.
//
// Arguments:
//
//  * <seq> - the sequence to be processed, eg "Seq(vec.size())".
//
//  * <pool> - the threadpool to use (in addition to the main thread). If this
//  parameter is nullptr, then ParallelFor will compile down to a vanilla
//  single-threaded for-loop. It is permissible for the calling thread to be a
//  member of <pool>.
//
//  * <function> - the method to call for each value in <seq>.
//
//  * <kItersPerBatch> [template param] - the number of calls to <function> that
//  each thread will perform before synchronizing access to the loop counter. If
//  not provided, the array will be divided into (n_threads * 4) work batches,
//  enabling good parallelism in most cases, while minimizing synchronization
//  overhead.
//
//
// Example: Compute 1M sqrts in parallel.
//
//    vector<double> sqrts(1000000);
//    ParallelFor(Seq(sqrts.size()), &pool, [&](size_t i) {
//      sqrts[i] = sqrt(i);
//    });
//
//
// Example: Only the evens, using SeqWithStride.
//
//    ParallelFor(SeqWithStride<2>(0, sqrts.size()), &pool, [&sqrts](size_t i) {
//      sqrts[i] = i;
//    });
//
//
// Example: Execute N expensive tasks in parallel, in batches of 1-at-a-time.
//
//    ParallelFor<1>(Seq(tasks.size()), &pool, [&](size_t j) {
//      DoSomethingExpensive(tasks[j]);
//    });
//
template <size_t kItersPerBatch = kDynamicBatchSize, typename SeqT,
          typename Function>
inline void ParallelFor(SeqT seq, ARThreadPool* pool, Function func,
                        ParallelForOptions opts = ParallelForOptions());

// ParallelForWithStatus - Similar to ParallelFor, except it can short circuit
// if an error occurred.
//
// Arguments:
//  * <seq> - the sequence to be processed, eg "Seq(vec.size())".
//
//  * <pool> - the threadpool to use (in addition to the main thread). If this
//  parameter is nullptr, then ParallelFor will compile down to a vanilla
//  single-threaded for-loop. It is permissible for the calling thread to be a
//  member of <pool>.
//
//  * <function> - the method to call for each value in <seq> with type
//  interface of std::function<absl::Status(size_t idx)>.
//
//  * <kItersPerBatch> [template param] - the number of calls to <function> that
//  each thread will perform before synchronizing access to the loop counter. If
//  not provided, the array will be divided into (n_threads * 4) work batches,
//  enabling good parallelism in most cases, while minimizing synchronization
//  overhead.
//
//  Example:
//
//    auto status = ParallelForWithStatus<1>(
//      Seq(tasks.size()), &pool, [&](size_t idx) -> absl::Status {
//      RETURN_IF_ERROR(RunTask(tasks[idx]));
//      return absl::OkStatus();
//    });
//
template <size_t kItersPerBatch = kDynamicBatchSize, typename SeqT,
          typename Function>
inline absl::Status ParallelForWithStatus(
    SeqT seq, ARThreadPool* pool, Function Func,
    ParallelForOptions opts = ParallelForOptions()) {
  absl::Status finite_check_status = absl::OkStatus();

  std::atomic_bool is_ok_status{true};
  absl::Mutex mutex;
  ParallelFor(
      seq, pool,
      [&](size_t idx) {
        if (!is_ok_status.load(std::memory_order_relaxed)) {
          return;
        }
        absl::Status status = Func(idx);
        if (!status.ok()) {
          absl::MutexLock lock(&mutex);
          finite_check_status = status;
          is_ok_status.store(false, std::memory_order_relaxed);
        }
      },
      opts);
  return finite_check_status;
}

////////////////////////////////////////////////////////////////////////////////
//  IMPLEMENTATION DETAILS
////////////////////////////////////////////////////////////////////////////////

namespace parallel_for_internal {

// ParallelForClosure - a single heap-allocated object that holds the loop's
// state. The object will delete itself when the final task completes.
template <size_t kItersPerBatch, typename SeqT, typename Function>
  class ParallelForClosure : public std::function<void()> {
 public:
  static constexpr bool kIsDynamicBatch = (kItersPerBatch == kDynamicBatchSize);
  ParallelForClosure(SeqT seq, Function func)
      : func_(func),
        index_(*seq.begin()),
        range_end_(*seq.end()),
        reference_count_(1) {}

  inline void RunParallel(ARThreadPool* pool, size_t desired_threads) {
    // Don't push more tasks to the pool than we have work for.  Also, if
    // parallelism is limited by desired_threads not thread pool size, subtract
    // 1 from the number of threads to push to account for the main thread.
    size_t n_threads = std::min<size_t>(desired_threads - 1,
                                        pool->NumThreads());

    // Handle dynamic batch size.
    if (kIsDynamicBatch) {
      batch_size_ =
          SeqT::Stride() * std::max(1ul, desired_threads / 4 / n_threads);
    }

    reference_count_ += n_threads;
    while (n_threads--) {
      pool->Schedule([this]() { Run(); });
    }

    // Do work on the main thread. Once this returns, we are guaranteed that all
    // batches have been assigned to some thread.
    DoWork();

    // Then wait for all worker threads to exit the core loop. Thus, once the
    // main thread is able to take a WriterLock, we are guaranteed that all
    // batches have finished, allowing the main thread to move on.
    //
    // The main thread does *NOT* wait for ARThreadPool tasks that haven't yet
    // entered the core loop. This is important for handling scenarios where
    // the ARThreadPool falls significantly behind and hasn't started some of
    // the tasks assigned to it. Once assigned, those tasks will quickly realize
    // that there is no work left, and the final task to schedule will delete
    // this heap-allocated object.
    //
    termination_mutex_.WriterLock();
    termination_mutex_.WriterUnlock();

    // Drop main thread's reference.
    if (--reference_count_ == 0) delete this;
  }

    void Run() {
    // Do work on a child thread. Before starting any work, each child thread
    // takes a reader lock, preventing the main thread from finishing while
    // any child threads are still executing in the core loop.
    termination_mutex_.ReaderLock();
    DoWork();
    termination_mutex_.ReaderUnlock();

    // Drop child thread's reference.
    if (--reference_count_ == 0) delete this;
  }

  // DoWork - the "core loop", executed in parallel on N threads.
  inline void DoWork() {
    // Performance Note: Copying constant values to the stack allows the
    // compiler to know that they are actually constant and can be assigned to
    // registers (the 'const' keyword is insufficient).
    const size_t range_end = range_end_;

    // Performance Note: when batch size is not dynamic, the compiler will treat
    // it as a constant that can be directly inlined into the code w/o consuming
    // a register.
    constexpr size_t kStaticBatchSize = SeqT::Stride() * kItersPerBatch;
    const size_t batch_size = kIsDynamicBatch ? batch_size_ : kStaticBatchSize;

    // The core loop:
    for (;;) {
      // The std::atomic index_ coordinates sequential batch assignment.
      const size_t batch_begin = index_.fetch_add(batch_size);
      // Once assigned, batches execute w/o further coordination.
      const size_t batch_end = std::min(batch_begin + batch_size, range_end);
      if (ABSL_PREDICT_FALSE(batch_begin >= range_end)) break;
      for (size_t idx : SeqWithStride<SeqT::Stride()>(batch_begin, batch_end)) {
        func_(idx);
      }
    }
  }

 private:
  Function func_;

  // The index_ is used by worker threads to coordinate sequential batch
  // assignment. This is the only coordination mechanism inside the core loop.
  std::atomic<size_t> index_;

  // The iteration stops at range_end_.
  const size_t range_end_;

  // The termination_mutex_ coordinates termination of the for-loop, allowing
  // the main thread to batch until all child threads have exited the core-loop.
  absl::Mutex termination_mutex_;

  // For smaller arrays, it's possible for the work to finish before some of
  // the ARThreadPool tasks have started running, and in some extreme cases, it
  // might take entire milliseconds before these tasks begin running. The main
  // thread will continue doing other work, and the last task to schedule and
  // terminate will delete this heap allocated object.
  std::atomic<uint32_t> reference_count_;

  // The batch_size_ member variable is only used when the batch size is
  // dynamic (io, kItersPerBatch == kDynamicBatchSize).
  size_t batch_size_ = kItersPerBatch;
};

}  // namespace parallel_for_internal

template <size_t kItersPerBatch, typename SeqT, typename Function>
inline void ParallelFor(SeqT seq, ARThreadPool* pool, Function func,
                        ParallelForOptions opts) {
  // Figure out how many batches of work we have.
  constexpr size_t kMinItersPerBatch =
      kItersPerBatch == kDynamicBatchSize ? 1 : kItersPerBatch;
  const size_t desired_threads = std::min(
      opts.max_parallelism, DivRoundUp(*seq.end() - *seq.begin(),
                                       SeqT::Stride() * kMinItersPerBatch));

    if (!pool || desired_threads <= 1) {
    for (size_t idx : seq) {
      func(idx);
    }
    return;
  }

  // Otherwise, fire up the threadpool. Note that the shared closure object is
  // heap-allocated, allowing this method to finish even if some tasks haven't
  // started running yet. The object will be deleted by the last finishing task,
  // or possibly by this thread, whichever is last to terminate.
  using parallel_for_internal::ParallelForClosure;
  auto closure =
      new ParallelForClosure<kItersPerBatch, SeqT, Function>(seq, func);
  closure->RunParallel(pool, desired_threads);
}

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_PARALLEL_FOR_H_
