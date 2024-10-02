#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <cassert>
#include <cstddef>
#include <functional>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"

// A simple ThreadPool implementation for tests.
class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ~ThreadPool();
  uint64_t NumThreads();

  // Schedule a function to be run on a ThreadPool thread immediately.
  void Schedule(absl::AnyInvocable<void()> func);

 private:
  bool WorkAvailable() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WorkLoop();
  uint64_t num_threads_;
  absl::Mutex mu_;
  std::queue<absl::AnyInvocable<void()>> queue_ ABSL_GUARDED_BY(mu_);
  std::vector<std::thread> threads_;
};

#endif  // THREAD_POOL_H_
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

#ifndef ARRAY_RECORD_CPP_THREAD_POOL_H_
#define ARRAY_RECORD_CPP_THREAD_POOL_H_

#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "unsupported/Eigen/CXX11/ThreadPool"

namespace array_record {

using ARThreadPool = ThreadPool;

ARThreadPool* ArrayRecordGlobalPool();

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_THREAD_POOL_H_
