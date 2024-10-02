#include "thread_pool.h"

ThreadPool::ThreadPool(int num_threads) : num_threads_(num_threads) {
  threads_.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads_.emplace_back(&ThreadPool::WorkLoop, this);
  }
}

ThreadPool::~ThreadPool() {
  {
    absl::MutexLock l(&mu_);
    for (size_t i = 0; i < threads_.size(); ++i) {
      queue_.push(nullptr);  // Shutdown signal.
    }
  }
  for (auto &t : threads_) {
    t.join();
  }
}

void ThreadPool::Schedule(absl::AnyInvocable<void()> func) {
  assert(func != nullptr);
  absl::MutexLock l(&mu_);
  queue_.push(std::move(func));
}

bool ThreadPool::WorkAvailable() const {
  return !queue_.empty();
}

void ThreadPool::WorkLoop() {
  while (true) {
    absl::AnyInvocable<void()> func;
    {
      absl::MutexLock l(&mu_);
      mu_.Await(absl::Condition(this, &ThreadPool::WorkAvailable));
      func = std::move(queue_.front());
      queue_.pop();
    }
    if (func == nullptr) {  // Shutdown signal.
      break;
    }
    func();
  }
}

uint64_t ThreadPool::NumThreads() {
  return this->num_threads_;
}


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

#include "cpp/thread_pool.h"
#include "absl/flags/flag.h"

ABSL_FLAG(uint32_t, array_record_global_pool_size, 64,
          "Number of threads for ArrayRecordGlobalPool");

namespace array_record {

ARThreadPool* ArrayRecordGlobalPool() {
  static ARThreadPool* pool_ = []() -> ARThreadPool* {
    ARThreadPool* pool = new
    ThreadPool(absl::GetFlag(FLAGS_array_record_global_pool_size));
    return pool;
  }();
  return pool_;
}

}  // namespace array_record
