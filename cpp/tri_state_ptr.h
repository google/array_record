/* Copyright 2024 Google LLC. All Rights Reserved.

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

#ifndef ARRAY_RECORD_CPP_TRI_STATE_PTR_H_
#define ARRAY_RECORD_CPP_TRI_STATE_PTR_H_

#include <stddef.h>

#include <atomic>
#include <cstdint>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "cpp/common.h"
#include "riegeli/base/dependency.h"
#include "riegeli/base/initializer.h"

namespace array_record {

template <typename BaseT>
class TriStatePtrBase {
 public:
  // TriStatePtrBase(BaseT* ptr) : ptr_(ptr) {
  //   if (ptr == nullptr)
  //   LOG(FATAL) << "ptr is null";
  // }

  ~TriStatePtrBase() {
    absl::MutexLock l(&mu_);
    mu_.Await(absl::Condition(
        +[](State* sharing_state) { return *sharing_state == State::kNoRef; },
        &state_));
  }

  class SharedRef {
   public:
    SharedRef(TriStatePtrBase<BaseT>* parent) : parent_(parent) {}

    SharedRef(const SharedRef& other) : parent_(other.parent_) {
      parent_->ref_count_++;
    }
    SharedRef& operator=(const SharedRef& other) {
      this->parent_ = other.parent_;
      this->parent_->ref_count_++;
      return *this;
    }

    SharedRef(SharedRef&& other) : parent_(other.parent_) {
      other.parent_ = nullptr;
    }
    SharedRef& operator=(SharedRef&& other) {
      this->parent_ = other.parent_;
      other.parent_ = nullptr;
      return *this;
    }

    ~SharedRef() {
      if (parent_ == nullptr) {
        return;
      }
      int32_t ref_count =
          parent_->ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
      if (ref_count == 0) {
        absl::MutexLock l(&parent_->mu_);
        parent_->state_ = State::kNoRef;
      }
    }

    const BaseT& operator*() const { return *parent_->ptr_; }
    const BaseT* operator->() const { return parent_->ptr_; }
    BaseT& operator*() { return *parent_->ptr_; }
    BaseT* operator->() { return parent_->ptr_; }

   private:
    TriStatePtrBase<BaseT>* parent_ = nullptr;
  };

  class UniqueRef {
   public:
    DECLARE_MOVE_ONLY_CLASS(UniqueRef);
    UniqueRef(TriStatePtrBase<BaseT>* parent) : parent_(parent) {}

    ~UniqueRef() {
      absl::MutexLock l(&parent_->mu_);
      parent_->state_ = State::kNoRef;
    }

    const BaseT& operator*() const { return *parent_->ptr_; }
    const BaseT* operator->() const { return parent_->ptr_; }
    BaseT& operator*() { return *parent_->ptr_; }
    BaseT* operator->() { return parent_->ptr_; }

   private:
    TriStatePtrBase<BaseT>* parent_;
  };

  SharedRef MakeShared() {
    absl::MutexLock l(&mu_);
    mu_.Await(absl::Condition(
        +[](State* sharing_state) { return *sharing_state != State::kUnique; },
        &state_));
    state_ = State::kSharing;
    ref_count_++;
    return SharedRef(this);
  }

  UniqueRef WaitAndMakeUnique() {
    absl::MutexLock l(&mu_);
    mu_.Await(absl::Condition(
        +[](State* sharing_state) { return *sharing_state == State::kNoRef; },
        &state_));
    state_ = State::kUnique;
    return UniqueRef(this);
  }

  enum class State {
    kNoRef = 0,
    kSharing = 1,
    kUnique = 2,
  };

  State state() const {
    absl::MutexLock l(&mu_);
    return state_;
  }

 protected:
  BaseT* ptr_;

 private:
  mutable absl::Mutex mu_;
  std::atomic_int32_t ref_count_ = 0;
  State state_ ABSL_GUARDED_BY(mu_) = State::kNoRef;
};

/** TriStatePtr is a wrapper around a pointer that allows for a unique and
 * shared reference.
 *
 * There are three states:
 *
 * - NoRef: The object does not have shared or unique references.
 * - Sharing: The object is shared.
 * - Unique: The object is referenced by a unique pointer wrapper.
 *
 * The state transition from NoRef to Shared when MakeShared is called.
 * An internal refernce count is incremented when a SharedRef is created.
 *
 *      SharedRef ref = MakeShared();           --
 * NoRef ----------------------------> Sharing /  | MakeShared()
 *         All SharedRef deallocated           <--
 *       <----------------------------
 *
 * The state can also transition to Unique when WaitAndMakeUnique is called.
 * We can only hold one unique reference at a time.
 *
 *      UniqueRef ref = WaitAndMakeUnique();
 * NoRef ----------------------------> Unique
 *         The UniqueRef is deallocated
 *       <----------------------------
 *
 * Other than the state transition above, state transitions methods would block
 * until the specified state is possible. On deallocation, the destructor blocks
 * until the state is NoRef.
 *
 * Example usage 1 (using rieglie::Maker):
 *
 *   TriStatePtr<FooBase, Foo> main(riegeli::Maker(...));
 *   // Create a shared reference to work on other threads.
 *   pool->Schedule([refobj = foo_ptr.MakeShared()] {
 *     refobj->FooMethod();
 *   });
 *
 *   // Blocks until refobj is out of scope.
 *   auto unique_ref = main.WaitAndMakeUnique();
 *   unique_ref->CleanupFoo();
 * 
 * Example usage 2 (using unique_ptr):
 *
 *   TriStatePtr<FooBase, std::unique_ptr<FooBase>> main(
 *       std::make_unique<Foo>(...));
 */
template <typename BaseT, typename T>
class TriStatePtr : public TriStatePtrBase<BaseT> {
 public:
  explicit TriStatePtr(riegeli::Initializer<T> base)
      : dependency_(std::move(base)) {
    absl::Nonnull<BaseT*> ptr = dependency_.get();
    TriStatePtrBase<BaseT>::ptr_ = ptr;
  }

 private:
  riegeli::Dependency<BaseT*, T> dependency_;
};

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_TRI_STATE_PTR_H_
