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
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "cpp/common.h"

namespace array_record {

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
 * Example usage:
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
 */
template <typename BaseT>
class TriStatePtr {
 public:
  DECLARE_IMMOBILE_CLASS(TriStatePtr);
  TriStatePtr() = default;

  ~TriStatePtr() {
    absl::MutexLock l(&mu_);
    mu_.Await(absl::Condition(
        +[](State* sharing_state) { return *sharing_state == State::kNoRef; },
        &state_));
  }

  // explicit TriStatePtr(std::unique_ptr<BaseT> ptr) : ptr_(std::move(ptr)) {}
  explicit TriStatePtr(std::unique_ptr<BaseT> ptr) : ptr_(std::move(ptr)) {}

  class SharedRef {
   public:
    SharedRef(TriStatePtr<BaseT>* parent) : parent_(parent) {}

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

    const BaseT& operator*() const { return *parent_->ptr_.get(); }
    const BaseT* operator->() const { return parent_->ptr_.get(); }
    BaseT& operator*() { return *parent_->ptr_.get(); }
    BaseT* operator->() { return parent_->ptr_.get(); }

   private:
    TriStatePtr<BaseT>* parent_ = nullptr;
  };

  class UniqueRef {
   public:
    DECLARE_MOVE_ONLY_CLASS(UniqueRef);
    UniqueRef(TriStatePtr<BaseT>* parent) : parent_(parent) {}

    ~UniqueRef() {
      absl::MutexLock l(&parent_->mu_);
      parent_->state_ = State::kNoRef;
    }

    const BaseT& operator*() const { return *parent_->ptr_.get(); }
    const BaseT* operator->() const { return parent_->ptr_.get(); }
    BaseT& operator*() { return *parent_->ptr_.get(); }
    BaseT* operator->() { return parent_->ptr_.get(); }

   private:
    TriStatePtr<BaseT>* parent_;
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

 private:
  mutable absl::Mutex mu_;
  std::atomic_int32_t ref_count_ = 0;
  State state_ ABSL_GUARDED_BY(mu_) = State::kNoRef;
  std::unique_ptr<BaseT> ptr_;
};

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_TRI_STATE_PTR_H_
