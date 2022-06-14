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

// `ThreadCompatibleSharedPtr` is a smart pointer for concurrent const access
// and blocks non-const methods until the owned object is unique.
//
// The main use case of `ThreadCompatibleSharedPtr` is to create const reference
// in detached thread, and allow the owner object to invoke non-const methods
// after all detached threads are finished.
//
// This is especially important for riegeli objects because we must call the
// non-const `Close()` on exit, and we cannot do that while other threads are
// accessing the object.
//
// Example usage:
//
//   {
//     auto owned = ThreadCompatibleSharedPtr<BaseClass>::Create(Foo(...));
//
//     // Detached thread with a refobj which increased the refcnt by 1.
//     pool->Schedule([refobj = owned]() {
//       refobj->BaseConstMethod(...);
//     });
//     owned->Close();  // Blocks until refobj goes out of scope.
//   }
//
// `owned` blocks on destruction when it is not unique. Therefore prevents
// refobj to be a dangling pointer.

#ifndef ARRAY_RECORD_CPP_THREAD_COMPATIBLE_SHARED_PTR_H_
#define ARRAY_RECORD_CPP_THREAD_COMPATIBLE_SHARED_PTR_H_

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/synchronization/mutex.h"

namespace array_record {

// `ThreadCompatibleSharedPtr` is a smart pointer for concurrent const access
// and blocks non-const methods until the owned object is unique.
template <typename BaseT>
class ThreadCompatibleSharedPtr {
 public:
  ThreadCompatibleSharedPtr() {}

  // Creates ThreadCompatibleSharedPtr by transferring the ownership of a
  // movable object.
  //
  // auto foo = ThreadCompatibleSharedPtr<FooBase>::Create(
  //   Foo(...));
  //
  // auto foo2 = ThreadCompatibleSharedPtr<FooBase>::Create(
  //   std::move(foo_obj));
  template <typename InstanceT>
  static ThreadCompatibleSharedPtr Create(InstanceT instance) {
    return ThreadCompatibleSharedPtr(MakeState<InstanceT>(std::move(instance)));
  }

  // Creates the object in-place from tuple of arguments.
  //
  // auto foo = ThreadCompatibleSharedPtr<FooBase>::Create<Foo>(
  //   std::forward_as_tuple(args...))
  template <typename InstanceT, typename... Args>
  static ThreadCompatibleSharedPtr Create(std::tuple<Args...> tuple_args) {
    return ThreadCompatibleSharedPtr(MakeState<InstanceT>(
        std::move(tuple_args), std::index_sequence_for<Args...>()));
  }

  // Creates ThreadCompatibleSharedPtr by transferring the ownership of a
  // std::unique_ptr object.
  //
  // auto foo = ThreadCompatibleSharedPtr<FooBase>::Create(
  //   std::make_unique<Foo>(args...))
  template <typename InstanceT>
  static ThreadCompatibleSharedPtr Create(std::unique_ptr<InstanceT> instance) {
    return ThreadCompatibleSharedPtr(MakeState<InstanceT>(std::move(instance)));
  }

  // is_owning == true  -- Blocks until refcnt == 0.
  // is_owning == false -- Does not block.
  ~ThreadCompatibleSharedPtr();

  // Copy: *this is a ref object. Increases refcnt.
  ThreadCompatibleSharedPtr(const ThreadCompatibleSharedPtr&);
  ThreadCompatibleSharedPtr& operator=(const ThreadCompatibleSharedPtr&);

  // Move: *this inherits the ownership of other. refcnt stays the same.
  ThreadCompatibleSharedPtr(ThreadCompatibleSharedPtr&&);
  ThreadCompatibleSharedPtr& operator=(ThreadCompatibleSharedPtr&&);

  // Thread compatible object can access const method without extra
  // synchronization.
  const BaseT* get() const { return state_->instance.get(); }
  const BaseT& operator*() const { return *get(); }
  const BaseT* operator->() const { return get(); }

  // is_owning == true  -- Blocks until refcnt == 0.
  // is_owning == false -- returns nullptr.
  BaseT* get();
  BaseT& operator*() { return *get(); }
  BaseT* operator->() { return get(); }

  // Tell if *this owned the underlying instance.
  bool is_owning() const { return is_owning_; }

  // Number of objects with is_owning = false.
  const int32_t refcnt() const { return state_->refcnt.load(); }

 private:
  struct State {
    mutable absl::Mutex mu;
    std::atomic<int32_t> refcnt = 0;
    std::unique_ptr<BaseT> instance = nullptr;
  };

  template <typename InstanceT, typename... Args, size_t... Indices>
  static std::shared_ptr<State> MakeState(std::tuple<Args...>&& tuple_args,
                                          std::index_sequence<Indices...>) {
    return MakeState<InstanceT, Args...>(
        std::forward<Args>(std::get<Indices>(tuple_args))...);
  }
  template <typename InstanceT, typename... Args>
  static std::shared_ptr<State> MakeState(Args... args) {
    auto state = std::make_shared<State>();
    state->instance = std::make_unique<InstanceT>(args...);
    return state;
  }

  template <typename InstanceT>
  static std::shared_ptr<State> MakeState(InstanceT instance) {
    auto state = std::make_shared<State>();
    state->instance.reset(new InstanceT(std::move(instance)));
    return state;
  }

  template <typename InstanceT>
  static std::shared_ptr<State> MakeState(std::unique_ptr<InstanceT> instance) {
    auto state = std::make_shared<State>();
    state->instance = std::move(instance);
    return state;
  }

  explicit ThreadCompatibleSharedPtr(std::shared_ptr<State> state)
      : state_(state), is_owning_(true) {}

  std::shared_ptr<State> state_;
  bool is_owning_ = false;
};

////////////////////////////////////////////////////////////////////////////////
//                       IMPLEMENTATION DETAILS
////////////////////////////////////////////////////////////////////////////////

template <typename BaseT>
ThreadCompatibleSharedPtr<BaseT>::~ThreadCompatibleSharedPtr() {
  if (state_) {
    if (is_owning_) {
      absl::MutexLock l(&state_->mu, absl::Condition(
                                         +[](std::atomic<int32_t>* refcnt) {
                                           return refcnt->load() == 0;
                                         },
                                         &state_->refcnt));
    } else {
      int32_t refcnt = state_->refcnt.fetch_sub(1) - 1;
      if (refcnt == 0) {
        // Unblocks the conditional lock.
        absl::MutexLock l(&state_->mu);
      }
    }
  }
}

template <typename BaseT>
BaseT* ThreadCompatibleSharedPtr<BaseT>::get() {
  if (state_ && is_owning_) {
    absl::MutexLock l(&state_->mu, absl::Condition(
                                       +[](std::atomic<int32_t>* refcnt) {
                                         return refcnt->load() == 0;
                                       },
                                       &state_->refcnt));
    return state_->instance.get();
  }
  return nullptr;
}

template <typename BaseT>
ThreadCompatibleSharedPtr<BaseT>::ThreadCompatibleSharedPtr(
    const ThreadCompatibleSharedPtr<BaseT>& other)
    : state_(other.state_) {
  state_->refcnt.fetch_add(1);
}
template <typename BaseT>
ThreadCompatibleSharedPtr<BaseT>& ThreadCompatibleSharedPtr<BaseT>::operator=(
    const ThreadCompatibleSharedPtr<BaseT>& other) {
  state_ = other.state_;
  state_->refcnt.fetch_add(1);
  return *this;
}

template <typename BaseT>
ThreadCompatibleSharedPtr<BaseT>::ThreadCompatibleSharedPtr(
    ThreadCompatibleSharedPtr<BaseT>&& other) {
  state_ = other.state_;
  is_owning_ = other.is_owning_;
  other.state_ = nullptr;
  other.is_owning_ = false;
}
template <typename BaseT>
ThreadCompatibleSharedPtr<BaseT>& ThreadCompatibleSharedPtr<BaseT>::operator=(
    ThreadCompatibleSharedPtr<BaseT>&& other) {
  state_ = other.state_;
  is_owning_ = other.is_owning_;
  other.state_ = nullptr;
  other.is_owning_ = false;
  return *this;
}

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_THREAD_COMPATIBLE_SHARED_PTR_H_
