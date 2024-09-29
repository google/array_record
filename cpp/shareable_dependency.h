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

// `ShareableDependency<Handle, Manager>` wraps a
// `riegeli::StableDependency<Handle, Manager>`. It allows creating concurrent
// shares of it of type `DependencyShare<Handle>`, and waiting until all shares
// are no longer in use. It waits explicitly or implicitly when the
// `ShareableDependency` is destroyed or reassigned.
//
// The main use case of `ShareableDependency` is to create a share for a
// detached thread, and allow the owner object to invoke non-const methods after
// all detached threads are finished.
//
// This is especially important for riegeli objects because we must call the
// non-const `Close()` on exit, and we cannot do that while other threads are
// accessing the object.
//
// Example usage:
//
//   ShareableDependency<FooBase*, Foo> main(riegeli::Maker(...));
//
//   // Detached thread with a refobj which increased the refcnt by 1.
//   pool->Schedule([refobj = main.Share()] {
//     refobj->FooMethod(...);
//   });
//
//   // Blocks until refobj goes out of scope.
//   auto& unique = main.WaitUntilUnique();
//   if (unique.IsOwning()) unique->Close();
//
// `main` blocks on destruction when it is not unique. Therefore prevents
// refobj to be a dangling pointer.

#ifndef ARRAY_RECORD_CPP_SHAREABLE_DEPENDENCY_H_
#define ARRAY_RECORD_CPP_SHAREABLE_DEPENDENCY_H_

#include <stddef.h>

#include <memory>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "riegeli/base/initializer.h"
#include "riegeli/base/intrusive_shared_ptr.h"
#include "riegeli/base/ref_count.h"
#include "riegeli/base/stable_dependency.h"
#include "riegeli/base/type_traits.h"

namespace array_record {

// `DependencyShare<Handle>` wraps a `Handle` and tracks the lifetime of all
// shares from the given `ShareableDependency<Handle, Manager>`.
template <typename Handle>
class DependencyShare {
 public:
  DependencyShare() = default;

  DependencyShare(const DependencyShare& other) = default;
  DependencyShare& operator=(const DependencyShare& other) = default;

  // The source is left empty.
  DependencyShare(DependencyShare&& other) = default;
  // The source is left empty.
  DependencyShare& operator=(DependencyShare&& other) = default;

  Handle get() const {
    DCHECK(sharing_ != nullptr);
    return sharing_->handle;
  }
  template <typename DependentHandle = Handle,
            std::enable_if_t<riegeli::HasDereference<DependentHandle>::value,
                             int> = 0>
  decltype(*std::declval<DependentHandle>()) operator*() const {
    return *get();
  }
  template <
      typename DependentHandle = Handle,
      std::enable_if_t<riegeli::HasArrow<DependentHandle>::value, int> = 0>
  Handle operator->() const {
    return get();
  }

 private:
  template <typename AnyHandle, typename Manager>
  friend class ShareableDependency;

  struct Sharing;

  explicit DependencyShare(Sharing* sharing);

  riegeli::IntrusiveSharedPtr<Sharing> sharing_;
};

// `ShareableDependency<Handle, Manager>` wraps a
// `riegeli::StableDependency<Handle, Manager>`. It allows creating concurrent
// shares of it of type `DependencyShare<Handle>`, and waiting until all shares
// are no longer in use. It waits explicitly or implicitly when the
// `ShareableDependency` is destroyed or reassigned.
template <typename Handle, typename Manager>
class ShareableDependency {
 public:
  // Creates an empty `ShareableDependency`.
  ShareableDependency() = default;

  // Creates a `ShareableDependency` storing the `Manager`.
  explicit ShareableDependency(riegeli::Initializer<Manager> manager)
      : dependency_(std::move(manager)),
        sharing_(new Sharing(dependency_.get())) {}

  // The source is left empty.
  ShareableDependency(ShareableDependency&& other) = default;
  // Waits until `*this` is empty or unique. The source is left empty.
  ShareableDependency& operator=(ShareableDependency&& other) = default;

  // Waits until `*this` is empty or unique.
  ~ShareableDependency() = default;

  // Makes `*this` equivalent to a newly constructed `ShareableDependency`. This
  // avoids constructing a temporary `ShareableDependency` and moving from it.
  ABSL_ATTRIBUTE_REINITIALIZES void Reset();
  ABSL_ATTRIBUTE_REINITIALIZES void Reset(
      riegeli::Initializer<Manager> manager);

  // Creates a `DependencyShare` sharing a pointer from `*this`.
  //
  // As long as the `DependencyShare` is alive, `*this` will wait in its
  // destructor, assignment, and `WaitUntilUnique()`.
  //
  // An empty `ShareableDependency` yields an empty `DependencyShare`.
  DependencyShare<Handle> Share() const;

  // Waits until `*this` is empty or unique. Returns a reference to a
  // `StableDependency` storing the `Manager`.
  riegeli::StableDependency<Handle, Manager>& WaitUntilUnique();

  // Returns `true` if there are no alive shares of `*this`.
  bool IsUnique() const;

 private:
  using Sharing = typename DependencyShare<Handle>::Sharing;

  struct Deleter;

  riegeli::StableDependency<Handle, Manager> dependency_;
  std::unique_ptr<Sharing, Deleter> sharing_;
};

////////////////////////////////////////////////////////////////////////////////
//                       IMPLEMENTATION DETAILS
////////////////////////////////////////////////////////////////////////////////

template <typename Handle>
struct DependencyShare<Handle>::Sharing {
  explicit Sharing(Handle handle) : handle(std::move(handle)) {}

  void Ref() const { 
    ref_count.Ref();
  }
  void Unref() const {
    // Notify the `ShareableDependency` if there are no more shares.
    absl::MutexLock l(&mu);
    if (ref_count.Unref()) {
      DLOG(FATAL)
          << "The last DependencyShare outlived the ShareableDependency";
    }
  }
  bool HasUniqueOwner() const { 
    return ref_count.HasUniqueOwner(); 
  }
  void WaitUntilUnique() const {    
    absl::MutexLock l(&mu, absl::Condition(this, &Sharing::HasUniqueOwner));
  }

  Handle handle;
  mutable absl::Mutex mu;
  riegeli::RefCount ref_count;
};

template <typename Handle>
DependencyShare<Handle>::DependencyShare(Sharing* sharing) : sharing_(sharing) {
  if (sharing_ != nullptr) sharing_->Ref();
}

template <typename Handle, typename Manager>
struct ShareableDependency<Handle, Manager>::Deleter {
  void operator()(Sharing* sharing) const {
    sharing->WaitUntilUnique();
    delete sharing;
  }
};

template <typename Handle, typename Manager>
void ShareableDependency<Handle, Manager>::Reset() {
  sharing_.reset();
  dependency_.Reset();
}

template <typename Handle, typename Manager>
void ShareableDependency<Handle, Manager>::Reset(
    riegeli::Initializer<Manager> manager) {
  WaitUntilUnique().Reset(std::move(manager));
  if (sharing_ == nullptr) {
    sharing_.reset(new Sharing(dependency_.get()));
  } else {
    sharing_->handle = dependency_.get();
  }
}

template <typename Handle, typename Manager>
DependencyShare<Handle> ShareableDependency<Handle, Manager>::Share() const {
  return DependencyShare<Handle>(sharing_.get());
}

template <typename Handle, typename Manager>
riegeli::StableDependency<Handle, Manager>&
ShareableDependency<Handle, Manager>::WaitUntilUnique() {
  if (sharing_ != nullptr) sharing_->WaitUntilUnique();
  return dependency_;
}

template <typename Handle, typename Manager>
bool ShareableDependency<Handle, Manager>::IsUnique() const {
  return sharing_ != nullptr && sharing_->HasUniqueOwner();
}

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_SHAREABLE_DEPENDENCY_H_
