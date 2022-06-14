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

#ifndef ARRAY_RECORD_CPP_COMMON_H_
#define ARRAY_RECORD_CPP_COMMON_H_

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace array_record {

////////////////////////////////////////////////////////////////////////////////
//                   Canonical Errors (with formatting!)
////////////////////////////////////////////////////////////////////////////////

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status FailedPreconditionError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::FailedPreconditionError(absl::StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status InternalError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::InternalError(absl::StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status InvalidArgumentError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::InvalidArgumentError(absl::StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status NotFoundError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::NotFoundError(absl::StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status OutOfRangeError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::OutOfRangeError(absl::StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status UnavailableError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::UnavailableError(absl::StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status UnimplementedError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::UnimplementedError(absl::StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT absl::Status UnknownError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return absl::UnknownError(absl::StrFormat(fmt, args...));
}

// TODO(fchern): Align with what XLA do.
template <typename Int, typename DenomInt>
constexpr Int DivRoundUp(Int num, DenomInt denom) {
  // Note: we want DivRoundUp(my_uint64, 17) to just work, so we cast the denom
  // to the numerator's type. The result of division always fits in the
  // numerator's type, so this is very safe.
  return (num + static_cast<Int>(denom) - static_cast<Int>(1)) /
         static_cast<Int>(denom);
}

////////////////////////////////////////////////////////////////////////////////
//                           Class Decorators
////////////////////////////////////////////////////////////////////////////////

#define DECLARE_COPYABLE_CLASS(ClassName)      \
  ClassName(ClassName&&) = default;            \
  ClassName& operator=(ClassName&&) = default; \
  ClassName(const ClassName&) = default;       \
  ClassName& operator=(const ClassName&) = default

#define DECLARE_MOVE_ONLY_CLASS(ClassName)     \
  ClassName(ClassName&&) = default;            \
  ClassName& operator=(ClassName&&) = default; \
  ClassName(const ClassName&) = delete;        \
  ClassName& operator=(const ClassName&) = delete

#define DECLARE_IMMOBILE_CLASS(ClassName)     \
  ClassName(ClassName&&) = delete;            \
  ClassName& operator=(ClassName&&) = delete; \
  ClassName(const ClassName&) = delete;       \
  ClassName& operator=(const ClassName&) = delete

////////////////////////////////////////////////////////////////////////////////
//                      Seq / SeqWithStride / IndicesOf
////////////////////////////////////////////////////////////////////////////////
//
// Seq facilitates iterating over [begin, end) index ranges.
//
//  * Avoids 3X stutter of the 'idx' variable, facilitating use of more
//    descriptive variable names like 'datapoint_idx', 'centroid_idx', etc.
//
//  * Unifies the syntax between ParallelFor and vanilla for-loops.
//
//  * Reverse iteration is much easier to read and less error prone.
//
//  * Strided iteration becomes harder to miss when skimming code.
//
//  * Reduction in boilerplate '=', '<', '+=' symbols makes it easier to
//    skim-read code with lots of small for-loops interleaed with operator heavy
//    logic (ie, most of ScaM).
//
//  * Zero runtime overhead.
//
//
// Equivalent for-loops (basic iteration):
//
//    for (size_t idx : Seq(collection.size()) { ... }
//    for (size_t idx : Seq(0, collection.size()) { ... }
//    for (size_t idx = 0; idx < collection.size(); idx++) { ... }
//
//
// In particular, reverse iteration becomes much simpler and more readable:
//
//    for (size_t idx : ReverseSeq(collection.size())) { ... }
//    for (ssize_t idx = collection.size() - 1; idx >= 0; idx--) { ... }
//
//
// Strided iteration works too:
//
//    for (size_t idx : SeqWithStride<8>(filenames.size())) { ... }
//    for (size_t idx = 0; idx < filenames.size(); idx += 8) { ... }
//
//
// Iteration count without using a variable:
//
//    for (auto _ : Seq(16)) { ... }
//
//
// Clarifies the ParallelFor syntax:
//
//    ParallelFor<1>(Seq(dataset.size()), &pool, [&](size_t datapoint_idx) {
//      ...
//    });
//
template <ssize_t kStride = 1>
class SeqWithStride {
 public:
  static constexpr size_t Stride() { return kStride; }

  // Constructor for iterating [0, end).
  inline explicit SeqWithStride(size_t end) : begin_(0), end_(end) {}

  // Constructor for iterating [begin, end).
  inline SeqWithStride(size_t begin, size_t end) : begin_(begin), end_(end) {
    static_assert(kStride != 0);
  }

  // SizeT is an internal detail that helps suppress 'unused variable' compiler
  // errors. It's implicitly convertible to size_t, but by virtue of having a
  // destructor, the compiler doesn't complain about unused SizeT variables.
  //
  // These are equivalent:
  //
  //     for (auto _ : Seq(10))   // Suppresses 'unused variable' error.
  //     for (SizeT _ : Seq(10))  // Suppresses 'unused variable' error.
  //
  // Prefer the 'auto' variant. Don't use SizeT directly.
  //
  class SizeT {
   public:
    // Implicit SizeT <=> SizeT conversions.
    inline SizeT(size_t val) : val_(val) {}          // NOLINT
    inline operator size_t() const { return val_; }  // NOLINT

    // Defining a destructor suppresses 'unused variable' errors for the
    // following pattern: for (auto _ : Seq(kNumIters)) { ... }
    inline ~SizeT() {}

   private:
    size_t val_;
  };

  // Iterator implements the "!=", "++" and "*" operators required to support
  // the C++ for-each syntax. Not intended for direct use.
  class Iterator {
   public:
    // Constructor.
    inline explicit Iterator(size_t idx) : idx_(idx) {}
    // The '*' operator.
    inline SizeT operator*() const { return idx_; }
    // The '++' operator.
    inline Iterator& operator++() {
      idx_ += kStride;
      return *this;
    }
    // The '!=' operator.
    inline bool operator!=(Iterator end) const {
      // Note: The comparison below is "<", not "!=", in order to generate the
      // correct behavior when (end - begin) is not a multiple of kStride; note
      // that the Iterator class only exists to support the C++ for-each syntax,
      // and is *not* intended for direct use.
      //
      // Consider the case where (end - begin) is not a multple of kStride:
      //
      //     for (size_t j : SeqWithStride<5>(9)) { ... }
      //     for (size_t j = 0; j < 9; j += 5) { ... }    // '==' wouldn't work.
      //
      if constexpr (kStride > 0) {
        return idx_ < end.idx_;
      }

      // The reverse-iteration case:
      //
      //     for (size_t j : ReverseSeq(sz)) { ... }
      //     for (ssize_t j = sz-1; j >= 0; j -= 5) { ... }
      //
      return static_cast<ssize_t>(idx_) >= static_cast<ssize_t>(end.idx_);
    }

   private:
    size_t idx_;
  };
  using iterator = Iterator;

  inline Iterator begin() const { return Iterator(begin_); }
  inline Iterator end() const { return Iterator(end_); }

 private:
  size_t begin_;
  size_t end_;
};

// Seq iterates [0, end)
inline auto Seq(size_t end) { return SeqWithStride<1>(0, end); }

// Seq iterates [begin, end).
inline auto Seq(size_t begin, size_t end) {
  return SeqWithStride<1>(begin, end);
}

// IndicesOf provides the following equivalence class:
//
//    for (size_t j : IndicesOf(container)) { ... }
//    for (size_t j : Seq(container.size()) { ... }
//
template <typename Container>
SeqWithStride<1> IndicesOf(const Container& container) {
  return Seq(container.size());
}

////////////////////////////////////////////////////////////////////////////////
//                             Enumerate
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename IdxType = size_t,
          typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto Enumerate(T&& iterable) {
  class IteratorWithIndex {
   public:
    IteratorWithIndex(IdxType idx, TIter it) : idx_(idx), it_(it) {}
    bool operator!=(const IteratorWithIndex& other) const {
      return it_ != other.it_;
    }
    void operator++() { idx_++, it_++; }
    auto operator*() const { return std::tie(idx_, *it_); }

   private:
    IdxType idx_;
    TIter it_;
  };
  struct iterator_wrapper {
    T iterable;
    auto begin() { return IteratorWithIndex{0, std::begin(iterable)}; }
    auto end() { return IteratorWithIndex{0, std::end(iterable)}; }
  };
  return iterator_wrapper{std::forward<T>(iterable)};
}

////////////////////////////////////////////////////////////////////////////////
//                           Profiling Helpers
////////////////////////////////////////////////////////////////////////////////

#define AR_ENDO_MARKER(...)
#define AR_ENDO_MARKER_TIMEOUT(...)
#define AR_ENDO_TASK(...)
#define AR_ENDO_JOB(...)
#define AR_ENDO_TASK_TIMEOUT(...)
#define AR_ENDO_JOB_TIMEOUT(...)
#define AR_ENDO_SCOPE(...)
#define AR_ENDO_SCOPE_TIMEOUT(...)
#define AR_ENDO_EVENT(...)
#define AR_ENDO_ERROR(...)
#define AR_ENDO_UNITS(...)
#define AR_ENDO_THREAD_NAME(...)
#define AR_ENDO_GROUP(...)

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_COMMON_H_
