/* Copyright 2025 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

==============================================================================
lrc.h
==============================================================================

LinearRegressionCoding (LRC) is a lossless integer compression format that
supports random access to compressed data. Data lookup is performed by combining
the linear regression prediction result with the compressed differences.

SYNOPSIS
--------

  const uint64_t* data = ...;
  LRCEncoder lrc_encoder;
  auto lrc_data = lrc_encoder.Encode(data, data_len);
  // The LRCDecoder does not own the 'lrc_data'.
  auto lrc_decoder(lrc_data);
  // Decodes the value at index i.
  uint64_t decode_value = lrc_decoder[i];


Only uint32_t and uint64_t integer types are supported for compression. Signed
integer arithmetic is not supported by design due to its potential for undefined
behavior, which can lead to inconsistent results across different compilers or
compiler flag settings.
*/

#ifndef ARRAY_RECORD_CPP_LRC_H_
#define ARRAY_RECORD_CPP_LRC_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace array_record {

template <typename T>
union alignas(64) LRCCacheLine;
template <typename T>
struct LRCData;

/**
 * LRCEncoder
 *
 * Encoder class for LRC. Currently it only contain the Encode method but we'll
 * add more options for compression in the near future.
 *
 * The encoded value does not handle endian conversion. Therefore user should
 * not use it in serialization.
 *
 *  TODO(fchern): Add multithread compression.
 *  TODO(fchern): Support not only the least squares regression, but also the
 *  minimax using the optimization library.
 *  TODO(fchern): Support endian conversion so that we can serialize it and
 *  use it on different platforms.
 */
class LRCEncoder {
 public:
  // Encodes unsigned integers into LRCData. Encoded data should use less bytes
  // than the original for large n. However for small n the encoded data may
  // take more memory space because paddings are added in the LRCData format.
  template <typename T>
  LRCData<T> Encode(const T* data, size_t n) const;
};

/**
 * LRCEncoder
 *
 * Decoder class for LRC. The decoder does not own the underlying encoded data.
 */
template <typename T>
class LRCDecoder {
 public:
  explicit LRCDecoder(const LRCData<T>& lrc_data)
      : num_elements_(lrc_data.num_elements),
        num_cls_(lrc_data.num_cls),
        data_(lrc_data.data.get()) {}

  // Decode value at index i. Throws error if the index is out of bound.
  T operator[](size_t i) const;

  // The number of elements stored in LRC.
  size_t size() const { return num_elements_; }

  // Average number of bits used per integer.
  float bit_rate() const {
    const float total_bits = num_cls_ * 64 * 8;
    return total_bits / num_elements_;
  }

 private:
  static_assert(std::is_unsigned_v<T>);
  const size_t num_elements_;
  const size_t num_cls_;
  const LRCCacheLine<T>* __restrict data_;
};

////////////////////////////////////////////////////////////////////////////////
//                           Implementation Details
////////////////////////////////////////////////////////////////////////////////

// CTAD
template <typename T>
LRCDecoder(const LRCData<T>&) -> LRCDecoder<T>;

template <typename T>
struct LRCData {
  size_t num_elements;
  size_t num_cls;
  std::unique_ptr<LRCCacheLine<T>[]> data;
};

template <typename T>
union alignas(64) LRCCacheLine {
  static_assert(std::is_unsigned_v<T>);

  // Fixed size data storage
  T data[64 / sizeof(T)];

  // Table that encodes the linear regression parameters and indices to
  // the underlying data storage.
  struct Table {
    // 16 x 4 bits. Packed as interleaved [(0, 8), (1, 9), (2, 10), ...].
    uint8_t block_ncls[8];
    // Linear regression parameters.
    // Two consecutive blocks share one LR parameter.
    uint32_t intercept[8];
    uint16_t slope[8];
    // 8 4bits data packed in interleaved format:
    // lsb [(0, 4), (1, 5), ...] msb
    // The 4bits data encodes (lsb to msb) 2 bits of intercept_scale and
    // 2 bits of slope_scale.
    uint8_t param_scales[4];
    // Memory offset to the first block of the table, measured in the number of
    // LRCCacheLines. Suppose we decoded a block_ncl from block_ncls, the memory
    // address to the block would be: (void*)base_addr + (block_cl_offset +
    // block_ncl) * sizeof(LRCCacheLine).
    uint32_t block_cl_offset;
  } table;
};

static_assert(sizeof(LRCCacheLine<uint32_t>) == 64);
static_assert(sizeof(LRCCacheLine<uint64_t>) == 64);
static_assert(sizeof(LRCCacheLine<uint32_t>[2]) == 128);
static_assert(sizeof(LRCCacheLine<uint64_t>[2]) == 128);
static_assert(alignof(LRCCacheLine<uint32_t>) == 64);
static_assert(alignof(LRCCacheLine<uint64_t>) == 64);
static_assert(alignof(LRCCacheLine<uint32_t>[2]) == 64);
static_assert(alignof(LRCCacheLine<uint64_t>[2]) == 64);

extern template struct LRCData<uint32_t>;
extern template struct LRCData<uint64_t>;
extern template class LRCDecoder<uint32_t>;
extern template class LRCDecoder<uint64_t>;

extern template LRCData<uint32_t> LRCEncoder::Encode(const uint32_t* data,
                                                     size_t n) const;
extern template LRCData<uint64_t> LRCEncoder::Encode(const uint64_t* data,
                                                     size_t n) const;

}  // namespace array_record

#endif  // _ARRAY_RECORD_CPP_LRC_H_
