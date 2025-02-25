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
lrc.cc
==============================================================================

LRC uses linear regression to predict an value at given index, and combine the
result with the compressed differences to restore the original value. The linear
regression is trained piecewise on slices of the data, each slice has fixed
number of elements (paddings may apply).

The data layout of the compressed format is chosen to be align to the common
cache line size (64 bytes). High level view of the layout is as follows:

  LRCCacheLine.table
  LRCCacheLine.table
  ...
  LRCCacheLine.table
  LRCCacheLines.data
  LRCCacheLines.data
  ...
  LRCCacheLines.data

The table encodes the LR parameters, the offset to the data, and the size of
data used by different encoding schemes. While not directly presented in the
LRCCacheLine struct, we use some extra units to locate compressed values in
the data field. Their relationships are as follows:

* A table map to 16 blocks (regardless of u32 or u64).

* Two blocks share one LR parameters (intercept and slope).

* A block of u32 map to decompressed 256 values, and 128 values for u64.

* A block contain varying number of cache lines, range in [1, 16].
  We use `block_ncl` (Number of Cache Lines used by the Block) in short.

* The encoding schemes are inferred by `block_ncl` and the integer type.
  Each bit in `block_ncl` corresponds to a power-of-two compression scheme:

  block_ncl value:       1, 2,  4,  8
  Num bits used for u32: 2, 4,  8, 16
  Num bits used for u64: 4, 8, 16, 32

  For `block_ncl` = 5, encoding u64 value, it uses the 4 bits and 16 bits
  encoding schemes to compress 128 values in 5 cache lines (CL).

  The exception is `block_ncl` = 16, where u32 uses 32 bits and u64 uses 64
  bits and the data is not compressed at all.

  When storing multiple `block_ncl` in LRCCacheLine.table.block_ncls, the
  block_ncl is subtracted by 1, so the values ranging in [0, 15] fits in
  4 bits and can be efficiently packed.

* We denote the data used in each compression scheme (e.g. u32-2bits,
  u64-16bits) as a "chunk". Chunks within a block is stored in the ascending
  order of the compression bits. For example, u32 with `block_ncl` = 5,
  the memory order is: chunk-2bits (1 CL), chunk-8bits, (4 CLs).

* The mapping of bits in the original value to the chunks uses the ascending
  order of the bit position. For example, u32 with `block_ncl` = 5, the least
  significant 2 bits of the original value are stored in chunk-2bits, and
  the bits in range [2, 9] (i.e. v & (0xff << 2)) are stored in chunk-8bits.

* Indexing compressed bits within a chunk go as follows:
  1. Locate the u32 or u64 by modulo the index by the chunk size (denoted as
    `chunk_nt`).
  2. The bits are offset from the lsb by (index / `chunk_nt`) * `nbits`, where
    `nbits` denotes the number of bits of the compression scheme.

Other naming conventions used in the codebase

 ncl := number of cache lines
 nt := nummber of T (u32 or u64)
 block_ncl := number of cls in a block
 chunk_ncl := number of cls in chunk
 chunk_nt := number of T (u32 or u64) in chunk

 i_of_unit := index to unit measured in unit.
              Typically computed as i / unit;
 i_in_unit := index to sub-unit within the unit.
              Typically computed as i % unit;
*/

#include "cpp/lrc.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "third_party/highway/hwy/highway.h"

namespace array_record {

namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Compute the prefix sum of block_ncls using the Hillis-Steele algorithm.
inline void BlockNCLPrefixSum(const uint8_t* __restrict block_ncls,
                              uint8_t* __restrict prefix_sum) {
  const hn::FixedTag<uint8_t, 8> d8_8;
  const hn::FixedTag<uint8_t, 16> d8_16;

  auto base = hn::LoadU(d8_8, block_ncls);
  auto low = hn::ShiftRight<4>(hn::ShiftLeft<4>(base));
  auto high = hn::ShiftRight<4>(base);
  // Combine(D, V2, V2): returns vector whose UpperHalf is the first argument
  // and whose LowerHalf is the second argument; D is Twice<DFromV<V2>>.
  auto full = hn::Combine(d8_16, high, low) + hn::Set(d8_16, 1);

  // Hillis-Steele scan prefix-sum algorithm.
  // However, we offset by one so each entry becomes
  // 0, a0, a0+a1, a0+a1+a2, ...
  full = hn::SlideUpLanes(d8_16, full, 1);
  full = full + hn::SlideUpLanes(d8_16, full, 1);
  full = full + hn::SlideUpLanes(d8_16, full, 2);
  full = full + hn::SlideUpLanes(d8_16, full, 4);
  full = full + hn::SlideUpLanes(d8_16, full, 8);
  hn::StoreU(full, d8_16, prefix_sum);
}

}  // namespace

// Utility struct that decodes the block information from a given table.
template <typename T>
struct BlockParams {
  // Number of cachelines to the block from the base address.
  uint32_t block_cl_offset;
  // Number of cachelines in the block.
  uint32_t block_ncl;
  // Rescaled intercept and slope.
  T intercept;
  T slope;

  BlockParams(const typename LRCCacheLine<T>::Table& table,
              uint32_t blk_i_in_tbl) {
    // Derive block_cl_offset
    uint8_t prefix_sum[16];
    BlockNCLPrefixSum(table.block_ncls, prefix_sum);
    block_cl_offset = table.block_cl_offset + prefix_sum[blk_i_in_tbl];

    // Derive block_ncl
    uint8_t u8_block_ncl = table.block_ncls[blk_i_in_tbl % 8];
    u8_block_ncl = blk_i_in_tbl / 8 ? u8_block_ncl >> 4 : u8_block_ncl & 0xf;
    block_ncl = u8_block_ncl + 1;

    // Derive intercept and slope
    uint32_t param_idx = blk_i_in_tbl / 2;
    const T ones = -1;
    T u32_sgn_ext = 0;
    if constexpr (sizeof(T) > 4) {
      u32_sgn_ext = ones << 32;
    }
    const T u16_sgn_ext = ones << 16;
    // cmov sign extension because we want to avoid undefined behaviors of int.
    intercept = table.intercept[param_idx];
    slope = table.slope[param_idx];
    intercept |= (intercept >> 31) ? u32_sgn_ext : 0;
    slope |= (slope >> 15) ? u16_sgn_ext : 0;
    uint8_t scales = table.param_scales[param_idx % 4];
    scales = param_idx / 4 ? scales >> 4 : scales & 0xf;
    uint8_t intercept_scale = scales & 0x03;
    uint8_t slope_scale = scales >> 2;
    intercept <<= (intercept_scale * 8);
    slope <<= (slope_scale * 8);
  }
};

// Decodes a single value from a chunk.
template <typename T, uint32_t ncl, uint32_t maskb>
T DecodeChunkAt(const T* block_base, uint32_t i_in_blk) {
  if constexpr ((ncl & maskb) != maskb) {
    return 0;
  }
  // u32: 256, u64: 128
  constexpr uint32_t cl_nt = sizeof(LRCCacheLine<T>) / sizeof(T);
  constexpr uint32_t t_nbits = sizeof(T) * 8;
  constexpr uint32_t base_nbits = sizeof(T) / 2;  // u32: 2, u64: 4
  constexpr uint32_t lower_ncls = ncl & (maskb - 1);
  constexpr uint32_t lower_nbits = base_nbits * lower_ncls;
  // Number of bits of the target value encoded in the chunk.
  // maskb:  1,  2,  4,   8
  //   u64:  4,  8, 16,  32
  //   u32:  2,  4,  8,  16
  constexpr uint32_t nbits = base_nbits * maskb;
  // Number of elements with type T in a chunk
  // maskb:  1,  2,  4,   8
  //   u64:  8, 16, 32,  64
  //   u32: 16, 32, 64, 128
  constexpr uint32_t chunk_nt = cl_nt * maskb;

  const T* chunk_base = block_base + cl_nt * lower_ncls;
  uint32_t chunk_t_idx = i_in_blk % chunk_nt;
  uint32_t sub_t_idx = i_in_blk / chunk_nt;

  T v = chunk_base[chunk_t_idx];
  v >>= nbits * sub_t_idx;
  v <<= t_nbits - nbits;

  // Arithmetic right shift if maskb map to the highest bit of ncl.
  if constexpr ((ncl & ~(maskb - 1)) == maskb) {
    auto vs = std::bit_cast<std::make_signed_t<T>>(v);
    v = std::bit_cast<T>(vs >> (t_nbits - nbits - lower_nbits));
  } else {
    v >>= (t_nbits - nbits - lower_nbits);
  }
  return v;
}

template <typename T, uint32_t ncl>
T DecodeBlockAt(const T* block_base, uint32_t i_in_blk) {
  T acc = 0;
  acc |= DecodeChunkAt<T, ncl, 1>(block_base, i_in_blk);
  acc |= DecodeChunkAt<T, ncl, 2>(block_base, i_in_blk);
  acc |= DecodeChunkAt<T, ncl, 4>(block_base, i_in_blk);
  acc |= DecodeChunkAt<T, ncl, 8>(block_base, i_in_blk);
  return acc;
}

template <typename T>
T LRCDecoder<T>::operator[](size_t i) const {
  assert(i < num_elements_);
  constexpr size_t block_nt = sizeof(LRCCacheLine<T>) / sizeof(T) * 16;
  const size_t blk_i = i / block_nt;
  const size_t blk_i_of_tbl = blk_i / 16;
  const size_t blk_i_in_tbl = blk_i % 16;
  const uint32_t i_in_blk = i % block_nt;
  const uint32_t i_in_2blks = i % (block_nt * 2);

  const auto blp = BlockParams<T>(data_[blk_i_of_tbl].table, blk_i_in_tbl);
  const T* block_base = data_[blp.block_cl_offset].data;

  T out = blp.intercept + blp.slope * i_in_2blks;
  out += [&]() -> T {
    switch (blp.block_ncl) {
      case 1:
        return DecodeBlockAt<T, 1>(block_base, i_in_blk);
      case 2:
        return DecodeBlockAt<T, 2>(block_base, i_in_blk);
      case 3:
        return DecodeBlockAt<T, 3>(block_base, i_in_blk);
      case 4:
        return DecodeBlockAt<T, 4>(block_base, i_in_blk);
      case 5:
        return DecodeBlockAt<T, 5>(block_base, i_in_blk);
      case 6:
        return DecodeBlockAt<T, 6>(block_base, i_in_blk);
      case 7:
        return DecodeBlockAt<T, 7>(block_base, i_in_blk);
      case 8:
        return DecodeBlockAt<T, 8>(block_base, i_in_blk);
      case 9:
        return DecodeBlockAt<T, 9>(block_base, i_in_blk);
      case 10:
        return DecodeBlockAt<T, 10>(block_base, i_in_blk);
      case 11:
        return DecodeBlockAt<T, 11>(block_base, i_in_blk);
      case 12:
        return DecodeBlockAt<T, 12>(block_base, i_in_blk);
      case 13:
        return DecodeBlockAt<T, 13>(block_base, i_in_blk);
      case 14:
        return DecodeBlockAt<T, 14>(block_base, i_in_blk);
      case 15:
        return DecodeBlockAt<T, 15>(block_base, i_in_blk);
      case 16:
        return block_base[i_in_blk];
      default:
        return 0;
    }
  }();
  return out;
}

template <typename T>
T CeilOfRatio(T numerator, T denominator) {
  const T round_toward_zero = numerator / denominator;
  const T rem = numerator % denominator;
  return (rem == 0) ? round_toward_zero : round_toward_zero + 1;
}

/* Ordinary least squares (OLS)
 * H @ \beta = x, where
 *   x = [x_0, x_1, ..., x_{n-1}]^T,
 *   H = [[1, 1, ..., 1], [0, 1, 2, ..., n-1]]^T, and
 *   \beta = [intercept, slope]^T.
 * To minimize ||H @ \beta - x||^2, we obtain the close form solution using
 * ordinary least squares (OLS):
 * \beta = (H^T H)^{-1} H^T x.
 *
 * H^T H = ( n                     \sum_{i=0}^{n-1} i  )
 *         (\sum_{i=0}^{n-1} i     \sum_{i=0}^{n-1} i^2)
 *
 *       = ( n                 n(n-1)/2)
 *         (n(n-1)/2     n(n-1)(2n-1)/6)
 *
 * (H^T H)^{-1} =
 *   1/(n^2(n-1)(2n-1)/6 - n^2(n-1)^2/4) ( n(n-1)(2n-1)/6    -n(n-1)/2)
 *                                       (-n(n-1)/2                  n)
 *
 *  = 12/(n(n-1)(4n-2 -3(n-1))) ((n-1)(2n-1)/6      -(n-1)/2)
 *                              (-(n-1)/2                  1)
 *
 *  = 12/(n(n-1)(n+1)) ((n-1)(2n-1)/6      -(n-1)/2)
 *                     (-(n-1)/2                  1)
 *
 * H^T x = [\sum_{i=0}^{n-1} x_i,  \sum_{i=0}^{n-1} i*x_i]^T
 *
 * (H^T H)^{-1} H^T x =
 *   [ 2*(2n-1)/(n+1) * Sx - 6n/(n+1) * Sxx,
 *     -6/(n+1) * Sx + 12n/((n-1)(n+1)) Sxx ]^T,
 * where
 *   Sx  := \sum_{i=0}^{n-1} x_i/n
 *   Sxx := \sum_{i=0}^{n-1} i * x_i / n^2
 */
template <typename T>
std::pair<double, double> LeastSquares(const T* x, size_t n) {
  if (n == 1) {
    return {x[0], 0};
  }
  T min_x = std::numeric_limits<T>::max();
  for (size_t i = 0; i < n; ++i) {
    min_x = std::min(min_x, x[i]);
  }
  double sx = 0, sxx = 0, dn = static_cast<double>(n);
  double dn_sq = dn * dn;
  for (size_t i = 0; i < n; ++i) {
    sx += static_cast<double>(x[i] - min_x) / dn;
    sxx += static_cast<double>(i * (x[i] - min_x)) / dn_sq;
  }
  double intercept = 2 * (2 * dn - 1) / (dn + 1) * sx - 6 * dn / (dn + 1) * sxx;
  double slope = -6 / (dn + 1) * sx + 12 * dn / ((dn - 1) * (dn + 1)) * sxx;
  return {intercept + min_x, slope};
}

// Truncate the floating point LR parameters to low bit unsigned integers.
//
// The Output tuple is
// * Updated input value with the truncation
// * Output with the truncation
// * scaling factor range from 0-3. updated_input = output << (scale * 8)
//   or, 4 if we failed to truncate the value.
template <typename DeT, typename EnT>
std::tuple<DeT, EnT, uint8_t> TruncatePrecision(double val) {
  using DeS = std::make_signed_t<DeT>;
  using EnS = std::make_signed_t<EnT>;

  const double max_val = std::nexttoward(std::numeric_limits<DeS>::max(), 0);
  const double min_val = std::nexttoward(std::numeric_limits<DeS>::min(), 0);
  if (val > max_val || val < min_val) {
    return {0, 0, 4};
  }

  DeS sval = static_cast<DeS>(val);

  for (int s = 0; s < 4; ++s) {
    DeS trimming_bits = s * 8;
    DeS denominator = 1 << trimming_bits;
    DeS rounded_sval =
        sval >= 0 ? sval + denominator / 2 : sval - denominator / 2;
    rounded_sval >>= trimming_bits;
    if (rounded_sval > std::numeric_limits<EnS>::max() ||
        rounded_sval < std::numeric_limits<EnS>::min()) {
      continue;
    }
    EnT en_val = std::bit_cast<EnT>(static_cast<EnS>(rounded_sval));
    return {static_cast<DeT>(en_val) << trimming_bits, en_val, s};
  }
  return {0, 0, 4};
}

template <typename T>
LRCData<T> LRCEncoder::Encode(const T* data, size_t n) const {
  // TODO(fchern): It is possible for the u32 block_cl_offset to overflow.
  // In the worst case, the following number of elements would overflow:
  // u32: 2^32 * 256*16/(16+1)
  // u64: 2^32 * 128*16/(16+1)
  // Therefore we currently limit the max number of elements to 1 << (32 + 6).
  // A potential solution is to create multiple compression blobs.
  if (n >= 1ULL << 38) {
    LOG(ERROR) << "Number of elements exceeding 1ULL << 38 is not supported.";
    return {0, 0, nullptr};
  }
  if (n == 0) {
    return {0, 0, nullptr};
  }
  // u32: 256, u64: 128
  constexpr size_t block_nt = sizeof(LRCCacheLine<T>) / sizeof(T) * 16;
  constexpr uint32_t t_nbits = sizeof(T) * 8;
  constexpr uint32_t base_nbits = sizeof(T) / 2;  // u32: 2, u64: 4
  const size_t n_blocks = CeilOfRatio(n, block_nt);
  const size_t n_tables = CeilOfRatio<size_t>(n_blocks, 16);
  std::vector<uint8_t> block_ncls(n_blocks);
  std::vector<LRCCacheLine<T>> cachelines(n_tables);

  // Two blocks share one linear regression weight, so we advance the loop by
  // two blocks instead of one. The inner loop need to handle even and odd
  // blocks.
  for (size_t blk_i = 0; blk_i * block_nt < n; blk_i += 2) {
    size_t len = std::min(block_nt, n - blk_i * block_nt);
    // TODO(fchern): Also implement minimax
    const auto [intercept_dbl, slope_dbl] =
        LeastSquares<T>(&data[blk_i * block_nt], len);
    auto [intercept, intercept_u32, intercept_scale] =
        TruncatePrecision<T, uint32_t>(intercept_dbl);
    auto [slope, slope_u16, slope_scale] =
        TruncatePrecision<T, uint16_t>(slope_dbl);
    const bool compress = intercept_scale != 4 && slope_scale != 4;
    size_t blk_i_of_table = blk_i / 16;
    size_t blk_i_in_table_param = (blk_i % 16) / 2;
    if (compress) {
      // Even and odd block
      for (size_t blk_p1 = 0; blk_p1 < 2; ++blk_p1) {
        uint32_t blk_hbit = 0;
        for (size_t j = 0; (blk_i + blk_p1) * block_nt + j < n; ++j) {
          T base = intercept + (blk_p1 * block_nt + j) * slope;
          T blk_diff = data[(blk_i + blk_p1) * block_nt + j] - base;
          T blk_abs = blk_diff >> (t_nbits - 1) ? ~blk_diff : blk_diff;
          // We need one extra bit to distinguish signs.
          blk_hbit = std::max<uint32_t>(
              blk_hbit, t_nbits + 1 - std::countl_zero(blk_abs));
        }
        if (blk_i + blk_p1 < n_blocks) {
          // block_ncl range in [1, 16], indicating the number of cache lines
          // required for the block. When stored in block_ncls, we subtract the
          // value by 1 so that it fit in 4bits [0, 15].
          auto block_ncl =
              std::min<uint16_t>(16, CeilOfRatio(blk_hbit, base_nbits));
          block_ncl = std::max<uint16_t>(block_ncl, 1);
          block_ncls[blk_i + blk_p1] = block_ncl - 1;
        }
      }
    } else {
      intercept = intercept_u32 = intercept_scale = 0;
      slope = slope_u16 = slope_scale = 0;
      block_ncls[blk_i] = 15;
      if (blk_i + 1 < n_blocks) {
        block_ncls[blk_i + 1] = 15;
      }
    }
    auto& table = cachelines[blk_i_of_table].table;
    table.intercept[blk_i_in_table_param] = intercept_u32;
    table.slope[blk_i_in_table_param] = slope_u16;
    uint8_t param_scale = (slope_scale << 2 | intercept_scale);
    param_scale <<= (blk_i_in_table_param / 4) * 4;
    table.param_scales[blk_i_in_table_param % 4] |= param_scale;
  }

  // Compute the prefix sum of block_ncls and store it into block_cl_offset.
  cachelines[0].table.block_cl_offset = n_tables;
  for (size_t tbl_i = 0; tbl_i < n_tables - 1; ++tbl_i) {
    uint32_t block_ncl_sum = 0;
    size_t blk_i_len = std::min<size_t>(16, n_blocks - tbl_i * 16);
    for (size_t blk_i = 0; blk_i < blk_i_len; ++blk_i) {
      block_ncl_sum += block_ncls[tbl_i * 16 + blk_i] + 1;
    }
    cachelines[tbl_i + 1].table.block_cl_offset =
        cachelines[tbl_i].table.block_cl_offset + block_ncl_sum;
  }
  uint32_t total_cls = cachelines[n_tables - 1].table.block_cl_offset;
  for (size_t j = (n_tables - 1) * 16; j < n_blocks; ++j) {
    total_cls += block_ncls[j] + 1;
  }
  LRCData<T> out;
  out.num_elements = n;
  out.num_cls = total_cls;
  out.data.reset(new LRCCacheLine<T>[total_cls]());

  // Copy the tables and block_ncls to out.
  for (size_t tbl_i = 0; tbl_i < n_tables; ++tbl_i) {
    out.data[tbl_i].table.block_cl_offset =
        cachelines[tbl_i].table.block_cl_offset;
    for (size_t j = 0; j < 8; ++j) {
      out.data[tbl_i].table.intercept[j] = cachelines[tbl_i].table.intercept[j];
      out.data[tbl_i].table.slope[j] = cachelines[tbl_i].table.slope[j];
    }
    for (size_t j = 0; j < 4; ++j) {
      out.data[tbl_i].table.param_scales[j] =
          cachelines[tbl_i].table.param_scales[j];
    }
    for (size_t j = 0; j < 8; ++j) {
      size_t idx_0 = tbl_i * 16 + j;
      size_t idx_1 = tbl_i * 16 + 8 + j;
      uint8_t val_0 = idx_0 < n_blocks ? block_ncls[idx_0] : 0;
      uint8_t val_1 = idx_1 < n_blocks ? block_ncls[idx_1] : 0;
      out.data[tbl_i].table.block_ncls[j] = val_0 | (val_1 << 4);
    }
  }

  // Pack the differences between the original value and the linear regression
  // result into the output data blob.
  for (size_t blk_i = 0; blk_i < n_blocks; ++blk_i) {
    size_t blk_i_of_tbl = blk_i / 16;
    size_t blk_i_in_tbl = blk_i % 16;
    const auto blp = BlockParams<T>(out.data[blk_i_of_tbl].table, blk_i_in_tbl);
    T* block_base = out.data[blp.block_cl_offset].data;
    if (blp.block_ncl == 16) {
      const size_t j_len = std::min<size_t>(block_nt, n - blk_i * block_nt);
      for (size_t j_out = 0; j_out < j_len; ++j_out) {
        size_t j_in = blk_i * block_nt + j_out;
        block_base[j_out] = data[j_in];
      }
      continue;
    }
    for (uint32_t lmaskb = 0; lmaskb < 4; ++lmaskb) {
      constexpr uint32_t cl_nt = sizeof(LRCCacheLine<T>) / sizeof(T);
      const uint32_t maskb = 1 << lmaskb;
      const uint32_t lower_ncls = blp.block_ncl & (maskb - 1);
      const uint32_t chunk_nt = cl_nt * maskb;
      const uint32_t nbits = base_nbits * maskb;
      const uint32_t lower_nbits = base_nbits * lower_ncls;
      const size_t j_len = std::min<size_t>(block_nt, n - blk_i * block_nt);

      T* __restrict chunk_base = block_base + cl_nt * lower_ncls;
      if (blp.block_ncl & maskb) {
        for (size_t j = 0; j < j_len; ++j) {
          size_t j_in = blk_i * block_nt + j;
          const uint32_t chunk_t_idx = j % chunk_nt;
          const uint32_t sub_t_idx = j / chunk_nt;
          // Two consecutive blocks share the interpret and slope, therefore
          // we must add the extra offset (blk_i & 1) * block_nt to the linear
          // regression arguments.
          T base = blp.intercept + blp.slope * (j + (blk_i & 1) * block_nt);
          T v = data[j_in] - base;
          v >>= lower_nbits;
          v <<= t_nbits - nbits;
          v >>= t_nbits - (sub_t_idx + 1) * nbits;
          chunk_base[chunk_t_idx] |= v;
        }
      }
    }
  }

  return out;
}

template struct LRCData<uint32_t>;
template struct LRCData<uint64_t>;
template class LRCDecoder<uint32_t>;
template class LRCDecoder<uint64_t>;

template LRCData<uint32_t> LRCEncoder::Encode(const uint32_t* data,
                                              size_t n) const;
template LRCData<uint64_t> LRCEncoder::Encode(const uint64_t* data,
                                              size_t n) const;

}  // namespace array_record
