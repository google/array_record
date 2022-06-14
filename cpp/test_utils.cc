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

#include "cpp/test_utils.h"

#include <cstring>
#include <random>

#include "cpp/common.h"

namespace array_record {

std::string MTRandomBytes(std::mt19937& bitgen, size_t length) {
  std::string result(length, '\0');

  size_t gen_bytes = sizeof(uint32_t);
  size_t rem = length % gen_bytes;
  std::mt19937::result_type val = bitgen();
  char* ptr = result.data();
  std::memcpy(ptr, &val, rem);
  ptr += rem;

  for (auto _ : Seq(length / gen_bytes)) {
    uint32_t val = bitgen();
    std::memcpy(ptr, &val, gen_bytes);
    ptr += gen_bytes;
  }
  return result;
}

}  // namespace array_record
