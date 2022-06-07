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

using ARThreadPool = Eigen::ThreadPoolInterface;

ARThreadPool* ArrayRecordGlobalPool();

}  // namespace array_record

#endif  // ARRAY_RECORD_CPP_THREAD_POOL_H_
