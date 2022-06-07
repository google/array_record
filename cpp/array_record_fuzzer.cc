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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/logging.h"
#include "base/sysinfo.h"
#include "fuzzer/FuzzedDataProvider.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/types/span.h"
#include "third_party/array_record/cpp/array_record_reader.h"
#include "third_party/array_record/cpp/array_record_writer.h"
#include "third_party/array_record/cpp/layout.proto.h"
#include "third_party/array_record/cpp/thread_pool.h"
#include "third_party/riegeli/bytes/string_reader.h"
#include "third_party/riegeli/bytes/string_writer.h"
#include "third_party/riegeli/records/record_reader.h"
#include "third_party/riegeli/records/records_metadata.proto.h"

namespace array_record {

void TestArrayRecord(absl::string_view options_text,
                     absl::Span<const std::string> ground_truth_records,
                     ARThreadPool* pool) {
  LOG(INFO) << "Testing with options: " << options_text
            << " num_records: " << ground_truth_records.size();
  auto options =
      ArrayRecordWriterBase::Options::FromString(options_text).ValueOrDie();
  std::string encoded;
  auto writer = ArrayRecordWriter<riegeli::StringWriter<>>(
      std::forward_as_tuple(&encoded), options, pool);

  for (const auto& record : ground_truth_records) {
    CHECK(writer.WriteRecord(record)) << writer.status();
  }
  CHECK(writer.Close()) << writer.status();

  auto reader = ArrayRecordReader<riegeli::StringReader<>>(
      std::forward_as_tuple(encoded),
      ArrayRecordReaderBase::Options().set_readahead_buffer_size(2048), pool);

  // Test sequential read
  for (auto i : Seq(reader.NumRecords())) {
    absl::string_view record;
    CHECK(reader.ReadRecord(&record));
    CHECK_EQ(record, ground_truth_records[i]);
  }

  // Test parallel read
  CHECK_OK(reader.ParallelReadRecords(
      [&](uint64_t record_idx, absl::string_view record) -> absl::Status {
        CHECK_EQ(record, ground_truth_records[record_idx]);
        return absl::OkStatus();
      }));

  CHECK(reader.Close()) << reader.status();
  LOG(INFO) << "TEST PASSED";
}

}  // namespace array_record

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  FuzzedDataProvider fuzzed_data_provider(data, size);
  std::vector<std::string> ground_truth_records;

  while (fuzzed_data_provider.remaining_bytes()) {
    ground_truth_records.push_back(
        fuzzed_data_provider.ConsumeRandomLengthString());
  }

  auto* pool = array_record::ArrayRecordGlobalPool();

  array_record::TestArrayRecord("uncompressed", ground_truth_records, pool);

  array_record::TestArrayRecord("brotli", ground_truth_records, pool);

  array_record::TestArrayRecord("zstd", ground_truth_records, pool);

  array_record::TestArrayRecord("snappy", ground_truth_records, pool);

  array_record::TestArrayRecord("uncompressed,transpose", ground_truth_records,
                                pool);

  array_record::TestArrayRecord("brotli,transpose", ground_truth_records, pool);

  array_record::TestArrayRecord("zstd,transpose", ground_truth_records, pool);

  array_record::TestArrayRecord("snappy,transpose", ground_truth_records, pool);
  return 0;
}
