#ifndef THIRD_PARTY_ARRAY_RECORD_PYTHON_READ_INTRUCTIONS_LIB_H_
#define THIRD_PARTY_ARRAY_RECORD_PYTHON_READ_INTRUCTIONS_LIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace array_record {

struct ReadInstruction {
  std::string filename;
  int64_t start = 0;  // Always >= 0.
  // Must be >= start or -1. -1 indicates that the end of the file.
  int64_t end = -1;

  static absl::StatusOr<ReadInstruction> Parse(absl::string_view path);

  int64_t NumRecords() const { return end - start; }
};

using GetNumRecords = std::function<uint64_t(const std::string&)>;

// Get the read instructions for a list of paths where each path can be:
// - A normal filename.
// - A filename with read instructions: filename[start:end].
// Unless the filename is given with read instruction the file will be opened
// to get the total number of records.
absl::StatusOr<std::vector<ReadInstruction>> GetReadInstructions(
    absl::Span<std::string> paths,
    const GetNumRecords& get_num_records);

}  // namespace array_record


#endif  // THIRD_PARTY_ARRAY_RECORD_PYTHON_READ_INTRUCTIONS_LIB_H_
