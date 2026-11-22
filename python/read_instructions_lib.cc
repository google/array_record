#include "python/read_instructions_lib.h"

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "cpp/parallel_for.h"
#include "third_party/re2/re2.h"
#include "thread/threadpool.h"
#include "absl/status/statusor.h"
#include "iostream"

namespace array_record {

// Getting the read instructions is cheap but IO bound. We create a temporary
// thread pool to get the number of records.
constexpr int kNumThreadsForReadInstructions = 256;

absl::StatusOr<ReadInstruction> ReadInstruction::Parse(absl::string_view path) {
  static const LazyRE2 kPattern = {R"((.+)\[(\d+):(\d+)\])"};
  std::string filename;
  int64_t start, end;
  if (RE2::FullMatch(path, *kPattern, &filename, &start, &end)) {
    return ReadInstruction{filename, start, end};
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Can't parse %s as ReadInstruction", path));
}

// Get the read instructions for a list of paths where each path can be:
// - A normal filename.
// - A filename with read instructions: filename[start:end].
// Unless the filename is given with read instruction, the file will be opened
// to get the total number of records.
absl::StatusOr<std::vector<ReadInstruction>> GetReadInstructions(
    absl::Span<std::string> paths,
    const GetNumRecords& get_num_records) {
  std::vector<ReadInstruction> read_instructions;

  // Step 1: Parse potential read instructions.
  bool missing_num_records = false;
  for (const std::string& path : paths) {
    absl::StatusOr<ReadInstruction> read_instruction =
        ReadInstruction::Parse(path);
    if (read_instruction.ok()) {
      read_instructions.push_back(read_instruction.value());
    } else {
      missing_num_records = true;
      const std::string pattern = path;
      read_instructions.push_back({pattern});
    }
  }
  if (!missing_num_records) {
    return read_instructions;
  }

  ThreadPool* pool = new ThreadPool(
      "ReadInstructionsPool", kNumThreadsForReadInstructions);
  pool->StartWorkers();

  std::vector<std::vector<ReadInstruction>> filled_instructions;
  filled_instructions.resize(read_instructions.size());

  // Step 2: Match any patterns.
  auto match_pattern = [&](int i) {
    const std::string& pattern = read_instructions[i].filename;
    if (read_instructions[i].end >= 0 || !absl::StrContains(pattern, '?')) {
      filled_instructions[i].push_back(std::move(read_instructions[i]));
      return;
    }
    const auto status_or_filenames = file::Match(pattern, file::Defaults());
    if (!status_or_filenames.ok() || status_or_filenames->empty()) {
      LOG(ERROR) << "Failed to find matching files for pattern " << pattern;
      return;
    }
    auto filenames = *status_or_filenames;
    // Make sure we always read files in the same order.
    absl::c_sort(filenames);
    filled_instructions[i].reserve(filenames.size());
    for (const std::string& filename : filenames) {
      filled_instructions[i].push_back({filename, 0, -1});
    }
  };

  array_record::ParallelFor(Seq(read_instructions.size()), pool, match_pattern);

  // Flatten filled_instructions into read_instructions;
  read_instructions.clear();
  for (const auto& instructions : filled_instructions) {
    read_instructions.insert(read_instructions.end(), instructions.begin(),
                             instructions.end());
  }

  // Step 3: Get number of records.
  auto add_num_records = [&](int i) {
    if (read_instructions[i].end >= 0) {
      return;
    }
    const std::string& filename = read_instructions[i].filename;

    std::cout << file::Exists(filename, file::Defaults()) << "\n";
    if (!file::Exists(filename, file::Defaults()).ok()) {
      LOG(ERROR) << "File " << filename << " not found.";
      return;
    }

    read_instructions[i].end =
        static_cast<int64_t>(get_num_records(filename));
  };
  array_record::ParallelFor(
      Seq(read_instructions.size()),
      pool,
      add_num_records);
  return read_instructions;
}

}  // namespace array_record
