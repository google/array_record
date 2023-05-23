#include "python/array_record_data_source_cpp.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "python/read_instructions_lib.h"
#include "cpp/array_record_reader.h"
#include "cpp/array_record_writer.h"


namespace array_record {

using ArrayRecordReaderOptions = ::array_record::ArrayRecordReaderBase::Options;
using ArrayRecordWriterOptions = ::array_record::ArrayRecordWriterBase::Options;

static uint64_t ArrayRecordGetNumRecords(const std::string& filename) {
  const array_record::ArrayRecordReader<riegeli::FileReader<>> reader(
      std::forward_as_tuple(filename));
  return reader.NumRecords();
}

ArrayRecordDataSource::ArrayRecordDataSource(absl::Span<std::string> paths_){
  absl::StatusOr<std::vector<ReadInstruction>> read_instructions_or_failure =
      GetReadInstructions(paths_, ArrayRecordGetNumRecords);
  CHECK_OK(read_instructions_or_failure);
  read_instructions_ = *read_instructions_or_failure;

  total_num_records_ = 0;
  for (const auto& ri : read_instructions_) {
    total_num_records_ += ri.NumRecords();
  }
  readers_.resize(read_instructions_.size());
}

uint64_t ArrayRecordDataSource::NumRecords() const {return total_num_records_;}

std::pair<int, uint64_t> ArrayRecordDataSource::GetReaderIndexAndPosition(
    uint64_t key) const {
  int reader_index = 0;
  CHECK(key < NumRecords()) << "Invalid key " << key;
  while (key >= read_instructions_[reader_index].NumRecords()) {
    key -= read_instructions_[reader_index].NumRecords();
    reader_index++;
  }
  key += read_instructions_[reader_index].start;
  return {reader_index, key};
}

absl::Status ArrayRecordDataSource::CheckGroupSize(
    const absl::string_view filename,
    const std::optional<std::string> options_string) {
  // Check that ArrayRecord files were created with group_size=1. Old files
  // (prior 2022-10) don't have this info.
  if (!options_string.has_value()) {
    return absl::OkStatus();
  }
  auto maybe_options = ArrayRecordWriterOptions::FromString(*options_string);
  if (!maybe_options.ok()) {
    return maybe_options.status();
  }
  const int group_size = maybe_options->group_size();
  if (group_size != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "File ", filename, " was created with group size ", group_size,
        ". Grain requires group size 1 for good performance. Please "
        "re-generate your ArrayRecord files with 'group_size:1'."));
  }
  return absl::OkStatus();
}

void ArrayRecordDataSource::CreateReader(const int reader_index) {
  // See b/262550570 for the readahead buffer size.
  ArrayRecordReaderOptions array_record_reader_options;
  array_record_reader_options.set_max_parallelism(0);
  array_record_reader_options.set_readahead_buffer_size(0);
  riegeli::FileReaderBase::Options file_reader_options;
  file_reader_options.set_buffer_size(1 << 15);
  // Copy is on purpose.
  std::string filename = read_instructions_[reader_index].filename;
  auto reader = std::make_unique<
      array_record::ArrayRecordReader<riegeli::FileReader<>>>(
          std::forward_as_tuple(filename, file_reader_options),
          array_record_reader_options, array_record::ArrayRecordGlobalPool());
  const auto status = CheckGroupSize(filename, reader->WriterOptionsString());
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  {
    const std::lock_guard<std::mutex> lock(create_reader_mutex_);
    if (readers_[reader_index] == nullptr) {
      readers_[reader_index] = std::move(reader);
    }
  }
}

absl::Status ArrayRecordDataSource::GetItem(
    uint64_t key, absl::string_view* record) {
  int reader_index;
  uint64_t position;
  std::tie(reader_index, position) = GetReaderIndexAndPosition(key);
  if (readers_[reader_index] == nullptr) {
      CreateReader(reader_index);
  }
  return readers_[reader_index]->ParallelReadRecordsWithIndices(
      {position},
      [&](uint64_t read_idx, absl::string_view value) -> absl::Status {
        // TODO(amrahmed): Follow on this
        *record = value;
        return absl::OkStatus();
      });
}
}  // namespace array_record
