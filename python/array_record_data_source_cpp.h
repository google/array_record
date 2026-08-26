#ifndef THIRD_PARTY_ARRAY_RECORD_PYTHON_ARRAY_RECORD_DATA_SOURCE_CPP_H_
#define THIRD_PARTY_ARRAY_RECORD_PYTHON_ARRAY_RECORD_DATA_SOURCE_CPP_H_


#include "python/read_instructions_lib.h"
#include "cpp/array_record_reader.h"
#include "riegeli/bytes/file_reader.h"


namespace array_record {

// A Datasource for multiple ArrayRecordFiles. It holds the file reader objects
// and implements the lookup logic. The constructor constructs the global index
// by reading the number of records per file. NumRecords() returns the total
// number of records. GetItem() looks up a single key and returns the record.
// If needed it will open file readers.
class ArrayRecordDataSource {
 public:
  explicit ArrayRecordDataSource(absl::Span<std::string> paths_);

  uint64_t NumRecords() const;

  absl::Status GetItem(uint64_t key, absl::string_view* record);

 private:
  const std::vector<std::string> paths_;
  std::vector<ReadInstruction> read_instructions_;
  uint64_t total_num_records_;

  void CreateReader(int reader_index);

  using Reader =
      std::unique_ptr<array_record::ArrayRecordReader<riegeli::FileReader<>>>;
  std::vector<Reader> readers_;
  std::mutex create_reader_mutex_;

  std::pair<int, uint64_t> GetReaderIndexAndPosition(uint64_t key) const;

  absl::Status CheckGroupSize(
    absl::string_view filename,
    std::optional<std::string> options_string);
};
}  // namespace array_record


#endif  // THIRD_PARTY_ARRAY_RECORD_PYTHON_ARRAY_RECORD_DATA_SOURCE_CPP_H_
