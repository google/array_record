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

#include "cpp/sequenced_chunk_writer.h"

#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "cpp/common.h"
#include "cpp/thread_pool.h"
#include "riegeli/bytes/chain_writer.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/chunk_encoding/chunk.h"
#include "riegeli/chunk_encoding/compressor_options.h"
#include "riegeli/chunk_encoding/simple_encoder.h"
#include "riegeli/records/record_reader.h"

namespace array_record {
namespace {

TEST(SequencedChunkWriterTest, RvalCtorTest) {
  // Constructs SequencedChunkWriter by taking the ownership of the other
  // riegeli writer.
  {
    std::string dest;
    auto str_writer = riegeli::StringWriter(&dest);
    auto to_string =
        SequencedChunkWriter<riegeli::StringWriter<>>(std::move(str_writer));
  }
  {
    absl::Cord cord;
    auto cord_writer = riegeli::CordWriter(&cord);
    auto to_cord =
        SequencedChunkWriter<riegeli::CordWriter<>>(std::move(cord_writer));
  }
  {
    std::string dest;
    auto str_writer = riegeli::StringWriter(&dest);
    auto to_string =
        std::make_unique<SequencedChunkWriter<riegeli::StringWriter<>>>(
            std::move(str_writer));
  }
  {
    absl::Cord cord;
    auto cord_writer = riegeli::CordWriter(&cord);
    auto to_cord =
        std::make_unique<SequencedChunkWriter<riegeli::CordWriter<>>>(
            std::move(cord_writer));
  }
}

TEST(SequencedChunkWriterTest, DestArgsCtorTest) {
  // Constructs SequencedChunkWriter by forwarding the constructor arguments to
  // templated riegeli writer.
  {
    std::string dest;
    auto to_string =
        SequencedChunkWriter<riegeli::StringWriter<>>(std::make_tuple(&dest));
  }
  {
    absl::Cord cord;
    auto to_cord =
        SequencedChunkWriter<riegeli::CordWriter<>>(std::make_tuple(&cord));
  }

  {
    std::string dest;
    auto to_string =
        std::make_unique<SequencedChunkWriter<riegeli::StringWriter<>>>(
            std::make_tuple(&dest));
  }
  {
    absl::Cord cord;
    auto to_cord =
        std::make_unique<SequencedChunkWriter<riegeli::CordWriter<>>>(
            std::make_tuple(&cord));
  }
}

class TestCommitChunkCallback
    : public SequencedChunkWriterBase::SubmitChunkCallback {
 public:
  void operator()(uint64_t chunk_seq, uint64_t chunk_offset,
                  uint64_t decoded_data_size, uint64_t num_records) override {
    chunk_offsets_.push_back(chunk_offset);
  }
  absl::Span<const uint64_t> get_chunk_offsets() const {
    return chunk_offsets_;
  }

 private:
  std::vector<uint64_t> chunk_offsets_;
};

TEST(SequencedChunkWriterTest, SanityTestCodeSnippet) {
  std::string encoded;
  auto callback = TestCommitChunkCallback();

  auto writer = std::make_shared<SequencedChunkWriter<riegeli::StringWriter<>>>(
      std::make_tuple(&encoded));
  writer->set_submit_chunk_callback(&callback);
  ASSERT_TRUE(writer->ok()) << writer->status();

  for (auto i : Seq(3)) {
    std::packaged_task<absl::StatusOr<riegeli::Chunk>()> encoding_task([i] {
      riegeli::Chunk chunk;
      riegeli::SimpleEncoder encoder(
          riegeli::CompressorOptions().set_uncompressed(), 1);
      std::string text_to_encode = std::to_string(i);
      EXPECT_TRUE(encoder.AddRecord(absl::string_view(text_to_encode)));
      riegeli::ChunkType chunk_type;
      uint64_t decoded_data_size;
      uint64_t num_records;
      riegeli::ChainWriter chain_writer(&chunk.data);
      EXPECT_TRUE(encoder.EncodeAndClose(chain_writer, chunk_type, num_records,
                                         decoded_data_size));
      EXPECT_TRUE(chain_writer.Close());
      chunk.header = riegeli::ChunkHeader(chunk.data, chunk_type, num_records,
                                          decoded_data_size);
      return chunk;
    });
    ASSERT_TRUE(writer->CommitFutureChunk(encoding_task.get_future()));
    encoding_task();
    writer->SubmitFutureChunks(false);
  }
  // Calling SubmitFutureChunks(true) blocks the current thread until all
  // encoding tasks complete.
  EXPECT_TRUE(writer->SubmitFutureChunks(true));
  // Paddings should not cause any failure.
  EXPECT_TRUE(writer->Close());

  // File produced by SequencedChunkWriter should be a valid riegeli file.
  auto reader = riegeli::RecordReader<riegeli::StringReader<>>(
      std::forward_as_tuple(encoded));
  ASSERT_TRUE(reader.CheckFileFormat());
  // Read sequentially
  absl::Cord result;
  EXPECT_TRUE(reader.ReadRecord(result));
  EXPECT_EQ(result, "0");
  EXPECT_TRUE(reader.ReadRecord(result));
  EXPECT_EQ(result, "1");
  EXPECT_TRUE(reader.ReadRecord(result));
  EXPECT_EQ(result, "2");
  EXPECT_FALSE(reader.ReadRecord(result));

  // We can use the chunk_offsets information to randomly access records.
  auto offsets = callback.get_chunk_offsets();
  EXPECT_TRUE(reader.Seek(offsets[1]));
  EXPECT_TRUE(reader.ReadRecord(result));
  EXPECT_EQ(result, "1");
  EXPECT_TRUE(reader.Seek(offsets[0]));
  EXPECT_TRUE(reader.ReadRecord(result));
  EXPECT_EQ(result, "0");
  EXPECT_TRUE(reader.Seek(offsets[2]));
  EXPECT_TRUE(reader.ReadRecord(result));
  EXPECT_EQ(result, "2");

  EXPECT_TRUE(reader.Close());
}

TEST(SequencedChunkWriterTest, SanityTestBadChunk) {
  std::string encoded;
  auto callback = TestCommitChunkCallback();

  auto writer = std::make_shared<SequencedChunkWriter<riegeli::StringWriter<>>>(
      std::make_tuple(&encoded));
  writer->set_submit_chunk_callback(&callback);
  ASSERT_TRUE(writer->ok()) << writer->status();
  std::packaged_task<absl::StatusOr<riegeli::Chunk>()> encoding_task(
      [] { return absl::InternalError("On purpose"); });
  EXPECT_TRUE(writer->CommitFutureChunk(encoding_task.get_future()));
  EXPECT_TRUE(writer->SubmitFutureChunks(false));
  encoding_task();
  // We should see the error being populated even when we try to run it with the
  // non-blocking version.
  EXPECT_FALSE(writer->SubmitFutureChunks(false));
  EXPECT_EQ(writer->status().code(), absl::StatusCode::kInternal);

  EXPECT_FALSE(writer->Close());
  EXPECT_TRUE(callback.get_chunk_offsets().empty());
}

}  // namespace
}  // namespace array_record
