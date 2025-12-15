C++ API Reference
==================

ArrayRecord provides a high-performance C++ API built on top of Riegeli. The C++ API offers the best performance for applications that can integrate directly with C++.

.. note::
   The C++ API is primarily for advanced users who need maximum performance or are integrating ArrayRecord into C++ applications. Most users should use the Python API.

Core Classes
------------

ArrayRecordReader
~~~~~~~~~~~~~~~~~

The ``ArrayRecordReader`` class provides high-performance reading of ArrayRecord files with support for parallel processing and random access.

**Header**: ``cpp/array_record_reader.h``

**Template Declaration**:

.. code-block:: cpp

   template <typename Src = riegeli::Reader>
   class ArrayRecordReader : public ArrayRecordReaderBase

**Basic Usage**:

.. code-block:: cpp

   #include "cpp/array_record_reader.h"
   #include "riegeli/bytes/fd_reader.h"

   // Read from file
   ArrayRecordReader reader(riegeli::Maker<riegeli::FdReader>("input.array_record"));
   
   if (!reader.ok()) {
       // Handle error
       return reader.status();
   }

   std::cout << "Total records: " << reader.NumRecords() << std::endl;

   // Read records sequentially
   absl::string_view record;
   while (reader.ReadRecord(&record)) {
       // Process record
       std::cout << "Record: " << record << std::endl;
   }

   if (!reader.Close()) {
       return reader.status();
   }

ArrayRecordReaderBase
~~~~~~~~~~~~~~~~~~~~~

Base class containing template-independent functionality.

**Key Methods**:

.. code-block:: cpp

   class ArrayRecordReaderBase : public riegeli::Object {
   public:
       // Get total number of records
       uint64_t NumRecords() const;
       
       // Get current record index
       uint64_t RecordIndex() const;
       
       // Seek to specific record
       bool SeekRecord(uint64_t record_index);
       
       // Read current record
       bool ReadRecord(absl::string_view* record);
       bool ReadRecord(google::protobuf::MessageLite* record);
       
       // Parallel reading methods
       absl::Status ParallelReadRecords(
           absl::FunctionRef<absl::Status(uint64_t, absl::string_view)> callback) const;
           
       absl::Status ParallelReadRecordsWithIndices(
           absl::Span<const uint64_t> indices,
           absl::FunctionRef<absl::Status(uint64_t, absl::string_view)> callback) const;
           
       absl::Status ParallelReadRecordsInRange(
           uint64_t begin, uint64_t end,
           absl::FunctionRef<absl::Status(uint64_t, absl::string_view)> callback) const;
   };

**Options Class**:

.. code-block:: cpp

   class ArrayRecordReaderBase::Options {
   public:
       // Parse options from string
       static absl::StatusOr<Options> FromString(absl::string_view text);
       
       // Set readahead buffer size (default: 16MB, set to 0 for random access)
       Options& set_readahead_buffer_size(uint64_t size);
       
       // Set maximum parallelism (default: auto, set to 0 for random access)
       Options& set_max_parallelism(std::optional<uint32_t> parallelism);
       
       // Set index storage option
       Options& set_index_storage_option(IndexStorageOption option);
   };

ArrayRecordWriter
~~~~~~~~~~~~~~~~~

The ``ArrayRecordWriter`` class provides high-performance writing of ArrayRecord files with configurable compression and parallel processing.

**Header**: ``cpp/array_record_writer.h``

**Template Declaration**:

.. code-block:: cpp

   template <typename Dest = riegeli::Writer*>
   class ArrayRecordWriter : public ArrayRecordWriterBase

**Basic Usage**:

.. code-block:: cpp

   #include "cpp/array_record_writer.h"
   #include "riegeli/bytes/fd_writer.h"

   // Write to file
   ArrayRecordWriter writer(riegeli::Maker<riegeli::FdWriter>("output.array_record"));
   
   if (!writer.ok()) {
       return writer.status();
   }

   // Write records
   for (int i = 0; i < 1000; ++i) {
       std::string record = absl::StrCat("Record ", i);
       if (!writer.WriteRecord(record)) {
           return writer.status();
       }
   }

   if (!writer.Close()) {
       return writer.status();
   }

ArrayRecordWriterBase
~~~~~~~~~~~~~~~~~~~~~

Base class containing template-independent functionality.

**Key Methods**:

.. code-block:: cpp

   class ArrayRecordWriterBase : public riegeli::Object {
   public:
       // Write records of various types
       bool WriteRecord(const google::protobuf::MessageLite& record);
       bool WriteRecord(absl::string_view record);
       bool WriteRecord(const absl::Cord& record);
       bool WriteRecord(const void* data, size_t num_bytes);
       
       template <typename T>
       bool WriteRecord(absl::Span<const T> record);
   };

**Options Class**:

.. code-block:: cpp

   class ArrayRecordWriterBase::Options {
   public:
       // Parse options from string
       static absl::StatusOr<Options> FromString(absl::string_view text);
       
       // Set group size (number of records per chunk)
       Options& set_group_size(uint32_t group_size);
       
       // Set maximum parallelism
       Options& set_max_parallelism(std::optional<uint32_t> parallelism);
       
       // Compression options
       Options& set_uncompressed();
       Options& set_brotli(int level = kDefaultBrotli);
       Options& set_zstd(int level = kDefaultZstd);
       Options& set_snappy();
       
       // Advanced options
       Options& set_transpose(bool transpose);
       Options& set_transpose_bucket_size(uint64_t size);
       Options& set_window_log(std::optional<int> window_log);
       Options& set_pad_to_block_boundary(bool pad);
       Options& set_metadata(const std::optional<riegeli::RecordsMetadata>& metadata);
   };

Usage Examples
--------------

Reading with Different Sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include "cpp/array_record_reader.h"
   #include "riegeli/bytes/string_reader.h"
   #include "riegeli/bytes/cord_reader.h"
   #include "riegeli/bytes/fd_reader.h"

   // Read from string
   std::string data = /* ... */;
   ArrayRecordReader string_reader(riegeli::Maker<riegeli::StringReader>(data));

   // Read from cord
   absl::Cord cord = /* ... */;
   ArrayRecordReader cord_reader(riegeli::Maker<riegeli::CordReader>(&cord));

   // Read from file
   ArrayRecordReader file_reader(riegeli::Maker<riegeli::FdReader>("file.array_record"));

Writing with Different Destinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include "cpp/array_record_writer.h"
   #include "riegeli/bytes/string_writer.h"
   #include "riegeli/bytes/cord_writer.h"
   #include "riegeli/bytes/fd_writer.h"

   // Write to string
   std::string output;
   ArrayRecordWriter string_writer(riegeli::Maker<riegeli::StringWriter>(&output));

   // Write to cord
   absl::Cord cord;
   ArrayRecordWriter cord_writer(riegeli::Maker<riegeli::CordWriter>(&cord));

   // Write to file
   ArrayRecordWriter file_writer(riegeli::Maker<riegeli::FdWriter>("output.array_record"));

Parallel Reading
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   ArrayRecordReader reader(riegeli::Maker<riegeli::FdReader>("large_file.array_record"));

   // Read all records in parallel
   absl::Status status = reader.ParallelReadRecords(
       [](uint64_t record_index, absl::string_view record_data) -> absl::Status {
           // Process record
           std::cout << "Record " << record_index << ": " << record_data << std::endl;
           return absl::OkStatus();
       });

   if (!status.ok()) {
       std::cerr << "Error: " << status << std::endl;
   }

   // Read specific indices
   std::vector<uint64_t> indices = {10, 100, 1000, 5000};
   status = reader.ParallelReadRecordsWithIndices(
       indices,
       [&indices](uint64_t indices_index, absl::string_view record_data) -> absl::Status {
           uint64_t record_index = indices[indices_index];
           std::cout << "Record " << record_index << ": " << record_data << std::endl;
           return absl::OkStatus();
       });

   // Read a range
   status = reader.ParallelReadRecordsInRange(
       1000, 2000,  // Read records 1000-1999
       [](uint64_t record_index, absl::string_view record_data) -> absl::Status {
           std::cout << "Record " << record_index << ": " << record_data << std::endl;
           return absl::OkStatus();
       });

Protocol Buffer Support
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include "your_proto.pb.h"

   // Writing protocol buffers
   ArrayRecordWriter writer(riegeli::Maker<riegeli::FdWriter>("protos.array_record"));
   
   YourProtoMessage message;
   message.set_field("value");
   writer.WriteRecord(message);

   // Reading protocol buffers
   ArrayRecordReader reader(riegeli::Maker<riegeli::FdReader>("protos.array_record"));
   
   // Parallel reading with automatic proto parsing
   absl::Status status = reader.ParallelReadRecords<YourProtoMessage>(
       [](uint64_t record_index, YourProtoMessage proto) -> absl::Status {
           std::cout << "Record " << record_index << ": " << proto.field() << std::endl;
           return absl::OkStatus();
       });

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Writer with custom options
   ArrayRecordWriterBase::Options writer_options;
   writer_options
       .set_group_size(1000)
       .set_brotli(9)  // High compression
       .set_max_parallelism(4)
       .set_transpose(true);  // For proto messages

   ArrayRecordWriter writer(
       riegeli::Maker<riegeli::FdWriter>("optimized.array_record"),
       writer_options);

   // Reader optimized for random access
   ArrayRecordReaderBase::Options reader_options;
   reader_options
       .set_readahead_buffer_size(0)  // Disable readahead
       .set_max_parallelism(0)        // Disable parallel readahead
       .set_index_storage_option(
           ArrayRecordReaderBase::Options::IndexStorageOption::kInMemory);

   ArrayRecordReader reader(
       riegeli::Maker<riegeli::FdReader>("data.array_record"),
       reader_options);

Error Handling
--------------

Always check for errors in C++ code:

.. code-block:: cpp

   ArrayRecordWriter writer(riegeli::Maker<riegeli::FdWriter>("output.array_record"));
   
   if (!writer.ok()) {
       std::cerr << "Failed to create writer: " << writer.status() << std::endl;
       return -1;
   }

   if (!writer.WriteRecord("test data")) {
       std::cerr << "Failed to write record: " << writer.status() << std::endl;
       return -1;
   }

   if (!writer.Close()) {
       std::cerr << "Failed to close writer: " << writer.status() << std::endl;
       return -1;
   }

Thread Safety
-------------

- ``ArrayRecordReader`` and ``ArrayRecordWriter`` are **thread-compatible** but not **thread-safe**
- Multiple readers can safely read from the same file simultaneously if each has its own instance
- Writers should not be accessed from multiple threads without external synchronization
- The parallel reading methods handle their own thread safety internally

Performance Considerations
--------------------------

Reading Performance
~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // For sequential access (default)
   ArrayRecordReaderBase::Options sequential_options;
   sequential_options.set_readahead_buffer_size(64 * 1024 * 1024);  // 64MB

   // For random access
   ArrayRecordReaderBase::Options random_options;
   random_options
       .set_readahead_buffer_size(0)
       .set_max_parallelism(0);

Writing Performance
~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // High throughput writing
   ArrayRecordWriterBase::Options high_throughput;
   high_throughput
       .set_group_size(10000)  // Larger groups for better compression
       .set_max_parallelism(8) // More parallel encoders
       .set_brotli(3);         // Lower compression for speed

   // Balanced performance
   ArrayRecordWriterBase::Options balanced;
   balanced
       .set_group_size(1000)
       .set_brotli(6);  // Default compression

Build Integration
-----------------

To use ArrayRecord in your C++ project, you'll need to integrate with the build system:

Bazel
~~~~~

.. code-block:: starlark

   cc_binary(
       name = "my_app",
       srcs = ["my_app.cc"],
       deps = [
           "//cpp:array_record_reader",
           "//cpp:array_record_writer",
           "@com_google_riegeli//riegeli/bytes:fd_reader",
           "@com_google_riegeli//riegeli/bytes:fd_writer",
       ],
   )

CMake
~~~~~

.. code-block:: cmake

   find_package(ArrayRecord REQUIRED)
   find_package(Riegeli REQUIRED)

   target_link_libraries(my_app 
       ArrayRecord::array_record_reader
       ArrayRecord::array_record_writer
       Riegeli::riegeli
   )
