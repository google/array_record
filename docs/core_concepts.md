# Core Concepts


This document explains the fundamental concepts behind ArrayRecord and how to use it efficiently.

## What is ArrayRecord?

ArrayRecord functions as a serialization format engineered to accommodate three
primary access methodologies: sequential, batch, and random read. To effectively
address these diverse operational requirements, ArrayRecord is designed to
support various configurations, each optimized for a specific access pattern.
Certain configurations are adjustable at runtime, permitting modifications
post-serialization, while others necessitate user decisions at the time of file
creation or serialization. To maximize the utility of ArrayRecord for a specific
application, a detailed examination of its internal architecture and the core
principles of advanced systems engineering is required.


## File Structure

An ArrayRecord file is structured as a sequence of individually compressed
chunks. A user data chunk may encapsulate one or more records, the quantity of
which is determined by the write configuration. Subsequent to the user data
chunks are the footer chunk, which stores the file offsets for each chunk, and a
postscript section designated for file metadata. This organization is
illustrated below.

```
┌─────────────────────┐
│    User Data        │
│      Chunk 1        │
├─────────────────────┤
│    User Data        │
│      Chunk 2        │
├─────────────────────┤
│       ...           │
├─────────────────────┤
│    User Data        │
│      Chunk N        │
├─────────────────────┤
│   Footer Chunk      │
│   (Index Data)      │
├─────────────────────┤
│   Postscript        │
│  (File Metadata)    │
└─────────────────────┘
```


## Write-time Configuration
The subsequent options govern the methodology by which ArrayRecord serializes
data to the disk medium. Within both the C++ and Python APIs, ArrayRecord
employs an option string for the encoding of these configurations. The option
string is comprised of one or more option specifications, delineated by commas.
Each individual option is inherently optional. Typically, each option is
succeeded by a colon (:) and its corresponding value, with the exception of
`uncompressed` and `snappy`, which function as boolean flags. The following
illustrates an example configuration:

```python
from array_record.python import array_record_module

zstd_writer = array_record_module.ArrayRecordWriter(
    'output.array_record',
    'group_size:1024,zstd:5,window_log:10'
)

snappy_writer = array_record_module.ArrayRecordWriter(
    'output.array_record',
    'group_size:1024,snappy'
)
```

The aforementioned code snippet initializes an `output.array_record` file where
`group_size` is set to 1024, the compression algorithm is Zstandard with level
5, and the compression tuning option `window_log` is set to 10. Adjusting these
parameters facilitates the optimization of the on-disk data representation for
various read access patterns.

### Group size

The `group_size` option dictates the quantity of records to be consolidated and
compressed into a single chunk. This parameter represents the most critical
tuning variable for accommodating diverse usage scenarios. Given that each group
is compressed autonomously, accessing a single record within a chunk
necessitates the complete decompression of that chunk. While the decompressed
chunk is subsequently cached for later access, this mechanism does not benefit
random access patterns unless the requested record resides in the currently
cached chunk. Consequently, as a general guideline, an ArrayRecord file
primarily intended for random access should consistently employ a `group_size`
of one.

In contrast to the random access use case, ArrayRecord achieves a superior
compression ratio and reduced I/O overhead when configured with a larger group
size. This configuration is particularly advantageous for sequential and batch
access of contiguous records.

```{attention}
When creating ArrayRecord files for `array_record_data_source`,
users must set the `group_size` to 1.
```

### Compression
ArrayRecord provides support for multiple compression algorithms. Users select
the algorithm by `algo[:level]`, where `algo` is one of the following: `zstd`,
`brotli`, `snappy`, or `uncompressed`, and the `level` specifies the compression
level which differs by the algorithm type.

#### Zstd (Default)
Zstandard (zstd) is a fast, lossless data compression algorithm developed by
Yann Collet at Facebook (now Meta). It is engineered to strike an effective
balance between compression speed and compression ratio, rendering it suitable
for a broad spectrum of applications. The configurable tuning parameters are:

- Levels -131072 to 22 (default: 3). A higher numerical value corresponds to a
  greater compression ratio, while negative values activate "fast" compression
  algorithms. Level 0 is functionally equivalent to the default compression
  level, which is 3.
- `window_log` 10 to 30 (default 20). This represents the logarithm of the LZ77
  sliding window size. This option tunes the trade-off between compression
  density and memory utilization (where a higher value results in improved
  density but increased memory consumption).

The following example configuration is optimized for rapid random access with a
minimal memory footprint for the compression sliding window:

```python
writer = array_record_module.ArrayRecordWriter(
    'output.array_record',
    'group_size:1,zstd:1,window_log:12'
)
```

#### Brotli
Brotli is an open-source compression algorithm developed by Google that
typically yields superior compression ratios compared to gzip for web content,
thereby facilitating faster loading times and reduced bandwidth consumption. In
comparison to zstd, Brotli often achieves a better compression ratio but may
entail slower compression and decompression speeds. The configurable tuning
parameters are:

- Levels 0 to 11 (default: 6). A higher numerical value indicates a greater
  compression ratio.
- `window_log` 10 to 30 (default 22). This represents the logarithm of the LZ77
  sliding window size. This option tunes the trade-off between compression
  density and memory utilization (where a higher value results in improved
  density but increased memory consumption).

The following is an example configuration tailored for a high compression ratio
suitable for batch access:

```python
writer = ArrayRecordWriter(
    'output.array_record',
    'group_size:65536,brotli:9,window_log:26'
)
```

#### Snappy
Snappy is a rapid and efficient compression/decompression library developed by
Google, where speed is prioritized over maximizing file size reduction. It is
extensively utilized in big data systems, such as Hadoop, Spark, and LevelDB,
where real-time processing capability is paramount. While Snappy possesses
certain tuning parameters, these are currently considered experimental, and thus
the tuning options have been disabled.

The following represents an example Snappy configuration for random access:

```python
writer = array_record_module.ArrayRecordWriter(
    'output.array_record',
    'group_size:1,snappy'
)
```

#### Uncompressed
Finally, ArrayRecord offers the uncompressed option. It is important to note
that for optimal fast random access, the user should still specify a group size
of 1 to minimize I/O operations.

```python
writer = ArrayRecordWriter('output.array_record', 'group_size:1,uncompressed')
```

## Read-time configuration

ArrayRecord furnishes two distinct APIs for data retrieval:
- `array_record_module.Reader`: This component offers a direct, one-to-one
  correspondence with the underlying C++ API, thereby providing comprehensive
  support for all operational scenarios, including random access, batch access,
  and sequential read operations.
- `array_record_data_source`: This serves as a multi-file adapter specifically
  designed for integration with pygrain, primarily architected to facilitate
  random access to the encapsulated records.

Configuration of the `array_record_module.Reader` options is executed using a
syntax analogous to that of the writer: a string of comma-separated key-value
pairs. Conversely, for `array_record_data_source`, the options are specified via a
Python dictionary. The following code snippet illustrates the initialization of
both reader types:

```python
from array_record.python import array_record_module
from array_record.python import array_record_data_source

reader = array_record_module.ArrayRecordReader(
    'output.array_record',
    'index_storage_option:offloaded,readahead_buffer_size=0'
)
ds = array_record_data_source(
    'output.array_record',
    reader_options={
        'index_storage_option': 'offloaded',
    }
)
```

### Index Storage Option

The `index_storage_option` constitutes the primary tuning variable for
optimizing read-time performance. Two options are currently available:

- `in_memory` (default): This configuration facilitates rapid random access by
  loading the chunk offset index directly into memory. However, this may result
  in a substantial memory footprint if the number of records is exceptionally
  large, particularly when utilized with ArrayRecordDataSource, where the
  indices from multiple ArrayRecord files would collectively reside in memory.
- `offloaded`: In this mode, the chunk offset index is not loaded into memory;
  instead, it is retrieved from the disk for each access operation.

### Read-Ahead for Sequential Access

```{caution}
This feature is only available in `array_record_module.Reader` and is not supported by `array_record_data_source`.
```

Sequential access is an access pattern employed in the majority of Google's
storage formats. This paradigm involves a user opening a file and iteratively
invoking a read() operation to retrieve the file content and advance the
internal cursor until the end-of-file is reached. See the demonstration below:

```python
from array_record.python import array_record_module

reader = array_record_module.ArrayRecordReader(
    'output.array_record',
    'readahead_buffer_size:65536,max_parallelism:8'
)
for _ in range(reader.num_records()):
  record = reader.read()
```

Given the predictable nature of this access pattern, ArrayRecord incorporates an
integrated read-ahead system to prefetch subsequent records. The following
parameters govern the configuration of this read-ahead functionality:

- `readahead_buffer_size`: This specifies the size of the read-ahead buffer per
  thread, measured in bytes. Setting this parameter to zero effectively disables
  the read-ahead mechanism.
- `max_parallelism`: This parameter dictates the number of dedicated threads
  utilized for read-ahead prefetching.


## Access Patterns

ArrayRecord is designed to efficiently support three primary data access
patterns. Following these best practices for each pattern can help you optimize
performance and resource utilization.

### Random Access
Random access is used when you need to quickly retrieve individual,
non-contiguous records from the dataset. To optimize for this:

- Configure the writer to set `group_size:1`. This ensures each record is a
  self-contained group, minimizing the data read for a single lookup.
- Enable compression (e.g., `zstd`) for text and numerical data to reduce file
  size and improve I/O speed. Compression is not typically needed for data
  already compressed by specific algorithms.
- For image data (e.g., JPEG, PNG), which is typically compressed using
  format-specific algorithms, it is better to keep it in its original binary
  form rather than applying an additional generic compression layer.
- The optimal compression algorithm and level require experimentation, but the
  default setting, `zstd:3`, provides a good balance of speed and ratio for most
  general-purpose data.
- If the memory footprint of the index becomes a concern, configure the reader
  to offload the index (`index_storage_option:offloaded`). This trades increase
  in latency for lower memory consumption.
- Whenever possible, prefer batch access (even for non-contiguous records) to
  leverage the underlying C++ thread pool for better overall performance due to
  parallel I/O and decompression.

```python
from array_record.python import array_record_module
from array_record.python import array_record_data_source

# Writer with the default compression option, which is zstd:3
writer = array_record_module.ArrayRecordWriter(
    'output.array_record',
    'group_size:1'
)

# Reader/data-source with in-memory index
reader = array_record_module.ArrayRecordReader( 'output.array_record')
ds = array_record_data_source('output.array_record')

# Reader/data-source with offloaded index
reader = array_record_module.ArrayRecordReader(
    'output.array_record',
    'index_storage_option:offloaded'
)
ds = array_record_data_source(
    'output.array_record',
    reader_options={
        'index_storage_option': 'offloaded',
    }
)
```

### Sequential Access
Sequential access is used for iterating through the dataset in order (e.g., for
training loops). To optimize for this:

- A larger `group_size` improves the compression ratio (by grouping more data)
  and reduces the number of I/O operations needed to read the file.
- While optimal compression requires tuning, the default `zstd:3` is generally a
  good starting point for generic data in sequential reads.
- Prefer using the Batch Access API over traditional prefetch mechanisms for
  better thread utilization and throughput.

```python
from array_record.python import array_record_module

# Writer with the default compression option, which is zstd:3
writer = array_record_module.ArrayRecordWriter(
    'output.array_record',
    'group_size:1024'
)

# Optimize sequential access with a read-ahead buffer.
reader = array_record_module.ArrayRecordReader(
    'output.array_record',
    'readahead_buffer_size:16M'
)

# Sequential access with read ahead buffer (slow)
for _ in range(reader.num_records()):
  record = reader.read()

# Read all data with a thread pool (fast)
records = reader.read_all()
```

### Batch Access
Batch access involves reading multiple records in a single function call, which
is the most efficient way to use ArrayRecord.

- Batch access is the recommended way to use ArrayRecord for high-performance
  reading.
- The API supports reading records using multiple methods: batch all (read the
  entire file), batch range (read a contiguous subset), and batch with indices
  (read an arbitrary set of records).
- When using an offloaded index reader with batch access, the index chunk is only
  read once per batch invocation. This greatly reduces the amortized latency per
  record compared to reading the index for every single record.

#### `array_record_module.ArrayRecordReader` batch access APIs

```python
from array_record.python import array_record_module

reader = array_record_module.ArrayRecordReader('output.array_record')

# Read all the records
records = reader.read_all()

# Read records by range
records = reader.read(0, 100)

# Read records by indices
records = reader.read([0, 1, 100])
```

#### `array_record_data_source` batch access APIs
```python
from array_record.python import array_record_data_source

ds = array_record_data_source.ArrayRecordDataSource('output.array_record')

# Read records by indices
records = ds.__getitems__([10, 100, 200])
```
