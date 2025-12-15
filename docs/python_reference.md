# Python API Reference


## `array_record.python.array_record_module.ArrayRecordWriter`

### `ArrayRecordWriter(path: str, options: str)`

* `path` (str): File path where the ArrayRecord to be written.
* `options` (str, optional): Comma-separated options string. Default ""

#### Options string format

The options string can contain the following comma-separated options:

* `group_size:N` - Number of records per chunk (default: 1)
* `uncompressed` - Disable compression
* `brotli[:N]` - Use Brotli compression with level N (0-11, default: 6)
* `zstd[:N]` - Use Zstd compression with level N (-131072 to 22, default: 3)
* `snappy` - Use Snappy compression
* `window_log:N` - LZ77 window size (10-31) for zstd and brotli.
* `pad_to_block_boundary:true/false` - Pad chunks to 64KB boundaries (default
  false)

User should only select one of the compression options `zstd`, `brotli`,
`snappy`, `uncompressed`, otherwise an error would be raised.

### `ok() -> bool`

Returns true when the writer object is having a healthy state.

### `close()`

Closes the file. May raise an error if it failed to do so.

### `is_open() -> bool`

Returns true when the file is opened.

### `write(record: bytes)`

Writes a record to the file. May raise an error if it failed to do so.


## `array_record.python.array_record_module.ArrayRecordReader`

### `ArrayRecordReader(path: str, options: str)`

* `path` (str): File path to read from.
* `options` (str, optional): Comma-separated options string. Default ""

#### Options string format

The options string can contain the following comma-separated options:

* `readahead_buffer_size:N` - Number of bytes for read-ahead buffer size per
  thread (default 0)
* `max_parallelism: N` - Number of read-ahead threads.
* `index_storage_options:in_memory/offloaded` - Specifies to store the record
  index in memory or on disk (default: `in_memory`)

### `ok() -> bool`

Returns true when the reader object is having a healthy state.

### `close()`

Closes the file. May raise an error if it failed to do so.

### `is_open() -> bool`

Returns true when the file was opened.

### `num_records() -> int`

Returns the number of records in the file.

### `record_index() -> int`

Returns the current record index. This field is only relevant in the sequential
reading mode.

### `writer_options_string() -> str`

Returns the writer options string that was used when creating the ArrayRecord
file.

### `seek(index: int)`

Update the cursor to the specified index. Throws an error if the index was out
of bound.

### `read() -> bytes`

Reads a record and advance the cursor index by one. Throws an error if the
cursor reaches the end of the file.

### `read(indices: Sequence) -> Sequence[bytes]`

Reads the set of records specified by the input indices with an internal thread
pool. Throws an error if any of the index was out of bound.

### `read(start: int, end: int) -> Sequence[bytes]`

Reads the set of records by range with an internal thread pool. Throws an error
if the index was out of bound.

### `read_all() -> Sequence[bytes]`

Reads all records with an internal thread pool. Throws an error if the index was
out of bound.

## `array_record.python.array_record_data_source.ArrayRecordDataSource`

### `ArrayRecordDataSource(paths: Sequence[str], reader_options: str)`

* `paths` (Sequence[str]): File paths to read from.
* `options` (str, optional): Comma-separated options string. Default "". See
  `ArrayRecordReader` constructor options for details.

### `__len__() -> int`

Returns the number of records of all the array record files specified in the
constructor.

```python
from array_record.python import array_record_data_source
ds = array_record_data_source.ArrayRecordDataSource(glob.glob("output.array_record*"))
len(ds)
```

### `__iter__() -> Iterator[bytes]`

Iterator interface for data access.

```python
from array_record.python import array_record_data_source
ds = array_record_data_source.ArrayRecordDataSource(glob.glob("output.array_record*"))
it = iter(ds)
record = next(it)
```

### `__getitem__(index: int) -> bytes`

Reads a record at the specified index.

```python
from array_record.python import array_record_data_source
ds = array_record_data_source.ArrayRecordDataSource(glob.glob("output.array_record*"))
ds[idx]
```

### `__getitems__(indices: Sequence[int]) -> Sequence[bytes]`

Reads a set of records of the specified indices.

```python
from array_record.python import array_record_data_source
ds = array_record_data_source.ArrayRecordDataSource(glob.glob("output.array_record*"))
ds.__getitems__(indices)
```
