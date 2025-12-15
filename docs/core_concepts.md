# Core Concepts

This document explains the fundamental concepts behind ArrayRecord and how to use them effectively.

## What is ArrayRecord?

ArrayRecord is a file format designed for high-performance storage and retrieval of sequential data. It's built on top of [Riegeli](https://github.com/google/riegeli) and provides:

- **Parallel I/O**: Multiple threads can read different parts of the file simultaneously
- **Random Access**: Jump to any record by index without scanning the entire file
- **Compression**: Multiple compression algorithms with configurable levels
- **Chunked Storage**: Data is organized in chunks for optimal compression and access patterns

## File Structure

ArrayRecord files are organized hierarchically:

```
┌─────────────────────┐
│    User Data        │
│  Riegeli Chunk 1    │
├─────────────────────┤
│    User Data        │
│  Riegeli Chunk 2    │
├─────────────────────┤
│       ...           │
├─────────────────────┤
│    User Data        │
│  Riegeli Chunk N    │
├─────────────────────┤
│   Footer Chunk      │
│  (Index Data)       │
├─────────────────────┤
│   Postscript        │
│ (File Metadata)     │
└─────────────────────┘
```

### Key Components

1. **User Data Chunks**: Contain your actual records, compressed using the specified algorithm
2. **Footer Chunk**: Contains index information for random access
3. **Postscript**: File metadata and chunk offsets (fits in 64KB)

## Records and Chunks

### Records

A record is the basic unit of data in ArrayRecord. Records can be:
- Raw bytes (any binary data)
- Protocol buffer messages
- Serialized objects
- Text data (encoded as bytes)

### Chunks (Groups)

Records are organized into chunks (also called "groups") before compression. The `group_size` parameter controls how many records are packed into each chunk.

**Trade-offs:**
- **Larger group_size**: Better compression ratio, but slower random access
- **Smaller group_size**: Faster random access, but less compression

```python
# High compression, slower random access
writer = ArrayRecordWriter('file.array_record', 'group_size:10000')

# Fast random access, less compression
writer = ArrayRecordWriter('file.array_record', 'group_size:100')
```

## Compression Options

ArrayRecord supports multiple compression algorithms:

### Brotli (Default)
- Best overall compression ratio
- Good balance of speed and size
- Levels 0-11 (default: 6)

```python
writer = ArrayRecordWriter('file.array_record', 'brotli:9')  # High compression
```

### Zstd
- Very fast compression/decompression
- Good compression ratio
- Levels -131072 to 22 (default: 3)

```python
writer = ArrayRecordWriter('file.array_record', 'zstd:1')  # Fast compression
```

### Snappy
- Extremely fast compression/decompression
- Lower compression ratio
- No compression levels

```python
writer = ArrayRecordWriter('file.array_record', 'snappy')
```

### Uncompressed
- No compression overhead
- Largest file size
- Fastest access

```python
writer = ArrayRecordWriter('file.array_record', 'uncompressed')
```

## Access Patterns

### Sequential Access

Optimized by default with readahead buffering:

```python
# Default settings optimize for sequential access
data_source = ArrayRecordDataSource('file.array_record')

for i in range(len(data_source)):
    record = data_source[i]
    # Process record sequentially
```

### Random Access

Disable readahead for pure random access:

```python
reader_options = {
    'readahead_buffer_size': '0',
    'max_parallelism': '0',
}

data_source = ArrayRecordDataSource('file.array_record', reader_options=reader_options)

# Now random access is optimized
import random
random_index = random.randint(0, len(data_source) - 1)
record = data_source[random_index]
```

### Batch Access

Read multiple records efficiently:

```python
# Read specific indices
indices = [10, 100, 1000, 5000]
batch = data_source[indices]

# Read a range
range_batch = data_source[100:200]  # If supported by implementation
```

## Parallel Processing

ArrayRecord supports parallel operations:

### Parallel Writing
```python
# Configure parallel writing
options = 'group_size:1000,max_parallelism:4'
writer = ArrayRecordWriter('file.array_record', options)
```

### Parallel Reading
```python
# Configure parallel reading
reader_options = {
    'max_parallelism': '4',
    'readahead_buffer_size': '16M',
}
data_source = ArrayRecordDataSource('file.array_record', reader_options=reader_options)
```

## Performance Considerations

### Group Size Selection

Choose group size based on your access pattern:

```python
# For mostly sequential access with some random access
writer = ArrayRecordWriter('file.array_record', 'group_size:1000')

# For heavy random access
writer = ArrayRecordWriter('file.array_record', 'group_size:100')

# For maximum compression (sequential only)
writer = ArrayRecordWriter('file.array_record', 'group_size:10000')
```

### Memory Usage

Control memory usage with buffer settings:

```python
# Low memory usage
reader_options = {
    'readahead_buffer_size': '1M',
    'max_parallelism': '1',
}

# High performance (more memory)
reader_options = {
    'readahead_buffer_size': '64M',
    'max_parallelism': '8',
}
```

## Data Types and Serialization

### Binary Data
```python
# Raw bytes
writer.write(b'binary data')

# Numpy arrays
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
writer.write(arr.tobytes())
```

### Protocol Buffers
```python
# Assuming you have a protobuf message
message = MyProtoMessage()
message.field = "value"
writer.write(message.SerializeToString())
```

### JSON Data
```python
import json
data = {"key": "value", "number": 42}
writer.write(json.dumps(data).encode('utf-8'))
```

## Best Practices

### Writing
1. Always call `close()` on writers
2. Choose appropriate group size for your access pattern
3. Use compression for storage efficiency
4. Consider using context managers for automatic cleanup

### Reading
1. Configure reader options based on access pattern
2. Use batch reading for better performance
3. Cache frequently accessed records
4. Use appropriate parallelism settings

### File Management
1. Use descriptive file extensions (e.g., `.array_record`)
2. Include metadata in filenames when helpful
3. Consider file size limits for your storage system
4. Plan for concurrent access if needed

## Integration with Other Systems

### Apache Beam
ArrayRecord integrates seamlessly with Apache Beam for large-scale data processing. See [Beam Integration](beam_integration.md) for details.

### Machine Learning Frameworks
ArrayRecord works well with:
- TensorFlow (via tf.data)
- JAX/Grain (native support)
- PyTorch (via custom datasets)

### Cloud Storage
ArrayRecord files can be stored on:
- Local filesystems
- Google Cloud Storage (via Beam)
- Amazon S3 (via appropriate readers)
- Network filesystems (NFS, etc.)
