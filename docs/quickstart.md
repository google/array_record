# Quick Start Guide

This guide will help you get started with ArrayRecord quickly. ArrayRecord is designed to be simple to use while providing high performance for both sequential and random access patterns.

## Basic Writing and Reading

### Writing Your First ArrayRecord File

```python
from array_record.python import array_record_module

# Create a writer with default settings
writer = array_record_module.ArrayRecordWriter('my_data.array_record')

# Write some records
for i in range(1000):
    data = f"Record number {i}".encode('utf-8')
    writer.write(data)

# Always close the writer to finalize the file
writer.close()
```

### Writer Configuration Options

ArrayRecord allows you to configure the writer for different performance characteristics:

```python
from array_record.python import array_record_module

# Maximum random access performance
# group_size:1 stores each record individually for fastest random access
writer = array_record_module.ArrayRecordWriter(
    'random_access.array_record', 
    'group_size:1'
)
for i in range(1000):
    data = f"Record {i}".encode('utf-8')
    writer.write(data)
writer.close()

# High compression configuration
# group_size:1000 groups records together for better compression
writer = array_record_module.ArrayRecordWriter(
    'compressed.array_record', 
    'group_size:1000,brotli:9'
)
for i in range(1000):
    data = f"Record {i}".encode('utf-8')
    writer.write(data)
writer.close()
```

**Key Configuration Parameters:**

- **`group_size:N`**: Number of records per group
  - `group_size:1` - Maximum random access speed (larger file size)
  - `group_size:100` - Balanced performance (recommended default)
  - `group_size:1000` - Better compression (slower random access)

- **Compression Options:**
  - `brotli:1-11` - Brotli compression (higher = better compression)
  - `zstd:1-22` - Zstandard compression (fast compression/decompression)
  - `snappy` - Very fast compression with moderate ratio
  - `uncompressed` - No compression (fastest write/read)

## Working with tf.train.Example

ArrayRecord integrates seamlessly with TensorFlow's `tf.train.Example` format for structured ML data:

```python
import tensorflow as tf
import grain
import dataclasses
from array_record.python import array_record_module, array_record_data_source

# Writing tf.train.Example records
def create_tf_example(text_data, is_tokens=False):
    if is_tokens:
        # Integer tokens
        features = {'text': tf.train.Feature(int64_list=tf.train.Int64List(value=text_data))}
    else:
        # String text
        features = {'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text_data.encode('utf-8')]))}
    
    return tf.train.Example(features=tf.train.Features(feature=features))

# Write examples to ArrayRecord
writer = array_record_module.ArrayRecordWriter('tf_examples.array_record', 'group_size:1')
for text in ["Sample text", "Another example"]:
    example = create_tf_example(text)
    writer.write(example.SerializeToString())  # Already bytes, no .encode() needed
writer.close()

# MaxText-style parsing with Grain
@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
    """Parse tf.train.Example records (from MaxText)."""
    def __init__(self, data_columns, tokenize):
        self.data_columns = data_columns
        self.dtype = tf.string if tokenize else tf.int64

    def map(self, element):
        return tf.io.parse_example(
            element,
            {col: tf.io.FixedLenSequenceFeature([], dtype=self.dtype, allow_missing=True) 
             for col in self.data_columns}
        )

@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
    """Normalize features (from MaxText)."""
    def __init__(self, column_names, tokenize):
        self.column_names = column_names
        self.tokenize = tokenize

    def map(self, element):
        if self.tokenize:
            return {col: element[col].numpy()[0].decode() for col in self.column_names}
        else:
            return {col: element[col].numpy() for col in self.column_names}

# Create MaxText-style training pipeline
data_source = array_record_data_source.ArrayRecordDataSource('tf_examples.array_record')
dataset = (
    grain.MapDataset.source(data_source)
    .map(ParseFeatures(['text'], tokenize=True))      # Parse tf.train.Example
    .map(NormalizeFeatures(['text'], tokenize=True))  # Normalize features
    .batch(batch_size=32)
    .shuffle(seed=42)
)
```

**Benefits**: Standard TensorFlow format + ArrayRecord performance + MaxText compatibility for production LLM training.

### Reading ArrayRecord Files

```python
from array_record.python import array_record_data_source

# Create a data source for reading
data_source = array_record_data_source.ArrayRecordDataSource('my_data.array_record')

# Get the total number of records
print(f"Total records: {len(data_source)}")

# Read the first record
first_record = data_source[0]
print(f"First record: {first_record.decode('utf-8')}")

# Read multiple records by index
batch = data_source[[0, 10, 100, 500]]
for i, record in enumerate(batch):
    print(f"Record: {record.decode('utf-8')}")
```

## Working with Binary Data

ArrayRecord excels at storing binary data like serialized protocol buffers:

```python
import json
from array_record.python import array_record_module

# Writing JSON data as bytes
writer = array_record_module.ArrayRecordWriter('json_data.array_record')

data_objects = [
    {"id": 1, "name": "Alice", "score": 95.5},
    {"id": 2, "name": "Bob", "score": 87.2},
    {"id": 3, "name": "Charlie", "score": 92.8},
]

for obj in data_objects:
    json_bytes = json.dumps(obj).encode('utf-8')
    writer.write(json_bytes)

writer.close()

# Reading JSON data back
from array_record.python import array_record_data_source

data_source = array_record_data_source.ArrayRecordDataSource('json_data.array_record')

for i in range(len(data_source)):
    json_bytes = data_source[i]
    obj = json.loads(json_bytes.decode('utf-8'))
    print(f"Object {i}: {obj}")
```

## Configuration Options

### Writer Options

ArrayRecord provides several configuration options for optimization:

```python
from array_record.python import array_record_module

# Configure writer options
options = {
    'group_size': '1000',      # Records per chunk (affects compression vs random access trade-off)
    'compression': 'brotli:6', # Compression algorithm and level
}

writer = array_record_module.ArrayRecordWriter(
    'optimized.array_record',
    ','.join([f'{k}:{v}' for k, v in options.items()])
)

# Write data...
writer.close()
```

### Reader Options

```python
from array_record.python import array_record_data_source

# Configure reader options for different access patterns
reader_options = {
    'readahead_buffer_size': '0',  # Disable readahead for pure random access
    'max_parallelism': '4',        # Number of parallel threads
}

data_source = array_record_data_source.ArrayRecordDataSource(
    'optimized.array_record',
    reader_options=reader_options
)
```

## Performance Tips

### Sequential Access

For sequential reading, use the default settings:

```python
# Default settings are optimized for sequential access
data_source = array_record_data_source.ArrayRecordDataSource('data.array_record')

# Iterate through all records
for i in range(len(data_source)):
    record = data_source[i]
    # Process record...
```

### Random Access

For random access, disable readahead:

```python
# Optimize for random access
reader_options = {
    'readahead_buffer_size': '0',
    'max_parallelism': '0',
}

data_source = array_record_data_source.ArrayRecordDataSource(
    'data.array_record',
    reader_options=reader_options
)

# Random access is now optimized
import random
indices = random.sample(range(len(data_source)), 100)
batch = data_source[indices]
```

### Batch Processing

Read multiple records at once for better performance:

```python
data_source = array_record_data_source.ArrayRecordDataSource('data.array_record')

# Process in batches
batch_size = 100
total_records = len(data_source)

for start in range(0, total_records, batch_size):
    end = min(start + batch_size, total_records)
    indices = list(range(start, end))
    batch = data_source[indices]
    
    # Process batch...
    for record in batch:
        # Process individual record...
        pass
```

## Context Manager Usage

Use context managers for automatic resource cleanup:

```python
from array_record.python import array_record_data_source

# Automatically handles cleanup
with array_record_data_source.ArrayRecordDataSource('data.array_record') as data_source:
    # Read data...
    for i in range(min(10, len(data_source))):
        record = data_source[i]
        print(f"Record {i}: {len(record)} bytes")
# File is automatically closed
```

## Error Handling

Always handle potential errors:

```python
from array_record.python import array_record_module, array_record_data_source

try:
    # Writing
    writer = array_record_module.ArrayRecordWriter('output.array_record')
    writer.write(b'test data')
    writer.close()
    
    # Reading
    data_source = array_record_data_source.ArrayRecordDataSource('output.array_record')
    record = data_source[0]
    print(f"Successfully read: {record}")
    
except Exception as e:
    print(f"Error: {e}")
```

## Next Steps

- Learn about [Core Concepts](core_concepts.md) for deeper understanding
- Explore [Python API Reference](python_reference.rst) for complete API documentation
- Check out [Apache Beam Integration](beam_integration.md) for large-scale processing
- See [Examples](examples.md) for real-world use cases
