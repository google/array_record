# Performance Guide

This guide covers performance optimization strategies for ArrayRecord, including configuration options, access patterns, and benchmarking techniques.

## Understanding ArrayRecord Performance

ArrayRecord's performance characteristics depend on several factors:

- **Access Pattern**: Sequential vs. random access
- **Group Size**: Number of records per compressed chunk
- **Compression Algorithm**: Trade-off between size and speed
- **Parallelism**: Number of concurrent operations
- **Buffer Sizes**: Memory usage vs. performance trade-offs

## Configuration for Different Use Cases

### Writer Configuration

The writer configuration significantly impacts both file size and read performance. Choose the right settings based on your access patterns:

#### Maximum Random Access Performance

Use `group_size:1` for applications requiring ultra-fast random access:

```python
from array_record.python import array_record_module

# Ultra-fast random access (larger file size)
writer = array_record_module.ArrayRecordWriter(
    'random_access.array_record', 
    'group_size:1'
)

# Each record is stored individually for instant access
for i in range(10000):
    data = f"Record {i}".encode('utf-8')
    writer.write(data)
writer.close()

# Benefits:
# - Zero decompression overhead for random access
# - Constant-time record retrieval O(1)
# - Ideal for ML training with random sampling
# 
# Trade-offs:
# - Larger file size (no compression grouping)
# - Higher memory usage for index
```

#### Balanced Performance

Use moderate group sizes for balanced performance:

```python
# Balanced performance and compression
writer = array_record_module.ArrayRecordWriter(
    'balanced.array_record', 
    'group_size:100,brotli:6'
)

# Benefits:
# - Good compression ratio
# - Reasonable random access performance
# - Lower memory usage
```

#### Maximum Compression

Use large group sizes for storage-optimized scenarios:

```python
# Maximum compression (slower random access)
writer = array_record_module.ArrayRecordWriter(
    'compressed.array_record', 
    'group_size:1000,brotli:9'
)

# Benefits:
# - Smallest file size
# - Best for archival storage
# - Efficient for sequential access
#
# Trade-offs:
# - Slower random access (must decompress groups)
# - Higher CPU usage during reads
```

#### Configuration Comparison

| Configuration | File Size | Random Access | Sequential | Use Case |
|---------------|-----------|---------------|------------|----------|
| `group_size:1` | Largest | Fastest | Fast | ML training, real-time |
| `group_size:100,brotli:6` | Medium | Good | Fast | General purpose |
| `group_size:1000,brotli:9` | Smallest | Slower | Fastest | Archival, batch processing |

### Sequential Access (Default)

Optimized for reading records in order:

```python
from array_record.python import array_record_data_source

# Default settings are optimized for sequential access
reader_options = {
    'readahead_buffer_size': '16MB',  # Large buffer for prefetching
    'max_parallelism': 'auto',       # Use available CPU cores
}

data_source = array_record_data_source.ArrayRecordDataSource(
    'sequential_data.array_record',
    reader_options=reader_options
)

# Efficient sequential iteration
for i in range(len(data_source)):
    record = data_source[i]
    # Process record...
```

### Random Access

Optimized for jumping to arbitrary record positions:

```python
# Random access optimization
reader_options = {
    'readahead_buffer_size': '0',    # Disable prefetching
    'max_parallelism': '0',          # Disable parallel readahead
    'index_storage_option': 'in_memory'  # Keep index in memory
}

data_source = array_record_data_source.ArrayRecordDataSource(
    'random_data.array_record',
    reader_options=reader_options
)

# Efficient random access
import random
indices = random.sample(range(len(data_source)), 1000)
for idx in indices:
    record = data_source[idx]
    # Process record...
```

### Batch Processing

Optimized for processing multiple records at once:

```python
# Batch access with moderate buffering
reader_options = {
    'readahead_buffer_size': '4MB',
    'max_parallelism': '2',
}

data_source = array_record_data_source.ArrayRecordDataSource(
    'batch_data.array_record',
    reader_options=reader_options
)

# Process in batches
batch_size = 1000
for start in range(0, len(data_source), batch_size):
    end = min(start + batch_size, len(data_source))
    indices = list(range(start, end))
    batch = data_source[indices]
    
    # Process batch efficiently
    for record in batch:
        # Process record...
        pass
```

## Writing Performance

### High Throughput Writing

```python
from array_record.python import array_record_module

# High throughput configuration
writer = array_record_module.ArrayRecordWriter(
    'high_throughput.array_record',
    'group_size:10000,max_parallelism:8,snappy'
)

# Large group size for better compression and fewer I/O operations
# More parallel encoders for CPU utilization
# Snappy for fast compression
```

### Balanced Performance

```python
# Balanced configuration
writer = array_record_module.ArrayRecordWriter(
    'balanced.array_record',
    'group_size:2000,max_parallelism:4,brotli:6'
)

# Medium group size balances compression and access speed
# Moderate parallelism
# Default Brotli compression
```

### Low Latency Writing

```python
# Low latency configuration
writer = array_record_module.ArrayRecordWriter(
    'low_latency.array_record',
    'group_size:100,max_parallelism:1,snappy'
)

# Small group size for immediate availability
# Single thread to avoid coordination overhead
# Fast compression
```

## Compression Performance Comparison

### Compression Algorithms

| Algorithm | Compression Ratio | Compression Speed | Decompression Speed | Use Case |
|-----------|------------------|-------------------|---------------------|----------|
| Uncompressed | 1.0x | Fastest | Fastest | Maximum speed, unlimited storage |
| Snappy | 2-4x | Very Fast | Very Fast | High throughput, moderate storage |
| Brotli (level 1) | 3-5x | Fast | Fast | Balanced performance |
| Brotli (level 6) | 4-6x | Medium | Fast | Default choice |
| Brotli (level 11) | 5-7x | Slow | Fast | Maximum compression |
| Zstd (level 1) | 3-5x | Very Fast | Very Fast | Alternative to Snappy |
| Zstd (level 3) | 4-6x | Fast | Fast | Alternative to Brotli |
| Zstd (level 22) | 6-8x | Very Slow | Fast | Maximum compression |

### Benchmark Example

```python
import time
import os
from array_record.python import array_record_module, array_record_data_source

def benchmark_compression(data, algorithms):
    """Benchmark different compression algorithms."""
    results = {}
    
    for name, options in algorithms.items():
        filename = f'benchmark_{name}.array_record'
        
        # Write benchmark
        start_time = time.time()
        writer = array_record_module.ArrayRecordWriter(filename, options)
        
        for item in data:
            writer.write(item)
        
        writer.close()
        write_time = time.time() - start_time
        
        # File size
        file_size = os.path.getsize(filename)
        
        # Read benchmark
        start_time = time.time()
        with array_record_data_source.ArrayRecordDataSource(filename) as ds:
            for i in range(len(ds)):
                _ = ds[i]
        read_time = time.time() - start_time
        
        results[name] = {
            'write_time': write_time,
            'read_time': read_time,
            'file_size_mb': file_size / (1024 * 1024),
        }
        
        # Cleanup
        os.remove(filename)
    
    return results

# Test data
test_data = [f"Test record {i}: " + "x" * 100 for i in range(10000)]

algorithms = {
    'uncompressed': 'uncompressed',
    'snappy': 'snappy',
    'brotli_1': 'brotli:1',
    'brotli_6': 'brotli:6',
    'brotli_11': 'brotli:11',
    'zstd_1': 'zstd:1',
    'zstd_3': 'zstd:3',
    'zstd_22': 'zstd:22',
}

results = benchmark_compression([data.encode() for data in test_data], algorithms)

# Print results
print(f"{'Algorithm':<15} {'Write (s)':<10} {'Read (s)':<9} {'Size (MB)':<10}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['write_time']:<10.2f} "
          f"{metrics['read_time']:<9.2f} {metrics['file_size_mb']:<10.2f}")
```

## Memory Usage Optimization

### Low Memory Configuration

```python
# Minimize memory usage
reader_options = {
    'readahead_buffer_size': '1MB',     # Small buffer
    'max_parallelism': '1',             # Single thread
    'index_storage_option': 'offloaded' # Store index on disk
}

writer_options = 'group_size:500,max_parallelism:1'

# Use context managers to ensure cleanup
with array_record_data_source.ArrayRecordDataSource(
    'data.array_record', reader_options=reader_options
) as ds:
    # Process data with minimal memory footprint
    for i in range(len(ds)):
        record = ds[i]
        # Process immediately and don't store
        process_record_immediately(record)
```

### High Memory Configuration

```python
# Use more memory for better performance
reader_options = {
    'readahead_buffer_size': '128MB',   # Large buffer
    'max_parallelism': '8',             # Many threads
    'index_storage_option': 'in_memory' # Keep index in memory
}

writer_options = 'group_size:5000,max_parallelism:8'

# Batch processing with large buffers
batch_size = 10000
with array_record_data_source.ArrayRecordDataSource(
    'data.array_record', reader_options=reader_options
) as ds:
    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        batch = ds[list(range(start, end))]
        process_batch(batch)
```

## Parallel Processing

### Multi-threaded Reading

```python
import concurrent.futures
from array_record.python import array_record_data_source

def process_chunk(args):
    """Process a chunk of records."""
    filename, start_idx, end_idx = args
    
    # Each thread gets its own data source
    with array_record_data_source.ArrayRecordDataSource(filename) as ds:
        results = []
        for i in range(start_idx, end_idx):
            record = ds[i]
            # Process record
            result = len(record)  # Example processing
            results.append(result)
        return results

def parallel_process_file(filename, num_threads=4):
    """Process file using multiple threads."""
    
    # Get file size
    with array_record_data_source.ArrayRecordDataSource(filename) as ds:
        total_records = len(ds)
    
    # Divide work among threads
    chunk_size = total_records // num_threads
    chunks = []
    
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_records)
        if i == num_threads - 1:  # Last chunk gets remainder
            end_idx = total_records
        chunks.append((filename, start_idx, end_idx))
    
    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    all_results = []
    for chunk_results in results:
        all_results.extend(chunk_results)
    
    return all_results

# Example usage
results = parallel_process_file('large_file.array_record', num_threads=8)
print(f"Processed {len(results)} records")
```

### Asynchronous Processing

```python
import asyncio
from array_record.python import array_record_data_source

async def async_process_records(filename, batch_size=1000):
    """Asynchronously process records in batches."""
    
    def process_batch_sync(batch):
        # CPU-intensive processing
        return [len(record) for record in batch]
    
    with array_record_data_source.ArrayRecordDataSource(filename) as ds:
        total_records = len(ds)
        
        tasks = []
        for start in range(0, total_records, batch_size):
            end = min(start + batch_size, total_records)
            indices = list(range(start, end))
            batch = ds[indices]
            
            # Process batch asynchronously
            task = asyncio.get_event_loop().run_in_executor(
                None, process_batch_sync, batch
            )
            tasks.append(task)
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for batch_results in results:
            all_results.extend(batch_results)
        
        return all_results

# Run async processing
async def main():
    results = await async_process_records('large_file.array_record')
    print(f"Processed {len(results)} records asynchronously")

# asyncio.run(main())  # Uncomment to run
```

## Benchmarking Tools

### Performance Measurement

```python
import time
import psutil
import os
from contextlib import contextmanager
from array_record.python import array_record_data_source, array_record_module

@contextmanager
def measure_performance():
    """Context manager to measure performance metrics."""
    process = psutil.Process()
    
    # Initial measurements
    start_time = time.time()
    start_cpu = process.cpu_percent()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    # Final measurements
    end_time = time.time()
    end_cpu = process.cpu_percent()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"CPU usage: {end_cpu:.1f}%")
    print(f"Memory usage: {end_memory:.1f} MB (Δ{end_memory - start_memory:+.1f} MB)")

def benchmark_access_patterns(filename):
    """Benchmark different access patterns."""
    
    with array_record_data_source.ArrayRecordDataSource(filename) as ds:
        total_records = len(ds)
        print(f"Benchmarking file with {total_records} records")
    
    # Sequential access
    print("\n1. Sequential Access:")
    with measure_performance():
        with array_record_data_source.ArrayRecordDataSource(filename) as ds:
            for i in range(min(1000, total_records)):
                _ = ds[i]
    
    # Random access
    print("\n2. Random Access:")
    reader_options = {'readahead_buffer_size': '0', 'max_parallelism': '0'}
    with measure_performance():
        with array_record_data_source.ArrayRecordDataSource(
            filename, reader_options=reader_options
        ) as ds:
            import random
            indices = random.sample(range(total_records), min(1000, total_records))
            for idx in indices:
                _ = ds[idx]
    
    # Batch access
    print("\n3. Batch Access:")
    with measure_performance():
        with array_record_data_source.ArrayRecordDataSource(filename) as ds:
            batch_size = 100
            for start in range(0, min(1000, total_records), batch_size):
                end = min(start + batch_size, total_records)
                indices = list(range(start, end))
                _ = ds[indices]

# Example usage
# Create test file first
test_file = 'performance_test.array_record'
if not os.path.exists(test_file):
    writer = array_record_module.ArrayRecordWriter(test_file)
    for i in range(10000):
        writer.write(f"Test record {i}: " + "x" * 200)
    writer.close()

benchmark_access_patterns(test_file)
```

## Performance Best Practices

### General Guidelines

1. **Choose the right group size**:
   - Small (100-500): Fast random access, larger files
   - Medium (1000-5000): Balanced performance
   - Large (10000+): Best compression, sequential access only

2. **Select appropriate compression**:
   - Snappy: Maximum speed, moderate compression
   - Brotli 1-3: Fast with good compression
   - Brotli 6: Default balanced choice
   - Brotli 9-11: Maximum compression, slower

3. **Configure parallelism**:
   - Sequential access: Use available CPU cores
   - Random access: Disable parallelism
   - Mixed access: Use moderate parallelism (2-4 threads)

4. **Optimize buffer sizes**:
   - Sequential: Large buffers (16-64MB)
   - Random: No buffering (0)
   - Batch: Medium buffers (4-16MB)

### Platform-Specific Optimizations

#### Linux
```python
# Use larger buffer sizes on Linux
reader_options = {
    'readahead_buffer_size': '64MB',
    'max_parallelism': '8'
}
```

#### macOS
```python
# More conservative settings for macOS
reader_options = {
    'readahead_buffer_size': '16MB',
    'max_parallelism': '4'
}
```

### Cloud Storage Optimizations

When working with cloud storage (GCS, S3):

```python
# Optimize for cloud storage latency
reader_options = {
    'readahead_buffer_size': '32MB',  # Larger buffers for network latency
    'max_parallelism': '4',           # Moderate parallelism
}

# Use larger group sizes for cloud storage
writer_options = 'group_size:5000,brotli:6,max_parallelism:4'
```

### Memory-Constrained Environments

```python
# Minimal memory configuration
reader_options = {
    'readahead_buffer_size': '512KB',
    'max_parallelism': '1',
    'index_storage_option': 'offloaded'
}

writer_options = 'group_size:200,snappy,max_parallelism:1'
```

## Troubleshooting Performance Issues

### Common Issues and Solutions

1. **Slow random access**:
   - Disable readahead: `'readahead_buffer_size': '0'`
   - Disable parallelism: `'max_parallelism': '0'`
   - Use smaller group sizes

2. **High memory usage**:
   - Reduce buffer sizes
   - Use offloaded index storage
   - Reduce parallelism

3. **Slow sequential access**:
   - Increase buffer sizes
   - Enable parallelism
   - Use larger group sizes

4. **Poor compression ratio**:
   - Increase group size
   - Use higher compression levels
   - Enable transposition for protocol buffers

### Profiling Tools

Use Python profiling tools to identify bottlenecks:

```python
import cProfile
import pstats

def profile_arrayrecord_operations():
    """Profile ArrayRecord operations."""
    
    def test_function():
        with array_record_data_source.ArrayRecordDataSource('test.array_record') as ds:
            for i in range(1000):
                _ = ds[i]
    
    # Profile the function
    pr = cProfile.Profile()
    pr.enable()
    test_function()
    pr.disable()
    
    # Print results
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

# profile_arrayrecord_operations()
```

By following these performance guidelines and using the provided benchmarking tools, you can optimize ArrayRecord for your specific use case and achieve maximum performance.
