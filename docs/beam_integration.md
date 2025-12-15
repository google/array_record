# Apache Beam Integration

ArrayRecord provides comprehensive Apache Beam integration for large-scale data processing and conversion workflows. This integration enables you to process ArrayRecord files in distributed Beam pipelines on various runners including Google Cloud Dataflow.

## Overview

The Beam integration provides:

- **PTransform for writing**: `WriteToArrayRecord` for disk-based output
- **DoFn for GCS**: `ConvertToArrayRecordGCS` for cloud storage output  
- **Pre-built pipelines**: Ready-to-use conversion utilities
- **Format conversion**: Seamless TFRecord to ArrayRecord conversion

## Installation

Install ArrayRecord with Beam support:

```bash
pip install array_record[beam]
```

This includes:
- Apache Beam with GCP support (>=2.53.0)
- Google Cloud Storage client library
- TensorFlow for TFRecord compatibility

## Quick Start

### Basic File Conversion

Convert TFRecord files to ArrayRecord format:

```python
from array_record.beam.pipelines import convert_tf_to_arrayrecord_disk
from apache_beam.options.pipeline_options import PipelineOptions

# Convert TFRecords to ArrayRecords on local disk
pipeline = convert_tf_to_arrayrecord_disk(
    num_shards=4,
    args=['--input', '/path/to/tfrecords/*', '--output', '/path/to/arrayrecords/output'],
    pipeline_options=PipelineOptions()
)

result = pipeline.run()
result.wait_until_finish()
```

### Cloud Storage Conversion

Convert files and upload to Google Cloud Storage:

```python
from array_record.beam.pipelines import convert_tf_to_arrayrecord_gcs
from apache_beam.options.pipeline_options import PipelineOptions

pipeline = convert_tf_to_arrayrecord_gcs(
    args=[
        '--input', 'gs://source-bucket/tfrecords/*',
        '--output', 'gs://dest-bucket/arrayrecords/'
    ],
    pipeline_options=PipelineOptions([
        '--runner=DataflowRunner',
        '--project=my-project',
        '--region=us-central1'
    ])
)

result = pipeline.run()
result.wait_until_finish()
```

## Core Components

### WriteToArrayRecord PTransform

For writing ArrayRecord files to disk-based filesystems:

```python
import apache_beam as beam
from array_record.beam.arrayrecordio import WriteToArrayRecord

with beam.Pipeline() as pipeline:
    # Create some data
    data = pipeline | beam.Create([
        b'record 1',
        b'record 2', 
        b'record 3'
    ])
    
    # Write to ArrayRecord files
    data | WriteToArrayRecord(
        file_path_prefix='/tmp/output',
        file_name_suffix='.array_record',
        num_shards=2
    )
```

**Important**: `WriteToArrayRecord` only works with local/disk-based paths, not cloud storage URLs.

### ConvertToArrayRecordGCS DoFn

For writing ArrayRecord files to Google Cloud Storage:

```python
import apache_beam as beam
from array_record.beam.dofns import ConvertToArrayRecordGCS

# Prepare data as (filename, records) tuples
file_data = [
    ('file1.tfrecord', [b'record1', b'record2']),
    ('file2.tfrecord', [b'record3', b'record4'])
]

with beam.Pipeline() as pipeline:
    data = pipeline | beam.Create(file_data)
    
    data | beam.ParDo(
        ConvertToArrayRecordGCS(),
        path='gs://my-bucket/arrayrecords/',
        file_path_suffix='.array_record'
    )
```

## Pre-built Pipelines

### Disk-based Conversion

```python
from array_record.beam.pipelines import convert_tf_to_arrayrecord_disk

# Convert with specific number of shards
pipeline = convert_tf_to_arrayrecord_disk(
    num_shards=10,
    args=['--input', 'gs://bucket/tfrecords/*', '--output', '/local/arrayrecords/output']
)
```

### Matching Shard Count

Convert while preserving the number of input files:

```python
from array_record.beam.pipelines import convert_tf_to_arrayrecord_disk_match_shards

# Output will have same number of files as input
pipeline = convert_tf_to_arrayrecord_disk_match_shards(
    args=['--input', '/path/to/tfrecords/*', '--output', '/path/to/arrayrecords/output']
)
```

### GCS Conversion

```python
from array_record.beam.pipelines import convert_tf_to_arrayrecord_gcs

pipeline = convert_tf_to_arrayrecord_gcs(
    overwrite_extension=True,  # Replace .tfrecord with .array_record
    args=[
        '--input', 'gs://input-bucket/tfrecords/*',
        '--output', 'gs://output-bucket/arrayrecords/'
    ]
)
```

## Command Line Usage

Run conversions directly from the command line:

```bash
# Local conversion
python -m array_record.beam.pipelines \
    --input /path/to/tfrecords/* \
    --output /path/to/arrayrecords/output \
    --num_shards 5

# GCS conversion with Dataflow
python -m array_record.beam.pipelines \
    --input gs://source-bucket/tfrecords/* \
    --output gs://dest-bucket/arrayrecords/ \
    --runner DataflowRunner \
    --project my-project \
    --region us-central1 \
    --temp_location gs://my-bucket/temp
```

## Google Cloud Dataflow

### Basic Dataflow Setup

```python
from apache_beam.options.pipeline_options import PipelineOptions
from array_record.beam.pipelines import convert_tf_to_arrayrecord_gcs

dataflow_options = PipelineOptions([
    '--runner=DataflowRunner',
    '--project=my-project',
    '--region=us-central1',
    '--temp_location=gs://my-bucket/temp',
    '--staging_location=gs://my-bucket/staging',
    '--max_num_workers=20',
    '--disk_size_gb=100'
])

pipeline = convert_tf_to_arrayrecord_gcs(
    args=[
        '--input', 'gs://large-dataset/tfrecords/*',
        '--output', 'gs://processed-data/arrayrecords/'
    ],
    pipeline_options=dataflow_options
)

result = pipeline.run()
result.wait_until_finish()
```

### Monitoring Dataflow Jobs

Monitor your conversion jobs through:
- [Google Cloud Console](https://console.cloud.google.com/dataflow)
- Beam metrics and logging
- Custom monitoring DoFns

```python
class MonitoringDoFn(beam.DoFn):
    def __init__(self):
        self.records_processed = Metrics.counter('conversion', 'records_processed')
    
    def process(self, element):
        self.records_processed.inc()
        yield element

# Add to your pipeline
data | beam.ParDo(MonitoringDoFn()) | ...
```

## Custom Pipelines

### Reading ArrayRecord Files

```python
import apache_beam as beam
from array_record.python import array_record_data_source

class ReadArrayRecordDoFn(beam.DoFn):
    def process(self, file_path):
        with array_record_data_source.ArrayRecordDataSource(file_path) as ds:
            for i in range(len(ds)):
                yield ds[i]

with beam.Pipeline() as pipeline:
    files = pipeline | beam.Create(['file1.array_record', 'file2.array_record'])
    records = files | beam.ParDo(ReadArrayRecordDoFn())
    
    # Process records further
    records | beam.Map(lambda x: len(x)) | beam.io.WriteToText('record_lengths.txt')
```

### Custom Conversion Logic

```python
import apache_beam as beam
from array_record.python import array_record_module
import tempfile
import os

class CustomArrayRecordWriterDoFn(beam.DoFn):
    def process(self, element):
        filename, records = element
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.array_record') as tmp:
            writer = array_record_module.ArrayRecordWriter(
                tmp.name, 
                'group_size:1000,brotli:9'  # Custom options
            )
            
            for record in records:
                # Apply custom transformation
                transformed = self.transform_record(record)
                writer.write(transformed)
            
            writer.close()
            
            # Yield the result
            yield (filename, tmp.name)
    
    def transform_record(self, record):
        # Custom record transformation logic
        return record.upper()

# Use in pipeline
with beam.Pipeline() as pipeline:
    file_data = pipeline | beam.Create([
        ('input1.txt', [b'hello', b'world']),
        ('input2.txt', [b'foo', b'bar'])
    ])
    
    transformed = file_data | beam.ParDo(CustomArrayRecordWriterDoFn())
```

## Performance Optimization

### Writer Configuration

Optimize ArrayRecord writer settings for your use case:

```python
# For high compression (slower)
high_compression_options = 'group_size:10000,brotli:11,max_parallelism:1'

# For fast writing (larger files)
fast_writing_options = 'group_size:1000,snappy,max_parallelism:8'

# Balanced
balanced_options = 'group_size:2000,brotli:6,max_parallelism:4'
```

### Dataflow Optimization

```python
dataflow_options = PipelineOptions([
    '--runner=DataflowRunner',
    '--max_num_workers=50',
    '--num_workers=10',
    '--worker_machine_type=n1-highmem-4',
    '--disk_size_gb=200',
    '--use_public_ips=false',  # For better network performance
    '--network=my-vpc',
    '--subnetwork=my-subnet'
])
```

### Batch Processing

Process files in batches for better resource utilization:

```python
class BatchProcessingDoFn(beam.DoFn):
    def process(self, element, batch_size=100):
        filename, records = element
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            yield self.process_batch(filename, batch)
    
    def process_batch(self, filename, batch):
        # Process batch of records
        pass
```

## Error Handling and Monitoring

### Robust Error Handling

```python
import logging
from apache_beam.transforms.util import Reshuffle

class RobustConversionDoFn(beam.DoFn):
    def process(self, element):
        try:
            filename, records = element
            
            # Conversion logic here
            result = self.convert_file(filename, records)
            
            yield beam.pvalue.TaggedOutput('success', result)
            
        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}")
            yield beam.pvalue.TaggedOutput('failed', (filename, str(e)))
    
    def convert_file(self, filename, records):
        # Your conversion logic
        pass

# Use with error handling
with beam.Pipeline() as pipeline:
    input_data = pipeline | beam.Create(file_data)
    
    results = input_data | beam.ParDo(RobustConversionDoFn()).with_outputs(
        'success', 'failed', main='success'
    )
    
    # Handle successful conversions
    results.success | beam.Map(lambda x: f"Converted: {x}")
    
    # Handle failures
    results.failed | beam.Map(lambda x: f"Failed: {x[0]} - {x[1]}")
```

### Progress Monitoring

```python
from apache_beam.metrics import Metrics

class MonitoredConversionDoFn(beam.DoFn):
    def __init__(self):
        self.files_processed = Metrics.counter('conversion', 'files_processed')
        self.records_written = Metrics.counter('conversion', 'records_written')
        self.bytes_written = Metrics.counter('conversion', 'bytes_written')
    
    def process(self, element):
        filename, records = element
        
        self.files_processed.inc()
        
        # Process file
        total_bytes = 0
        for record in records:
            # Write record
            total_bytes += len(record)
            self.records_written.inc()
        
        self.bytes_written.inc(total_bytes)
        
        yield f"Processed {filename}: {len(records)} records, {total_bytes} bytes"
```

## Best Practices

### File Organization

```python
# Use meaningful file patterns
input_pattern = 'gs://data-bucket/year=2024/month=*/day=*/tfrecords/*.tfrecord'
output_prefix = 'gs://processed-bucket/year=2024/arrayrecords/data'

# Include metadata in filenames
output_filename = f"{output_prefix}-{datetime.now().strftime('%Y%m%d')}"
```

### Resource Management

```python
# Use appropriate machine types
# For CPU-intensive compression: n1-highcpu-*
# For memory-intensive operations: n1-highmem-*
# For balanced workloads: n1-standard-*

worker_options = [
    '--worker_machine_type=n1-standard-4',
    '--disk_size_gb=100',
    '--max_num_workers=20'
]
```

### Testing

Test your pipelines locally before running on Dataflow:

```python
# Local testing
local_options = PipelineOptions(['--runner=DirectRunner'])

# Test with small dataset
test_pipeline = convert_tf_to_arrayrecord_disk(
    num_shards=1,
    args=['--input', 'test_data/*.tfrecord', '--output', 'test_output/'],
    pipeline_options=local_options
)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `array_record[beam]` is installed
2. **Permission errors**: Check GCS bucket permissions
3. **Out of disk space**: Increase worker disk size
4. **Memory errors**: Use appropriate machine types
5. **Slow performance**: Tune parallelism and batch sizes

### Debug Tips

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Add debug outputs
debug_data = input_data | beam.Map(lambda x: logging.info(f"Processing: {x}"))
```

### Performance Monitoring

Use Dataflow's built-in monitoring or add custom metrics:

```python
# Custom timing metrics
from apache_beam.metrics import Metrics
import time

class TimedConversionDoFn(beam.DoFn):
    def __init__(self):
        self.conversion_time = Metrics.distribution('conversion', 'time_ms')
    
    def process(self, element):
        start_time = time.time()
        
        # Conversion logic
        result = self.convert(element)
        
        end_time = time.time()
        self.conversion_time.update(int((end_time - start_time) * 1000))
        
        yield result
```
