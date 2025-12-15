Apache Beam Integration Reference
===================================

ArrayRecord provides comprehensive Apache Beam integration for large-scale data processing. This integration allows you to read from and write to ArrayRecord files in distributed Beam pipelines.


Overview
--------

The ArrayRecord Beam integration provides:

* **WriteToArrayRecord**: PTransform for writing ArrayRecord files to disk
* **ConvertToArrayRecordGCS**: DoFn for writing ArrayRecord files to Google Cloud Storage
* **Pipeline utilities**: Pre-built pipelines for common conversion tasks

Installation
------------

Install ArrayRecord with Beam support:

.. code-block:: bash

   pip install array_record[beam]

This includes Apache Beam with GCP support and Google Cloud Storage client libraries.

Core Components
---------------

WriteToArrayRecord
~~~~~~~~~~~~~~~~~~

A PTransform for writing data to ArrayRecord files on disk-based filesystems.

   A PTransform for writing data to ArrayRecord files on disk-based filesystems.

   **Important**: This sink only works with disk-based paths. It does not support cloud storage URLs (gs://, s3://, etc.) directly.

   **Parameters:**
   
   * ``file_path_prefix`` (str): Path prefix for output files
   * ``file_name_suffix`` (str, optional): Suffix for output files (default: "")
   * ``num_shards`` (int, optional): Number of output shards (default: 0 for auto)
   * ``shard_name_template`` (str, optional): Template for shard naming
   * ``coder`` (Coder, optional): Beam coder for encoding records
   * ``compression_type`` (str, optional): Compression type

   **Example:**

   .. code-block:: python

      import apache_beam as beam
      from array_record.beam.arrayrecordio import WriteToArrayRecord

      with beam.Pipeline() as pipeline:
          data = pipeline | beam.Create([b'record1', b'record2', b'record3'])
          data | WriteToArrayRecord(
              file_path_prefix='/tmp/output',
              file_name_suffix='.array_record',
              num_shards=2
          )

DoFn Components
~~~~~~~~~~~~~~~

The DoFn components provide custom processing functions for Beam pipelines.

ConvertToArrayRecordGCS
~~~~~~~~~~~~~~~~~~~~~~~

A DoFn that writes ArrayRecord files to Google Cloud Storage.

   A DoFn that writes ArrayRecord files to Google Cloud Storage. This DoFn processes
   tuples of (filename, records) and uploads the resulting ArrayRecord files to GCS.

   **Parameters:**
   
   * ``path`` (str): GCS bucket path prefix (e.g., "gs://my-bucket/path/")
   * ``write_dir`` (str, optional): Local temporary directory (default: "/tmp/")
   * ``file_path_suffix`` (str, optional): File suffix (default: ".arrayrecord")
   * ``overwrite_extension`` (bool, optional): Replace existing extension (default: False)

   **Example:**

   .. code-block:: python

      import apache_beam as beam
      from array_beam.beam.dofns import ConvertToArrayRecordGCS

      def read_tfrecord_with_filename(file_pattern):
          # Custom function to read TFRecords and return (filename, records) tuples
          pass

      with beam.Pipeline() as pipeline:
          file_data = pipeline | beam.Create([
              ('file1.tfrecord', [b'record1', b'record2']),
              ('file2.tfrecord', [b'record3', b'record4']),
          ])
          
          file_data | beam.ParDo(
              ConvertToArrayRecordGCS(),
              path='gs://my-bucket/arrayrecords/',
              file_path_suffix='.array_record'
          )

Pipeline Utilities
------------------

The pipelines module provides several ready-to-use pipeline functions for common data conversion tasks.

Pre-built Pipelines
~~~~~~~~~~~~~~~~~~~~

The pipelines module provides several ready-to-use pipeline functions:

convert_tf_to_arrayrecord_disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts TFRecord files to ArrayRecord files on disk with configurable sharding.

   Converts TFRecord files to ArrayRecord files on disk with configurable sharding.

   **Parameters:**
   
   * ``num_shards`` (int): Number of output shards
   * ``args`` (list): Command line arguments
   * ``pipeline_options`` (PipelineOptions): Beam pipeline options

   **Example:**

   .. code-block:: python

      from array_record.beam.pipelines import convert_tf_to_arrayrecord_disk
      from apache_beam.options.pipeline_options import PipelineOptions

      # Convert with 4 output shards
      pipeline = convert_tf_to_arrayrecord_disk(
          num_shards=4,
          args=['--input', 'gs://bucket/tfrecords/*', '--output', '/tmp/arrayrecords/output'],
          pipeline_options=PipelineOptions()
      )
      pipeline.run().wait_until_finish()

convert_tf_to_arrayrecord_gcs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts TFRecord files to ArrayRecord files on Google Cloud Storage.

   Converts TFRecord files to ArrayRecord files on Google Cloud Storage.

   **Parameters:**
   
   * ``overwrite_extension`` (bool): Whether to replace file extensions
   * ``file_path_suffix`` (str): Suffix for output files
   * ``args`` (list): Command line arguments
   * ``pipeline_options`` (PipelineOptions): Beam pipeline options

   **Example:**

   .. code-block:: python

      from array_record.beam.pipelines import convert_tf_to_arrayrecord_gcs

      pipeline = convert_tf_to_arrayrecord_gcs(
          overwrite_extension=True,
          file_path_suffix='.array_record',
          args=['--input', 'gs://input-bucket/tfrecords/*', 
                '--output', 'gs://output-bucket/arrayrecords/'],
          pipeline_options=PipelineOptions()
      )

convert_tf_to_arrayrecord_disk_match_shards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts TFRecord files to ArrayRecord files with matching number of shards.

   Converts TFRecord files to ArrayRecord files with matching number of shards.

   **Example:**

   .. code-block:: python

      from array_record.beam.pipelines import convert_tf_to_arrayrecord_disk_match_shards

      # Output will have same number of files as input
      pipeline = convert_tf_to_arrayrecord_disk_match_shards(
          args=['--input', '/path/to/tfrecords/*', '--output', '/path/to/arrayrecords/output']
      )

Command Line Usage
------------------

The pipeline utilities can be run from the command line:

.. code-block:: bash

   # Convert TFRecords to ArrayRecords on disk
   python -m array_record.beam.pipelines \
       --input gs://bucket/tfrecords/* \
       --output /tmp/arrayrecords/output \
       --num_shards 10

   # Convert to GCS
   python -m array_record.beam.pipelines \
       --input gs://input-bucket/tfrecords/* \
       --output gs://output-bucket/arrayrecords/ \
       --runner DataflowRunner \
       --project my-project \
       --region us-central1

Configuration Options
---------------------

Writer Configuration
~~~~~~~~~~~~~~~~~~~~

Configure ArrayRecord writer options in your pipelines:

.. code-block:: python

   from array_record.beam.arrayrecordio import _ArrayRecordSink

   # The sink uses 'group_size:1' by default
   # You can modify this by subclassing _ArrayRecordSink

Reader Configuration
~~~~~~~~~~~~~~~~~~~~

When reading ArrayRecord files in Beam pipelines, you can use the standard
ArrayRecord API within your DoFns:

.. code-block:: python

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

Google Cloud Dataflow
----------------------

ArrayRecord Beam integration works with Google Cloud Dataflow:

.. code-block:: python

   from apache_beam.options.pipeline_options import PipelineOptions

   dataflow_options = PipelineOptions([
       '--runner=DataflowRunner',
       '--project=my-project',
       '--region=us-central1',
       '--temp_location=gs://my-bucket/temp',
       '--staging_location=gs://my-bucket/staging',
   ])

   pipeline = convert_tf_to_arrayrecord_gcs(
       args=['--input', 'gs://input/tfrecords/*', '--output', 'gs://output/arrayrecords/'],
       pipeline_options=dataflow_options
   )

Best Practices
--------------

1. **File Size Management**: Use appropriate sharding to avoid very large files

2. **Temporary Storage**: Ensure sufficient disk space for temporary files when using GCS DoFn

3. **Error Handling**: Implement proper error handling in custom DoFns

4. **Resource Management**: Use context managers for file operations

5. **Monitoring**: Monitor Dataflow jobs through the Google Cloud Console

Example: Complete Conversion Pipeline
-------------------------------------

Here's a complete example of converting TFRecord files to ArrayRecord:

.. code-block:: python

   import apache_beam as beam
   from apache_beam.options.pipeline_options import PipelineOptions
   from array_record.beam.pipelines import convert_tf_to_arrayrecord_gcs

   def main():
       pipeline_options = PipelineOptions([
           '--runner=DataflowRunner',
           '--project=my-project',
           '--region=us-central1',
           '--temp_location=gs://my-bucket/temp',
           '--max_num_workers=10',
       ])

       pipeline = convert_tf_to_arrayrecord_gcs(
           overwrite_extension=True,
           file_path_suffix='.array_record',
           args=[
               '--input', 'gs://source-bucket/tfrecords/*.tfrecord',
               '--output', 'gs://dest-bucket/arrayrecords/'
           ],
           pipeline_options=pipeline_options
       )

       result = pipeline.run()
       result.wait_until_finish()

   if __name__ == '__main__':
       main()

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **"Module not found" errors**: Ensure you installed with ``pip install array_record[beam]``

2. **GCS permission errors**: Check that your service account has proper GCS permissions

3. **Disk space errors**: Increase disk size for Dataflow workers or use smaller batch sizes

4. **Memory errors**: Reduce parallelism or increase worker memory

Performance Tips
~~~~~~~~~~~~~~~~

1. Use appropriate worker machine types for your data size
2. Tune the number of workers based on your input data
3. Use regional persistent disks for better I/O performance
4. Monitor resource usage through Dataflow monitoring
