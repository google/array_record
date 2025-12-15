Python API Reference
====================

This section documents the Python API for ArrayRecord. The Python interface provides high-level access to ArrayRecord functionality through two main modules.

Core Module
-----------

The core Python module provides the fundamental ArrayRecord functionality through C++ bindings.

ArrayRecordWriter
~~~~~~~~~~~~~~~~~

The ArrayRecordWriter class is used to create ArrayRecord files with various compression options and parallel writing capabilities.

   The ArrayRecordWriter class is used to create ArrayRecord files. It supports various compression
   options and parallel writing capabilities.

   **Constructor Parameters:**
   
   * ``path`` (str): File path where the ArrayRecord will be written
   * ``options`` (str, optional): Comma-separated options string. Default: ""

   **Options String Format:**
   
   The options string can contain the following comma-separated options:

   * ``group_size:N`` - Number of records per chunk (default: 65536)
   * ``max_parallelism:N`` - Maximum parallel threads (default: auto)
   * ``saturation_delay_ms:N`` - Delay when queue is saturated (default: 10)
   * ``uncompressed`` - Disable compression
   * ``brotli:N`` - Use Brotli compression with level N (0-11, default: 6)
   * ``zstd:N`` - Use Zstd compression with level N (-131072 to 22, default: 3)
   * ``snappy`` - Use Snappy compression
   * ``transpose:true/false`` - Enable/disable transposition for proto messages
   * ``transpose_bucket_size:N`` - Bucket size for transposition
   * ``window_log:N`` - LZ77 window size (10-31)
   * ``pad_to_block_boundary:true/false`` - Pad chunks to 64KB boundaries

   **Example:**

   .. code-block:: python

      from array_record.python import array_record_module

      # Basic usage
      writer = array_record_module.ArrayRecordWriter('output.array_record')
      writer.write(b'Hello, World!')
      writer.close()

      # With options
      writer = array_record_module.ArrayRecordWriter(
          'compressed.array_record',
          'group_size:1000,brotli:9,max_parallelism:4'
      )

ArrayRecordReader
~~~~~~~~~~~~~~~~~

The ArrayRecordReader class provides low-level sequential access to ArrayRecord files.

   The ArrayRecordReader class provides low-level sequential access to ArrayRecord files.

   **Constructor Parameters:**
   
   * ``path`` (str): File path to read from
   * ``options`` (str, optional): Comma-separated options string. Default: ""

   **Options String Format:**
   
   * ``readahead_buffer_size:N`` - Buffer size for readahead (default: 16MB, set to 0 for random access)
   * ``max_parallelism:N`` - Maximum parallel threads (default: auto, set to 0 for random access)
   * ``index_storage_option:in_memory/offloaded`` - How to store the index (default: in_memory)

   **Example:**

   .. code-block:: python

      from array_record.python import array_record_module

      # Sequential reading
      reader = array_record_module.ArrayRecordReader('input.array_record')
      
      print(f"Total records: {reader.num_records()}")
      
      # Read records sequentially
      reader.seek(0)
      while True:
          record = reader.read_record()
          if not record:
              break
          print(f"Record: {record}")
      
      reader.close()

      # Random access optimized
      reader = array_record_module.ArrayRecordReader(
          'input.array_record',
          'readahead_buffer_size:0,max_parallelism:0'
      )

Data Source Module
------------------

The data source module provides high-level random access to ArrayRecord files.

ArrayRecordDataSource
~~~~~~~~~~~~~~~~~~~~~

The ArrayRecordDataSource class provides high-level random access to ArrayRecord files with support for indexing and slicing.

   The ArrayRecordDataSource class provides high-level random access to ArrayRecord files.
   It implements a Python sequence interface with support for indexing and slicing.

   **Constructor Parameters:**
   
   * ``paths`` (str, pathlib.Path, FileInstruction, or list): Path(s) to ArrayRecord file(s)
   * ``reader_options`` (dict, optional): Dictionary of reader options

   **Reader Options:**
   
   * ``readahead_buffer_size`` (str): Buffer size (e.g., "16MB", "0" for random access)
   * ``max_parallelism`` (str): Number of parallel threads (e.g., "4", "0" for random access)
   * ``index_storage_option`` (str): "in_memory" or "offloaded"

   **Example:**

   .. code-block:: python

      from array_record.python import array_record_data_source

      # Basic usage
      data_source = array_record_data_source.ArrayRecordDataSource('data.array_record')
      
      # Get number of records
      print(f"Total records: {len(data_source)}")
      
      # Read single record
      first_record = data_source[0]
      
      # Read multiple records
      batch = data_source[[0, 10, 100]]
      
      # Context manager usage
      with array_record_data_source.ArrayRecordDataSource('data.array_record') as ds:
          records = ds[0:10]  # Read first 10 records

      # Multiple files
      files = ['part-00000.array_record', 'part-00001.array_record']
      data_source = array_record_data_source.ArrayRecordDataSource(files)

      # Optimized for random access
      reader_options = {
          'readahead_buffer_size': '0',
          'max_parallelism': '0'
      }
      data_source = array_record_data_source.ArrayRecordDataSource(
          'data.array_record',
          reader_options=reader_options
      )

FileInstruction
~~~~~~~~~~~~~~~

FileInstruction allows you to specify a subset of records to read from a file, which can significantly speed up initialization when you only need part of a large file.

   FileInstruction allows you to specify a subset of records to read from a file,
   which can significantly speed up initialization when you only need part of a large file.

   **Example:**

   .. code-block:: python

      from array_record.python.array_record_data_source import FileInstruction

      # Read only records 1000-2000 from the file
      instruction = FileInstruction(
          filename='large_file.array_record',
          start=1000,
          num_records=1000
      )
      
      data_source = array_record_data_source.ArrayRecordDataSource([instruction])

Utility Functions
-----------------

The module also provides several utility functions for working with ArrayRecord files, including functions for processing file instructions and validating group sizes.

Error Handling
--------------

ArrayRecord operations can raise various exceptions:

* ``ValueError``: Invalid parameters or file paths
* ``RuntimeError``: File I/O errors or corruption
* ``OSError``: System-level file access errors

**Example:**

.. code-block:: python

   try:
       data_source = array_record_data_source.ArrayRecordDataSource('nonexistent.array_record')
   except (ValueError, RuntimeError, OSError) as e:
       print(f"Error opening file: {e}")

Performance Tips
----------------

1. **For Sequential Access**: Use default settings or increase buffer size
   
   .. code-block:: python

      reader_options = {
          'readahead_buffer_size': '64MB',
          'max_parallelism': '8'
      }

2. **For Random Access**: Disable readahead and parallelism
   
   .. code-block:: python

      reader_options = {
          'readahead_buffer_size': '0',
          'max_parallelism': '0'
      }

3. **For Large Files**: Use FileInstruction to read subsets

4. **For Batch Processing**: Read multiple records at once using list indexing

5. **Memory Management**: Use context managers to ensure proper cleanup
