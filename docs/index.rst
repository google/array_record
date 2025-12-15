ArrayRecord - High-Performance Data Storage
==================================================

ArrayRecord is a high-performance file format designed for machine learning
workloads, derived from `Riegeli <https://github.com/google/riegeli>`_ and
achieving a new frontier of IO efficiency. ArrayRecord supports parallel read,
write, and random access by record index, making it ideal for machine learning
workloads and large-scale data processing.

.. image:: https://img.shields.io/pypi/v/array-record-python.svg
   :target: https://pypi.org/project/array-record-python/
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/bzantium/array_record.svg
   :target: https://github.com/bzantium/array_record/blob/main/LICENSE
   :alt: License

Installation
------------

.. code-block:: bash

   pip install array_record

For Apache Beam integration:

.. code-block:: bash

   pip install array_record[beam]

Basic Usage
-----------

Writing Records
~~~~~~~~~~~~~~~

.. code-block:: python

   from array_record.python import array_record_module

   # Create a writer
   # Use `group_size` to configure the access granularity.
   # `group_size:1` is optimized for random access, while a larger group size
   # increases compression ratio and improves the performance of sequential and
   # batch access.
   writer = array_record_module.ArrayRecordWriter('output.array_record', 'group_size:1')

   # Write some records
   for i in range(10000):
       data = f"Record {i}".encode('utf-8')
       writer.write(data)

   # Close the writer (important!)
   writer.close()

Reading Records
~~~~~~~~~~~~~~~

ArrayRecord provides two read APIs. One is at the file level
(``array_record_module``) which has one-to-one mapping of the underlying C++ API;
the other wraps around ``array_record_module`` for convenience to access
multiple ArrayRecord files (``array_record_data_source``).

File-level read API ``array_record_module.reader``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from array_record.python import array_record_module

   reader = array_record_module.ArrayRecordReader('output.array_record')

   # Read records sequentially with a read-ahead thread pool.
   for _ in range(reader.num_records()):
     record = reader.read()

   # Read all the records at once with an internal thread pool
   records = reader.read_all()

   # Read records by range with an internal thread pool
   records = reader.read(0, 100)

   # Read records by indices with an internal thread pool
   records = reader.read([0, 1, 100])

   # Releases the thread pool and the file handle.
   reader.close()

Multi-file read API ``array_record_data_source``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ArrayRecord data source is designed specifically for pygrain.
It interfaces with multiple ArrayRecord files for random access.
To use ArrayRecord data source, the writer must specify the
``group_size`` to be 1.

.. code-block:: python

   import glob
   from array_record.python import array_record_module
   from array_record.python import array_record_data_source

   num_files = 5
   for i in range(num_files):
     # array_record_data_source strictly require the group_size
     # to be 1.
     writer = array_record_module.ArrayRecordWriter(
         f"output.array_record-{i:05d}-of-{num_files:05d}",
         "group_size:1"
     )
     for j in range(100):
         data = f"File {i:05d}, Record {j}".encode("utf-8")
         writer.write(data)
     writer.close()

   # The data source object accepts multiple array record files
   # as the input and creates a virtual concatenated view over the files
   with array_record_data_source.ArrayRecordDataSource(glob.glob("output.array_record*")) as ds:
       # Get total number of records
       total_records = len(ds)

       # Read a specific record from the files as if all the files were
       # concatenated into a single virtual array.
       record_1000 = ds[150]

       # Read multiple records. The records indexed in the same file would
       # use a thread pool to speed up the reading performance.
       batch = ds.__getitems__([10, 100, 200])


Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   core_concepts
   performance
   python_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
