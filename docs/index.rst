ArrayRecord - High-Performance Data Storage
==================================================

ArrayRecord is a high-performance file format designed for machine learning workloads, derived from `Riegeli <https://github.com/google/riegeli>`_ and achieving a new frontier of IO efficiency. ArrayRecord supports parallel read, write, and random access by record index, making it ideal for machine learning workloads and large-scale data processing.

.. image:: https://img.shields.io/pypi/v/array-record-python.svg
   :target: https://pypi.org/project/array-record-python/
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/bzantium/array_record.svg
   :target: https://github.com/bzantium/array_record/blob/main/LICENSE
   :alt: License

Features
--------

* **High Performance**: Optimized for both sequential and random access patterns
* **Parallel Processing**: Built-in support for concurrent read and write operations
* **Compression**: Multiple compression algorithms (Brotli, Zstd, Snappy) with configurable levels
* **Random Access**: Efficient random access by record index without full file scanning
* **Apache Beam Integration**: Seamless integration with Apache Beam for large-scale data processing
* **Cross-Platform**: Available for Linux (x86_64, aarch64) and macOS (aarch64)

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install array_record

For Apache Beam integration:

.. code-block:: bash

   pip install array_record[beam]

Basic Usage
~~~~~~~~~~~

Writing Records
^^^^^^^^^^^^^^^

.. code-block:: python

   from array_record.python import array_record_module

   # Create a writer
   writer = array_record_module.ArrayRecordWriter('output.array_record', 'group_size:1000')
   
   # Write some records
   for i in range(10000):
       data = f"Record {i}".encode('utf-8')
       writer.write(data)
   
   # Close the writer (important!)
   writer.close()

Reading Records
^^^^^^^^^^^^^^^

.. code-block:: python

   from array_record.python import array_record_data_source

   # Create a data source
   data_source = array_record_data_source.ArrayRecordDataSource('output.array_record')
   
   # Read all records
   for i in range(len(data_source)):
       record = data_source[i]
       print(f"Record {i}: {record.decode('utf-8')}")

   # Or read specific records
   records = data_source[[0, 100, 500]]  # Read records at indices 0, 100, and 500

Random Access
^^^^^^^^^^^^^

.. code-block:: python

   # ArrayRecord supports efficient random access
   with array_record_data_source.ArrayRecordDataSource('output.array_record') as ds:
       # Get total number of records
       total_records = len(ds)
       
       # Read a specific record
       record_1000 = ds[1000]
       
       # Read multiple specific records
       batch = ds[[10, 20, 30, 40, 50]]

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   performance

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   core_concepts
   beam_integration
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   python_reference
   beam_reference
   cpp_reference

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
