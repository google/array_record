## Apache Beam Integration for ArrayRecord

### Quickstart

#### Convert TFRecord in a GCS bucket to ArrayRecord
```
pip install array-record[beam]
git clone https://github.com/google/array_record.git
cd array_record/beam/examples
# Fill in the required fields in example_gcs_conversion.py
# If use DataFlow, set pipeline_options as instructed in example_gcs_conversion.py
python example_gcs_conversion.py
```
If DataFlow is used, you can monitor the run from the DataFlow job monitoring UI (https://cloud.google.com/dataflow/docs/guides/monitoring-overview)

### Summary

This submodule provides some Apache Beam components and lightweight pipelines for converting different file formats (TFRecord at present) into ArrayRecords. The intention is to provide a variety of fairly seamless tools for migrating existing TFRecord datasets, allowing a few different choices regarding sharding and write location.

There are two core components of this module:

1. A Beam `PTransform` with a `FileBasedSink` for writing ArrayRecords. It's modeled after similar components like `TFRecordIO` and `FileIO`. Worth noting in this implementation is that `array_record`'s `ArrayRecordWriter` object requires a file-path-like string to initialize, and the `.close()` method is required to make the file usable. This characteristic forces the overriding of Beam's default `.open()` functionality, which is where its schema and file handling functionality is housed. In short, it means this sink is **only usable for ArrayRecord writes to disk or disk-like paths, e.g. FUSE, NFS mounts, etc.** All writes using schema prefixes (e.g. `gs://`) will fail.

2. A Beam `DoFn` that accepts a single tuple consisting of a filename key and an entire set of serialized records. The function writes the serialized content to an ArrayRecord file in an on-disk path, uploads it to a specified GCS bucket, and removes the temporary file. This function has no inherent file awareness, making its primary goal the writing of a single file per PCollection. As such, it requires the file content division logic to be provided to the function elsewhere in the Beam pipeline.

In addition to these components, there are a number of simple pipelines included in this module that provide basic likely implementations of the above components. A few of those pipelines are as follows:

1. **Conversion from a set number of TFRecord files in either GCS or on-disk to a flexible number of ArrayRecords on disk:** Leverages the PTransform/Sink, and due to Beam's file handling capabilities allows for a `num_shards` argument that supports redistribution of the bounded dataset across an arbitrary number of files. However, due to overriding the `open()` method, writes to GCS don't work.

2. **Conversion from a set number of TFRecord files in either GCS or on-disk to a matching number of ArrayRecords on GCS:** Levarages the `ReadAllFromTFRecord` and `GroupByKey` Beam functions to organize a set of filename:content pairs, which are then passed to the ArrayRecord `DoFn`. The end result is that TFRecords are converted to ArrayRecords one-to-one.

3. **Conversion from a set number of TFRecord files in either GCS or on-disk to a matching number of ArrayRecords on disk:** Identical to pipeline 1, it just reads the number of shards first and sets the number of ArrayRecord shards to match.

In addition to all of that, there are a handful of dummy data generation functions used for testing and validation.

### Usage

**Basics and 'Getting Started'**

Please note that in an attempt to keep the array_record library lightweight, Apache Beam (and some of the underlying data generation dependencies like Tensorflow) are not installed by default when you run `pip install array-record`. To get the extra packages automatically, run `pip install array-record[beam]`.

Once installed, all of the Beam components are available to import from `array_record.beam`.

**Importing the PTransform or the DoFn**

If you're familiar with Apache Beam and want to build a custom pipeline around its core constructs, you can import the native Beam objects and implement them as you see fit.

To import the PTransform with the disk-based sink, use `from array_record.beam.arrayrecordio import WriteToArrayRecord`. You may then use it as a standard step in Beam Pipeline. It accepts a variety of different inputs including `file_path_prefix`, `file_path_suffix`, `coder`, and `num_shards`. For more detail, as well as options for extensibility, please refer to [Apache Beam's Documentation for FileBasedSink](https://beam.apache.org/releases/pydoc/current/apache_beam.io.filebasedsink.html)


To import the custom DoFn, use `from array_record.beam.dofns import ConvertToArrayRecordGCS`. You may then use it as a parameter for a Beam `ParDo`. It takes a handful of side inputs as described below:

- **path:** REQUIRED (and positional). The intended path prefix for the GCS bucket in "gs://..." format
- **overwrite_extension:** FALSE by default. Boolean making the DoFn attempt to overwrite any file extension after "."
- **file_path_suffix:** ".arrayrecord" by default. Intended suffix for overwrite or append

Note that by default, the DoFn will APPEND an existing filename/extension with ".arrayrecord". Setting `file_path_suffix` to `""` will leave the file names as-is and thus expect you to be passing in a different `path` than the source.

You can see usage details for each of these implementations in `pipelines.py`.

**Using the Helper Functions**

Several helper functions have been packaged to make the functionality more accessible to those with less comfort building Apache Beam pipelines. All of these pipelines take `input` and `output` arguments, which are intended as the respective source and destination paths of the TFRecord files and the ArrayRecord files. Wildcards are accepted in these paths. By default, these parameters can either be passed as CLI arguments when executing a pipeline as `python -m <python_module> --input <path> --output <path>`, or as an override to the `args` argument if executing programmatically. Additionally, extra arguments can be passed via CLI or programmatically in the `pipeline_options` argument if you want to control the behavior of Beam. The likely reason for this would be altering the Runner to Google Cloud Dataflow, which these examples support (with caveats; see the section below on Dataflow).

There are slight variations in execution when running these either from an interpreter or the CLI, so familiarize yourself with the files in the `examples/` directory along with `demo.py`, which show the different invocation methods. The available functions can all be imported `from array_record.beam.pipelines import *` and are as follows:

- **convert_tf_to_arrayrecord_disk:** Converts TFRecords at `input` path to ArrayRecords at `output` path for disk-based writes only. Accepts an extra `num_shards` argument for resharding ArrayRecords across an arbitrary number of files.
- **convert_tf_to_arrayrecord_disk_match_shards:** Same as above, except it reads the number of source files and matches them to the destination. There is no `num_shards` argument.
- **convert_tf_to_arrayrecord_gcs:** Converts TFRecords at `input` path to ArrayRecords at `output` path, where the `output` path **must** be a GCS bucket in "gs://" format. This function accepts the same `overwrite_extension` and `file_path_suffix` arguments as the DoFn itself, allowing for customization of file naming.

### Examples and Demos

See the examples in the `examples/` directory for different invocation techniques. One of the examples invokes `array_record.beam.demo` as a module, which is a simple pipeline that generates some TFRecords and then converts them to ArrayRecord in GCS. You can see the implementation in `demo.py`, which should serve as a guide for implementing your own CLI-triggered pipelines.

You'll also note commented sections in each example, which are the configuration parameters for running the pipelines on Google Cloud Dataflow. There is also a `requirements.txt` in there, which at present is a requirement for running these on Dataflow as is. See below for more detail.

### Dataflow Usage

These pipelines have all been tested and are compatible with Google Cloud Dataflow. Uncomment the sections in the example files and set your own bucket/project information to see it in action.

Note, however, the `requirements.txt` file. This is necessary because the `array-record` PyPl installation does not install the Apache Beam or Tensorflow components by default to keep the library lightweight. A `requirements.txt` passed as an argument to the Dataflow job is required to ensure everything is installed correctly on the runner.


Allow to simmer uncovered for 5 minutes. Plate, serve, and enjoy.
