"""Various opinionated Beam pipelines for testing different functionality."""

import apache_beam as beam
from apache_beam.coders import coders
from . import arrayrecordio
from . import dofns
from . import example
from . import options


## Grab CLI arguments.
## Override by passing args/pipeline_options to the function manually.
def_args, def_pipeline_options = options.get_arguments()


def example_to_tfrecord(
    num_shards=1,
    args=def_args,
    pipeline_options=def_pipeline_options):
  """Beam pipeline for creating example TFRecord data.

  Args:
    num_shards: Number of files
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  p1 = beam.Pipeline(options=pipeline_options)
  initial = (p1
             | 'Create' >> beam.Create(example.generate_movie_examples())
             | 'Write' >> beam.io.WriteToTFRecord(
                 args['output'],
                 coder=coders.ToBytesCoder(),
                 num_shards=num_shards,
                 file_name_suffix='.tfrecord'))

  return p1, initial


def example_to_arrayrecord(
    num_shards=1,
    args=def_args,
    pipeline_options=def_pipeline_options):
  """Beam pipeline for creating example ArrayRecord data.

  Args:
    num_shards: Number of files
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  p1 = beam.Pipeline(options=pipeline_options)
  initial = (p1
             | 'Create' >> beam.Create(example.generate_movie_examples())
             | 'Write' >> arrayrecordio.WriteToArrayRecord(
                 args['output'],
                 coder=coders.ToBytesCoder(),
                 num_shards=num_shards,
                 file_name_suffix='.arrayrecord'))

  return p1, initial


def convert_tf_to_arrayrecord_disk(
    num_shards=1,
    args=def_args,
    pipeline_options=def_pipeline_options):
  """Convert TFRecords to ArrayRecords using sink/sharding functionality.

  THIS ONLY WORKS FOR DISK ARRAYRECORD WRITES

  Args:
    num_shards: Number of files
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  p1 = beam.Pipeline(options=pipeline_options)
  initial = (p1
             | 'Read TFRecord' >> beam.io.ReadFromTFRecord(args['input'])
             | 'Write ArrayRecord' >> arrayrecordio.WriteToArrayRecord(
                 args['output'],
                 coder=coders.ToBytesCoder(),
                 num_shards=num_shards,
                 file_name_suffix='.arrayrecord'))

  return p1, initial


def convert_tf_to_arrayrecord_disk_match_shards(
    args=def_args,
    pipeline_options=def_pipeline_options):
  """Convert TFRecords to matching number of ArrayRecords.

  THIS ONLY WORKS FOR DISK ARRAYRECORD WRITES

  Args:
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  p1 = beam.Pipeline(options=pipeline_options)
  initial = (p1
             | 'Start' >> beam.Create([args['input']])
             | 'Read' >> beam.io.ReadAllFromTFRecord(with_filename=True))

  file_count = (initial
                | 'Group' >> beam.GroupByKey()
                | 'Count Shards' >> beam.combiners.Count.Globally())

  write_files = (initial
                 | 'Drop Filename' >> beam.Map(lambda x: x[1])
                 | 'Write ArrayRecord' >> arrayrecordio.WriteToArrayRecord(
                     args['output'],
                     coder=coders.ToBytesCoder(),
                     num_shards=beam.pvalue.AsSingleton(file_count),
                     file_name_suffix='.arrayrecord'))

  return p1, write_files


def convert_tf_to_arrayrecord_gcs(
    overwrite_extension=False,
    file_path_suffix='.arrayrecord',
    args=def_args,
    pipeline_options=def_pipeline_options):
  """Convert TFRecords to ArrayRecords in GCS 1:1.
  
  Args:
    overwrite_extension: Boolean making DoFn attempt to overwrite extension
    file_path_suffix: Intended suffix for overwrite or append
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  p1 = beam.Pipeline(options=pipeline_options)
  initial = (p1
             | 'Start' >> beam.Create([args['input']])
             | 'Read' >> beam.io.ReadAllFromTFRecord(with_filename=True)
             | 'Group' >> beam.GroupByKey()
             | 'Write to ArrayRecord in GCS' >> beam.ParDo(
                 dofns.ConvertToArrayRecordGCS(),
                 args['output'],
                 file_path_suffix=file_path_suffix,
                 overwrite_extension=overwrite_extension))

  return p1, initial
