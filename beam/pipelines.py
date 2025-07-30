"""Various opinionated Beam pipelines for testing different functionality."""

from typing import Any, Callable

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
  """
  Beam pipeline for creating example TFRecord data.

  Args:
    num_shards: Number of files
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  p1 = beam.Pipeline(options=pipeline_options)
  _ = (
      p1
      | "Create" >> beam.Create(example.generate_movie_examples())
      | "Write"
      >> beam.io.WriteToTFRecord(
          args["output"],
          coder=coders.ToBytesCoder(),
          num_shards=num_shards,
          file_name_suffix=".tfrecord",
      )
  )
  return p1


def example_to_arrayrecord(
    num_shards=1, args=def_args, pipeline_options=def_pipeline_options
):
  """Beam pipeline for creating example ArrayRecord data.

  Args:
    num_shards: Number of files
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  p1 = beam.Pipeline(options=pipeline_options)
  _ = (
      p1
      | "Create" >> beam.Create(example.generate_movie_examples())
      | "Write"
      >> arrayrecordio.WriteToArrayRecord(
          args["output"],
          coder=coders.ToBytesCoder(),
          num_shards=num_shards,
          file_name_suffix=".arrayrecord",
      )
  )
  return p1


def _convert_to_array_record_disk(
    num_shards, args, pipeline_options, file_type, beam_fn
):
  p1 = beam.Pipeline(options=pipeline_options)
  _ = (
      p1
      | f"Read {file_type}" >> beam_fn(args["input"])
      | "Write ArrayRecord"
      >> arrayrecordio.WriteToArrayRecord(
          args["output"],
          coder=coders.ToBytesCoder(),
          num_shards=num_shards,
          file_name_suffix=".arrayrecord",
      )
  )
  return p1


def convert_tf_to_arrayrecord_disk(
    *, num_shards=1, args=def_args, pipeline_options=def_pipeline_options
):
  """
  Convert TFRecords to ArrayRecords using sink/sharding functionality.

  THIS ONLY WORKS FOR DISK ARRAYRECORD WRITES

  Args:
    num_shards: Number of files
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  return _convert_to_array_record_disk(
    num_shards, args, pipeline_options, "TFRecord", beam.io.ReadFromTFRecord
  )


def convert_text_to_array_record_disk(
    *, num_shards=1, args=def_args, pipeline_options=def_pipeline_options
):
  """
  Convert Text files to ArrayRecords using sink/sharding functionality.

  THIS ONLY WORKS FOR DISK ARRAYRECORD WRITES

  Args:
    num_shards: Number of files
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  return _convert_to_array_record_disk(
    num_shards, args, pipeline_options, "Text file", beam.io.ReadFromText
  )


def _convert_to_arrayrecord_disk_match_shards(args, pipeline_options, beam_fn):
  p1 = beam.Pipeline(options=pipeline_options)
  initial = (
      p1
      | "Start" >> beam.Create([args["input"]])
      | "Read" >> beam_fn(with_filename=True)
  )

  file_count = (
      initial
      | "Group" >> beam.GroupByKey()
      | "Count Shards" >> beam.combiners.Count.Globally()
  )

  _ = (
      initial
      | "Drop Filename" >> beam.Map(lambda x: x[1])
      | "Write ArrayRecord"
      >> arrayrecordio.WriteToArrayRecord(
          args["output"],
          coder=coders.ToBytesCoder(),
          num_shards=beam.pvalue.AsSingleton(file_count),
          file_name_suffix=".arrayrecord",
      )
  )
  return p1


def convert_tf_to_arrayrecord_disk_match_shards(
    *, args=def_args, pipeline_options=def_pipeline_options
):
  """
  Convert TFRecords to matching number of ArrayRecords.

  THIS ONLY WORKS FOR DISK ARRAYRECORD WRITES

  Args:
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  return _convert_to_arrayrecord_disk_match_shards(
    args, pipeline_options, beam.io.ReadAllFromTFRecord
  )


def convert_text_to_arrayrecord_disk_match_shards(
    *, args=def_args, pipeline_options=def_pipeline_options
):
  """
  Convert Text files to matching number of ArrayRecords.

  THIS ONLY WORKS FOR DISK ARRAYRECORD WRITES

  Args:
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  return _convert_to_arrayrecord_disk_match_shards(
    args, pipeline_options, beam.io.ReadAllFromText
  )


def _convert_to_arrayrecord_gcs(
    overwrite_extension, file_path_suffix, args, pipeline_options, beam_fn
):
  p1 = beam.Pipeline(options=pipeline_options)
  _ = (
      p1
      | "Start" >> beam.Create([args["input"]])
      | "Read" >> beam_fn(with_filename=True)
      | "Group" >> beam.GroupByKey()
      | "Write to ArrayRecord in GCS"
      >> beam.ParDo(
          dofns.ConvertToArrayRecordGCS(),
          args["output"],
          file_path_suffix=file_path_suffix,
          overwrite_extension=overwrite_extension,
      )
  )
  return p1


def convert_tf_to_arrayrecord_gcs(
    *,
    overwrite_extension=False,
    file_path_suffix=".arrayrecord",
    args=def_args,
    pipeline_options=def_pipeline_options
):
  """
  Convert TFRecords to ArrayRecords in GCS 1:1.

  Args:
    overwrite_extension: Boolean making DoFn attempt to overwrite extension
    file_path_suffix: Intended suffix for overwrite or append
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  return _convert_to_arrayrecord_gcs(
    overwrite_extension, file_path_suffix, args, pipeline_options, beam.io.ReadAllFromTFRecord
  )


def convert_text_to_arrayrecord_gcs(
    *,
    overwrite_extension=False,
    file_path_suffix=".arrayrecord",
    args=def_args,
    pipeline_options=def_pipeline_options
):
  """
  Convert Text files to ArrayRecords in GCS 1:1.

  Args:
    overwrite_extension: Boolean making DoFn attempt to overwrite extension
    file_path_suffix: Intended suffix for overwrite or append
    args: Custom arguments
    pipeline_options: Beam arguments in dict format

  Returns:
    Beam Pipeline object
  """

  return _convert_to_arrayrecord_gcs(
    overwrite_extension, file_path_suffix, args, pipeline_options, beam.io.ReadAllFromText
  )
