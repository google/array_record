"""An IO module for ArrayRecord.

CURRENTLY ONLY SINK IS IMPLEMENTED, AND IT DOESN'T WORK WITH NON-DISK WRITES
"""

from apache_beam import io
from apache_beam import transforms
from apache_beam.coders import coders
from apache_beam.io import filebasedsink
from apache_beam.io.filesystem.CompressionTypes import AUTO
from array_record.python.array_record_module import ArrayRecordWriter


class _ArrayRecordSink(filebasedsink.FileBasedSink):
  """Sink Class for use in Arrayrecord PTransform."""

  def __init__(
      self,
      file_path_prefix,
      file_name_suffix=None,
      num_shards=0,
      shard_name_template=None,
      coder=coders.ToBytesCoder(),
      compression_type=AUTO):

    super().__init__(
        file_path_prefix,
        file_name_suffix=file_name_suffix,
        num_shards=num_shards,
        shard_name_template=shard_name_template,
        coder=coder,
        mime_type='application/octet-stream',
        compression_type=compression_type)

  def open(self, temp_path):
    array_writer = ArrayRecordWriter(temp_path, 'group_size:1')
    return array_writer

  def close(self, file_handle):
    file_handle.close()

  def write_encoded_record(self, file_handle, value):
    file_handle.write(value)


class WriteToArrayRecord(transforms.PTransform):
  """PTransform for a disk-based write to ArrayRecord."""

  def __init__(
      self,
      file_path_prefix,
      file_name_suffix='',
      num_shards=0,
      shard_name_template=None,
      coder=coders.ToBytesCoder(),
      compression_type=AUTO):

    self._sink = _ArrayRecordSink(
        file_path_prefix,
        file_name_suffix,
        num_shards,
        shard_name_template,
        coder,
        compression_type)

  def expand(self, pcoll):
    return pcoll | io.iobase.Write(self._sink)
