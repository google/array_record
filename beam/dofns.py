"""DoFn's for parallel processing."""

import os
import urllib
import apache_beam as beam
from array_record.python.array_record_module import ArrayRecordWriter
from google.cloud import storage


class ConvertToArrayRecordGCS(beam.DoFn):
  """Write a tuple consisting of a filename and records to GCS ArrayRecords."""

  _WRITE_DIR = '/tmp/'

  def process(
      self,
      element,
      path,
      write_dir=_WRITE_DIR,
      file_path_suffix='.arrayrecord',
      overwrite_extension=False,
    ):

    ## Upload to GCS
    def upload_to_gcs(bucket_name, filename, prefix='', source_dir=self._WRITE_DIR):
      source_filename = os.path.join(source_dir, filename)
      blob_name = os.path.join(prefix, filename)
      storage_client = storage.Client()
      bucket = storage_client.get_bucket(bucket_name)
      blob = bucket.blob(blob_name)
      blob.upload_from_filename(source_filename)

    ## Simple logic for stripping a file extension and replacing it
    def fix_filename(filename):
      base_name = os.path.splitext(filename)[0]
      new_filename = base_name + file_path_suffix
      return new_filename

    parsed_gcs_path = urllib.parse.urlparse(path)
    bucket_name = parsed_gcs_path.hostname
    gcs_prefix = parsed_gcs_path.path.lstrip('/')

    if overwrite_extension:
      filename = fix_filename(os.path.basename(element[0]))
    else:
      filename = '{}{}'.format(os.path.basename(element[0]), file_path_suffix)

    write_path = os.path.join(write_dir, filename)
    writer = ArrayRecordWriter(write_path, 'group_size:1')

    for item in element[1]:
      writer.write(item)

    writer.close()

    upload_to_gcs(bucket_name, filename, prefix=gcs_prefix)
    os.remove(os.path.join(write_dir, filename))
