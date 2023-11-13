"""Demo Pipeline.

This file creates a TFrecord dataset and converts it to ArrayRecord on GCS
"""

import apache_beam as beam
from apache_beam.coders import coders
from . import dofns
from . import example
from . import options


## Grab CLI arguments.
## Override by passing args/pipeline_options to the function manually.
args, pipeline_options = options.get_arguments()


def main():
  p1 = beam.Pipeline(options=pipeline_options)
  initial = (p1
             | 'Create a set of TFExamples' >> beam.Create(
                 example.generate_movie_examples()
             )
             | 'Write TFRecords' >> beam.io.WriteToTFRecord(
                 args['input'],
                 coder=coders.ToBytesCoder(),
                 num_shards=4,
                 file_name_suffix='.tfrecord'
             )
             | 'Read shards from GCS' >> beam.io.ReadAllFromTFRecord(
                 with_filename=True)
             | 'Group with Filename' >> beam.GroupByKey()
             | 'Write to ArrayRecord in GCS' >> beam.ParDo(
                 dofns.ConvertToArrayRecordGCS(),
                 args['output'],
                 overwrite_extension=True))

  return p1, initial


if __name__ == '__main__':
  demo_pipeline = main()
  demo_pipeline.run()
