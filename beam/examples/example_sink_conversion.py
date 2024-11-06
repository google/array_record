"""Execute this to convert TFRecords to ArrayRecords using the disk Sink."""


from apache_beam.options import pipeline_options
from array_record.beam.pipelines import convert_tf_to_arrayrecord_disk_match_shards

## Set input and output patterns as specified
input_pattern = 'gs://<YOUR_INPUT_BUCKET>/records/*.tfrecord'
output_path = 'records/movies'

args = {'input': input_pattern, 'output': output_path}

## If run in Dataflow, set pipeline options and uncomment in main()
## If run pipeline_options is not set, you will use a local runner
pipeline_options = pipeline_options.PipelineOptions(
    runner='DataflowRunner',
    project='<YOUR_PROJECT>',
    region='<YOUR_REGION>',
    requirements_file='requirements.txt'
)


def main():
  convert_tf_to_arrayrecord_disk_match_shards(
      args=args,
      # pipeline_options=pipeline_options,
  ).run()

if __name__ == '__main__':
  main()
