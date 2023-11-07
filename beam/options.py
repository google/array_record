"""Handler for Pipeline and Beam options that allows for cleaner importing."""


import argparse
from apache_beam.options import pipeline_options


def get_arguments():
  """Simple external wrapper for argparse that allows for manual construction.

  Returns:
    1. A dictionary of known args for use in pipelines
    2. The remainder of the arguments in PipelineOptions format

  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      help='The file pattern for the input TFRecords.',)
  parser.add_argument(
      '--output',
      help='The path prefix for output ArrayRecords.')

  args, beam_args = parser.parse_known_args()
  return(args.__dict__, pipeline_options.PipelineOptions(beam_args))
