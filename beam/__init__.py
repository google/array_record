"""Apache Beam module for array_record.

This module provides both core components and
helper functions to enable users to convert different file formats to AR.

To keep dependencies light, we'll import Beam on module usage so any errors
occur early.
"""

import apache_beam as beam

# I'd really like a PEP8 compatible conditional import here with a more
# explicit error message. Example below:

# try:
#   import apache_beam as beam
# except Exception as e:
#   raise ImportError(
#       ('Beam functionality requires extra dependencies. '
#        'Install apache-beam or run "pip install array_record[beam]".')) from e
