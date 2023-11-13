"""Helper file for generating TF/ArrayRecords and writing them to disk."""

import os
from array_record.python.array_record_module import ArrayRecordWriter
import tensorflow as tf
from . import testdata


def generate_movie_examples():
  """Create a list of TF examples from the dummy data above and return it.
  
  Returns:
    TFExample object
  """

  examples = []
  for example in testdata.data:
    examples.append(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'Age': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[example['Age']])),
                    'Movie': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[
                                m.encode('utf-8') for m in example['Movie']])),
                    'Movie Ratings': tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=example['Movie Ratings'])),
                    'Suggestion': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[example['Suggestion'].encode('utf-8')])),
                    'Suggestion Purchased': tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[example['Suggestion Purchased']])),
                    'Purchase Price': tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[example['Purchase Price']]))
                }
            )
        )
    )

  return(examples)


def generate_serialized_movie_examples():
  """Return a serialized version of the above data for byte insertion."""

  return [example.SerializeToString() for example in generate_movie_examples()]


def write_example_to_tfrecord(example, file_path):
  """Write example(s) to a single TFrecord file."""

  with tf.io.TFRecordWriter(file_path) as writer:
    writer.write(example.SerializeToString())


# Write example(s) to a single ArrayRecord file
def write_example_to_arrayrecord(example, file_path):
  writer = ArrayRecordWriter(file_path, 'group_size:1')
  writer.write(example.SerializeToString())
  writer.close()


def kitty_tfrecord(prefix=''):
  """Create a TFRecord from a cat pic on the Internet.

  This is mainly for testing; probably don't use it.

  Args:
    prefix: A file directory in string format.
  """

  cat_in_snow = tf.keras.utils.get_file(
      '320px-Felis_catus-cat_on_snow.jpg',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')

  image_labels = {
      cat_in_snow: 0
  }

  image_string = open(cat_in_snow, 'rb').read()
  label = image_labels[cat_in_snow]
  image_shape = tf.io.decode_jpeg(image_string).shape

  feature = {
      'height': tf.train.Feature(int64_list=tf.train.Int64List(
          value=[image_shape[0]])),
      'width': tf.train.Feature(int64_list=tf.train.Int64List(
          value=[image_shape[1]])),
      'depth': tf.train.Feature(int64_list=tf.train.Int64List(
          value=[image_shape[2]])),
      'label': tf.train.Feature(int64_list=tf.train.Int64List(
          value=[label])),
      'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[image_string]))
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature))

  record_file = os.path.join(prefix, 'kittymeow.tfrecord')
  with tf.io.TFRecordWriter(record_file) as writer:
    writer.write(example.SerializeToString())
