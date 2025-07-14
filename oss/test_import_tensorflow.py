"""Smoke test for importing TensorFlow and ArrayRecord together."""

# pylint: disable=unused-import,g-import-not-at-top
from array_record.python import array_record_module
print("Imported ArrayRecord.")
import tensorflow as tf
print(f"Imported TensorFlow {tf.__version__}.")
