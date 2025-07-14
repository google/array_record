"""Smoke test for importing TensorFlow and ArrayRecord through Grain."""

# pylint: disable=unused-import,g-import-not-at-top
import grain
print("Imported Grain.")
import tensorflow as tf
print(f"Imported TensorFlow {tf.__version__}.")
