# Execute this via BASH to run a full demo that creates TFRecords and converts them

#!/bin/bash


# Set bucket info below. Uncomment lower lines and set values to use Dataflow.
python -m array_record.beam.demo \
  --input="gs://<YOUR_INPUT_BUCKET>/records/movies" \
  --output="gs://<YOUR_OUTPUT_BUCKET>/records/" \
  # --region="<YOUR_REGION>" \
  # --runner="DataflowRunner" \
  # --project="<YOUR_PROJECT>" \
  # --requirements_file="requirements.txt"
