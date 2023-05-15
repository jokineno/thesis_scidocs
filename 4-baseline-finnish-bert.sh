#!/bin/bash

set -x

python setup.py install > /dev/null

echo "Starting to predict with ###BASELINE 1 FINNISH BERT####"

python scripts/run.py \
  --user-citation thesis_data/finnish_bert/user-citations.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path thesis_data \
  --debug true \
  --is_thesis True


echo "Done"