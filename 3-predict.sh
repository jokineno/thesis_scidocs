#!/bin/bash

set -x

python setup.py install > /dev/null

echo "Starting to predict"
python scripts/run.py \
  --cls data/specter-embeddings/cls.jsonl \
  --user-citation data/specter-embeddings/user-citation.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --debug true
echo "Done"