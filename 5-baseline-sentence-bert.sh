#!/bin/bash

set -x

python setup.py install > /dev/null

echo "Starting to predict with ###BASELINE MODEL 2 - SENTENCE BERT###"
python scripts/run.py \
  --user-citation thesis_data/sbert/user-citation.jsonl \
  --cls thesis_data/sbert/cls.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path thesis_data \
  --debug true \
  --is_thesis True
fi

echo "Done"