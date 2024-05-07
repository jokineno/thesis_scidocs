#!/bin/bash

set -x

python setup.py install > /dev/null

echo "Starting to predict with ###BASELINE 1 FINNISH BERT####"

set -u
DATA_DIR="thesis_data"

python scripts/run.py \
  --cls thesis_data/wiki_cls/top/20240507_110618_baseline_finnishbert_cls.jsonl \
  --cls2 "./thesis_data/wiki_cls/second/20240507_155022_baseline_finnishbert_cls.jsonl" \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path $DATA_DIR

echo "Done"