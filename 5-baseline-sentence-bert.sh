#!/bin/bash

set -x

python setup.py install > /dev/null

echo "Starting to predict with BASELINE 2 FINNISH Sentence BERT####"

set -u
DATA_DIR="thesis_data"

python scripts/run.py \
  --cls thesis_data/wiki_cls/top/20240507_111634_baseline_sbert_cls.jsonl \
  --cls2 thesis_data/wiki_cls/second/20240507_155733_baseline_sbert_cls.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path $DATA_DIR

echo "Done"