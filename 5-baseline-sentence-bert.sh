#!/bin/bash

set -x

python setup.py install > /dev/null

echo "Starting to predict with BASELINE 2 FINNISH Sentence BERT####"

set -u
DATA_DIR="thesis_data"

MODE="--no-debug"
if [[ $* == *--debug* ]]
then
 MODE="--debug"
fi

python scripts/run.py \
  --cls $DATA_DIR/sbert-embeddings/20231117_192643_baseline_sbert_cls.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path $DATA_DIR

echo "Done"