#!/bin/bash

set -x

python setup.py install > /dev/null

echo "Starting to predict"

if [[ ! $* == *--thesis* ]]
then
  echo "Running ***ORIGINAL*** "
  python scripts/run.py \
  --cls data/specter-embeddings/cls.jsonl \
  --user-citation data/specter-embeddings/user-citation.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path data
else

  echo "Running ***THESIS DEMO*** "
  python scripts/run.py \
  --user-citation thesis_data/myownfinedtuned/user-citation.jsonl \
  --cls thesis_data/myownfinedtuned/cls.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path thesis_data \
  --debug true \
  --is_thesis True
fi



echo "Done"