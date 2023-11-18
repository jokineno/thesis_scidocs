#!/bin/bash

set -x

python setup.py install > /dev/null

echo "[*] Starting to predict"
#

# ===========ORIGINAL SCRIPT==============
#  echo "[*]  Running ***ORIGINAL*** "
#  python scripts/run.py \
#  --cls data/specter-embeddings/cls.jsonl \
#  --user-citation data/specter-embeddings/user-citation.jsonl \
#  --val_or_test test \
#  --n-jobs 12 \
#  --cuda-device -1 \
#  --data-path data
#===========ORIGINAL SCRIPT==============

echo "[*] Running scripts/run.py ...... "
set -x
set -u
DATA_DIR="thesis_data"

MODE="--no-debug"
if [[ $* == *--debug* ]]
then
 MODE="--debug"
fi
python scripts/run.py \
  --cls $DATA_DIR/myown-embeddings/20231118_140627_myown_embeddings_cls.jsonl \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path $DATA_DIR \
  $MODE \

echo "[*] Done"
