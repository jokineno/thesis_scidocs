#!/bin/bash

set -x

python setup.py install > /dev/null

echo "[*] Starting to predict"

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

python scripts/run.py \
  --cls ./thesis_data/wiki_cls/top/20240507_152325_myown_embeddings_cls.jsonl \
  --cls2 "./thesis_data/wiki_cls/second/_myown_embeddings_cls.jsonl" \
  --val_or_test test \
  --n-jobs 12 \
  --cuda-device -1 \
  --data-path $DATA_DIR



echo "[*] Done"
