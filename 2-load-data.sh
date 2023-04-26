#!/bin/bash

set -x

echo "Starting to load data.."
aws s3 sync --no-sign-request s3://ai2-s2-research-public/specter/scidocs/ data/
echo "Done."
