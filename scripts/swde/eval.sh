#!/bin/bash

# conda env: s2j
# command: bash scripts/swde/eval.sh

API_SOURCE=azure
BACKEND=fan-gpt4

python run.py \
    --run_type eval \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task swde \
    --prompt_setting one_by_one \
    --data_split test \
    --metric f1 \
    --threshold 0.25 \
    --verbose