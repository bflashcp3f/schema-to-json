#!/bin/bash

# conda env: s2j
# command: bash scripts/discomat/eval.sh

API_SOURCE=azure
BACKEND=fan-gpt4

python run.py \
    --run_type eval \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task discomat \
    --prompt_setting error_recovery \
    --template template_filling \
    --data_split test \
    --metric f1 \
    --threshold 0.25 \
    --verbose