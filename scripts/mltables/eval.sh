#!/bin/bash

# conda env: s2j
# command: bash scripts/mltables/eval.sh

API_SOURCE=azure
BACKEND=fan-gpt4

# python run.py \
#     --run_type eval \
#     --api_source $API_SOURCE \
#     --backend $BACKEND \
#     --task mltables \
#     --prompt_setting error_recovery \
#     --template template_filling \
#     --data_split test \
#     --metric f1 \
#     --threshold 0.25 \
#     --verbose

python run.py \
    --run_type eval \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task mltables \
    --prompt_setting error_recovery \
    --template template_filling_gpt4_typo \
    --data_split test \
    --metric exact_match \
    --verbose
