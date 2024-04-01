#!/bin/bash

# conda env: s2j
# command: bash scripts/chemtables/eval_os.sh

API_SOURCE=open_source

BACKEND=codellama-13b-instruct

python run.py \
    --run_type eval \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task chemtables \
    --prompt_setting error_recovery \
    --template template_filling \
    --data_split test \
    --metric f1 \
    --threshold 0.25 \
    --verbose

python run.py \
    --run_type eval \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task chemtables \
    --prompt_setting error_recovery \
    --template template_filling \
    --data_split test \
    --metric exact_match \
    --verbose

# BACKEND=llama2-chat-13b

# python run.py \
#     --run_type eval \
#     --api_source $API_SOURCE \
#     --backend $BACKEND \
#     --task mltables \
#     --prompt_setting error_recovery \
#     --template template_filling_llama2chat \
#     --data_split test \
#     --metric f1 \
#     --threshold 0.25 \
#     --verbose

# python run.py \
#     --run_type eval \
#     --api_source $API_SOURCE \
#     --backend $BACKEND \
#     --task mltables \
#     --prompt_setting error_recovery \
#     --template template_filling_llama2chat \
#     --data_split test \
#     --metric exact_match \
#     --verbose