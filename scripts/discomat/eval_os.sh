#!/bin/bash

# conda env: s2j
# command: bash scripts/discomat/eval_os.sh

API_SOURCE=open_source

BACKEND=codellama-13b-instruct

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