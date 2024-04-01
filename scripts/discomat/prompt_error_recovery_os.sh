#!/bin/bash

# conda env: s2j
# command: bash scripts/discomat/prompt_error_recovery_os.sh

API_SOURCE=open_source

BACKEND=codellama-13b-instruct

MAX_TOKENS=256

python run.py \
    --run_type prompt \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task discomat \
    --prompt_setting error_recovery \
    --template template_filling \
    --task_start_index 0 \
    --task_end_index 200 \
    --max_tokens $MAX_TOKENS \
    --data_split test \
    --num_passes 200

