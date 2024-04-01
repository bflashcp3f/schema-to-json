#!/bin/bash

# conda env: s2j
# command: bash scripts/swde/prompt_os.sh

API_SOURCE=open_source
MAX_TOKENS=128

BACKEND=codellama-13b-instruct

python run.py \
    --run_type prompt \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task swde \
    --prompt_setting one_by_one \
    --task_start_index 0 \
    --task_end_index 1600 \
    --max_tokens $MAX_TOKENS \
    --data_split test \
    --batch_size 2 \
    --num_passes 1 \
