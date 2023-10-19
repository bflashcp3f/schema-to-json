#!/bin/bash

# conda env: s2j
# command: bash scripts/swde/prompt.sh

API_SOURCE=openai
# BACKEND=gpt-3.5-turbo
# MAX_TOKENS=256

BACKEND=gpt-4
MAX_TOKENS=256

# API_SOURCE=azure
# # BACKEND=gpt35-turbo
# # MAX_TOKENS=256

# BACKEND=gpt4
# MAX_TOKENS=256

python run.py \
    --run_type prompt \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task swde \
    --prompt_setting one_by_one \
    --task_start_index 0 \
    --task_end_index 1600 \
    --max_tokens $MAX_TOKENS \
    --stop $'\n' \
    --data_split test \
    --batch_size 2 \
    --num_passes 1 \
    --sleep_time 60 \
    --async_prompt
