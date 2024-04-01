#!/bin/bash

# conda env: s2j
# command: bash scripts/chemtables/prompt_error_recovery_os.sh

API_SOURCE=open_source
MAX_TOKENS=256

BACKEND=codellama-13b-instruct

python run.py \
    --run_type prompt \
    --api_source $API_SOURCE \
    --backend $BACKEND \
    --task chemtables \
    --prompt_setting error_recovery \
    --template template_filling \
    --task_start_index 0 \
    --task_end_index 100 \
    --max_tokens $MAX_TOKENS \
    --data_split test \
    --num_passes 200

# BACKEND=llama2-chat-13b

# python run.py \
#     --run_type prompt \
#     --api_source $API_SOURCE \
#     --backend $BACKEND \
#     --task chemtables \
#     --prompt_setting error_recovery \
#     --template template_filling_llama2chat \
#     --task_start_index 0 \
#     --task_end_index 100 \
#     --max_tokens $MAX_TOKENS \
#     --data_split test \
#     --num_passes 100