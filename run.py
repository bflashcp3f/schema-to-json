import sys
import os
import re
import openai
import json
import time
import argparse
import tiktoken

from collections import defaultdict, Counter
from pathlib import Path

from schema2json.tasks import get_task
from schema2json.methods.prompt import prompt_one_by_one, prompt_error_recovery
from schema2json.methods.eval import eval, eval_discomat, eval_swde


def main(args):

    print(f"Arguments: {vars(args)}")

    # Get the task
    task = get_task(args)

    if args.run_type == 'prompt':

        # Set the OpenAI API key
        if args.api_source == 'openai':
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif args.api_source == 'azure':
            openai.api_type = "azure"
            openai.api_base = "https://inference.openai.azure.com/"
            openai.api_version = "2023-07-01-preview"
            openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        else:
            raise ValueError(f"API source {args.api_source} not supported")

        # Run the task
        if args.prompt_setting == 'error_recovery':
            prompt_error_recovery(args, task)
        elif args.prompt_setting == 'one_by_one':
            prompt_one_by_one(args, task)
        else:
            raise ValueError(f"Prompt setting {args.prompt_setting} not supported")
    
    else:

        # Evaluate the task
        if args.task in ['mltables', 'chemtables']:
            eval(args, task)
        elif args.task == 'discomat':
            eval_discomat(args, task)
        elif args.task == 'swde':
            eval_swde(args, task)
        else:
            raise ValueError(f"Task {args.task} not supported")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_type', type=str, choices=['prompt', 'eval'], required=True)

    parser.add_argument('--api_source', type=str, choices=['openai', 'azure'], required=True)
    parser.add_argument('--backend', type=str, required=True)
    parser.add_argument('--prompt_setting', type=str, default='error_recovery', choices=['error_recovery', 'one_by_one'], required=True)
    
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--template', type=str)
    parser.add_argument('--task_start_index', type=int, default=-1)
    parser.add_argument('--task_end_index', type=int, default=-1)
    parser.add_argument('--data_split', type=str, default='dev', required=True)

    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--stop', type=str, default='\n')

    parser.add_argument('--sleep_time', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_passes', type=int, default=1)
    parser.add_argument('--num_fail', type=int, default=3)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--async_prompt', action='store_true')

    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'exact_match'])
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    main(args)