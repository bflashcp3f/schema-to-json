
import os
import json
import time
import openai
import backoff 
import asyncio
import tiktoken
import torch
import transformers

from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from schema2json.tasks.base import Task, DATA_PATH


CHATCOMPLETION_MODEL = ['gpt4', 'gpt35-turbo', 'gpt-4', 'gpt-3.5-turbo', 'gpt-35-turbo-0613', 'gpt-35-turbo-16k', 'gpt4', 'gpt35-turbo']

AZURE_MODELS = {
    'gpt4': 'gpt-4',
    'gpt35-turbo': 'gpt-3.5-turbo',
    'gpt-35-turbo-0613': 'gpt-3.5-turbo',
    'gpt-35-turbo-16k': 'gpt-3.5-turbo-16k',
}

OPEN_SOURCE_MODELS = {
    'tablellama': 'osunlp/TableLlama',
    'llama2-chat-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-chat-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'codellama-7b-instruct': 'codellama/CodeLlama-7b-Instruct-hf',
    'codellama-13b-instruct': 'codellama/CodeLlama-13b-Instruct-hf',
    'codellama-34b-instruct': 'codellama/CodeLlama-34b-Instruct-hf',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.1',
}

def num_tokens_from_string(string: str, encoding, open_source=False) -> int:
    """Returns the number of tokens in a text string."""

    if not open_source:
        num_tokens = len(encoding.encode(string))
    else:
        num_tokens = len(encoding.encode(string, return_tensors='pt')[0])
    return num_tokens


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def chatcompletions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def prompt_sync(prompt_list, api_source, sleep_time, model, temperature, top_p, max_tokens, stop) -> list:

    response_list = []
    for prompt in prompt_list:

        print(f'Start prompting...')
        start_time = time.time()
        
        try:
            if model in CHATCOMPLETION_MODEL:
                messages = [{"role": "user", "content": prompt}]
                if api_source == 'openai':
                    response = chatcompletions_with_backoff(model=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
                elif api_source == 'azure':
                    response = chatcompletions_with_backoff(engine=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
                else:
                    raise ValueError(f"api_source {api_source} not supported")
                response = response["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"model {model} not supported")
            response_list.append((True, response))
        except Exception as e:
            print(e)
            response_list.append((False, e))

        end_time = time.time()
        print(f'Finished prompting in {end_time - start_time} seconds')

        # Sleep to avoid hitting the API rate limit
        print(f"Sleep for {sleep_time} seconds")
        time.sleep(sleep_time)
        
    return response_list


def prompt_os(prompt_list, model_name, model, encoding, max_tokens) -> list:

    response_list = []
    for prompt in prompt_list:

        try:
            if model_name in OPEN_SOURCE_MODELS:

                start_time = time.time()
                outputs = model(
                    prompt,
                    do_sample=True,
                    top_p=1,
                    temperature=0.001,
                    num_return_sequences=1,
                    eos_token_id=encoding.eos_token_id,
                    max_new_tokens=max_tokens,
                )
                end_time = time.time()
                print(f'Finish prompting in {end_time - start_time} seconds')

                response = outputs[0]['generated_text'][len(prompt):]
                response_list.append((True, response))
                
            else:
                raise ValueError(f"model {model} not supported")
        except Exception as e:
            print(e)
            response_list.append((False, e))

    return response_list


async def dispatch_chatcompletion_requests(
    api_source: str,
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: str,
) -> list[str]:
    # Dispatches requests to OpenAI API asynchronously.
    
    if api_source == 'openai':
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)
    elif api_source == 'azure':
        async_responses = [
            openai.ChatCompletion.acreate(
                engine=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)
    else:
        raise ValueError(f"api_source {api_source} not supported")


async def dispatch_completion_requests(
    api_source: str,
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: str,
) -> list[str]:
    # Dispatches requests to OpenAI API asynchronously.

    if api_source == 'openai':
        async_responses = [
            openai.Completion.acreate(
                model=model,
                prompt=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)
    elif api_source == 'azure':
        async_responses = [
            openai.Completion.acreate(
                engine=model,
                prompt=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)
    else:
        raise ValueError(f"api_source {api_source} not supported")


def prompt_async(prompt_list, api_source, sleep_time, model, temperature, top_p, max_tokens, stop, batch_size) -> list:

    response_list = []
    for i in range(0, len(prompt_list), batch_size):
        print(f"Running batch {i} to {i + min(batch_size, len(prompt_list)-i)}")
        prompt_list_batch = prompt_list[i : i + batch_size]

        if model in CHATCOMPLETION_MODEL:
            messages_list = [[{"role": "user", "content": prompt}] for prompt in prompt_list_batch]

            response_list_batch = asyncio.run(
                dispatch_chatcompletion_requests(
                    api_source,
                    messages_list,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
            )
            response_list_batch_no_content = [item for item in response_list_batch if "content" not in item["choices"][0]["message"]]
            if len(response_list_batch_no_content) > 0:
                for item in response_list_batch_no_content:
                    print(item)

            response_list_batch = [item["choices"][0]["message"]["content"] if "content" in item["choices"][0]["message"] else "" for item in response_list_batch]
            response_list_batch = [(True, response) for response in response_list_batch]
            
        else:
            raise ValueError(f"model {model} not supported")

        response_list.extend(response_list_batch)

        # Sleep to avoid hitting the API rate limit
        print(f"Sleep for {sleep_time} seconds")
        time.sleep(sleep_time)

    return response_list


def prompt_naive(args, task):
    
    template = task.template
    output_prefix = template.split("\n")[-1]
    prompt_list = task.get_input_list()

    if not args.async_prompt:
        response_list = prompt_sync(prompt_list, args.api_source, args.sleep_time, args.backend, args.temperature, args.top_p, args.max_tokens, args.stop)
    else:
        response_list = prompt_async(prompt_list, args.api_source, args.sleep_time, args.backend, args.temperature, args.top_p, args.max_tokens, args.stop, args.batch_size)

    # Update the task with the response
    task.update_output(response_list)

    # Save the task output
    task.save_output()


def prompt_error_recovery(args, task, open_source=False):

    template = task.template

    if not open_source:
        if args.api_source == 'openai':
            encoding = tiktoken.encoding_for_model(args.backend)
            max_length = 4000 if 'gpt-3.5' in args.backend else 8000
        else:
            encoding = tiktoken.encoding_for_model(AZURE_MODELS[args.backend])
            max_length = 4000 if 'gpt-3.5' in AZURE_MODELS[args.backend] else 8000
    else:
        model_name = OPEN_SOURCE_MODELS[args.backend]
        encoding = AutoTokenizer.from_pretrained(model_name)
        encoding.pad_token = encoding.eos_token
        max_length = 6000 if encoding.model_max_length > 6000 else encoding.model_max_length

        model = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            model_kwargs= {
                "device_map": "auto", 
                "load_in_4bit": True,
                # "load_in_8bit": True
            }
        )

    for table in task.get_data():

        if args.task == 'discomat' and not table.comp_table_pred:
            continue

        num_table_code_tokens = num_tokens_from_string(table.table_code, encoding, open_source)

        if table.table_text is not None:
            num_table_text_tokens = num_tokens_from_string(table.table_text, encoding, open_source)
            print(f"{table.table_id}: code - {num_table_code_tokens} tokens, text - {num_table_text_tokens} tokens, total - {num_table_code_tokens+num_table_text_tokens} tokens")
        else:
            print(f"{table.table_id}: code - {num_table_code_tokens} tokens")

        if table.last_pass_idx == args.num_passes-1:
            print(f"Skip {table.table_id} for reaching the pass limit.")
            continue
        else:
            while table.current_pass_idx < args.num_passes:

                if len(table.cell_list_pred) == len(table.cells_extracted):
                    print(f"Skip {table.table_id} for extracting all the extracted cells.")
                    break

                # Check whether the fail count has reached the threshold of 5
                if table.fail_count >= args.num_fail:
                    break

                print(f"Extract cell descriptions for pass {table.current_pass_idx}")
                prompt = table.prompt_wrap(template, encoding, 1024 if table.current_pass_idx==0 and args.max_tokens>1024 else args.max_tokens, max_length, open_source)
                prompt_list = [prompt]

                if not open_source:
                    response_list = prompt_sync(prompt_list, args.api_source, args.sleep_time, args.backend, args.temperature, args.top_p, args.max_tokens, args.stop)
                else:
                    response_list = prompt_os(prompt_list, args.backend, model, encoding, args.max_tokens)

                # breakpoint()
                # Check whether the responses are valid (w/o prompting errors)
                if sum([item[0] for item in response_list]) == len(prompt_list):

                    # Only one prompt each time for error recovery
                    assert len(response_list) == 1

                    # Update the responses
                    table.update_output(response_list[0][1])
                    # breakpoint()

                    # Save the responses
                    table.save_output(args)

                    # Update the current pass index
                    table.update_pass_idx()
                    table.reset_fail_count()

                else:

                    table.save_error_message(prompt_list, response_list)
                    table.increment_fail_count()


def prompt_one_by_one(args, task, open_source=False):

    if not open_source:
        if args.api_source == 'openai':
            # encoding = tiktoken.encoding_for_model(args.backend)
            encoding = None
            max_length = 4000 if 'gpt-3.5' in args.backend else 8000
        else:
            # encoding = tiktoken.encoding_for_model(AZURE_MODELS[args.backend])
            encoding = None
            max_length = 4000 if 'gpt-3.5' in AZURE_MODELS[args.backend] else 8000
    else:
        model_name = OPEN_SOURCE_MODELS[args.backend]
        encoding = AutoTokenizer.from_pretrained(model_name)
        encoding.pad_token = encoding.eos_token
        max_length = 8000 if encoding.model_max_length > 8000 else encoding.model_max_length

        model = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            model_kwargs= {
                "device_map": "auto", 
                "load_in_4bit": True,
                # "load_in_8bit": True
            }
        )

    prompt_list = task.get_prompt_list(encoding, max_length)

    if not prompt_list:
        print("All pages have been processed")
        return

    if not open_source:

        if not args.async_prompt:
            response_list = prompt_sync(prompt_list, args.api_source, args.sleep_time, args.backend, args.temperature, args.top_p, args.max_tokens, args.stop)
        else:
            response_list = prompt_async(prompt_list, args.api_source, args.sleep_time, args.backend, args.temperature, args.top_p, args.max_tokens, args.stop, args.batch_size)
                    
    else:
        response_list = prompt_os(prompt_list, args.backend, model, encoding, args.max_tokens)

    # Update the task with the response
    task.update_output(response_list)
