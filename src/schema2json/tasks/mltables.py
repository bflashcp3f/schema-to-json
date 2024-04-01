
import re
import os
import json
import sympy
import ast
import time

import pandas as pd
import networkx as nx

from collections import Counter, defaultdict

from schema2json.tasks.base import Task, DATA_PATH
from schema2json.methods.prompt import *
from schema2json.templates.mltables import * 


ANNOTATION_ATTRIBUTES = {
    "Result": ["value", "char_index", "type", "training data/set", "test data/set", "task", "metric", "model", "experimental settings", "model settings"],
    "Data Stat.": ["value", "char_index", "type", "dataset", "attribute name", "sub-set/group name", "dataset features"],
    "Hyper-parameter/Architecture": ["value", "char_index", "type", "model", "parameter/architecture name", "dataset"],
}


TEMPLATE_ATTRIBUTES = {
    "Result": ["value", "char_index", "type", "training data/set", "test data/set", "task", "metric", "model/method", "experimental settings", "model/method settings"],
    "Data Stat.": ["value", "char_index", "type", "dataset", "attribute name", "sub-set/group name", "dataset features"],
    "Hyper-parameter/Architecture": ["value", "char_index", "type", "model", "parameter/architecture name", "dataset"],
}


TEMP_ANNO_ALIGNMENT = {
    "type": "type", "char_index": "char_index", "value": "value", "training data/set": "training data/set", "test data/set": "test data/set", 
    "task": "task", "metric": "metric", "model/method": "model", "experimental settings": "experimental settings", 
    "model/method settings": "model settings", "dataset": "dataset", "attribute name": "attribute name", "model": "model",
    "sub-set/group name": "sub-set/group name", "dataset features": "dataset features", "parameter/architecture name": "parameter/architecture name"
}


FREE_FORM_ANNO = ["experimental settings", "model settings", "dataset features"]
FREE_FORM_PROMPT = ["experimental settings", "model/method settings", "dataset features"]

def process_eval_string(eval_string):

    # Remove \\cite{...} and \\ref{...} using regex
    try:
        eval_string = re.sub(r'\\cite.*?\{.*?\}', '', eval_string)
        eval_string = re.sub(r'\\ref\{.*?\}', '', eval_string)
    except:
        print([eval_string])
        raise

    return ' '.join([item for item in re.sub(r'(?<!\\)([{}[\],()@$\-+%&#_^~|<>\\/])', r' ', eval_string).split() if item]).lower()


def word_f1(gold_attribute_value, pred_attribute_value):
    tp = len(list((Counter(gold_attribute_value.split()) & Counter(pred_attribute_value.split())).elements()))
    fp = len(pred_attribute_value.split()) - tp
    fn = len(gold_attribute_value.split()) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1



class CellItem:
    def __init__(self, item: list):
        cell_value, cell_raw, cell_value_start, cell_value_end, cell_raw_start, cell_raw_end = item
        self.cell_value = cell_value
        self.cell_raw = cell_raw
        self.cell_value_start = cell_value_start
        self.cell_value_end = cell_value_end
        self.cell_raw_start = cell_raw_start
        self.cell_raw_end = cell_raw_end


class MlTablesItem:
    def __init__(self, table_id: str, table_item: dict, output_dir: str):
        self.paper_id = table_id.split('_')[0]
        self.table_id = table_id
        self.table_code = table_item['table_code']
        self.table_code_full = table_item['table_source']
        self.table_label = table_item['table_label']
        self.cells_extracted = [CellItem(cell_item) for cell_item in table_item['table_numeric_cells']]
        self.table_text = table_item['text_chunk_selected']
        self.cell_list_gold = table_item['cell_list_gold']

        self.output_dir = os.path.join(output_dir, self.table_id)
        
        self.cell_list_pred, self.last_pass_idx = self.load_predictions()
        self.current_pass_idx = self.last_pass_idx + 1
        self.fail_count = 0
        self.prompt = None
        self.prompt_response = None

    def load_predictions(self):
        if not os.path.exists(f'{self.output_dir}/pass_0_processed.jsonl'):
            return [], -1
        else:
            # Get the last pass
            last_pass_idx = max([int(file_name.split('_')[-2]) for file_name in os.listdir(self.output_dir) if file_name.startswith('pass') and file_name.endswith('processed.jsonl')])

            with open(f'{self.output_dir}/pass_{last_pass_idx}_processed.jsonl') as f:
                cell_list_pred = [json.loads(line) for line in [str(line) for line in f]]
                
            return cell_list_pred, last_pass_idx

    def get_next_prompt_cell(self) -> CellItem:
        if len(self.cell_list_pred) == 0:
            return self.cells_extracted[0]
        else:
            assert len(self.cell_list_pred) < len(self.cells_extracted)
            return self.cells_extracted[len(self.cell_list_pred)]
        
    def prompt_wrap(self, template: str, encoding, max_output_tokens, max_length, open_source=False) -> str:
        
        # Control the length of the table code and table text
        table_text = self.table_text
        table_code_text = f"{table_text}\n\n{self.table_code}"

        while num_tokens_from_string(table_code_text, encoding, open_source) > (max_length/2) and table_text.strip() != '':
            print(f"Table code length: {num_tokens_from_string(table_code_text, encoding, open_source)}")
            table_text = '\n\n'.join(table_text.split('\n\n')[1:])
            table_code_text = f"{table_text}\n\n{self.table_code}"

        # Control the length of the predicted cell description
        next_prompt_cell = self.get_next_prompt_cell()
        prompt_cell_prefix = '{"value": ' + f'"{next_prompt_cell.cell_value}", "type":'

        cell_describe_idx = 0
        prompt_prefix = '\n'.join([json.dumps(cell_describe) for cell_describe in self.cell_list_pred[cell_describe_idx:]] + [prompt_cell_prefix])

        while num_tokens_from_string(prompt_prefix, encoding, open_source) + max_output_tokens + num_tokens_from_string(template, encoding, open_source) > (max_length-num_tokens_from_string(table_code_text, encoding, open_source)) and cell_describe_idx < len(self.cell_list_pred):
            print(f"prompt_prefix length: {num_tokens_from_string(prompt_prefix, encoding, open_source)}")
            # Remove the first cell description in the prompt prefix
            cell_describe_idx += 1
            prompt_prefix = '\n'.join([json.dumps(cell_describe) for cell_describe in self.cell_list_pred[cell_describe_idx:]] + [prompt_cell_prefix])

        if len(self.cell_list_pred) > 0:
            assert cell_describe_idx < len(self.cell_list_pred)
                                       
        assert "{{table_code_text}}" in template
        prompt = template.replace("{{table_code_text}}", table_code_text)

        assert "{{prompt_prefix}}" in prompt
        prompt = prompt.replace("{{prompt_prefix}}", prompt_prefix)

        num_prompt_tokens = num_tokens_from_string(prompt, encoding, open_source)
        print(f"Number of prompt tokens: {num_prompt_tokens}")
        assert num_prompt_tokens + max_output_tokens <= max_length

        self.prompt = prompt

        return prompt
    
    def update_pass_idx(self) -> None:
        self.last_pass_idx = self.current_pass_idx
        self.current_pass_idx = self.current_pass_idx + 1

    def increment_fail_count(self) -> None:
        self.fail_count = self.fail_count + 1

    def reset_fail_count(self) -> None:
        self.fail_count = 0

    def process_output(self, response: str) -> None:

        if response.startswith('{"value":'):
            response = response.lstrip().lstrip('{"value":')
        elif response.startswith('{'):
            response = response.lstrip('{')

        next_prompt_cell = self.get_next_prompt_cell()
        prompt_cell_prefix = '{"value": ' + f'"{next_prompt_cell.cell_value}", "type":'

        output_raw = prompt_cell_prefix + response
        print("Output raw:\n", output_raw, '\n')

        # Parse the output to a list of JSON objects
        output_raw = output_raw.split('\n\n')[0]
        output_list = output_raw.split('\n')
        output_processed = []
        for output in output_list:

            # Handle potential escape characters for parsing
            if output.count('\\') > output.count('\\\\')*2:
                output = re.sub(r'\\{1}', r'\\\\', output)
                output = re.sub(r'\\{2,}', r'\\\\', output)

            # replace '\\\\"' in output with "'" (e.g., "the \\\\" is" -> "the ' is")
            if '\\\\"' in output:
                output = output.replace('\\\\"', "'")

            output = output.strip().strip('.')
            # breakpoint()
            if output.count('"value"') > 1:
                output = output[:output.index('"value":')+len('"value":')] + output[output.index('"value":')+len('"value":'):].replace('"value"', '"random"')

            try:
                output = json.loads(output)
            except:
                if 'Other' in output and '"type":' in output and '"' not in output[output.index('"value":')+8:output.index('"type":')].strip().strip(',"'):
                    output = output[:output.index('"type":')] + '"type": "Other"}'
                    output = json.loads(output)

                elif len(output_list) > 1:
                    continue
                else:
                    assert '"type":' in output
                    output = output[:output.index('"type":')] + '"type": "Other"}'
                    output = json.loads(output)

            if 'value' not in output or output['value'] in ['xx', '']:
                continue

            output_processed.append(output)
        return output_processed

    def update_output(self, response):

        self.prompt_response = response

        # Process the raw response
        output_processed = self.process_output(response)

        # Check if the output align with the extracted cells. If not, skip the output
        cell_idx, output_idx = 0, 0
        cell_list_expected = self.cells_extracted[len(self.cell_list_pred):]
        cell_values_expected = [cell.cell_value for cell in cell_list_expected]

        output_cleaned = []
        while cell_idx < len(cell_list_expected) and output_idx < len(output_processed):
            
            output = output_processed[output_idx]

            cell_exp_value = cell_list_expected[cell_idx].cell_value
            cell_exp_value_start = cell_list_expected[cell_idx].cell_value_start
            cell_exp_value_end = cell_list_expected[cell_idx].cell_value_end
            # assert cell_exp_value == self.table_code[cell_exp_value_start:cell_exp_value_end]

            # Check the predicted cell value is the same as the desired cell value
            if output['value'] == cell_exp_value:
                output_cleaned.append(self.check_free_form_attribute(output))
                cell_idx += 1
                output_idx += 1
            else:
                assert cell_idx > 0 # The first cell should always be correct
                print(f"Cell {output['value']} not follow the desired order")
                print(f"Desired cells: {cell_values_expected}")
                break

        # Update the cell_list_pred
        self.cell_list_pred.extend(output_cleaned)

        for item in self.cell_list_pred:
            print(item)

    def check_free_form_attribute(self, output: dict) -> bool:

        # Check whether the model output "experimental settings" or "model settings" not in the table
        output_checked = {}
        for key, value in output.items():
            # Check whether value is a dictionary
            if isinstance(value, dict) and not ("xx" in value and len(value["xx"]) == 1):
                value_checked = {}
                for key1, value1 in value.items():
                    word_list = (str(key1) + ' ' + str(value1)).lower().split()
                    word_list_filtered = [word for word in word_list if word not in ['of', 'number']]
                    # Check whether the word is in the table. If yes, add it to the dictionary value_checked
                    # if any(word in table_code.lower().split() for word in word_list_filtered):
                    if any(word in self.table_code.lower() for word in word_list_filtered):
                        value_checked[key1] = value1
                    
                    # Check whether the length of the value_checked is 5. If yes, break the loop
                    if len(value_checked) == 5:
                        break
                    
                output_checked[key] = value_checked if len(value_checked) > 0 else {"xx": "yy"}

            else:
                output_checked[key] = value

        return output_checked

    def save_output(self, args):
                
        # Create the directory to store the output
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Dump the processed output to a jsonl file, where each line is a JSON object
        output_processed_path = os.path.join(self.output_dir, f"pass_{self.current_pass_idx}_processed.jsonl")
        with open(output_processed_path, 'w') as f:
            for output in self.cell_list_pred:
                json.dump(output, f)
                f.write('\n')

        # Dump the raw output to a txt file
        output_raw_path = os.path.join(self.output_dir, f"pass_{self.current_pass_idx}_output.txt")
        with open(output_raw_path, 'w') as f:
            f.write(self.prompt.strip() + self.prompt_response)

        # Dump the prompt input to a txt file
        prompt_input_path = os.path.join(self.output_dir, f"pass_{self.current_pass_idx}_prompt.txt")
        with open(prompt_input_path, 'w') as f:
            f.write(self.prompt)

        # Dump the prompting setting to a json file
        prompt_setting_path = os.path.join(self.output_dir, f"pass_{self.current_pass_idx}_setting.json")
        with open(prompt_setting_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

    def save_error_message(self, prompt_list: list, response_list: list) -> None:

        # Create the directory to store the output
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        error_path = os.path.join(self.output_dir, '..', f"error.txt")
        
        for prompt, response in zip(prompt_list, response_list):

            if response[0] == False:
                error_message = response[1]
                with open(error_path, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {self.table_id} error message: {error_message}\n\n")
                    f.write(f"Prompt: {prompt}\n\n")

    def generate_cell_index(self):

        # Generate the cell index for the predicted cells based on the extracted cells
        cell_extract_idx, pred_idx = 0, 0
        cell_list_pred_indexed = []

        while cell_extract_idx < len(self.cells_extracted) and pred_idx < len(self.cell_list_pred):

            cell_extract = self.cells_extracted[cell_extract_idx]
            cell_pred = self.cell_list_pred[pred_idx]

            if cell_extract.cell_value == cell_pred['value']:
                cell_pred['char_index'] = [cell_extract.cell_value_start, cell_extract.cell_value_end]
                cell_list_pred_indexed.append(cell_pred)
                cell_extract_idx += 1
                pred_idx += 1
            else:
                assert cell_extract_idx > 0
                raise ValueError(f"Cell {cell_pred['value']} not follow the desired order")
            
        return cell_list_pred_indexed
    
    def count_valid_attributes(self, json_obj):
        num_valid_attributes = 0
        for attribute, value in json_obj.items():

            if attribute in ['char_index', 'value']:
                continue

            if attribute == 'type' and value == 'Other':
                num_valid_attributes += 1
                continue

            assert attribute in ANNOTATION_ATTRIBUTES[json_obj['type']]
            if attribute not in FREE_FORM_ANNO:
                if value != 'xx':
                    num_valid_attributes += 1
            else:
                if value != {"xx": "yy"}:
                    num_valid_attributes += 1
        return num_valid_attributes

    def get_pair_attribute_result(self, cell_gold, cell_pred, metric='f1', threshold=0.25):

        TP, FP, FN = 0, 0, 0

        # Find the shared attributes
        shared_attributes = [item for item in set(cell_gold.keys()) & set(cell_pred.keys()) if item not in ['value', 'char_index']]

        # Calculate the TP, FP, FN for each attribute
        for attribute in shared_attributes:
            if attribute not in FREE_FORM_ANNO:
                if cell_gold[attribute] == 'xx' and cell_pred[attribute] == 'xx':
                    continue
                elif cell_gold[attribute] == 'xx' and cell_pred[attribute] != 'xx':
                    FP += 1
                elif cell_gold[attribute] != 'xx' and cell_pred[attribute] == 'xx':
                    FN += 1
                else:
                    gold_attribute_value = cell_gold[attribute]
                    pred_attribute_value = cell_pred[attribute]

                    gold_attribute_value = [process_eval_string(item) for item in (gold_attribute_value if isinstance(gold_attribute_value, list) else [gold_attribute_value])]
                    pred_attribute_value = process_eval_string(pred_attribute_value)

                    if metric == 'exact_match':
                        if pred_attribute_value in gold_attribute_value:
                            TP += 1
                        else:
                            FP += 1
                            FN += 1
                    elif metric == 'f1':
                        if max([word_f1(gold_attribute, pred_attribute_value) for gold_attribute in gold_attribute_value]) >= threshold:
                            TP += 1
                        else:
                            FP += 1
                            FN += 1
                    else:
                        raise ValueError(f'Unknown metric: {metric}')
            else:
                if cell_gold[attribute] == {"xx": "yy"} and cell_pred[attribute] == {"xx": "yy"}:
                    continue
                elif cell_gold[attribute] == {"xx": "yy"} and cell_pred[attribute] != {"xx": "yy"}:
                    FP += 1
                elif cell_gold[attribute] != {"xx": "yy"} and cell_pred[attribute] == {"xx": "yy"}:
                    FN += 1
                else:
                    gold_attribute_value = cell_gold[attribute]
                    pred_attribute_value = cell_pred[attribute]

                    gold_attribute_value = [process_eval_string(' '.join([key, value])) for key, value in gold_attribute_value.items()]
                    pred_attribute_value = [process_eval_string(' '.join([key, value])) for key, value in [(key, value) for key, value in pred_attribute_value.items() if isinstance(value, str)]] # Prevent the nested dict

                    # Match the gold and pred attributes using bipartite matching
                    edges = []
                    edge_score_dict = {}
                    for i, gold_attribute in enumerate(gold_attribute_value):
                        for j, pred_attribute in enumerate(pred_attribute_value):
                            if metric == 'exact_match':
                                score = int(sorted(set(gold_attribute.split())) == sorted(set(pred_attribute)))
                            elif metric == 'f1':
                                score = word_f1(gold_attribute, pred_attribute)
                            else:
                                raise ValueError(f'Unknown metric: {metric}')

                            if score > threshold:
                                edges.append((f'gold_{i}', f'pred_{j}', score))
                                edge_score_dict[f'{i}-{j}'] = 1 if score >= threshold else 0

                    # Bipartite matching
                    G = nx.Graph()
                    G.add_weighted_edges_from(edges)
                    matched_pairs = nx.max_weight_matching(G, maxcardinality=True)

                    paired_gold_pred_idx = [[int(matched_item[0].split('_')[1]), int(matched_item[1].split('_')[1])] if matched_item[0].split('_')[0].startswith('gold') else [int(matched_item[1].split('_')[1]), int(matched_item[0].split('_')[1])] for matched_item in matched_pairs]

                    tp = 0
                    for gold_idx, pred_idx in paired_gold_pred_idx:
                        assert f'{gold_idx}-{pred_idx}' in edge_score_dict
                        tp += 1 if edge_score_dict[f'{gold_idx}-{pred_idx}'] >= threshold else 0

                    precision = tp / len(pred_attribute_value) if len(pred_attribute_value) > 0 else 0
                    recall = tp / len(gold_attribute_value) if len(gold_attribute_value) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    if f1 >= 0.5:
                        TP += 1
                    else:
                        FP += 1
                        FN += 1

        unshared_attributes_pred = [item for item in set(cell_pred.keys()) - set(cell_gold.keys()) if item not in ['value', 'char_index']]
        for attribute in unshared_attributes_pred:
            if attribute not in FREE_FORM_ANNO:
                if cell_pred[attribute] != 'xx':
                    FP += 1
            else:
                if cell_pred[attribute] != {"xx": "yy"}:
                    FP += 1

        unshared_attributes_gold = [item for item in set(cell_gold.keys()) - set(cell_pred.keys()) if item not in ['value', 'char_index']]
        for attribute in unshared_attributes_gold:
            if attribute not in FREE_FORM_ANNO:
                if cell_gold[attribute] != 'xx':
                    FN += 1
            else:
                if cell_gold[attribute] != {"xx": "yy"}:
                    FN += 1

        return TP, FP, FN


    def eval(self, metric, threshold, verbose=False):

        # Evaluate per table
        pred_data = self.generate_cell_index()
        gold_data = self.cell_list_gold

        TP_per_table, FP_per_table, FN_per_table = 0, 0, 0

        # Post-process the predicted data to match the annotation attribute names
        pred_data_mod = []
        for pred_item in pred_data:
            if 'type' in pred_item and pred_item['type'] in TEMPLATE_ATTRIBUTES.keys():
                pred_item_mod ={}
                for attribute in TEMPLATE_ATTRIBUTES[pred_item['type']]:
                    if attribute in pred_item:
                        pred_item_mod[TEMP_ANNO_ALIGNMENT[attribute]] = pred_item[attribute]
                    else:
                        pred_item_mod[TEMP_ANNO_ALIGNMENT[attribute]] = 'xx' if attribute not in FREE_FORM_PROMPT else {"xx": "yy"}
                pred_data_mod.append(pred_item_mod)
            else:
                # Change the type of the cell description to "Other"
                pred_item_mod ={}
                pred_item_mod["value"] = pred_item["value"]
                pred_item_mod["char_index"] = pred_item["char_index"]
                pred_item_mod["type"] = "Other"
                pred_data_mod.append(pred_item_mod)

        pred_data = pred_data_mod

        pred_char_index_dict = dict([('-'.join([str(char_index) for char_index in item['char_index']]), item) for item in pred_data])
        assert len(pred_char_index_dict) == len(pred_data)

        gold_char_index_dict = dict([('-'.join([str(char_index) for char_index in item['char_index']]), item) for item in gold_data])
        assert len(gold_char_index_dict) == len(gold_data)

        pred_cells_unmatched = [pred_item for char_index, pred_item in pred_char_index_dict.items() if char_index not in gold_char_index_dict]
        pred_cells_matched = [pred_item for char_index, pred_item in pred_char_index_dict.items() if char_index in gold_char_index_dict]
        FP_per_table += sum([self.count_valid_attributes(item) for item in pred_cells_unmatched])

        gold_cells_unmatched = [gold_item for char_index, gold_item in gold_char_index_dict.items() if char_index not in pred_char_index_dict]
        gold_cells_matched = [gold_item for char_index, gold_item in gold_char_index_dict.items() if char_index in pred_char_index_dict]
        FN_per_table += sum([self.count_valid_attributes(item) for item in gold_cells_unmatched])
        assert len(pred_cells_matched) == len(gold_cells_matched)

        pred_gold_pairs_matched = [(pred_item, gold_char_index_dict[char_index]) for char_index, pred_item in pred_char_index_dict.items() if char_index in gold_char_index_dict]
        assert len(pred_gold_pairs_matched) == len(pred_cells_matched)

        for pred_item, gold_item in pred_gold_pairs_matched:
            TP_per_cell, FP_per_cell, FN_per_cell = self.get_pair_attribute_result(gold_item, pred_item, metric=metric, threshold=threshold)   
            TP_per_table += TP_per_cell
            FP_per_table += FP_per_cell
            FN_per_table += FN_per_cell  

        return TP_per_table, FP_per_table, FN_per_table


class MLTables(Task):
    def __init__(self, args) -> None:
        self.data_path = os.path.join(DATA_PATH, 'mltables' if args.task == 'mltables' else 'mltables_html', f'{args.data_split}.json')
        self.template = globals()[args.template]
        self.args = args
        if args.api_source == 'openai':
            model_name = args.backend
        elif args.api_source == 'azure':
            model_name = AZURE_MODELS[args.backend]
        elif args.api_source == 'open_source':
            model_name = args.backend
        else:
            raise ValueError(f'Unknown api source: {args.api_source}')
        self.output_dir = os.path.join(DATA_PATH, 'predict', self.args.task, self.args.data_split, model_name, args.template)
        self.data = self.load_data(args.task_start_index, args.task_end_index)

    def load_data(self, start_index: int, end_index: int) -> list:
        with open(self.data_path) as f:
            data = json.load(f)
            data = sorted(list(data.items()), key=lambda x: x[0])
            
        if start_index != -1:
            return [MlTablesItem(item[0], item[1], self.output_dir) for item in data[start_index:end_index]]
        else:
            return [MlTablesItem(item[0], item[1], self.output_dir) for item in data]
        
    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx].prompt
    
    def get_input_list(self) -> list:
        return [self.get_input(idx) for idx in range(len(self.data))]
    
    def get_data(self) -> list:
        return self.data
    
