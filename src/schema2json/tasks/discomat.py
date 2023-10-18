

import re
import os
import json
import time

from collections import Counter, defaultdict

from schema2json.tasks.base import Task, DATA_PATH
from schema2json.methods.prompt import *
from schema2json.templates.discomat import * 


class CellItem:
    def __init__(self, item: dict):
        self.cell_value = item['cell_value_processed']
        self.cell_index = [item['i'], item['j'], item['k']]


class DiSCoMaTItem:
    def __init__(self, paper_table_id: str, table_item: dict, output_dir: str):
        self.paper_id = paper_table_id.split('::')[0]
        self.table_id = paper_table_id
        self.table_only_id = paper_table_id.split('::')[1]
        self.table_code = table_item['table_processed']
        self.table_text = None
        self.cells_extracted = [CellItem(cell_item) for cell_item in table_item['numeric_cells']]
        self.cell_list_gold = table_item['cell_list_gold']
        self.table_data_org = table_item['table_data_org']
        self.comp_table_pred = table_item['comp_table_pred']

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
        
    def get_next_prompt_cell(self) -> CellItem:
        if len(self.cell_list_pred) == 0:
            return self.cells_extracted[0]
        else:
            assert len(self.cell_list_pred) < len(self.cells_extracted)
            return self.cells_extracted[len(self.cell_list_pred)]

    def prompt_wrap(self, template: str, encoding, max_output_tokens, max_length) -> str:
        
        # Control the length of the table code and table text
        table_text = self.table_text
        if table_text is not None:
            table_code_text = f"{table_text}\n\n{self.table_code}"
        else:
            table_code_text = self.table_code

        if table_text is not None:
            while num_tokens_from_string(table_code_text, encoding) > (max_length/2) and table_text.strip() != '':
                print(f"Table code length: {num_tokens_from_string(table_code_text, encoding)}")
                table_text = '\n\n'.join(table_text.split('\n\n')[1:])
                table_code_text = f"{table_text}\n\n{self.table_code}"

        # Control the length of the predicted cell description
        next_prompt_cell = self.get_next_prompt_cell()
        prompt_cell_prefix = '{"value": ' + f'"{next_prompt_cell.cell_value}", "type":'

        cell_describe_idx = 0
        prompt_prefix = '\n'.join([json.dumps(cell_describe) for cell_describe in self.cell_list_pred[cell_describe_idx:]] + [prompt_cell_prefix])

        while num_tokens_from_string(prompt_prefix, encoding) + max_output_tokens + num_tokens_from_string(template, encoding) > (max_length-num_tokens_from_string(table_code_text, encoding)) and cell_describe_idx < len(self.cell_list_pred):
            print(f"prompt_prefix length: {num_tokens_from_string(prompt_prefix, encoding)}")
            # Remove the first cell description in the prompt prefix
            cell_describe_idx += 1
            prompt_prefix = '\n'.join([json.dumps(cell_describe) for cell_describe in self.cell_list_pred[cell_describe_idx:]] + [prompt_cell_prefix])

        if len(self.cell_list_pred) > 0:
            assert cell_describe_idx < len(self.cell_list_pred)
                                       
        assert "{{table_code_text}}" in template
        prompt = template.replace("{{table_code_text}}", table_code_text)

        assert "{{prompt_prefix}}" in prompt
        prompt = prompt.replace("{{prompt_prefix}}", prompt_prefix)

        num_prompt_tokens = num_tokens_from_string(prompt, encoding)
        assert num_prompt_tokens + max_output_tokens <= max_length
        print(f"Number of prompt tokens: {num_prompt_tokens}")

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
            if '"type":' not in response:
                response = response.lstrip().lstrip('{"value":')
            else:
                response = response[response.index('"type":')+len('"type":'):].lstrip()
        elif response.startswith('{'):
            response = response.lstrip('{')
        elif not response.lstrip().startswith('"'):
            response = '"' + response.lstrip()

        next_prompt_cell = self.get_next_prompt_cell()
        prompt_cell_prefix = '{"value": ' + f'"{next_prompt_cell.cell_value}", "type":'

        output_raw = prompt_cell_prefix + response
        print("Output raw:\n", output_raw, '\n')

        # Parse the output to a list of JSON objects
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

            try:
                output = json.loads(output)
            except:
                if 'Other' in output and '"type":' in output:
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

            # Check the predicted cell value is the same as the desired cell value
            if output['value'] == cell_exp_value:
                output_cleaned.append(output)
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
                cell_pred['cell_index'] = '-'.join([str(idx) for idx in cell_extract.cell_index])
                cell_list_pred_indexed.append(cell_pred)
                cell_extract_idx += 1
                pred_idx += 1
            else:
                assert cell_extract_idx > 0
                raise ValueError(f"Cell {cell_pred['value']} not follow the desired order")
            
        return cell_list_pred_indexed
    
    def preprocess_pred_data(self, pred_data: list) -> list:
        pred_data_process = []
        for item in pred_data:
            if item['type'] == 'Other' or 'unit' not in item or item['unit'].lower() in ['xx', 'g/l', 'mol ratio', 'degc']:
                continue
            else:
                pred_data_process.append(item)
        return pred_data_process
    
    def process_unit(self, unit):
        unit = unit.replace('%', '').replace('.', '').strip() if unit.replace('%', '').replace('.', '').strip() else 'wt'

        if unit.lower().strip().startswith('mol') or unit.lower().strip() in ['at']:
            unit = 'mol'
        elif unit.lower().strip() in ['mass', 'weight', 'weight fraction', 'weight oxide', 'wt fraction']:
            unit = 'wt'
        
        return unit.lower()

    def identify_material(self, material_pred, i, j, table, constituent, value):

        if material_pred.lower().strip() in ['xx', None, 'material', 'sample', 'batch'] or any(item in material_pred.lower() for item in ['measurement', 'composition', 'nominal', 'analyzed']):
            return None, None, None, None, None
        else:
            material_by_row = []
            for row_index in range(0, i):
                if material_pred.lower() in table['act_table'][row_index][j].lower():
                    material_by_row.append((row_index, j, table['act_table'][row_index][j], len(material_pred.lower())/len(table['act_table'][row_index][j].replace(' ', '').replace('[', '').replace(']', '').lower())))
                elif table['act_table'][row_index][j].lower() in material_pred.lower():
                    if len(table['act_table'][row_index][j].replace(' ', '').replace('[', '').replace(']', '').lower()) and len(material_pred.lower()):
                        material_by_row.append((row_index, j, table['act_table'][row_index][j], len(table['act_table'][row_index][j].replace(' ', '').replace('[', '').replace(']', '').lower())/len(material_pred.lower())))

            material_by_col = []
            for col_index in range(0, j):
                if material_pred.lower() in table['act_table'][i][col_index].lower():
                    material_by_col.append((i, col_index, table['act_table'][i][col_index], len(material_pred.lower())/len(table['act_table'][i][col_index].replace(' ', '').replace('[', '').replace(']', '').lower())))
                elif table['act_table'][i][col_index].lower() in material_pred.lower():
                    if len(table['act_table'][i][col_index].replace(' ', '').replace('[', '').replace(']', '').lower()) and len(material_pred.lower()):
                        material_by_col.append((i, col_index, table['act_table'][i][col_index], len(table['act_table'][i][col_index].replace(' ', '').replace('[', '').replace(']', '').lower())/len(material_pred.lower())))

            if len(material_by_row) == 0 and len(material_by_col) == 0:
                return None, None, None, None, None
            else:
                material_by_row_by_col_sorted = sorted(material_by_row+material_by_col, key=lambda x: (x[3], -x[1]), reverse=True)
                mat_i, mat_j, material_pred, _ = material_by_row_by_col_sorted[0]

                if mat_j == j:
                    # material is in the same column
                    material_on_same_col = True
                else:
                    material_on_same_col = False

                if mat_i == i:
                    # material is in the same row
                    material_on_same_row = True
                else:
                    material_on_same_row = False

                if material_on_same_col and material_pred.lower() in ['glass', 'glass sample'] and mat_i+1 < i:
                    mat_i, mat_j, material_pred = mat_i+1, mat_j, table['act_table'][mat_i+1][mat_j].strip()

                return material_pred, mat_i, mat_j, material_on_same_col, material_on_same_row
    
    def get_pred_tuples(self, pred_data: list) -> list:

        tuples = []
        table = self.table_data_org

        k = 0
        pred_data_dict = defaultdict(list)
        for item in pred_data:
            
            value = float(item['value'])
            unit = self.process_unit(item['unit'])

            # GPT usually overpredicts 'wt' unit when it is not specified
            unit = 'mol' if unit == 'wt' and all(item not in table['caption']+str(table['act_table']).lower() for item in ['wt', 'weight', item['unit'].lower()]) else unit

            if unit not in ['wt', 'mol']:
                continue

            cell_index = item['cell_index']
            i, j, _ = cell_index.split('-')
            constituent = item['constituent compound name']
            material = item['glass material/sample name/id/code']
            material, mat_i, mat_j, material_on_same_col, material_on_same_row = self.identify_material(material, int(i), int(j), table, constituent, value)

            pred_data_dict[material].append((i, j, constituent, value, unit, material, mat_i, mat_j, material_on_same_col, material_on_same_row))


        if len(pred_data_dict) == 0:
            return []

        for material, pred_items in pred_data_dict.items():

            smallest_i = min([int(item[0]) for item in pred_items])
            row_continuity = True if sorted(set([int(item[0]) for item in pred_items])) == list(range(smallest_i, smallest_i+len(set([int(item[0]) for item in pred_items])))) else False
    
            for item in pred_items:

                i, j, constituent, value, unit, material, mat_i, mat_j, material_on_same_col, material_on_same_row = item
            
                if material is not None:
                    mid_index = 1

                if material_on_same_col or (material_on_same_row and row_continuity):
                    prefix = f'{self.paper_id}_{self.table_only_id}_{smallest_i}'
                else:
                    prefix = f'{self.paper_id}_{self.table_only_id}_{i}'

                if material:
                    gid = prefix + '_' + material
                else:
                    gid = prefix

                if value != 0:
                    pred_item_processed = [gid, constituent, round(float(value), 5), unit]
                    tuples.append(pred_item_processed)
        
        return tuples
    
    def get_tuples_metrics(self, pred_tuples: list, gold_tuples: list) -> tuple:
    
        TP = 0
        for p in pred_tuples:
            if p in gold_tuples:
                TP += 1
        FP = len(pred_tuples) - TP
        FN = len(gold_tuples) - TP
        return TP, FP, FN
    
    def eval(self, verbose=False):

        # Evaluate per table
        pred_data = self.generate_cell_index()
        pred_data = self.preprocess_pred_data(pred_data)
        pred_tuples = self.get_pred_tuples(pred_data)
        gold_tuples = self.cell_list_gold

        if len(pred_tuples) > 0 or len(gold_tuples) > 0:

            print(f'\n{self.table_id}')
            print(f'{len(pred_tuples)} predicted cells, {len(gold_tuples)} gold cells')

            TP, FP, FN = self.get_tuples_metrics(pred_tuples, gold_tuples)
            print(f"TP: {TP}, FP: {FP}, FN: {FN}")

            # if FP == FN and FP > 0:
            #     print("false positive tuples:\n", [item for item in pred_tuples if item not in gold_tuples])
            #     print("false negative tuples:\n", [item for item in gold_tuples if item not in pred_tuples])
            
        else:
            TP, FP, FN = 0, 0, 0
        
        return TP, FP, FN


class DiSCoMaT(Task):
    def __init__(self, args) -> None:
        self.data_path = os.path.join(DATA_PATH, 'discomat', f'{args.data_split}.json')
        self.template = globals()[args.template]
        self.args = args
        model_name = args.backend if args.api_source == 'openai' else AZURE_MODELS[args.backend]
        self.output_dir = os.path.join(DATA_PATH, 'predict', self.args.task, self.args.data_split, model_name, args.template)
        self.data = self.load_data(args.task_start_index, args.task_end_index)

    def load_data(self, start_index: int, end_index: int) -> list:
        with open(self.data_path) as f:
            data = json.load(f)
            data = sorted(list(data.items()), key=lambda x: x[0])
            
        if start_index != -1:
            return [DiSCoMaTItem(item[0], item[1], self.output_dir) for item in data[start_index:end_index]]
        else:
            return [DiSCoMaTItem(item[0], item[1], self.output_dir) for item in data]
        
    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx].prompt
    
    def get_input_list(self) -> list:
        return [self.get_input(idx) for idx in range(len(self.data))]
    
    def get_data(self) -> list:
        return self.data
    

    
    