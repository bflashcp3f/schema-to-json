
import re
import os
import json
import time

from collections import Counter, defaultdict

from schema2json.tasks.base import Task, DATA_PATH
from schema2json.methods.prompt import *
from schema2json.templates.swde import * 


def nested_dict_factory_3():
    return defaultdict(dict)

def nested_dict_factory_4():
    return defaultdict(lambda: defaultdict(dict))


TEMPLATE_PREFIX_DICT = {
    'nbaplayer': 'player name',
    'university': 'university name',
    'movie': 'movie title',
    'auto': 'automobile model (year)',
    'book': 'book title',
    'job': 'job title',
    'camera': 'camera model (full)',
    'restaurant': 'restaurant name',
}


TEMP_ANNO_ALIGNMENT = {
    "nbaplayer": {"player name": "name", "team": "team", "height": "height", "weight": "weight"},
    "university": {"university name": "name", "university type (by fund source)": "type", "website": "website", "phone number": "phone"},
    "movie": {"movie title": "title", "director": "director", "mpaa rating": "mpaa_rating", "genre": "genre"},
    "auto": {"automobile model (year)": "model", "price": "price", "engine type": "engine", "fuel economy": "fuel_economy"},
    "book": {"book title": "title", "author": "author", "isbn_13": "isbn_13", "publisher": "publisher", "publication date": "publication_date"},
    "job": {"job title": "title", "company": "company", "location": "location", "date posted": "date_posted"},
    "camera": {"camera model (full)": "model", "price": "price", "manufacturer": "manufacturer"},
    "restaurant": {"restaurant name": "name", "cuisine type": "cuisine", "address": "address", "phone": "phone"},
}


def string_process(string):

    # Remove all non-ascii characters
    string = string.encode("ascii", errors="ignore").decode() 

    string = string.replace('&nbsp;', ' ').replace('\xa0', ' ').replace('&#47;', ' ').replace('&#43;', ' ').replace('&#34;', ' ').replace('&#38;', ' ').replace('&#40;', ' ').replace('&#41;', ' ').replace('&#160;', ' ').replace('&reg;', ' ').replace('&#39;', ' ').replace('&#039;', "'").replace('&amp;', '&').replace('&quot;', '"').replace('&#150;', ' ').replace('\x96', ' ').strip('\xa0').replace('gt;', ' ').replace('lt;', ' ').replace('8226;', ' ')
    string = ' '.join(re.sub(r'(?<!\\)([{}[\]:,()@$\-+%&#_^~|<>\\/\"\'])', r' ', string).replace('(', ' ').replace(')', ' ').strip().split()).lower()

    return string


def word_f1(gold_attribute_value, pred_attribute_value):
    tp = len(list((Counter(gold_attribute_value.split()) & Counter(pred_attribute_value.split())).elements()))
    fp = len(pred_attribute_value.split()) - tp
    fn = len(gold_attribute_value.split()) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1


def string_match(pred_str, gold_str, metric='em', threshold=0.25):
    if metric == 'em':
        return pred_str == gold_str
    elif metric == 'f1':
        return word_f1(gold_str, pred_str) >= threshold


class SWDEItem:
    def __init__(self, vertical_name: str, website_name: str, page_id: str, page_item: dict, output_dir: str):
        self.vertical_name = vertical_name
        self.website_name = website_name
        self.page_id = page_id
        self.page_title = page_item['title']
        self.page_code = page_item['html_cleaned']
        self.template = globals()[vertical_name]
        self.tag_content_dict = page_item['tag_content_dict']
        self.gold_attributes = page_item['gold_attributes']

        self.output_dir = os.path.join(output_dir, self.vertical_name, self.website_name)
        self.output_path = os.path.join(self.output_dir, self.page_id.split('.')[0]+'.json')

        if os.path.exists(self.output_path):
            with open(self.output_path) as f:
                pred_data = json.load(f)
            self.output_processed = pred_data['output_processed']
            self.prompt = pred_data['prompt']
            self.prompt_prefix = pred_data['prompt_prefix']
            self.prompt_response = pred_data['prompt_response']
        else:
            self.prompt = None
            self.prompt_prefix = None
            self.prompt_response = None
            self.output_processed = None

    def prompt_wrap(self, encoding, max_length: int) -> str:
        template = self.template

        assert "{{page_code}}" in template
        prompt = template.replace("{{page_code}}", self.page_code)

        assert "{{prompt_prefix}}" in prompt
        prefix_attribute = TEMPLATE_PREFIX_DICT[self.vertical_name]
        prompt_prefix = '{"webpage title": ' + f'"{self.page_title}", "{prefix_attribute}":'
        prompt = prompt.replace("{{prompt_prefix}}", prompt_prefix)

        self.prompt = prompt
        self.prompt_prefix = prompt_prefix
        return prompt
    
    def update_output(self, prompt_response: tuple) -> None:

        prompt_response = prompt_response[1].rstrip().split('\n')[0]
        self.prompt_response = prompt_response

        try: 
            output_processed = json.loads(self.prompt_prefix+prompt_response)
        except Exception as e:
            print(f"Error {e} when processing {self.prompt_prefix+prompt_response}")
            output_processed = {"webpage title": self.page_title}

        self.output_processed = output_processed
        self.save_output()

    def save_output(self) -> None:
        # Create the directory to store the output
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save all the attributes in a json file
        output_path = os.path.join(self.output_dir, self.page_id.split('.')[0]+'.json')

        with open(output_path, 'w') as f:
            json.dump({
                'vertical_name': self.vertical_name,
                'website_name': self.website_name,
                'page_id': self.page_id,
                'template': self.template,
                'prompt': self.prompt,
                'prompt_prefix': self.prompt_prefix,
                'prompt_response': self.prompt_response,
                'output_processed': self.output_processed,
            }, f, indent=4)
    

class SWDE(Task):
    def __init__(self, args) -> None:
        self.data_path = os.path.join(DATA_PATH, 'swde', f'{args.data_split}.json')
        
        self.args = args
        if args.api_source == 'openai':
            model_name = args.backend
        elif args.api_source == 'azure':
            model_name = AZURE_MODELS[args.backend]
        elif args.api_source == 'open_source':
            model_name = args.backend
        else:
            raise ValueError(f'Unknown api source: {args.api_source}')
        self.output_dir = os.path.join(DATA_PATH, 'predict', self.args.task, self.args.data_split, model_name)
        self.data = self.load_data(args.task_start_index, args.task_end_index)
        self.data_processed = self.load_data_processed()

    def load_data(self, start_index: int, end_index: int) -> list:
        with open(self.data_path) as f:
            data = json.load(f)
            
        if start_index != -1:
            data_all = [SWDEItem(vertical_name, website_name, page_id, page_item, self.output_dir) for vertical_name in data.keys() for website_name in data[vertical_name].keys() for page_id, page_item in data[vertical_name][website_name].items()][start_index:end_index]

        else:
            data_all = [SWDEItem(vertical_name, website_name, page_id, page_item, self.output_dir) for vertical_name in data.keys() for website_name in data[vertical_name].keys() for page_id, page_item in data[vertical_name][website_name].items()]

        data_unprocessed = [item for item in data_all if item.prompt_response is None]
        
        return data_unprocessed
    
    def load_data_processed(self) -> list:
        with open(self.data_path) as f:
            data = json.load(f)

        data_all = [SWDEItem(vertical_name, website_name, page_id, page_item, self.output_dir) for vertical_name in data.keys() for website_name in data[vertical_name].keys() for page_id, page_item in data[vertical_name][website_name].items()]

        data_processed = [item for item in data_all if item.prompt_response is not None]
        return data_processed
        
    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx].prompt
    
    def get_input_list(self) -> list:
        return [self.get_input(idx) for idx in range(len(self.data))]
    
    def get_data(self) -> list:
        return self.data
    
    def get_prompt_list(self, encoding, max_length: int) -> list:
        if len(self.data) > 0:
            return [item.prompt_wrap(encoding, max_length) for item in self.data]  
        else:
            return []
    
    def update_output(self, responses: list) -> list:
        for item, response in zip(self.data, responses):
            item.update_output(response)

    def gather_attributes_pred(self) -> dict:
        attr_pred_dict = defaultdict(nested_dict_factory_4)
        for item in self.data_processed:
            vertical_name = item.vertical_name
            website_name = item.website_name
            page_id = item.page_id
            output_processed = item.output_processed
            gold_attributes = item.gold_attributes
            tag_content_dict = item.tag_content_dict

            attr_pred_dict[vertical_name][website_name][page_id] = {
                'output_processed': output_processed,
                'gold_attributes': gold_attributes,
                'tag_content_dict': tag_content_dict,
            }
        return attr_pred_dict
    
    def process_attribute(self, pred_attribute: str, gold_attribute: list) -> dict:

        gold_attr_value = [string_process(item) if item != '<NULL>' else '<NULL>' for item in gold_attribute]
        gold_attr_value = [item for item in gold_attr_value if item]
        gold_attr_value = ['<NULL>'] if not gold_attr_value else gold_attr_value

        pred_attribute = '<NULL>' if pred_attribute in ['', 'xx'] else pred_attribute
        pred_attr_value = pred_attribute if type(pred_attribute) == str else pred_attribute[0]
        assert type(pred_attr_value) == str

        pred_attr_value = string_process(pred_attr_value)

        return pred_attr_value, gold_attr_value
            
    def get_attr_result(self, vertical_name: str):
        attr_pred_dict = self.gather_attributes_pred()

        attr_result_dict = defaultdict(nested_dict_factory_4)
        for website_name in attr_pred_dict[vertical_name].keys():
            for page_id in attr_pred_dict[vertical_name][website_name].keys():
                pred_attributes = attr_pred_dict[vertical_name][website_name][page_id]['output_processed']
                gold_attributes = attr_pred_dict[vertical_name][website_name][page_id]['gold_attributes']
                tag_content_dict = attr_pred_dict[vertical_name][website_name][page_id]['tag_content_dict']

                for attr in sorted(pred_attributes.keys()):
                    if attr in ["webpage title"]:
                        continue
                    else:
                        if attr not in TEMP_ANNO_ALIGNMENT[vertical_name]:
                            # print(f"Attribute {attr} not in {TEMP_ANNO_ALIGNMENT[vertical_name]}")
                            continue

                    pred_attr_value, gold_attr_value = self.process_attribute(pred_attributes[attr], gold_attributes[TEMP_ANNO_ALIGNMENT[vertical_name][attr]])

                    # Find the cloest tag content
                    def find_cloest_dom_node(attr_value, tag_content_dict, theshold=0.25, num_return=1):
                        if attr_value == '<NULL>'.lower():
                            return ['<NULL>']
                        else:
                            attr_value_f1_scores = []
                            for each_tag_content in tag_content_dict.keys():
                                attr_value_f1_scores.append((each_tag_content, word_f1(attr_value, each_tag_content)))
                            attr_value_f1_scores = sorted(attr_value_f1_scores, key=lambda x: x[1], reverse=True)
                            
                            attr_value_f1_scores = [item for item in attr_value_f1_scores if item[1] >= theshold][:num_return]
                            
                            if attr_value_f1_scores:
                                return [item[0] for item in attr_value_f1_scores]
                            else:
                                return ['<NULL>']
                            
                    pred_attr_value = find_cloest_dom_node(pred_attr_value, tag_content_dict, theshold=0.25, num_return=1)

                    pred_attr_value = [''.join(item.split()) for item in pred_attr_value]
                    gold_attr_value = [''.join(item.split()) for item in gold_attr_value]

                    tp, fp, fn, tn = 0, 0, 0, 0
                    if pred_attr_value != '<NULL>' and set(pred_attr_value) & set(gold_attr_value):
                        tp += 1
                    elif pred_attr_value != '<NULL>' and (not set(pred_attr_value) & set(gold_attr_value)):
                        fp += 1
                    elif pred_attr_value == '<NULL>' and gold_attr_value[0].lower() != '<null>':
                        fn += 1
                    elif pred_attr_value == '<NULL>' and len(gold_attr_value) == 1 and gold_attr_value[0].lower() == '<null>':
                        tn += 1
                    else:
                        raise ValueError
                    attr_result = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

                    attr_result_dict[vertical_name][website_name][page_id][TEMP_ANNO_ALIGNMENT[vertical_name][attr]] = attr_result

                # Handle unmatched attributes
                assert sorted(gold_attributes.keys()) == sorted(TEMP_ANNO_ALIGNMENT[vertical_name].values())
                matched_attributes = [TEMP_ANNO_ALIGNMENT[vertical_name][pred_attr] for pred_attr in pred_attributes.keys() if pred_attr != 'webpage title' and pred_attr in TEMP_ANNO_ALIGNMENT[vertical_name]]

                for attr in gold_attributes.keys():

                    gold_attr_value = gold_attributes[attr]

                    if attr == 'webpage title' or attr in matched_attributes:
                        continue
                    elif len(gold_attr_value) == 1 and gold_attr_value[0].lower() == '<null>':
                        attr_result_dict[vertical_name][website_name][page_id][attr] = attr_result = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                    else:
                        attr_result_dict[vertical_name][website_name][page_id][attr] = attr_result = {'tp': 0, 'fp': 0, 'fn': 1, 'tn': 0}


        return attr_result_dict

    def eval_vertical(self, vertical_name: str, verbose: bool = False) -> dict:
        attr_metrics_dict = defaultdict(nested_dict_factory_3)
        attr_result_dict = self.get_attr_result(vertical_name)

        attr_list = list(TEMP_ANNO_ALIGNMENT[vertical_name].values())

        for website_name in sorted(attr_result_dict[vertical_name].keys()):
            for each_attr in attr_list:
                tp = sum([attr_result_dict[vertical_name][website_name][webpage_name][each_attr]['tp'] for webpage_name in attr_result_dict[vertical_name][website_name].keys()])
                fp = sum([attr_result_dict[vertical_name][website_name][webpage_name][each_attr]['fp'] for webpage_name in attr_result_dict[vertical_name][website_name].keys()])
                fn = sum([attr_result_dict[vertical_name][website_name][webpage_name][each_attr]['fn'] for webpage_name in attr_result_dict[vertical_name][website_name].keys()])

                page_precision = tp / (tp + fp) if tp + fp != 0 else 0  
                page_recall = tp / (tp + fn) if tp + fn != 0 else 0
                page_f1 = 2 * page_precision * page_recall / (page_precision + page_recall) if page_precision + page_recall != 0 else 0

                attr_metrics_dict[vertical_name][website_name][each_attr] = {'page_precision': page_precision, 'page_recall': page_recall, 'page_f1': page_f1}

            # Average over all attrs
            web_precision = sum([attr_metrics_dict[vertical_name][website_name][each_attr]['page_precision'] for each_attr in attr_list]) / len(attr_list)
            web_recall = sum([attr_metrics_dict[vertical_name][website_name][each_attr]['page_recall'] for each_attr in attr_list]) / len(attr_list)
            web_f1 = sum([attr_metrics_dict[vertical_name][website_name][each_attr]['page_f1'] for each_attr in attr_list]) / len(attr_list)

        # Average over all websites
        vertical_precision = sum([attr_metrics_dict[vertical_name][website_name][each_attr]['page_precision'] for website_name in attr_metrics_dict[vertical_name].keys() for each_attr in attr_list]) / (len(attr_metrics_dict[vertical_name].keys()) * len(attr_list))
        vertical_recall = sum([attr_metrics_dict[vertical_name][website_name][each_attr]['page_recall'] for website_name in attr_metrics_dict[vertical_name].keys() for each_attr in attr_list]) / (len(attr_metrics_dict[vertical_name].keys()) * len(attr_list))
        vertical_f1 = sum([attr_metrics_dict[vertical_name][website_name][each_attr]['page_f1'] for website_name in attr_metrics_dict[vertical_name].keys() for each_attr in attr_list]) / (len(attr_metrics_dict[vertical_name].keys()) * len(attr_list))
        print(f"Vertical: {vertical_name}, Precision: {vertical_precision}, Recall: {vertical_recall}, F1: {vertical_f1}", "\n")

        return vertical_f1