template_filling_davinci = """{{table_code_text}}

Here are JSON templates for four types of numeric cells: "Other", "Result", "Data Stat.", and "Hyper-parameter/Architecture":
{"value": "xx", "type": "Other"}
{"value": "xx", "type": "Result", "task": "xx", "metric": "xx", "training data/set": "xx", "test data/set": "xx", "model/method": "xx", "model/method settings": {"xx": "yy"}, "experimental settings": {"xx": "yy"}}
{"value": "xx", "type": "Data Stat.", "dataset": "xx", "attribute name": "xx", "sub-set/group name": "xx", "dataset features": {"xx": "yy"}}
{"value": "xx", "type": "Hyper-parameter/Architecture", "model": "xx", "parameter/architecture name": "xx", "dataset": "xx"}

Please describe all numeric cells in the above latex table following the JSON templates (proceeding by row in a left-right, top-down direction). For each cell, output one JSON description per line. For any unanswerable attributes in the templates, set their value to the placeholder "xx" if it is of string type and {"xx": "yy"} if it is of dictionary type.

Cell Description:
{{prompt_prefix}}"""

template_filling = """{{table_code_text}}

Here are JSON templates for four types of numeric cells: "Other", "Data Stat.", "Hyper-parameter/Architecture", and "Result":
{"value": "xx", "type": "Other"}
{"value": "xx", "type": "Data Stat.", "dataset": "xx", "attribute name": "xx", "sub-set/group name": "xx", "dataset features": {"xx": "yy"}}
{"value": "xx", "type": "Hyper-parameter/Architecture", "model": "xx", "parameter/architecture name": "xx", "dataset": "xx"}
{"value": "xx", "type": "Result", "task": "xx", "metric": "xx", "training data/set": "xx", "test data/set": "xx", "model/method": "xx", "model/method settings": {"xx": "yy"}, "experimental settings": {"xx": "yy"}}

Please describe all numeric cells in the above latex table following the JSON templates (proceeding by row in a left-right, top-down direction). For each cell, output one JSON description per line. For any unanswerable attributes in the templates, set their value to the placeholder "xx" if it is of string type and {"xx": "yy"} if it is of dictionary type. Numeric cells, which describe performance/error analysis, should be labeled as "Other".

Cell Description:
{{prompt_prefix}}"""