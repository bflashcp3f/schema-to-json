template_filling = """{{table_code_text}}

Here are JSON templates for six types of numeric cells: "Other", "IC50", "EC50", "CC50", "MIC", and "GI50":
{"value": "xx", "type": "Other"}
{"value": "xx", "type": "IC50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "EC50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "CC50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "MIC", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "GI50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}

Please describe all numeric cells in the above XML table following the JSON templates (proceeding by row in a left-right, top-down direction). For each cell, output one JSON description per line. For any unanswerable attributes in the templates, set their value to the placeholder "xx".

Cell Description:
{{prompt_prefix}}"""

template_filling_llama2chat = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{{table_code_text}}

Here are JSON templates for six types of numeric cells: "Other", "IC50", "EC50", "CC50", "MIC", and "GI50":
{"value": "xx", "type": "Other"}
{"value": "xx", "type": "IC50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "EC50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "CC50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "MIC", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}
{"value": "xx", "type": "GI50", "unit": "xx", "treatment compound": "xx", "target compound": "xx"}

Please describe all numeric cells in the above XML table following the JSON templates (proceeding by row in a left-right, top-down direction). For each cell, output one JSON description per line. For any unanswerable attributes in the templates, set their value to the placeholder "xx".[/INST]Cell Description:
{{prompt_prefix}}"""