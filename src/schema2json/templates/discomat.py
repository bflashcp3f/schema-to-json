template_filling = """{{table_code_text}}

Here are JSON templates for two types of numeric cells: "Other" and "Glass_Compound_Amount":
{"value": "xx", "type": "Other"}
{"value": "xx", "type": "Glass_Compound_Amount", "constituent compound name": "xx", "unit": "xx", "glass material/sample name/id/code": "xx"}

Please describe all numeric cells in the above table following the JSON templates (proceeding by row in a left-right, top-down direction). For each cell, output one JSON description per line. For any unanswerable attributes in the templates, set their value to the placeholder "xx".

Cell Description:
{{prompt_prefix}}"""