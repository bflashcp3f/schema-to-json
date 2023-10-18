import os
import json
import time
import openai
import backoff 
import asyncio
import tiktoken


def eval(args, task):

    TP_FP_FN_list = []
    for table in task.get_data():

        print(f'\n{table.table_id}')
        print(f'{len(table.cell_list_pred)} predicted cells, {len(table.cell_list_gold)} gold cells')

        # Evaluat per table
        TP_per_table, FP_per_table, FN_per_table = table.eval(metric=args.metric, threshold=args.threshold, verbose=args.verbose)
        TP_FP_FN_list.append([TP_per_table, FP_per_table, FN_per_table])

        if args.verbose:
            precision = TP_per_table / (TP_per_table + FP_per_table) if (TP_per_table + FP_per_table) > 0 else 0
            recall = TP_per_table / (TP_per_table + FN_per_table) if (TP_per_table + FN_per_table) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

    # Compute the overall score
    # Micro average
    TP_micro = sum([item[0] for item in TP_FP_FN_list])
    FP_micro = sum([item[1] for item in TP_FP_FN_list])
    FN_micro = sum([item[2] for item in TP_FP_FN_list])
    precision_micro = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) > 0 else 0
    recall_micro = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) > 0 else 0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

    # Macro average
    precision_list = [item[0] / (item[0] + item[1]) if (item[0] + item[1]) > 0 else 0 for item in TP_FP_FN_list]
    recall_list = [item[0] / (item[0] + item[2]) if (item[0] + item[2]) > 0 else 0 for item in TP_FP_FN_list]
    f1_list = [2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 for precision, recall in zip(precision_list, recall_list)]
    precision_macro = sum(precision_list) / len(precision_list) if len(precision_list) > 0 else 0
    recall_macro = sum(recall_list) / len(recall_list) if len(recall_list) > 0 else 0
    f1_macro = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0

    print(f'\nThe micro-average attribute-level Precision over {len(TP_FP_FN_list)} tables: {precision_micro*100:.1f}')
    print(f'The micro-average attribute-level Recall over {len(TP_FP_FN_list)} tables: {recall_micro*100:.1f}')
    print(f'The micro-average attribute-level F1 over {len(TP_FP_FN_list)} tables: {f1_micro*100:.1f}')

    print(f'\nThe macro-average attribute-level Precision over {len(TP_FP_FN_list)} tables: {precision_macro*100:.1f}')
    print(f'The macro-average attribute-level Recall over {len(TP_FP_FN_list)} tables: {recall_macro*100:.1f}')
    print(f'The macro-average attribute-level F1 over {len(TP_FP_FN_list)} tables: {f1_macro*100:.1f}')


def eval_discomat(args, task):
    
    TP_FP_FN_list = []
    for table in task.get_data():

        if len(table.cell_list_pred) == 0 and len(table.cell_list_gold) == 0:
            continue

        # Evaluat per table
        TP_per_table, FP_per_table, FN_per_table = table.eval(verbose=args.verbose)
        TP_FP_FN_list.append([TP_per_table, FP_per_table, FN_per_table])

    # Micro average
    TP_micro = sum([item[0] for item in TP_FP_FN_list])
    FP_micro = sum([item[1] for item in TP_FP_FN_list])
    FN_micro = sum([item[2] for item in TP_FP_FN_list])
    precision_micro = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) > 0 else 0
    recall_micro = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) > 0 else 0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

    print(f'\nThe Precision over {len(TP_FP_FN_list)} tables: {precision_micro*100:.1f}')
    print(f'The Recall over {len(TP_FP_FN_list)} tables: {recall_micro*100:.1f}')
    print(f'The F1 over {len(TP_FP_FN_list)} tables: {f1_micro*100:.1f}')


def eval_swde(args, task):

    SWDE_VERTICALS = ['auto', 'book', 'camera', 'job', 'movie', 'nbaplayer', 'restaurant', 'university']

    vertical_f1_dict = {}
    for vertical_name in SWDE_VERTICALS:

        vertical_f1 = task.eval_vertical(vertical_name, args.verbose)
        vertical_f1_dict[vertical_name] = vertical_f1
        # break

    print(f"Average F1: {sum(vertical_f1_dict.values()) / len(vertical_f1_dict)}")