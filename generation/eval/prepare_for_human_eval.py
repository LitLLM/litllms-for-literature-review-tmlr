#!/usr/bin/env python3
# Run as python -m eval.prepare_for_human_eval
# PYTHONPATH=. python eval/prepare_for_human_eval.py
"""
Prepare files for eval
"""
import json, pathlib
import random
import argparse
import pickle as pkl
from datasets import load_dataset, Dataset, set_caching_enabled
from autoreview.models.data_utils import (create_model_input, get_base_name_from_hf_path, dump_to_jsonl, get_pandas_percentile,
                                          pkl_load, pkl_dump, get_hf_dataset) 
from autoreview.models.langchain_openai_agent import OpenAIAgent
import concurrent.futures
from tqdm import tqdm
from eval_utils import (concurrent_requests, dump_jsonl, load_eval_prompts, append_to_jsonl, flatten_ref_papers,
                        find_cite_in_reference,
                        find_avg_across_two_col, is_length_within_range, get_json_list)
import time
from huggingface_hub import InferenceClient
import shortuuid
import pandas as pd
from functools import partial
import random


class PrepareData: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        self.dataset_name = config.dataset_name
        self.cite_list = None    


    def prepare_outputs_for_eval(self, file_path, out_name, use_plan=False):
        print(f"Preparing outputs from function of {self.class_name}")
        indices_file_path = self.get_indices_pkl_path(savedir)
        indices = pkl_load(indices_file_path)
        # pred_df -- pd df, result_list -- dataset, result_json
        results_json = pkl_load(file_path)[2] # We store [1] as pkl object
        print(f"Length of results: {len(results_json)}")
        # results_df = pd.DataFrame(results_json)
        output_preds = results_json["preds"]
        cite_file_path = f"{savedir}/cite_list_{self.dataset_name}.pkl"
        cite_data = pkl_load(cite_file_path)
        cite_list = cite_data["cite_list"]
        missing_cites = []
        for index, pred in enumerate(output_preds):
            missing_cites.append(find_cite_in_reference(cite_list[index], pred))
        data = {"diff": missing_cites}
        diff_df = pd.DataFrame(data)
        describe_df = get_pandas_percentile(diff_df, "diff")
        print(describe_df)
        same_lines = missing_cites.count(0)  # We count example with same number of sentences 
        print(f"Total % of same cites for {out_name}: {same_lines/len(missing_cites)*100}")
        if use_plan:
            # dataset = pkl_load(file_path)[1]
            # dataset = dataset.select(indices)
            # print(dataset.column_names)
            # selected_plans = [plans[i] for i in indices]
            df = pkl_load(file_path)[1]
            subset_df = df.iloc[indices]
            plans = subset_df["plan"].tolist()
        selected_preds = [output_preds[i] for i in indices]
        print(f"Length of results: {len(results_json)}")
        examples = []
        # shortuuid.uuid(),
        for index, row in enumerate(selected_preds):
            sample = {
                "example_id": indices[index], 
                "model_id": out_name,
                "response": row
            } 
            if use_plan:
                sample["plan"] = plans[index]
            examples.append(sample)
        filepath = f"{savedir}/{out_name}_{self.dataset_name}.jsonl"
        dump_jsonl(filepath, examples)
        return

    def get_indices_pkl_path(self, savedir):
        pkl_file_path = f"{savedir}/subset_indices_{self.dataset_name}.pkl"
        return pkl_file_path

    def subset_indices_and_data(self, dataset_name, savedir, human_eval: bool = True, subset_count: int = 100):
        print(f"Calling the main function of {self.class_name}")
        dataset = get_hf_dataset(dataset_name)
        set_caching_enabled(False)                
        # dataset = dataset.filter(is_length_within_range)
        dataset = dataset.map(flatten_ref_papers)        
        print(f"Length of dataset: {len(dataset)}")
        total_examples = len(dataset)
        df = dataset.to_pandas()
        cite_list = df['cite_list'].tolist()
        num_cites = df['num_cites'].tolist()
        pkl_file_path = f"{savedir}/cite_list_{self.dataset_name}.pkl"
        pkl_dump(pkl_file_path, object_list={"cite_list":cite_list, "num_cites": num_cites})
        if human_eval:
            # Select rows with <4 citations
            indices = df.index[df["num_cites"]==3].tolist()
        else:
            # Select random sample
            indices = df.index[df["num_cites"]>1].tolist()
            # indices = list(range(total_examples))
        random.shuffle(indices)
        subset_indices = indices[:subset_count]
        print(f"Length of indices: {len(subset_indices)}")
        # dataset = dataset.select(subset_indices)
        # print(dataset.column_names)
        # print(f"Length of dataset: {len(dataset)}")
        examples = []
        
        subset_df = df.iloc[subset_indices].reset_index(drop=True)
        print(f"Length of dataset: {len(subset_df)}")
        # for new_index, row in enumerate(dataset):
        for new_index, row in subset_df.iterrows():
            sample = {
                "example_id": shortuuid.uuid(),
                "model_id": "gt",
                "num_cites": row["num_cites"],
                "aid": row["aid"],
                "abstract": row["abstract"],
                "gt_related_work": row["related_work"],
                "original_index": subset_indices[new_index],
                "citation_text": row["ref_text"]
            } 
            examples.append(sample)
        filepath = f"{savedir}/ref_data_{self.dataset_name}.jsonl"
        dump_jsonl(filepath, examples)
        pkl_file_path = self.get_indices_pkl_path(savedir)
        pkl_dump(pkl_file_path, object_list=subset_indices)

    def read_model_outputs(self, savedir, out_name):
        filepath = f"{savedir}/{out_name}_{self.dataset_name}.jsonl"
        result_list = get_json_list(filepath)
        return result_list

    def generate_eval_pairs(self, savedir, model1_name, model2_name, vs_gt: bool = False, shuffle: bool = True):
        context_jsons = self.read_model_outputs(savedir=savedir, out_name="ref_data")
        response_1_jsons = self.read_model_outputs(savedir=savedir, out_name=model1_name)
        response_2_jsons = self.read_model_outputs(savedir=savedir, out_name=model2_name)            
        # ratings_filepath = f"{savedir}/{prompt_type}_{model1_name}_judge_{self.judge}_reviews_{self.dataset_name}.jsonl"

        ratings_filepath = f"{savedir}/{model1_name}_vs_{model2_name}_{self.dataset_name}_data.jsonl"

        assert len(context_jsons) == len(response_1_jsons) == len(response_2_jsons), f"Found: {len(context_jsons)}, {len(response_1_jsons)}, {len(response_2_jsons)}"
        preds = []
        ratings_list = []
        total_len = len(context_jsons)
        question_idx_list = list(range(total_len))
        if vs_gt:
            shuffle_index = [1,2,3]
        else:
            shuffle_index = [1,2]
        for idx in tqdm(question_idx_list):
            response_1 = response_1_jsons[idx]["response"]
            response_2 = response_2_jsons[idx]["response"]
            example_id = response_1_jsons[idx]["example_id"]
            response_3 = context_jsons[idx]["gt_related_work"]
            model3_name = "gt"
            abstract = context_jsons[idx]['abstract']
            citation_text = context_jsons[idx]['citation_text']
            
            random.shuffle(shuffle_index)
            # model_a = globals()[f'model{choice}_name']
            model_a, response_a = eval(f'model{shuffle_index[0]}_name'), eval(f'response_{shuffle_index[0]}')
            model_b, response_b = eval(f'model{shuffle_index[1]}_name'), eval(f'response_{shuffle_index[1]}')
            if vs_gt:
                model_c, response_c = eval(f'model{shuffle_index[2]}_name'), eval(f'response_{shuffle_index[2]}')            
            else:
                model_c = model3_name
                response_c = response_3

            data_input = f"Main abstract: {context_jsons[idx]['abstract']}\n\n{context_jsons[idx]['citation_text']}"
            out_line = {"example_id":example_id, "abstract": abstract, "citation_text": citation_text, "model_a": model_a, "model_b": model_b,
                        "model_c": model_c, "data_input": data_input, "response_a": response_a, "response_b": response_b, "response_c": response_c}
            # out_line = {"example_id":example_id, "abstract": abstract, "citation_text": citation_text, "model_a": model_a, "model_b": model_b, 
            #             "response_a": response_a, "response_b": response_b}
            preds.append(out_line)
        dump_jsonl(ratings_filepath, preds)

    def create_mix_batch_eval(self, savedir,file_a_names, file_b_names):
        ratings1_filepath = f"{savedir}/{file_a_names[0]}_vs_{file_a_names[1]}_{self.dataset_name}_data.jsonl"
        ratings2_filepath = f"{savedir}/{file_b_names[0]}_vs_{file_b_names[1]}_{self.dataset_name}_data.jsonl"
        output_prefix = f"{savedir}/input"
        subset_lines_batches(input_file1=ratings1_filepath, input_file2=ratings2_filepath, output_prefix=output_prefix, num_lines=10)

def subset_lines_batches(input_file1, input_file2, output_prefix, num_lines=10, shuffle:bool = True, num_batches = 10):
    # Read lines from the first file
    with open(input_file1, 'r') as file1:
        lines1 = [json.loads(line) for line in file1.readlines()]

    # Read lines from the second file
    with open(input_file2, 'r') as file2:
        lines2 = [json.loads(line) for line in file2.readlines()]

    # Shuffle the lines from both files
    random.shuffle(lines1)
    random.shuffle(lines2)


    # Combine the subsets

    for batch_num in range(num_batches):
        start_index = batch_num * num_lines
        end_index = start_index + num_lines

        subset_lines1 = lines1[start_index:end_index]
        subset_lines2 = lines2[start_index:end_index]

        # batch_lines = combined_lines[start_index:end_index]
        # Shuffle the combined lines
        # random.shuffle(batch_lines)

        combined_lines = subset_lines1 + subset_lines2
        random.shuffle(combined_lines)


        output_file_path = f"{output_prefix}_batch{batch_num + 1}.jsonl"

        # Write the combined lines to the output file
        with open(output_file_path, 'w') as output_file:
            for line in combined_lines:
                output_file.write(json.dumps(line) + '\n')
        print(f"Saving file to {output_file_path}")


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dp",
        "--prepare_data",
        default=False,
        help="Prepare data",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="multi_x_science_sum",
        choices=["multi_x_science_sum", "shubhamagarwal92/rw_2308_filtered"],
        help="Dataset name",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    sample = PrepareData(parsed_args)
    cur_dir = pathlib.Path(__file__).parent.resolve()
    savedir = f"{cur_dir}/outputs"
    if parsed_args.prepare_data:
        sample.subset_indices_and_data(dataset_name=parsed_args.dataset_name, savedir=savedir)
        # gpt_4_plan
        file_path = "results/auto_review/multixscience/gpt-4-plan/results_gpt-4.pkl"
        out_name = "gpt_4_plan"
        sample.prepare_outputs_for_eval(file_path, out_name, use_plan=True)

    # sample.generate_eval_pairs(savedir, model1_name="gpt_4_plan", model2_name="gpt_4")
    # sample.generate_eval_pairs(savedir, model1_name="llama2_70b_plan", model2_name="llama2_70b")

    sample.create_mix_batch_eval(savedir, file_a_names=["gpt_4_plan", "gpt_4"], file_b_names=["llama2_70b_plan", "llama2_70b"])