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


class DataAnalysis: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        self.dataset_name = "multi_x_science_sum"

    def get_indices_pkl_path(self, savedir):
        pkl_file_path = f"{savedir}/subset_indices_{self.dataset_name}.pkl"
        return pkl_file_path


    def calculate_avg_words(self, file_path, out_name, use_plan=False):
        print(f"Preparing outputs from function of {self.class_name}")
        indices_file_path = self.get_indices_pkl_path(savedir)
        indices = pkl_load(indices_file_path)
        # pred_df -- pd df, result_list -- dataset, result_json
        results_json = pkl_load(file_path)[2] # We store [1] as pkl object
        print(f"Length of results: {len(results_json)}")
        # results_df = pd.DataFrame(results_json)
        output_preds = results_json["preds"]
        calculate_words_percentile(output_preds)
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
        calculate_words_percentile(selected_preds)
        # print(f"Length of results: {len(results_json)}")


        return

def calculate_words_percentile(output_preds):
    words_list = []
    for pred in output_preds:
        words_list.append(len(pred.split(" ")))
    data = {"words": words_list}
    diff_df = pd.DataFrame(data)
    describe_df = get_pandas_percentile(diff_df, "words")
    print(describe_df)
    return 



def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_path",
        default="results/auto_review/rw_2308_filtered/gpt-4/results_gpt-4.pkl",
        help="File path",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    sample = DataAnalysis(parsed_args)
    cur_dir = pathlib.Path(__file__).parent.resolve()
    savedir = f"{cur_dir}/outputs"


    # gpt_4_plan
    # file_path = "results/auto_review/multixscience/gpt-4-plan/results_gpt-4.pkl"
    # out_name = "gpt_4_plan"
    # sample.calculate_avg_words(file_path, out_name)