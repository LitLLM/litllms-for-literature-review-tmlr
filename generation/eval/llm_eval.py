# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/qa_baseline_gpt35.py

#!/usr/bin/env python3
# Run as python -m eval.prepare_files
# PYTHONPATH=. python eval/prepare_files.py
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
MODEL="gpt-4"
MODEL_ID="gpt-4:20230924"


class PrepareData: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        self.dataset_name = config.dataset_name
        self.prompts = load_eval_prompts()
        # self.base_prompt = self.prompts[self.config.prompt_type]
        # self.judge = self.config.model_name
        self.judge = MODEL
        self.ml_model = self.load_model()
        self.rating_list = []
        self.winner_list = []
        self.cite_list = None    

    def load_model(self):
        ml_model = OpenAIAgent(self.judge)
        # ml_model = OpenAIAgent(self.config.model_name)
        return ml_model

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

    def subset_indices_and_data(self, dataset_name, savedir, human_eval: bool = False, subset_count: int = 100):
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
            indices = df.index[df["num_cites"]<4].tolist()
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
                "model_id": self.judge,
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


    def generate_llm_scores(self, savedir, model1_name="", model2_name="", prompt_type="single"):
        print(f"Calling the llm score function of {self.class_name}")
        context_jsons = self.read_model_outputs(savedir=savedir, out_name="ref_data")
        response_1_jsons = self.read_model_outputs(savedir=savedir, out_name=model1_name)
        ratings_filepath = f"{savedir}/{prompt_type}_{model1_name}_judge_{self.judge}_reviews_{self.dataset_name}.jsonl"
        if prompt_type == "pairwise":
            ratings_filepath = f"{savedir}/{prompt_type}_{model1_name}_vs_{model2_name}_judge_{self.judge}_reviews_{self.dataset_name}.jsonl"
            response_2_jsons = self.read_model_outputs(savedir=savedir, out_name=model2_name)            
            assert len(context_jsons) == len(response_1_jsons) == len(response_2_jsons), f"Found: {len(context_jsons)}, {len(response_1_jsons)}, {len(response_2_jsons)}"
        preds = []
        ratings_list = []
        prompt_template_key = f"{prompt_type}_prompt_template"
        base_prompt = self.prompts[prompt_type]
        prompt_template = self.prompts[prompt_template_key]
        print(f"Doing eval for prompt type: {prompt_type}")
        total_len = len(context_jsons)
        question_idx_list = list(range(total_len))
        for idx in tqdm(question_idx_list):
            response_1=response_1_jsons[idx]["response"]
            example_id = response_1_jsons[idx]["example_id"]
            if prompt_type == "single":
                prompt = prompt_template.format(base_prompt=base_prompt, abstract=context_jsons[idx]["abstract"], 
                related_work=context_jsons[idx]["gt_related_work"], response_1=response_1_jsons[idx]["response"])            
            elif prompt_type == "pairwise":
                response_2=response_2_jsons[idx]["response"]
                prompt = prompt_template.format(base_prompt=base_prompt, abstract=context_jsons[idx]["abstract"], 
                related_work=context_jsons[idx]["gt_related_work"], response_1=response_1_jsons[idx]["response"],
                response_2=response_2)                  
            if self.config.model_name.startswith("gpt"):
                response_dict = self.ml_model.get_response(prompt)
                response_with_multiple_outputs = response_dict["response"]
                # print(prompt)
                # print(response_with_multiple_outputs)
                # a= "Source:Human \n\n Review:Very nice \n\n Rating:5"
                # We will try till we get separators of "Rating:" or "\n\n"
                while True:
                    try:
                        reviews = response_with_multiple_outputs.split("Rating:")[0]
                        ratings = response_with_multiple_outputs.split("Rating:")[1]
                        break
                    except:
                        response_dict = self.ml_model.get_response(prompt)
                        response_with_multiple_outputs = response_dict["response"]                            
                try:
                    if prompt_type == "pairwise":
                        # winner rating
                        ratings_list.append(int(ratings.strip().replace("\n","")))
                    else:
                        ratings_list.append(float(ratings.strip().replace("\n","")))
                except:
                    print(f"Please manually fix the score")
                # Abstract and gt text
                # data_input = f"Main abstract: {context_jsons[idx]['abstract']}\n\nOriginal related_work: {context_jsons[idx]['gt_related_work']}"
                # original_abstract = f"Main abstract: {context_jsons[idx]['abstract']}"
                # Citations text
                data_input = f"Main abstract: {context_jsons[idx]['abstract']}\n\n{context_jsons[idx]['citation_text']}"
                if prompt_type == "pairwise":
                    out_line = {"example_id":example_id, "judge": self.judge, "model_a": model1_name, "model_b": model2_name, "winner": ratings, 
                                "judge_review": response_with_multiple_outputs, "data_input": data_input, "response_a": response_1, "response_b": response_2}
                else:
                    out_line = {"example_id":example_id, "judge": self.judge, "model": model1_name, "rating": ratings, "judge_review": response_with_multiple_outputs, 
                                "data_input": data_input, "response": response_1}
                preds.append(out_line)
        # print(ratings_list)
        if prompt_type == "pairwise":
            win = ratings_list.count(1)
            draw = ratings_list.count(3)
            lost = ratings_list.count(2)
            print(f"Win % for {model1_name} vs {model2_name} is {win}, draw: {draw}, lost: {lost}")
            winner_json = {"judge": self.judge, "model1":model1_name, "model2":model2_name, "win_1": win, "win_2": lost, "draw": draw}
            self.winner_list.append(winner_json)
            avg_win_filepath = f"{savedir}/{prompt_type}_judge_{self.judge}_reviews_{self.dataset_name}.jsonl"
            append_to_jsonl(winner_json, avg_win_filepath, do_json_dump=True)        
        else:
            mean = sum(ratings_list) / len(ratings_list)
            print(f"Mean rating for {model1_name} is {mean}")
            rating_json = {"judge": self.judge, "model":model1_name, "mean_rating": mean}
            self.rating_list.append(rating_json)
            avg_rating_filepath = f"{savedir}/{prompt_type}_judge_{self.judge}_reviews_{self.dataset_name}.jsonl"
            append_to_jsonl(rating_json, avg_rating_filepath, do_json_dump=True)
        dump_jsonl(ratings_filepath, preds)

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
    parser.add_argument(
        "-m",
        "--model_name",
        default="gpt-3.5-turbo",
        help="Name for file utils",
    )
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
        help="Dataset name",
    )
    # shubhamagarwal92/rw_2308_filtered
    # multi_x_science_sum
    parser.add_argument(
        "-p",
        "--prompt_type",
        default="single",
        choices=["pairwise", "single", "source"],
        help="Type of prompt to choose from all the prompt templates",
    )
    # parser.add_argument(
    #     "-t",
    #     "--prompt_template",
    #     default="plan_template",
    #     choices=["pairwise", "single", "source"],
    #     help="Type of prompt to choose from all the prompt templates",
    # )
    # parser.add_argument(
    #     "-s",
    #     "--savedir",
    #     default="results/auto_review/multixscience/eval_results",
    #     help="File path",
    # )
    # SA: TODO change reviewer


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
        # gpt_3.5_plan
        file_path = "results/auto_review/multixscience//gpt-3.5-turbo-plan/results_gpt-3.5-turbo.pkl"
        out_name = "gpt_3.5_plan"
        sample.prepare_outputs_for_eval(file_path, out_name, use_plan=True)
        # gpt_4
        file_path = "results/auto_review/multixscience//gpt-4_vanilla/results_gpt-4.pkl"
        out_name = "gpt_4"
        sample.prepare_outputs_for_eval(file_path, out_name, use_plan=False)
        # gpt_3.5
        file_path = "results/auto_review/multixscience//results_chatgpt_gpt-3.5-turbo.pkl"
        out_name = "gpt_3.5"
        sample.prepare_outputs_for_eval(file_path, out_name, use_plan=False)
        # llama70bplan
        file_path = "results/auto_review/multixscience//Llama-2-70b-chat-hf_plan_based_gen/results_Llama-2-70b-chat-hf.pkl"
        out_name = "llama2_70b_plan"
        sample.prepare_outputs_for_eval(file_path, out_name, use_plan=True)
        # llama70b
        file_path = "results/auto_review/multixscience/results_Llama-2-70b-chat-hf.pkl"
        out_name = "llama2_70b"
        sample.prepare_outputs_for_eval(file_path, out_name, use_plan=False)



    sample.generate_llm_scores(savedir=savedir, model1_name="llama2_70b_plan", model2_name="llama2_70b", prompt_type="pairwise")
