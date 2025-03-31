# Run as python -m autoreview.models.plan_based_generation
# PYTHONPATH=. python autoreview/models/plan_based_generation.py

import concurrent.futures
import requests
import time
from autoreview.models.toolkit_utils import set_env_variables
set_env_variables(do_hf_login=False)
import argparse
from datasets import load_dataset, set_caching_enabled
from functools import partial
from autoreview.models.ml_utils import load_all_prompts, Dict2Class
from autoreview.models.data_utils import (create_model_input, get_complete_mapped_prompt, 
                                          get_sentences_spacy, get_pandas_percentile,
                                          truncate_single_paper_or_input, get_base_name_from_hf_path,
                                          pkl_load, find_strings_with_word_positions) 
from autoreview.models.pipeline import MDSGen
from autoreview.models.langchain_openai_agent import OpenAIAgent
from autoreview.models.anyscale_endpoint import anyscale_chat_complete
from autoreview.evaluation.compute_score import Metrics
import pandas as pd
from tqdm import tqdm
import os
from os import path
import spacy
import warnings
import numpy as np
warnings.filterwarnings("ignore")



class PlanBasedInference: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        self.nlp_spacy = self.load_spacy(spacy_module='en_core_web_sm')
        self.prompts = load_all_prompts()
        self.base_prompt = self.prompts[self.config.prompt_type]        
        config={"model_name": self.config.model_name}
        config = Dict2Class(config)
        score_pipeline = MDSGen(config, inference=False)
        if self.config.model_name.startswith("gpt"):
            self.ml_model = self.load_model()
        else:
            print("Not GPT based. Using Anyscale endpoints")
        self.metric = Metrics(config={"metric": "rouge"})

    def load_model(self):
        ml_model = OpenAIAgent(self.config.model_name)
        return ml_model

    def load_spacy(self, spacy_module: str='en_core_web_sm'):
        nlp_spacy = spacy.load(spacy_module)
        return nlp_spacy

    def generate_sentence_by_sentence(self, dataset_name, savedir, max_len_ctx: int = 1200, small_dataset: bool = False):
        dataset = self.get_dataset(dataset_name, small_dataset)
        preds = []
        for row in tqdm(dataset):
            draft_related_work = ""
            gt_related_work = row["related_work"]
            original_abstract = row["abstract"]
            abstract_text = f"Main Abstract: {original_abstract}\n"
            references = row["ref_abstract"]  # refs
            citations = references["cite_N"]
            abstracts = references["abstract"]
            # Use list comprehension to filter out elements with empty strings in list2
            filtered_lists = [(item1, item2) for item1, item2 in zip(citations, abstracts) if item2] # != ""]
            # Unpack the filtered elements back into separate lists
            citations, abstracts = zip(*filtered_lists)
            gt_sentences = get_sentences_spacy(gt_related_work, nlp_spacy=self.nlp_spacy)
            num_gt_words = len(gt_related_work.split())
            for _, sent in enumerate(gt_sentences):
                # cite_list = find_strings_in_reference(citations, sent)
                cite_list, _ = find_strings_with_word_positions(citations, sent)
                ref_text = ""
                if cite_list:
                    for element in cite_list:
                        # Find the index of the element in the first list
                        index_of_element = citations.index(element)
                        # Get the corresponding element from the second list
                        individual_abstract = abstracts[index_of_element]
                        ref_text = f"{ref_text}Reference {citations[index_of_element]}: {individual_abstract}\n"
                        # ref_text = truncate_single_paper(ref_text)
                        ref_text = truncate_single_paper_or_input(ref_text, max_len_cite=240, truncate_cite=True)
                # We will truncate text. So draft first then refs
                # data_input = f"{abstract_text}\n{ref_text}\nDraft: {draft_related_work}"
                data_input = f"{abstract_text}\nDraft: {draft_related_work}\n{ref_text}" 
                data_input = truncate_single_paper_or_input(data_input, max_len_cite=max_len_ctx, truncate_cite=True, single_paper=False)

                suffix = f"""Remember, overall please write the related work section in max {len(gt_sentences)} sentences using {num_gt_words} words. 
                Cite the reference as (@cite_#) only whenever provided. Do not write as author et al or Reference (@cite_#).\nRelated Work:\n"""
                complete_prompt = get_complete_mapped_prompt(sample=None, base_prompt=self.base_prompt, data=data_input, suffix=suffix, mapped=False)
                # print(complete_prompt)
                response_dict = self.ml_model.get_response(complete_prompt)
                draft_related_work = response_dict["response"] # Overwrite related work
            preds.append(draft_related_work)
        refs = dataset["related_work"]
        # print("\n".join(map(str, preds)))
        print("Total amount spent: ", self.ml_model.get_state_dict()["budget_spent"])
        self.metrics_wrapper(savedir, preds=preds, refs=refs, dataset=dataset, model_name=f"{self.config.model_name}-sentence-level")
        print("\n-------------------------------------------------\n")
        self.compute_sentence_rouge(preds=preds, refs=refs)

    def get_dataset(self, dataset_name, small_dataset: bool = False, split: str= "test", redownload: bool = False):
        if redownload:
            dataset = load_dataset(dataset_name, split=split, download_mode='force_redownload')
        else:
            dataset = load_dataset(dataset_name, split=split)
        hf_column_names = dataset.column_names
        if "ref_abstract_full_text" in hf_column_names:
            dataset = dataset.remove_columns(['ref_abstract_full_text_original', 'ref_abstract_full_text', "ref_abstract_original"])
        if small_dataset:
            dataset = dataset.select(list(range(3)))
        set_caching_enabled(False)
        return dataset

    def main(self, dataset_name, savedir, max_len_ctx: int = 1600, small_dataset: bool = False, max_len_cite: int = 250, max_tries:int = 5):
        print(f"Calling the main function of {self.class_name}")
        dataset = self.get_dataset(dataset_name, small_dataset)
        if self.config.gen_type == "vanilla":
            dataset = dataset.map(partial(create_model_input, plan_based_gen=False, max_len_ctx=max_len_ctx, max_len_cite=max_len_cite))
            dataset = dataset.map(partial(get_complete_mapped_prompt, base_prompt=self.base_prompt, data_column="text", use_llama_chat=False))
        elif self.config.gen_type == "vanilla_word_based":
            dataset = dataset.map(partial(create_model_input, plan_based_gen=False, max_len_ctx=max_len_ctx, max_len_cite=max_len_cite))
            dataset = dataset.map(partial(get_complete_mapped_prompt, base_prompt=self.base_prompt, data_column="text", use_llama_chat=False, word_based=True))
        elif self.config.gen_type == "learned_plan":
            # plan_based_gen=False bcoz we learn plan
            dataset = dataset.map(partial(create_model_input, plan_based_gen=False, max_len_ctx=max_len_ctx))
            suffix_prompt = self.prompts[self.config.suffix_prompt_type]
            dataset = dataset.map(partial(get_complete_mapped_prompt, base_prompt=self.base_prompt, suffix = suffix_prompt, data_column="text", use_llama_chat=False))
        elif self.config.gen_type == "plan_based_gen":
            dataset = dataset.map(partial(create_model_input, plan_based_gen=True, nlp_spacy=self.nlp_spacy, max_len_ctx=max_len_ctx, max_len_cite=max_len_cite))
            dataset = dataset.map(partial(get_complete_mapped_prompt, base_prompt=self.base_prompt, data_column="text", use_llama_chat=False))
        # print(dataset["num_gt_lines"][:3])
        # print("\n".join(dataset["text"][:4]))
        # print("\n".join(dataset["text"][10:12]))
        print("Dataset loaded! Now generating!")
        preds = []
        generated_plan = []
        for row in tqdm(dataset):
            if self.config.model_name.startswith("gpt"):
                response_dict = self.ml_model.get_response(row["text"])
            else:
                response = anyscale_chat_complete(prompt=row["text"], engine=self.config.model_name)
                response_dict = {"response": response}
            if self.config.gen_type == "learned_plan":
                response_with_plan = response_dict["response"]
                # print(response_with_plan)
                # We will try till we get separators of "Related Work:" or "\n\n"

                plan = ""
                gen_output = ""
                try_counter = 0
                while try_counter < max_tries:
                # while True:
                    try:
                        plan = response_with_plan.split("Related Work:")[0]
                        gen_output = response_with_plan.split("Related Work:")[1]
                        break
                    except:
                        # https://stackoverflow.com/questions/17322208/multiple-try-codes-in-one-block
                        try_counter += 1
                        try:
                            print(f"Split with Related Work failed")
                            plan = response_with_plan.split("\n\n")[0]
                            gen_output = response_with_plan.split("\n\n")[1]
                            break
                        except:
                            print(f"Split with \n\n failed. Trying the response again")
                            if self.config.model_name.startswith("gpt"):
                                response_dict = self.ml_model.get_response(row["text"])
                            else:
                                response = anyscale_chat_complete(prompt=row["text"], engine=self.config.model_name)
                                response_dict = {"response": response}                            
                            response_with_plan = response_dict["response"]                            

                generated_plan.append(plan)
                preds.append(gen_output)    
            else:
                preds.append(response_dict["response"])
        # print("\n".join(map(str, preds)))
        # print("\n Plans:")
        # print("\n".join(map(str, generated_plan)))
        # Common        
        # print(preds)
        refs = dataset["related_work"]
        if self.config.model_name.startswith("gpt"):
            print("Total amount spent: ", self.ml_model.get_state_dict()["budget_spent"])
        self.metrics_wrapper(savedir, preds, refs, dataset=dataset, generated_plan=generated_plan)
        # if not self.config.learned_plan:
        if self.config.gen_type == "plan_based_gen":
            print("\n-------------------------------------------------\n")
            self.compute_sentence_rouge(preds=preds, refs=refs)
        return

    def metrics_wrapper(self, savedir, preds, refs, dataset, generated_plan=[], model_name: str=""):
        if not model_name:
            model_name = self.config.model_name
        print(f"Total responses are {len(preds)}")
        config={"model_name": model_name} # Used for saving file
        config = Dict2Class(config)
        score_pipeline = MDSGen(config, inference=False)
        dataset = pd.DataFrame(dataset)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if generated_plan:
            print("Saving metrics with generated plan")
            score_pipeline.calculate_metrics(preds, refs, dataset, savedir, plans=generated_plan)
        else:
            print("Saving metrics with no generated plan")
            score_pipeline.calculate_metrics(preds, refs, dataset, savedir)

    def compute_sentence_rouge(self, preds, refs):
        assert len(preds) == len(refs), f"Length of refs: {len(refs)}, while length of preds: {len(preds)}"
        diff_lines = []  # store diff number of lines in output
        rouge_scores = {"rouge1":[], "rouge2":[], "rougeL":[], "rougeLsum":[]}
        for index, pred in enumerate(preds):
            pred_sentences = get_sentences_spacy(pred, self.nlp_spacy)
            refs_sentences = get_sentences_spacy(refs[index], self.nlp_spacy)
            num_lines_preds = len(pred_sentences)
            num_lines_refs = len(refs_sentences)
            
            min_length = min(num_lines_preds, num_lines_refs)
            pred_sentences = pred_sentences[:min_length]
            refs_sentences = refs_sentences[:min_length]
            diff = num_lines_preds - num_lines_refs
            # print(f"Difference: {num_lines_preds} vs {num_lines_refs}")
            diff_lines.append(diff)

            result_json = {"preds": pred_sentences, "refs": refs_sentences}
            # rouge_scores_eval_hf - use evaluate library while the other one use datasets
            _, _, rouge_scores_eval_hf = self.metric.compute_score(result_json, print_scores=False)
            for key, value in rouge_scores_eval_hf.items():
                rouge_scores[key].append(value)

        average_scores = {}
        for metric, values in rouge_scores.items():
            average_scores[metric] = np.mean(values)

        # print("\n".join(map(str, refs)))
        print(f"Average scores: {average_scores}")
        data = {"diff": diff_lines}
        diff_df = pd.DataFrame(data)
        describe_df = get_pandas_percentile(diff_df, "diff")
        print(describe_df)
        # TODO 
        # https://stackoverflow.com/questions/29229600/counting-number-of-zeros-per-row-by-pandas-dataframe
        # (df == 0).sum(axis=1)
        # print(same_lines)
        # same_lines = (diff_lines == 0).sum()
        # List of diff lines
        same_lines = diff_lines.count(0)  # We count example with same number of sentences 
        print(f"Total % of same lines: {same_lines/len(diff_lines)*100}")
        # print(f"Total % of same lines: {same_lines.values[0]/len(diff_lines)*100}")

    def read_outputs(savedir, model_name):
        pkl_file_path = path.join(savedir, f"results_{os.path.split(model_name)[1]}.pkl")
        data = pkl_load(pkl_file_path)
        result_json = data[2]


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="gpt-4",
        choices=["gpt-3.5-turbo", "gpt-4", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf",
                 "meta-llama/Llama-2-70b-chat-hf", "codellama/CodeLlama-34b-Instruct-hf"],
        help="Model name",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="multi_x_science_sum",
        choices=["multi_x_science_sum", "shubhamagarwal92/rw_2308_filtered"],
        help="Dataset name",
    )
    parser.add_argument(
        "-p",
        "--prompt_type",
        default="plan_learned_template",
        choices=["plan_learned_template", "plan_template", "per_sentence_template", "vanilla_template", "vanilla_template_word_based"],
        help="Type of prompt to choose from all the prompt templates",
    )
    parser.add_argument(
        "-t",
        "--gen_type",
        default="learned_plan",
        choices=["learned_plan", "plan_based_gen", "vanilla", "vanilla_word_based"],
        help="Type of generation",
    )
    parser.add_argument(
        "-sf",
        "--suffix_prompt_type",
        default="suffix_plan_learned_template",
        help="Type of prompt to choose from all the prompt templates",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default="/results/auto_review/multixscience/gpt4-new-learned-plan",
        help="Path to save dir",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    sample = PlanBasedInference(parsed_args)
    
    # data_name = os.path.split(parsed_args.dataset_name)[1]
    # out_dir = f"{parsed_args.savedir}/{data_name}/{parsed_args.model_name}_{parsed_args.gen_type}"
    
    model_name = get_base_name_from_hf_path(parsed_args.model_name)
    out_dir = f"{parsed_args.savedir}/{model_name}_{parsed_args.gen_type}_ctx_3800"
    print(f"Outputs would be saved in {out_dir}")
    # Plan based
    if parsed_args.model_name == "gpt-4":
        sample.main(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=3000, max_len_cite=330)
    elif parsed_args.model_name == "gpt-3.5-turbo":
        sample.main(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=1600, max_len_cite=250)
    else:
        # 408 example
        # 2300 - 4216 tokens
        sample.main(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=1900, max_len_cite=330)
        # sample.main(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=3800, max_len_cite=330)  

    # sample.generate_sentence_by_sentence(parsed_args.dataset_name, parsed_args.savedir)
    # sample.debug_data(parsed_args.dataset_name, parsed_args.savedir)