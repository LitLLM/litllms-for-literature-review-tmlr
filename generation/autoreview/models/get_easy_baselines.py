#!/usr/bin/env python3
# Run as python -m autoreview.models.get_easy_baselines

"""
This file defines class to get baseline summaries
"""
from .toolkit_utils import set_env_variables
from .data_utils import create_model_input, compute_length, get_pandas_percentile, flatten_ref_papers_from_citations
from .pipeline import MDSGen
from .ml_utils import get_gpu_memory, Dict2Class, write_excel_df
set_env_variables(do_hf_login=True)
import argparse
from datasets import load_dataset
from functools import partial
import pandas as pd
import os
from os import path
from datasets import set_caching_enabled
import spacy
import pickle as pkl
from tqdm import tqdm


class OneLineBaseline: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        self.nlp_spacy = self.load_spacy(spacy_module='en_core_web_sm')

    def load_spacy(self, spacy_module: str='en_core_web_sm'):
        nlp_spacy = spacy.load(spacy_module)
        return nlp_spacy

    def generate_one_line_baseline(self, dataset_name, savedir):        
        dataset = load_dataset(dataset_name, split="test")
        set_caching_enabled(False)
        dataset = dataset.map(partial(create_model_input, one_line_baseline=True, nlp_spacy=self.nlp_spacy))
        preds = dataset["text"]
        refs = dataset["related_work"]
        config={"model_name": "one_line_baseline"}
        config = Dict2Class(config)
        score_pipeline = MDSGen(config, inference=False)
        dataset = pd.DataFrame(dataset)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        score_pipeline.calculate_metrics(preds, refs, dataset, savedir)

    def get_data_stats(self, dataset_name, savedir):
        splits = ["train", "validation", "test"]
        df_list = []
        for split in splits:
            dataset = load_dataset(dataset_name, split=split)
            # set_caching_enabled(False)
            dataset = dataset.map(partial(compute_length, col_name="abstract"))
            dataset = dataset.map(partial(compute_length, col_name="related_work"))
            dataset = dataset.map(partial(flatten_ref_papers_from_citations, mapped=True))
            df_pandas = dataset.to_pandas()
            all_df = get_pandas_percentile(df_pandas, ["total_ref_words", "abstract_length", "related_work_length"])
            df_list.append(all_df)
        # print(df_list)
        xls_file_path = path.join(savedir, f"data_stats.xlsx")
        write_excel_df(
            df_list=df_list,
            sheet_name_list=splits,
            save_file_path=xls_file_path,
            close_writer=True,
        )

    def citation_by_citation_baseline(self, dataset_name, savedir):
        dataset = load_dataset(dataset_name, split="test")
        set_caching_enabled(False)
        dataset = dataset.map(partial(create_model_input, one_line_baseline=True, nlp_spacy=self.nlp_spacy))
        preds = dataset["text"]
        refs = dataset["related_work"]
        result_list = []
            # Dataset specific
        for row in tqdm(dataset):
            abstract = row["abstract"]
            gt_related_work = row["related_work"]
            references = row["ref_abstract"]  # abstract
            citations = references["cite_N"]
            abstracts = references["abstract"]

            data_text = f"Abstract: {abstract}"
            for index in range(len(citations)):
                data_text = (
                    f"Abstract: {abstract} \n Reference {citations[index]}: {abstracts[index]}"
                )
            response_txt = ""
            prompts = load_all_prompts()
            base_prompt = prompts[prompt_type]
            prompt = self.get_complete_prompt(base_prompt, data_text)
            
            try:
                response_txt = self.ml_model.run_pipeline(prompt)
            except Exception as e:
                response_txt = ""
                print("The error is: ", e)


            print("\n\n Generated related work: ", response_txt, "\n\n")

            response_json = {
                "gen_related_work": response_txt,
                "gt_related_work": gt_related_work,
                "original_abstract": abstract,
                "citations": citations,
                "abstracts": abstracts,
            }
            preds.append(response_txt)
            refs.append(gt_related_work)
            result_list.append(response_json)


        config={"model_name": "citation_by_citation"}
        config = Dict2Class(config)
        score_pipeline = MDSGen(config, inference=False)
        dataset = pd.DataFrame(dataset)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        score_pipeline.calculate_metrics(preds, refs, result_list, savedir)



def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="one_line_baseline",
        help="Model name",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default="/results/auto_review/multixscience/one_line_baseline",
        help="Path to save dir",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="multi_x_science_sum",
        help="Dataset name",
    )
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    model = OneLineBaseline(parsed_args)
    print("Getting data stats first!")
    model.get_data_stats(parsed_args.dataset_name, parsed_args.savedir)
    print("Generating one line baseline now")
    model.generate_one_line_baseline(parsed_args.dataset_name, parsed_args.savedir)    