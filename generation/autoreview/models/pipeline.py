#!/usr/bin/env python3

"""
This file defines class to calculate summaries
"""
import os
# os.environ["HF_HOME"] = "/mnt/home/cached/"
# os.environ["TORCH_HOME"] = "/mnt/home/cached/"
HF_TOKEN = os.environ['HF_TOKEN']

from json.decoder import JSONDecodeError
from typing import Any
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from os import path
import pickle as pkl
from .ml_utils import load_all_prompts, write_excel_df, get_gpu_memory
from .chatgpt_model import ChatGPTModel
from .langchain_openai_agent import OpenAIAgent
from .llama2_zero_shot import Llamav2Pipeline
from ..evaluation.compute_score import Metrics
from .parse_all_args import parse_args
import time


class MDSGen:
    """
    Class to generate auto insights for text based on LLMs
    """

    def __init__(self, config, inference: bool = True):
        self.class_name = "MDSGen"
        self.config = config
        self.model_name = config.model_name
        self.metric = Metrics(config={"metric": "rouge"})
        if inference:
            self.use_langchain = config.use_langchain
            self.ml_model = self.get_model()
            # Load all prompts from resources/prompts.json
            self.prompts = load_all_prompts()            
            self.dataset_name = config.dataset_name 

    def get_model(self) -> str:
        ml_model = None
        print(f"Running inference for {self.model_name}")
        if self.config.mode=="chatgpt":
            if self.use_langchain:
                ml_model = OpenAIAgent(self.model_name)
            else:
                ml_model = ChatGPTModel(self.model_name)
        elif self.config.mode=="llama":
            # Login to HF to access llama models
            from huggingface_hub import login as hf_login
            hf_login(token=HF_TOKEN)
            ml_model = Llamav2Pipeline(self.config) 
            get_gpu_memory()               
        return ml_model

    def get_base_prompt(self, prompt_type) -> str:
        """
        Get base prompt from prompts json in resources directory
        """
        base_prompt = self.prompts[prompt_type]

        return base_prompt


    def get_complete_prompt(self, base_prompt: str, data: str) -> str:
        """
        This function gets the complete prompt filling the base prompt with data

        # https://huggingface.co/blog/codellama#conversational-instructions
        """
        complete_prompt = f"{base_prompt}\n```{data}```\nRelated work: "

        if "use_llama_chat" in self.config:
            # Llama chat expected prompt
            # NOTE: this is for chat model, have to verify for non chat models
            complete_prompt = f"<s>[INST] {complete_prompt.strip()} [/INST]"

            # # https://huggingface.co/blog/codellama#conversational-instructions
            # complete_prompt = f"[INST] <<SYS>>\n{complete_prompt}\n<</SYS>>\n\n"
            # prompt = f"<s>[INST] {prompt.strip()} [/INST]"
            # prompt = f"<s><<SYS>>\\n{sys_msg}\\n<</SYS>>\\n\\n{complete_prompt}"
            # llama_prompt = f"[INST] <<SYS>>\n{complete_prompt}\n<</SYS>>\n\n[/INST]"
        return complete_prompt

    def generate_response(self, prompt):
        response_txt = ""
        if self.config.mode=="chatgpt":
            if self.use_langchain:
                response_dict = self.ml_model.get_response(prompt)
                response_txt = response_dict["response"]
            else:
                # Truncating tokens to max tokens allowed by the API
                prompt = self.ml_model.truncate_tokens(input_txt=prompt)
                # print(prompt)
                api_data = {"prompt": prompt}
                try_counter = 0
                # ChatGPT can produce incorrect json output sometimes
                # To catch these corner cases we use try, except clause
                # TODO: temperature
                while try_counter < self.config.max_tries:
                    try:
                        response_txt = self.ml_model.gen_response(api_data)
                        break
                    except JSONDecodeError:
                        print("Incorrect json output")
                        try_counter += 1
        elif self.config.mode=="llama":
            try:
                response_txt = self.ml_model.run_pipeline(prompt)
            except Exception as e:
                response_txt = ""
                print("The error is: ", e)
        response_txt = response_txt.strip()
        return response_txt


    def generate_summary(self, savedir="", per_citation: bool=False, per_citation_summarize: bool = False) -> Any:
        """
        per_citation: if we want to generate results per citation
        per_citation_summarize: if we also want to paraphrase / summarize the per citation generated summary again
        :return:
        """
        get_gpu_memory()
        start = time.time()
        # Default values here
        result_list = []
        preds = []
        refs = []
        preds_per_cite = []
        if per_citation:
            print(f"Summarizing for each citation individually. Also CoT style summary set to {per_citation_summarize}")
        if "zero_shot" in self.config:
            base_prompt = self.get_base_prompt(self.config.prompt_type)
            dataset = load_dataset(self.config.dataset_name, split="test")
            # If we want to create small dataset: 
            if "small_dataset" in self.config:
                if "shuffle" in self.config:
                    import random
                    data_idx = random.choices(range(len(dataset["test"])), k=self.config.small_dataset_num)
                    # or do shuffled_dataset = dataset.shuffle(seed=42)
                    dataset = dataset["test"].select(data_idx)
                # dataset = dataset.select(list(range(100)))
                dataset = dataset.select(list(range(self.config.small_dataset_num)))
                print("Running on small dataset")

            # Dataset specific
            for row in tqdm(dataset):
                abstract = row["abstract"]
                gt_related_work = row["related_work"]
                references = row["ref_abstract"]  # abstract
                citations = references["cite_N"]
                abstracts = references["abstract"]

                data_text = f"Abstract: {abstract}"
                # per_citation -> Generate response for each citation individually and combine them together
                if not per_citation:
                    for index in range(len(citations)):
                        # Some are empty abstracts
                        if abstracts[index] != "":
                            data_text = (
                                f"{data_text} \n Reference {citations[index]}: {abstracts[index]}"
                            )
                    prompt = self.get_complete_prompt(base_prompt, data_text)
                    response_txt = self.generate_response(prompt=prompt)
                else:
                    per_cite_response_list = []
                    for index in range(len(citations)):                        
                        if abstracts[index] != "":
                            per_citation_text = (f"{data_text} \n Reference {citations[index]}: {abstracts[index]}")
                            prompt = self.get_complete_prompt(base_prompt, per_citation_text)
                            response_txt = self.generate_response(prompt=prompt)
                            # print(f"\n Per cite text: {per_citation_text}\n")
                            # print(f"\n Individual response: {response_txt}\n")
                            per_cite_response_list.append(response_txt)
                    response_txt = " ".join(per_cite_response_list)
                    # Summarize again -- kind of CoT
                    if per_citation_summarize:
                        prompt_cot = self.get_base_prompt(self.config.prompt_cot)
                        # Here we are summarizing again the combined generated text
                        prompt = self.get_complete_prompt(prompt_cot, response_txt)
                        response_cot_txt = self.generate_response(prompt=prompt)
                        # print(f"\n CoT response: {response_cot_txt}\n")

                # print("\n\n Generated related work: ", response_txt, "\n\n")

                response_json = {
                    "gen_related_work": response_txt,
                    "gt_related_work": gt_related_work,
                    "original_abstract": abstract,
                    "citations": citations,
                    "abstracts": abstracts,
                }
                if per_citation_summarize:
                    response_json["gen_response_cot"] = response_cot_txt
                    preds_per_cite.append(response_cot_txt)
                preds.append(response_txt)
                refs.append(gt_related_work)
                result_list.append(response_json)
        done = time.time()
        elapsed = done - start
        print(f"Elapsed time in seconds: {elapsed}")
        # Calculate metrics and save results
        self.calculate_metrics(preds, refs, result_list, savedir, per_citation=per_citation)
        if per_citation_summarize:
            self.calculate_metrics(preds_per_cite, refs, result_list, savedir, per_citation_summarize, per_citation)

    def calculate_metrics(self, preds, refs, result_list, savedir, plans = None, per_citation_summarize: bool = False, per_citation: bool = False):
        result_json = {"preds": preds, "refs": refs}
        # rouge_scores_eval_hf - use evaluate library while the other one use datasets
        all_scores, rouge_scores_dataset_hf, rouge_scores_eval_hf = self.metric.compute_score(result_json)
        rouge_scores_eval_hf = pd.DataFrame.from_dict(rouge_scores_eval_hf, orient='index')
        rouge_scores_dataset_hf = pd.DataFrame.from_dict(rouge_scores_dataset_hf, orient='index')
        all_scores = pd.DataFrame.from_dict(all_scores, orient='index')

        if isinstance(result_list, list):
            print(f"Total number of responses are: {len(result_list)}")
            # Creating df from the list
            pred_df = pd.DataFrame(result_list)
        else:
            pred_df = result_list
        
        if plans:
            print(f"Also saving plans!")
            result_json["gen_plans"] = plans
        results_df = pd.DataFrame(result_json)
        # Use os.path.split(self.model_name)[1] to get model name from "meta/Llama_X_*" 
        if per_citation_summarize:
            xls_file_path = path.join(savedir, f"results_{os.path.split(self.model_name)[1]}_per_cite_{per_citation}_summarize_{per_citation_summarize}.xlsx")
            pkl_file_path = path.join(savedir, f"results_{os.path.split(self.model_name)[1]}_per_cite_{per_citation}_summarize_{per_citation_summarize}.pkl")
        else:
            xls_file_path = path.join(savedir, f"results_{os.path.split(self.model_name)[1]}.xlsx")
            pkl_file_path = path.join(savedir, f"results_{os.path.split(self.model_name)[1]}.pkl")
        write_excel_df(
            df_list=[results_df, pred_df, rouge_scores_eval_hf, rouge_scores_dataset_hf, all_scores],
            sheet_name_list=["responses", "data", "rouge_scores_eval_hf", "rouge_scores_dataset_hf", "all_scores"],
            save_file_path=xls_file_path,
            close_writer=True,
        )
        # pred_df -- data, result_json -- preds, refs, plans
        # Todo: Store as pkl_dump = {"pred_df": pred_df, "result_list": result_list, "preds_with_refs": result_json}
        with open(pkl_file_path, 'wb') as fp:
            pkl.dump([pred_df, result_list, result_json], fp)


        print(f"Writing the predictions at: {xls_file_path}")
        print(f"Writing the list at: {pkl_file_path}")

        return pred_df


if __name__ == "__main__":
    parsed_args = parse_args()
    generator = MDSGen(parsed_args)
    generator.generate_summary(savedir=parsed_args.savedir)

