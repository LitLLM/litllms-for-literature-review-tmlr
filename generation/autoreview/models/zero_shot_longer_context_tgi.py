# Run as python -m autoreview.models.zero_shot_longer_context
# or
# PYTHONPATH=. python autoreview/models/zero_shot_longer_context.py
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, GenerationConfig
from autoreview.models.ml_utils import get_gpu_memory, load_all_prompts, Dict2Class, concurrent_requests, hf_tgi
from autoreview.models.toolkit_utils import set_env_variables
set_env_variables(do_hf_login=True)
# import torch
# import os
import argparse
from autoreview.models.data_utils import (create_model_input, get_complete_mapped_prompt, get_base_name_from_hf_path,
                                          pkl_load, metrics_wrapper, get_hf_dataset, postprocess_output) 
from autoreview.models.pipeline import MDSGen
from functools import partial
from tqdm import tqdm


class ZeroShotHFModel:
    """
    Class to generate auto insights for text based on LLMs
    """
    def __init__(self, config, batch_size:int = 4):
        self.class_name = self.__class__.__name__
        self.config = config
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name
        self.model_type = self.config.model_type
        self.prompts = load_all_prompts()
        self.base_prompt = self.prompts[self.config.prompt_type]        

    def generate_outputs(self, dataset_name, model_name, savedir, max_len_ctx: int = 1600, small_dataset: bool = False, max_len_cite: int = 250,
                         max_workers = 2):
        preds = []
        dataset = get_hf_dataset(dataset_name, small_dataset)
        if self.config.gen_type == "vanilla":
            dataset = dataset.map(partial(create_model_input, use_full_text=True, plan_based_gen=False, max_len_ctx=max_len_ctx, max_len_cite=max_len_cite))
            dataset = dataset.map(partial(get_complete_mapped_prompt, base_prompt=self.base_prompt, data_column="text", use_llama_chat=True))
        prompts = dataset["text"]
        config = {"service_url": "http://0.0.0.0:80"}
        config = Dict2Class(config)
        preds = concurrent_requests(prompts, config, hf_tgi, max_workers=max_workers)

        refs = dataset["related_work"]
        metrics_wrapper(model_name, savedir, preds, refs, dataset)
        # text = "What is DL?"
        # results,_ = hf_tgi(text, service_url, headers= False)
        # print(results)

def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="meta-llama/Llama-2-70b-chat-hf",
        help="Model name",
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        default="llama",
        choices = ["oa", "upstage", "togetherai", "llama"],
        help="Model name",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default="/outputs",
        help="Path to save dir",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="shubhamagarwal92/rw_2308_filtered",
        help="Dataset name",
    )
    # shubhamagarwal92/multi_x_science_test_full
    # shubhamagarwal92/rw_2308_filtered
    # multi_x_science_sum
    parser.add_argument(
        "-gt",
        "--gen_type",
        default="vanilla",
        help="Type of generation",
    )
    # plan_based_gen
    parser.add_argument(
        "-p",
        "--prompt_type",
        default="vanilla_template_full_text",
        help="Type of prompt to choose from all the prompt templates",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parsed_args = parse_args()
    model = ZeroShotHFModel(parsed_args, batch_size=1)
    print("Generating summaries now")
    dataset_name = get_base_name_from_hf_path(parsed_args.dataset_name)
    model_name = get_base_name_from_hf_path(parsed_args.model_name)
    prefix_out_dir = f"{parsed_args.savedir}/{dataset_name}/{model_name}_{parsed_args.gen_type}"
    
    if parsed_args.model_type == "llama": # 10k
        # 38G -- 4 GPUs - 70B
        # 6k 7b 4bit 32GB
        # # 5k 7b 4bit 28GB
        # for ctx_length in [5000, 7000, 9000, 10000]:
        # 3000, 5000, 7000
        for ctx_length in [4000]:
            max_workers=4 # For 7b model
            # max_workers=8 # For 7b model
            # max_workers=6 # For 13b model
            out_dir = f"{prefix_out_dir}_{ctx_length}_words"
            print(f"Outputs would be saved in {out_dir}")
            model.generate_outputs(parsed_args.dataset_name, model_name=model_name, savedir=out_dir, max_len_ctx=ctx_length, max_len_cite=800, 
                                   max_workers=max_workers)
            print("\n-----------------")
        
        # [8000, 9000, 11000]
        for ctx_length in [6000]:
            max_workers=3
            out_dir = f"{prefix_out_dir}_{ctx_length}_words"
            print(f"Outputs would be saved in {out_dir}")
            model.generate_outputs(parsed_args.dataset_name, model_name=model_name, savedir=out_dir, max_len_ctx=ctx_length, max_len_cite=800, 
                                    max_workers=max_workers)


