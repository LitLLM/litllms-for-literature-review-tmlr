# Run as python -m autoreview.models.zero_shot_longer_context
# or
# PYTHONPATH=. python autoreview/models/zero_shot_longer_context.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, GenerationConfig
from transformers.pipelines.base import KeyDataset
from autoreview.models.ml_utils import get_gpu_memory
from autoreview.models.ml_utils import get_gpu_memory, load_all_prompts, Dict2Class
from autoreview.models.toolkit_utils import set_env_variables
set_env_variables(do_hf_login=True)
import torch
import os
import argparse
from datasets import load_dataset, set_caching_enabled
import torch
import transformers
from autoreview.models.data_utils import (create_model_input, get_complete_mapped_prompt, get_base_name_from_hf_path,
                                          pkl_load, metrics_wrapper, get_hf_dataset, postprocess_output) 
from autoreview.models.pipeline import MDSGen
from functools import partial
from tqdm import tqdm


class ZeroShotHFModel:
    """
    Class to generate auto insights for text based on LLMs
    """
    def __init__(self, config, batch_size:int = 2):
        self.class_name = self.__class__.__name__
        self.config = config
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name
        self.model_type = self.config.model_type
        self.tokenizer, self.model, self.pipeline = self.get_tokenizer_pipeline(self.config.model_name, batch_size=batch_size)
        self.prompts = load_all_prompts()
        self.base_prompt = self.prompts[self.config.prompt_type]        

    def get_tokenizer_pipeline(self, model_name, batch_size: int = 1):
        get_gpu_memory()
        self.model, self.tokenizer = get_model_tokenizer(self.config)
        get_gpu_memory()
        pipe = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
            batch_size=batch_size,
            device_map="auto"
        )
        #             batch_size=batch_size,
        #             torch_dtype=torch.float16,
        # pipe.tokenizer.pad_token_id = self.model.config.eos_token_id
        return self.tokenizer, self.model, pipe


    def generate_outputs(self, dataset_name, savedir, max_len_ctx: int = 1600, small_dataset: bool = False, max_len_cite: int = 250, batch_size: int = 2):
        preds = []
        dataset = get_hf_dataset(dataset_name, small_dataset)
        if self.config.gen_type == "vanilla":
            dataset = dataset.map(partial(create_model_input, use_full_text=True, plan_based_gen=False, max_len_ctx=max_len_ctx, max_len_cite=max_len_cite))
            dataset = dataset.map(partial(get_complete_mapped_prompt, base_prompt=self.base_prompt, data_column="text", use_llama_chat=True))

        total_examples = len(dataset)
        for chunk in tqdm(range(total_examples // batch_size + 1)):
            prompt = dataset[batch_size * chunk: (batch_size+1) * chunk]["text"]
            # descr = test_df[batch_size * chunk: (batch_size+1) * chunk]['description'].to_list()
        # for row in tqdm(dataset):
            # prompt = row["text"]
            sequences = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False,
                top_k=10,
                num_return_sequences=1,
                max_new_tokens=500,
                temperature=0.7)
                # eos_token_id=self.tokenizer.eos_token_id,
            # output_text = sequences[0]['generated_text']
            # output_text = postprocess_output(output_text)
            # preds.append(output_text)
            preds.extend([postprocess_output(out["generated_text"]) for out in sequences])            

        refs = dataset["related_work"]
        metrics_wrapper(model_name, savedir, preds, refs, dataset)

    def generate_w_pipeline(self, dataset):
        preds = []
        generation_config = GenerationConfig.from_pretrained(self.config.model_name)
        generation_config = GenerationConfig.from_dict({**generation_config.to_dict(), "do_sample":True,
                                                        'max_new_tokens':512, "temperature":0.6, 
                                                        "repetition_penalty":1.2})
        for outputs in tqdm(self.pipeline(KeyDataset(dataset, "text"), return_full_text=False, eos_token_id=self.tokenizer.eos_token_id,
                                        generation_config=generation_config)):
            preds.extend([postprocess_output(out["generated_text"]) for out in outputs])
        return preds


    def generate_wo_batch(self, dataset):
        preds= []
        for row in tqdm(dataset):
            prompt = row["text"]
            # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") #.to(model.device)
            # # input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()
            # # output = model.generate(input_ids, max_length=128, temperature=0.7)
            # # input_ids = self.tokenizer(batch[field], padding='max_length', truncation = True, return_tensors="pt").input_ids.to("cuda")
            # generation_kwargs = {"num_beams": 7, "max_length": 512,
            #                     "num_return_sequences":2, "temperature":0.7,
            #                     "do_sample": True, "top_k": 90, "top_p": 0.95,
            #                     "no_repeat_ngram_size": 2, "early_stopping": True}
            # generated_text = self.model.generate(**tokenized_inputs, **generation_kwargs)
            # output = self.model.generate(input_ids, return_full_text=False, max_new_tokens=500, temperature=0.7, repetition_penalty=1.1, top_p=0.7, top_k=50)
            # # output = self.model.generate(input_ids, max_length=500, temperature=0.7, repetition_penalty=1.1, top_p=0.7, top_k=50)
            # output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            # tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
            # output_text = output_text.lstrip('<pad>')
            # output_text = postprocess_output(output_text)
            # del input_ids
            # # del input_ids["token_type_ids"]

            sequences = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False,
                top_k=10,
                num_return_sequences=1,
                max_new_tokens=500,
                temperature=0.7)
                # eos_token_id=self.tokenizer.eos_token_id,
            output_text = sequences[0]['generated_text']
            output_text = postprocess_output(output_text)
            preds.append(output_text)
            del sequences
        return preds


def get_model_tokenizer(parsed_args):
    if parsed_args.model_type == "oa": # 8k
        # https://huggingface.co/OpenAssistant/llama2-13b-orca-8k-3319
        tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", load_in_4bit=True,
                                                     torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        # output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
    elif parsed_args.model_type == "upstage": # 10k
        # https://huggingface.co/upstage/SOLAR-0-70b-8bit - 10k
        tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-0-70b-8bit")
        model = AutoModelForCausalLM.from_pretrained(
            "upstage/SOLAR-0-70b-8bit",
            device_map="auto")
        #     load_in_8bit=True,
        #     low_cpu_mem_usage=True,
        #     rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
        # )        
        # #             load_in_8bit_fp32_cpu_offload=False,
        # torch_dtype=torch.float16,
        # load_in_8bit=True,
        # load_in_8bit_fp32_cpu_offload=False
    elif parsed_args.model_type == "togetherai": # 32k
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K",device_map="auto", 
                                                     load_in_4bit=True, trust_remote_code=False, torch_dtype=torch.float16)
        # trust_remote_code=False if you prefer not to use flash attention
        # output = model.generate(input_ids, max_length=128, temperature=0.7, repetition_penalty=1.1, top_p=0.7, top_k=50)
    elif parsed_args.model_type == "llama": # 10k
        tokenizer = AutoTokenizer.from_pretrained(parsed_args.model_name)
        model = AutoModelForCausalLM.from_pretrained(parsed_args.model_name, device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
        )

        # load_in_8bit=True,
    elif parsed_args.model_type == "gptq": # 14k
        model = AutoModelForCausalLM.from_pretrained(parsed_args.model_name,
                                                    device_map="auto",
                                                    trust_remote_code=False,
                                                    revision="main")
        tokenizer = AutoTokenizer.from_pretrained(parsed_args.model_name, use_fast=True)

    # https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020
    # tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation 
    # tokenizer.padding_side = "right"
    # tokenizer.pad_token = "<PAD>"
    # Define PAD Token = BOS Token
    # tokenizer.pad_token = tokenizer.bos_token
    # model.config.pad_token_id = model.config.bos_token_id
    # # https://github.com/facebookresearch/llama/issues/380
    model = model.bfloat16()

    return model, tokenizer


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="oa",
        help="Model name",
    )
    # meta-llama/Llama-2-7b-chat-hf
    # meta-llama/Llama-2-13b-chat-hf
    # meta-llama/Llama-2-70b-chat-hf    
    # TheBloke/Llama-2-70B-GPTQ
    # togethercomputer/LLaMA-2-7B-32K
    parser.add_argument(
        "-mt",
        "--model_type",
        default="oa",
        choices = ["oa", "upstage", "togetherai", "llama"],
        help="Model name",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default="/datadrive2/shubham/outputs",
        help="Path to save dir",
    )
    # /mnt/colab_public/results/auto_review/multixscience/
    # /mnt/colab_public/results/auto_review/rw_2308_filtered/
    # /datadrive2/shubham/outputs
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
    out_dir = f"{parsed_args.savedir}/{dataset_name}/{model_name}_{parsed_args.gen_type}"
    print(f"Outputs would be saved in {out_dir}")
    if parsed_args.model_type == "oa": # 8k
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=7000, max_len_cite=400)
    elif parsed_args.model_type == "upstage": # 10k
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=9000, max_len_cite=400)
    elif parsed_args.model_type == "togetherai": # 32k
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=5000, max_len_cite=400)
        # model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=15000, max_len_cite=400)
    elif parsed_args.model_type == "llama": # 10k
        # 6k 7b 4bit 32GB
        # # 5k 7b 4bit 28GB
        # model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=13000, max_len_cite=400, small_dataset=True)
        print("Generating with 7000 tokens now")
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=7000, max_len_cite=400)
        print("Generating with 9000 tokens now")
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=9000, max_len_cite=400)
        print("Generating with 11000 tokens now")
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=11000, max_len_cite=400)
        print("Generating with 13000 tokens now")
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=13000, max_len_cite=400)
        print("Generating with 15000 tokens now")
        model.generate_outputs(parsed_args.dataset_name, savedir=out_dir, max_len_ctx=15000, max_len_cite=400)
