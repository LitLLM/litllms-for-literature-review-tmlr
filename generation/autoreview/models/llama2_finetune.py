# https://www.philschmid.de/instruction-tune-llama-2
# https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
# https://github.com/mlabonne/llm-course/blob/main/Fine_tune_Llama_2_in_Google_Colab.ipynb
# https://www.mlexpert.io/machine-learning/tutorials/alpaca-fine-tuning
# https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19 
# TODO: https://together.ai/blog/llama-2-7b-32k https://huggingface.co/togethercomputer/LLaMA-2-7B-32K 

# Getting llama
# https://medium.com/@indirakrigan/lessons-i-learned-while-i-arrived-at-llama-2-the-long-way-4a9a0c903bf

from .toolkit_utils import set_env_variables, SNOWTrainer
set_env_variables(do_hf_login=True)
import os
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
import bitsandbytes as bnb
from functools import partial
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, get_peft_model
from trl import SFTTrainer
import json
from typing import Any, List, Dict
import pathlib
import argparse
from os import path
import pickle as pkl
# from autoreview.models.pipeline import MDSGen
from .pipeline import MDSGen
from .configs_llama import train_config
from .ml_utils import get_gpu_memory, Dict2Class
from .quant_utils import (
    find_all_linear_names, print_trainable_parameters, 
    create_bnb_config, create_peft_config)
from .data_utils import (compute_length, create_llama_chat_prompt, create_prompt_formats, get_base_name_from_hf_path,
    get_max_length, length_stats, preprocess_dataset, preprocess_batch, postprocess_output)

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{30000}MB'
    print(f"No. of GPUs: {n_gpus}, max_memory: {max_memory}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    # tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=model_max_length, truncation=truncation)
    # https://discuss.huggingface.co/t/where-to-put-use-auth-token-in-the-code-if-you-cant-run-hugginface-cli-login-command/11701
    # https://github.com/facebookresearch/llama/issues/374
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name,use_auth_token='token_value')
    # tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token = 'token_value')
    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def train(model, tokenizer, dataset, output_dir, savedir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = SNOWTrainer(savedir=savedir,
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=3,
            gradient_accumulation_steps=3,
            warmup_steps=2,
            num_train_epochs=2,
            max_grad_norm = 0.3,
            learning_rate=5e-6,
            bf16=True,
            save_strategy="steps",
            save_steps=5000,
            save_total_limit=3,
            logging_steps=100,
            lr_scheduler_type="cosine",
            output_dir=output_dir,
            logging_dir=f"{output_dir}/logs",
            optim="paged_adamw_32bit"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # fp16 = True
    # optim="adamw_torch_fused", paged_adamw_32bit
    # learning_rate=1e-5,

    # evaluation_strategy="steps",
    # eval_steps=200,
    # https://discuss.huggingface.co/t/save-only-best-model-in-trainer/8442
    # save_strategy="steps",
    # save_strategy="no",
    # save_total_limit = 2
    # save_strategy = “no”
    # load_best_model_at_end=False

    # Also see: https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
    # https://colab.research.google.com/drive/1mV9sAY4QBKLmS58dpFGHgwCXQKRASR31?usp=sharing#scrollTo=qf1qxbiF-x6p
    # max_steps=2, warmup_steps=2, logging_steps=1, num_train_epochs=1, gradient_accumulation_steps=2, learning_rate=2e-4,
    # max_grad_norm=0.3, warmup_ratio=0.03, lr_scheduler_type="constant", optim="paged_adamw_8bit"

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    do_train = True
    
    # Launch training
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()
    best_ckpt_path = trainer.state.best_model_checkpoint
    print(f"best_ckpt_path: {best_ckpt_path}")



class LlamaV2Finetune:
    """
    Class to generate auto insights for text based on LLMs
    """
    def __init__(self, config):
        self.class_name = "LlamaV2Finetune"
        self.config = config
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name

    def llama_finetune(self, save_dir, do_train: bool = True, output_merged_dir = None, eval_batch_size: int =1):
        print(f"Training is set to {do_train}")

        if do_train:
            dataset = load_dataset(self.dataset_name, split="train")
            print(f"Training on {self.dataset_name}")
            bnb_config = create_bnb_config()
            model, tokenizer = load_model(self.model_name, bnb_config)
            ## Preprocess dataset
            max_length = get_max_length(model)
            # TODO: SA: debugging llama 70b model
            max_length = 2800
            seed=42
            dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
            output_dir = f"{save_dir}/final_checkpoint"
            os.makedirs(output_dir, exist_ok=True)
            train(model, tokenizer, dataset, output_dir, save_dir)
            model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
            model = model.merge_and_unload()
            get_gpu_memory()
            output_merged_dir = f"{save_dir}/final_merged_checkpoint"
            os.makedirs(output_merged_dir, exist_ok=True)
            model.save_pretrained(output_merged_dir, safe_serialization=True)
            # SA: TODO
            # model.config.to_json_file("config.json")
            print(f"Saving the model at {output_merged_dir}")
            # save tokenizer for easy inference
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=max_length, truncation=True)
            tokenizer.save_pretrained(output_merged_dir)
            print("Training done!")
        else:
            # ValueError: Can't find 'adapter_config.json' at '/mnt/colab_public/results/auto_review/ad6df82733208405381eeed4471b896b/final_merged_checkpoint'
            #model = AutoPeftModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)
            # TODO:
            # model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
            # model = model.merge_and_unload()
            # TODO: https://colab.research.google.com/drive/1mV9sAY4QBKLmS58dpFGHgwCXQKRASR31?usp=sharing#scrollTo=xg6nHPsLzMw-
            model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)

            max_length = 3000
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=max_length, truncation=True)
            print(f"Model and tokenizer loaded!")
        # https://github.com/facebookresearch/llama/issues/540
        # https://github.com/huggingface/transformers/issues/8452
        tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation 
        tokenizer.padding_side = "right"                       
        get_gpu_memory()
        self.generate(model, tokenizer, save_dir, eval_batch_size)

    def generate_wo_pipeline(model, tokenizer, prompt):
        # TODO: Follow https://colab.research.google.com/drive/1X1z9Q6domMKl2CnEM0QGHNwidLfR4dW2?usp=sharing#scrollTo=THqfvzHIjSK9
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        output = model.generate(**model_inputs)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        # inputs = tokenizer(text, return_tensors="pt").to(device)
        # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), 
        # attention_mask=inputs["attention_mask"], max_new_tokens=50, 
        #                          pad_token_id=tokenizer.eos_token_id)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return

    def generate(self, model, tokenizer, savedir, batch_size: int = 3, batched: bool = True):
        dataset = load_dataset(self.dataset_name, split="test")
        # dataset = dataset.select(list(range(5)))
        # https://discuss.huggingface.co/t/dataset-map-method-how-to-pass-argument-to-the-function/16274/2
        # dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
        # max_length = get_max_length(model)
        dataset = dataset.map(partial(create_prompt_formats, eval=True, truncate_context=True, max_len_context=1900))
        # df_pandas = pd.DataFrame(dataset)
        inputs = dataset["text"]
        refs = dataset["related_work"]
        preds = []
        print("Generating sentences from pipeline")
        pipe = pipeline("text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            batch_size=batch_size)
        #             torch_dtype=torch.float16,
        print("Pipeline defined. Generating outputs")
        if batched: 
            total_examples = len(dataset)
            for chunk in tqdm(range(total_examples // batch_size + 1)):
                prompt = dataset[batch_size * chunk: (batch_size+1) * chunk]["text"]
                sequences = pipe(
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
        # SA: TODO batch size
        # Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with `pipe.tokenizer.pad_token_id = model.config.eos_token_id`.
        else:
            for row in tqdm(dataset):
                # print(f"Length of sequence {len(row['text'].split())}")
                sequences = pipe(
                    row["text"],
                    do_sample=True,
                    return_full_text=False,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=500,
                    temperature=0.9)
                response_txt = sequences[0]['generated_text']
                preds.append(response_txt)
        print(f"Total length of preds: {len(preds)}")

        config={"model_name": "finetuned_llama"}
        config = Dict2Class(config)
        score_pipeline = MDSGen(config, inference=False)
        dataset = pd.DataFrame(dataset)
        score_pipeline.calculate_metrics(preds, refs, dataset, savedir)


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
        "-s",
        "--save_dir",
        default="/outputs",
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
    config = parse_args()
    llama_trainer = LlamaV2Finetune(config)
    dataset_name = get_base_name_from_hf_path(config.dataset_name)
    model_name = get_base_name_from_hf_path(config.model_name)
    save_dir = f"{config.savedir}/{dataset_name}/{model_name}_finetuning_lr_5e-6"
    llama_trainer.llama_finetune(save_dir=config.save_dir)