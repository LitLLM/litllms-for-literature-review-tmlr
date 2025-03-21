# https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=d8aa5db3
# https://github.com/pacman100/DHS-LLM-Workshop/tree/main/chat_assistant/training
# https://www.philschmid.de/gptq-llama
# https://www.philschmid.de/instruction-tune-llama-2
# https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
# https://www.philschmid.de/fine-tune-flan-t5-deepspeed
# https://www.philschmid.de/deepspeed-lora-flash-attention
import os
from dataclasses import dataclass, field
from typing import Optional
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments, pipeline, GenerationConfig
)
from tqdm import tqdm
from transformers.pipelines.base import KeyDataset
from trl import SFTTrainer
from autoreview.models.toolkit_utils import set_env_variables, SNOWTrainer, CombinedArguments
set_env_variables(do_hf_login=True)
from autoreview.models.pipeline import MDSGen
from autoreview.models.data_utils import get_complete_mapped_prompt, get_hf_dataset, create_model_input
import numpy as np

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-70b-chat-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    # Llama-2-7b-hf
    dataset_name: Optional[str] = field(
        default="multi_x_science_sum",
        metadata={"help": "The preference dataset to use."},
    )
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=5e-6)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    # float16",
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    # SA: TODO setting it to true
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=100000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=True,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="/datadrive2/shubham/outputs/multi_x_science_sum/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

def get_bnb_config(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    return compute_dtype, bnb_config

def create_and_prepare_model(args):
    compute_dtype, bnb_config = get_bnb_config(args)
    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)
    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    # device_map = {"": 0}
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        use_auth_token=True
    )
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1 
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def get_training_arguments(script_args):
    training_arguments = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        max_grad_norm=script_args.max_grad_norm,
        max_steps=script_args.max_steps,
        warmup_ratio=script_args.warmup_ratio,
        group_by_length=script_args.group_by_length,
        lr_scheduler_type=script_args.lr_scheduler_type,
        save_strategy="steps",
        save_total_limit=3
    )
        # save_strategy="steps",
        # save_steps=5000,
        # save_total_limit=3,
    return training_arguments


def train(script_args):
    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False
    dataset = load_dataset(script_args.dataset_name, split="train")
    # Fix weird overflow issue with fp16 training
    tokenizer.padding_side = "right"
    training_arguments = get_training_arguments()
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=script_args.packing,
    )
    # max_seq_length=script_args.max_seq_length,
    trainer.train()
    if script_args.merge_and_push:
        output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
        trainer.model.save_pretrained(output_dir)
        # Free memory for merging weights
        del model
        torch.cuda.empty_cache()
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)

def compute_metrics(eval_pred, tokenizer):
    # https://discuss.huggingface.co/t/trainer-never-invokes-compute-metrics/11440
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

def llama_pipeline(model_name = "meta-llama/Llama-2-7b-chat-hf", model_max_length: int = 1500, truncation: bool = True, load_in_4bit: bool = True, batch_size: int=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length, truncation=truncation)
    tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation 
    tokenizer.padding_side = "right"
    tokenizer.pad_token = "<PAD>"
    generation_config = GenerationConfig.from_pretrained(model_name)
    print(generation_config)
    _, bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, 
                                                    device_map="auto", trust_remote_code=True)        
    print(f"Loading {model} in 4 bit")
    # model = AutoModelForCausalLM.from_pretrained(model, device_map='auto', load_in_4bit=load_in_4bit)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id
    preds = []
    max_length = 200

    generation_config = GenerationConfig.from_dict({**generation_config.to_dict(), 'max_new_tokens':max_length, "temperature":0.4, "repetition_penalty":1.2})

    dataset = load_dataset("multi_x_science_sum", split="test").select(list(range(9)))
    # for outputs in tqdm(pipeline(KeyDataset(dataset, "text"), return_full_text=False, eos_token_id=tokenizer.eos_token_id, max_new_tokens=200)):
    for outputs in tqdm(pipe(KeyDataset(dataset, "abstract"), return_full_text=False, eos_token_id=tokenizer.eos_token_id, generation_config=generation_config)):
        preds.extend([out["generated_text"] for out in outputs])
        print(outputs[0]["generated_text"])

    # Can also do https://discuss.huggingface.co/t/trainer-never-invokes-compute-metrics/11440/6
    # for outputs in tqdm(self.pipeline(KeyDataset(dataset, "text"), return_full_text=False, max_new_tokens=max_length, 
    #                                   do_sample=True,top_p=0.8, temperature=0.8, repetition_penalty=1.2)):
    #     preds.extend([self.postprocess_code_output(out["generated_text"]) for out in outputs])


class LlamaV2Finetune:
    """
    Class to generate auto insights for text based on LLMs
    """
    def __init__(self, config):
        self.class_name = "LlamaV2Finetune"
        self.config = config
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name
    
    def finetune_model(self):
        train(self.config)

    def generate_outputs(self, dataset_name, savedir, max_len_ctx: int = 1600, small_dataset: bool = False, max_len_cite: int = 250, batch_size: int = 2):
        preds = []
        dataset = get_hf_dataset(dataset_name, small_dataset)
        




def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="meta-llama/Llama-2-7b-hf",
        help="Model name",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        default="/mnt/home/results/llama2/",
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


def print_args_values(combined_args):
    print("\n---------------------------------\n")
    print("All Values of Combined Arguments:")
    for attr, value in combined_args.__dict__.items():
        print(f"{attr}: {value}")
    print("\n---------------------------------\n")

if __name__ == "__main__":
    parsed_args = parse_args()
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    combined_args = CombinedArguments(script_args, parsed_args)
    print_args_values(combined_args=combined_args)
    # print(combined_args)
    # training_arguments = get_training_arguments(script_args)
    # print(training_arguments)
    # create_and_prepare_model(combined_args)
    # dataset_name = get_base_name_from_hf_path(config.dataset_name)
    # model_name = get_base_name_from_hf_path(config.model_name)
    # save_dir = f"{config.savedir}/{dataset_name}/{model_name}_finetuning_lr_5e-6"
    llama_trainer = LlamaV2Finetune(parsed_args)
    llama_trainer.finetune_model()