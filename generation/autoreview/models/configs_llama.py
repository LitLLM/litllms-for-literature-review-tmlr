# https://github.com/facebookresearch/llama-recipes/blob/main/configs/training.py
# Also see: https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da

# https://discuss.huggingface.co/t/trainingarguments-now-immutable-why/52565
# training_args = replace(training_args, **kwargs_to_set)


from dataclasses import dataclass, field
from typing import ClassVar, List, Optional


@dataclass
class train_config:
    # model_name: str="PATH/to/LLAMA/7B"
    # dataset = "samsum_dataset"    
     enable_fsdp: bool=False
     low_cpu_fsdp: bool=False
     run_validation: bool=True
     batch_size_training: int=4
     num_epochs: int=3
     num_workers_dataloader: int=1
     lr: float=1e-4
     weight_decay: float=0.0
     gamma: float= 0.85
     seed: int=42
     use_fp16: bool=False
     mixed_precision: bool=True
     val_batch_size: int=1
     micro_batch_size: int=4
     peft_method: str = "lora" # None , llama_adapter, prefix
     use_peft: bool=False
     output_dir: str = "PATH/to/save/PEFT/model"
     freeze_layers: bool = False
     num_freeze_layers: int = 1
     quantization: bool = False
     one_gpu: bool = False
     save_model: bool = True
     dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
     dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
     save_optimizer: bool=False # will be used if using FSDP
     use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels

     # per_device_train_batch_size=1,
     # gradient_accumulation_steps=4,
     # warmup_steps=2,
     # num_train_epochs=1,
     # max_grad_norm = 0.3,
     # learning_rate=1e-6,
     # fp16=True,
     # logging_steps=5,
     # lr_scheduler_type="cosine",
     # output_dir=output_dir,
     # optim="paged_adamw_32bit"



@dataclass
class lora_config:
     r: int=8
     lora_alpha: int=32
     target_modules: ClassVar[List[str]]= ["q_proj", "v_proj"]
     bias= "none"
     task_type: str= "CAUSAL_LM"
     lora_dropout: float=0.05
     inference_mode: bool = False

@dataclass
class llama_adapter_config:
     adapter_len: int= 10
     adapter_layers: int= 30
     task_type: str= "CAUSAL_LM"

@dataclass
class prefix_config:
     num_virtual_tokens: int=30
     task_type: str= "CAUSAL_LM"    


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048


# Taken from https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
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
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
