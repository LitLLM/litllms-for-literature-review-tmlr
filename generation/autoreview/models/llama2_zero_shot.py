#!/usr/bin/env python3
"""
This file defines class to run Llama in inference mode
"""

import torch
import transformers
from transformers import AutoTokenizer
from typing import Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

class Llamav2Pipeline:
    """
    Class to generate auto insights for text based on LLMs
    """
    # Also see
    # # https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19
    # https://huggingface.co/blog/llama2

    def __init__(self, config):
        self.class_name = "Llamav2"
        self.config = config
        self.tokenizer, self.pipeline = self.get_tokenizer_pipeline(self.config.model_name, load_in_4bit=self.config.load_in_4bit)

    def get_tokenizer_pipeline(self, model, model_max_length: int = 3500, truncation: bool = True, load_in_4bit: bool = False):
        # See https://github.com/huggingface/transformers/issues/4501
        tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=model_max_length, truncation=truncation)

        # If quanitzed, we will load model separately from pipeline, otherwise provide the "model name" directly
        # https://github.com/facebookresearch/llama/issues/425#issuecomment-1652217175
        # https://github.com/facebookresearch/llama/issues/540 
        # https://huggingface.co/TheBloke/Llama-2-70B-chat-GPTQ/discussions/2
        # https://www.reddit.com/r/LocalLLaMA/comments/13mxq66/13b_4bit_or_7b_8bits/

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                            bnb_4bit_compute_dtype=torch.float16, 
                                            bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, 
                                                         device_map="auto", trust_remote_code=True)
            
            print(f"Loading {model} in 4 bit")

            # model = AutoModelForCausalLM.from_pretrained(model, device_map='auto', load_in_4bit=load_in_4bit)

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, pipeline

    def run_pipeline(self, input, temperature=0.6, max_new_tokens=500): #, tokenizer, pipeline):
        # https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
        sequences = self.pipeline(
            input,
            do_sample=True,
            return_full_text=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature)
        # Why use high temperature: 
        # https://discuss.huggingface.co/t/llama-2-generation-config-top-p-0-6/49916

        # This is a list - return 0 element
        # For batch, might have to do [index][0] to avoid TypeError: list indices
        return sequences[0]['generated_text']