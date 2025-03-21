# Run as python -m tests.test_galactica
# PYTHONPATH=. python tests/test_galactica.py
# https://huggingface.co/facebook/galactica-1.3b
# https://huggingface.co/theblackcat102/galactica-1.3b-v2
import torch
from autoreview.models.toolkit_utils import set_env_variables
set_env_variables(do_hf_login=False)
from autoreview.models.ml_utils import get_gpu_memory, Dict2Class
from autoreview.models.data_utils import create_model_input
from autoreview.models.pipeline import MDSGen
from datasets import load_dataset, set_caching_enabled
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, OPTForCausalLM, T5Tokenizer, 
                          T5ForConditionalGeneration, AutoModelForSeq2SeqLM)
import transformers
import os
import pandas as pd
from typing import Any, List
import argparse
from functools import partial
from tqdm import tqdm


class HFPipeline:
    """
    Class to generate auto insights for text based on LLMs
    """
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        self.tokenizer, self.pipeline = self.get_tokenizer_pipeline(self.config.model_name, return_obj="pipeline")
        # self.tokenizer, self.model = self.get_tokenizer_pipeline(self.config.model_name, return_obj="model")

    def get_tokenizer_pipeline(self, model, model_max_length: int = 2048, truncation: bool = True, load_in_4bit: bool = False,
                               return_obj: str = "model", model_type: str="t5"):

        # https://github.com/paperswithcode/galai/issues/39
        tokenizer = AutoTokenizer.from_pretrained(model, truncation=truncation)
        if model_type == "galactica":
            model = OPTForCausalLM.from_pretrained(model, device_map="auto")
            tokenizer.pad_token_id = 1
            tokenizer.padding_side = 'left'
            tokenizer.model_max_length = model_max_length
            # model = OPTForCausalLM.from_pretrained("facebook/galactica-30b")
            # model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto", torch_dtype=torch.float16)
            # model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto", load_in_8bit=True)
        elif model_type == "t5":
            # model = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="auto")
            model = T5ForConditionalGeneration.from_pretrained(model, device_map="auto")

        get_gpu_memory()
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                            bnb_4bit_compute_dtype=torch.float16, 
                                            bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, 
                                                         device_map="auto", trust_remote_code=True)            
            print(f"Loading {model} in 4 bit")

        if return_obj == "model":
            return tokenizer, model
        elif return_obj == "pipeline":
            pipeline = transformers.pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # "text-generation",
            return tokenizer, pipeline
        else:
            print("Specify one value!")

    def run_pipeline(self, input, temperature=0.6, max_new_tokens=500):
        # https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
        sequences = self.pipeline(
            input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature)
        print(sequences[0]['generated_text'])
            # eos_token_id=self.tokenizer.eos_token_id,
                    # return_full_text=False,

        return sequences[0]['generated_text']    

    def generate_answers(self, batch, field = "text", max_length=2048):
        input_ids = self.tokenizer(batch[field], padding='max_length', truncation = True, return_tensors="pt").input_ids.to("cuda")
        # inputs_dict = self.tokenizer(batch[field], padding='max_length', truncation = True, return_tensors="pt")
        # input_ids = inputs_dict.input_ids.to("cuda")
        # attention_mask = inputs_dict.attention_mask.to("cuda")
        # output_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=2)
        output_ids = self.model.generate(input_ids, max_new_tokens=500,
                                do_sample=True,
                                temperature=0.7,
                                top_k=25,
                                top_p=0.9,
                                no_repeat_ngram_size=10,
                                early_stopping=True)
        batch["preds"] = self.tokenizer.decode(output_ids[0]).lstrip('<pad>')
        # print(batch["preds"])
        # batch["preds"] = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].lstrip('<pad>')
        return batch






    def generate_outputs(self, dataset_name, savedir, batch_size: int = 10):        
        preds = []
        dataset = load_dataset(dataset_name, split="test")
        set_caching_enabled(False)
        # prompts "TLDR" and "Summarize the text above"
        # Summarize the text above for related work
        # model.generate(TEXT + "\n\nTLDR:", max_length=400)
        if self.config.model_type == "galactica": 
            dataset = dataset.map(partial(create_model_input, suffix="\n\nSummarize the text above:"))
            # print(dataset["text"][:3])
            # TODO: SA
            dataset = dataset.map(self.generate_answers) #, batched=True, batch_size=batch_size)
            preds = dataset["preds"]
        elif self.config.model_type == "t5": 
            dataset = dataset.map(partial(create_model_input, prefix="Summarize the text:"))
            preds = []
            for row in tqdm(dataset):
                response = self.run_pipeline(row["text"])
                preds.append(response)
                print(response)            
        print(f"Total responses are {len(preds)}")
        refs = dataset["related_work"]
        config={"model_name": self.config.model_name}
        config = Dict2Class(config)
        score_pipeline = MDSGen(config, inference=False)
        dataset = pd.DataFrame(dataset)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        score_pipeline.calculate_metrics(preds, refs, dataset, savedir)

def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="google/flan-t5-base",
        help="Model name",
    )
    # facebook/galactica-1.3b
    # galactica-6.7b
    # galactica-30b
    # google/flan-t5-large
    # google/flan-t5-base
    parser.add_argument(
        "-t",
        "--model_type",
        default="t5",
        help="Model type",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default="results/auto_review/multixscience/flan_t5",
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
    model = HFPipeline(parsed_args)
    print("Generating summaries now")
    model.generate_outputs(parsed_args.dataset_name, parsed_args.savedir)    

# input_text = "The Transformer architecture [START_REF]"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))

# input_text = '# Introduction \n\n The main idea of the paper "Supervised hashing for image retrieval via image representation learning" is'
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
# padding='max_length', 
# outputs = model.generate(input_ids, max_new_tokens=500,
#                          do_sample=True,
#                          temperature=0.7,
#                          top_k=25,
#                          top_p=0.9,
#                          no_repeat_ngram_size=10,
#                          early_stopping=True)
# print(tokenizer.decode(outputs[0]))
# with open('galactica_2000.md', 'w') as result_file:
#     result_file.write(tokenizer.decode(outputs[0]).lstrip('<pad>'))
