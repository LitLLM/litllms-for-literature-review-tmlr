# Run as python -m tests.hf_zero_shot
# PYTHONPATH=. python tests/hf_zero_shot.py
import torch
from autoreview.models.toolkit_utils import set_env_variables
set_env_variables(do_hf_login=False)
from autoreview.models.ml_utils import get_gpu_memory, Dict2Class
from autoreview.models.data_utils import create_model_input
from autoreview.models.pipeline import MDSGen
from datasets import load_dataset, set_caching_enabled
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, OPTForCausalLM, T5Tokenizer, 
                          T5ForConditionalGeneration, AutoModelForSeq2SeqLM, LongT5ForConditionalGeneration, GenerationConfig)
from transformers.pipelines.base import KeyDataset
import transformers
import os
import pandas as pd
from typing import Any, List
import argparse
from functools import partial
from tqdm import tqdm
import re


class HFPipeline:
    """
    Class to generate auto insights for text based on LLMs
    """
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        self.model_type = self.config.model_type
        # self.tokenizer, self.pipeline = self.get_tokenizer_pipeline(self.config.model_name, return_obj="pipeline")
        self.tokenizer, self.model, self.pipeline = self.get_tokenizer_pipeline(self.config.model_name, return_obj="model")

    def get_tokenizer_pipeline(self, model, model_max_length: int = 2048, truncation: bool = True, load_in_4bit: bool = False,
                               return_obj: str = "model", batch_size: int = 6):
        print(f"Loading the type {self.model_type} from {model}")
        # https://github.com/paperswithcode/galai/issues/39
        tokenizer = AutoTokenizer.from_pretrained(model, truncation=truncation)
        if self.model_type == "galactica":
            model = OPTForCausalLM.from_pretrained(model, device_map="auto")
            tokenizer.pad_token_id = 1
            tokenizer.padding_side = 'left'
            tokenizer.model_max_length = model_max_length
        elif self.model_type == "t5":
            # model = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="auto")
            # model = T5ForConditionalGeneration.from_pretrained(model, device_map="auto")
            model = (
                LongT5ForConditionalGeneration.from_pretrained(model, device_map="auto")
                .half()
            )
        elif self.model_type == "flan-t5":
            # load_in_8bit=True
            model = T5ForConditionalGeneration.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)
        elif self.model_type == "starcoder":
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left" # padding side = left for a decoder-only architecture
            model_kwargs = {"device_map": "auto", "load_in_8bit": True}
            model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            # model = AutoModelForCausalLM.from_pretrained(model, force_download=True, resume_download=False, **model_kwargs)

            # model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        elif load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                            bnb_4bit_compute_dtype=torch.float16, 
                                            bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, 
                                                         device_map="auto", trust_remote_code=True)            
            print(f"Loading {model} in 4 bit")

        get_gpu_memory()
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            batch_size=batch_size,
            return_full_text=False,
            device_map="auto"
        )
        return tokenizer, model, pipeline


    def run_pipeline(self, input, temperature=0.6, max_new_tokens=500):
        # https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
        sequences = self.pipeline(
            input,
            do_sample=True,
            top_k=10,
            return_full_text=False,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature)
        print(sequences[0]['generated_text'])
            # eos_token_id=self.tokenizer.eos_token_id,
                    # return_full_text=False,

        return sequences[0]['generated_text']    

    def generate_answers(self, batch, field = "text", max_length=2048):
        if self.model_type == "galactica":
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
        elif self.config.model_type in ["t5", "flan-t5"]: 
            # max_length=16384 for longt5
            inputs_dict = self.tokenizer(
                batch[field], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids = inputs_dict.input_ids.to("cuda")
            attention_mask = inputs_dict.attention_mask.to("cuda")
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=512, temperature=0.7, top_k=25, top_p=0.9, num_beams=2)
            batch["preds"] = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        elif self.config.model_type in ["starcoder"]: 
            inputs_dict = self.tokenizer(
                batch[field], truncation=True, return_tensors="pt"
            )
            output_ids = self.model.generate(**inputs_dict, max_new_tokens=500, temperature=0.7, top_k=25, top_p=0.9, num_beams=2)
            batch["preds"] = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)            

        # print(batch[field])
        print(batch["preds"])

        return batch


    # helper function to postprocess text
    def postprocess_text(self, preds):
        # https://www.philschmid.de/fine-tune-flan-t5
        preds = [pred.strip() for pred in preds]
        # labels = [label.strip() for label in labels]
        # # rougeLSum expects newline after each sentence
        # preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(sent_tokenize(label)) for label in labels]
        return preds

    def postprocess_code_output(self, input_string):
        # Remove extra spaces
        output_string = re.sub(r'\s+', ' ', input_string)
        # Remove extra newline characters
        output_string = re.sub(r'\n+', '\n', output_string)
        # Remove leading and trailing whitespace
        output_string = output_string.strip()
        return output_string


    def generate_outputs(self, dataset_name, savedir, batch_size: int = 2):        
        preds = []
        dataset = load_dataset(dataset_name, split="test")
        set_caching_enabled(False)
        if self.config.model_type == "galactica": 
            # prompts "TLDR" and "Summarize the text above"
            # Summarize the text above for related work
            # model.generate(TEXT + "\n\nTLDR:", max_length=400)
            dataset = dataset.map(partial(create_model_input, suffix="\n\nSummarize the text above"))
            # print(dataset["text"][:3])
            # TODO: SA
            dataset = dataset.map(self.generate_answers)
            preds = dataset["preds"]
        elif self.config.model_type in ["t5", "flan-t5"]: 
            # dataset = dataset.map(partial(create_model_input, prefix="Summarize the text:"))
            dataset = dataset.map(partial(create_model_input, prefix="Summarize the text in 200 words and cite sources"))
            dataset = dataset.map(self.generate_answers, batched=True, batch_size=batch_size)
            preds = dataset["preds"]
        elif self.config.model_type in ["starcoder"]:
            # https://github.com/bigcode-project/starcoder/issues/38
            # SA: TODO try this
            preds = []
            max_length = 512
            prefix = """Given the abstract and the relevant papers, provide a related work section for a research paper citing the sources correctly. 
            Provide the scientific related work section in max 200 words. Do not provide the output in bullet points. Do not provide references at the end. 
            Do not provide code. Provide only the text.\n\n"""
            dataset = dataset.map(partial(create_model_input, prefix=prefix, max_len_ctx=1000))
            # print(dataset["text"][:4])

            if self.config.model_name == "bigcode/starcoder":
            # https://github.com/bigcode-project/starcoder/issues/66                    
                generation_config = GenerationConfig.from_pretrained(self.config.model_name)
                generation_config = GenerationConfig.from_dict({**generation_config.to_dict(), 'max_new_tokens':max_length, "temperature":0.4, "repetition_penalty":1.2})
                for outputs in tqdm(self.pipeline(KeyDataset(dataset, "text"), return_full_text=False,
                                                generation_config=generation_config)):
                    preds.extend([self.postprocess_code_output(out["generated_text"]) for out in outputs])                    
            elif self.config.model_name == "bigcode/starcoderplus":
                for outputs in tqdm(self.pipeline(KeyDataset(dataset, "text"), return_full_text=False, max_new_tokens=max_length, 
                                                  do_sample=True,top_p=0.8, temperature=0.8, repetition_penalty=1.2)):
                    preds.extend([self.postprocess_code_output(out["generated_text"]) for out in outputs])
                    print(self.postprocess_code_output(outputs[0]["generated_text"]))

        # Common
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
        default="bigcode/starcoderplus",
        help="Model name",
    )
    # bigcode/starcoder
    # bigcode/starcoderplus
    # google/long-t5-tglobal-base
    # google/flan-t5-base
    # google/flan-t5-large
    # facebook/galactica-1.3b
    # galactica-6.7b
    # galactica-30b
    # google/flan-t5-large
    # google/flan-t5-base
    parser.add_argument(
        "-t",
        "--model_type",
        default="starcoder",
        help="Model type",
    )
    # t5
    # flan-t5
    # galactica
    parser.add_argument(
        "-s",
        "--savedir",
        default="/results/auto_review/multixscience/starcoderplus",
        help="Path to save dir",
    )
    # results/auto_review/multixscience/galactica
    # results/auto_review/multixscience/long_t5
    # results/auto_review/multixscience/flan_t5
    # results/auto_review/multixscience/starcoder
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