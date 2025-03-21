# https://discuss.huggingface.co/t/progress-bar-for-hf-pipelines/20498/8
# https://saturncloud.io/blog/displaying-summarization-progress-percentage-with-hugging-face-transformers/
# https://huggingface.co/docs/accelerate/usage_guides/big_modeling

# Run as python -m tests.test_hf_pipeline
# PYTHONPATH=. python tests/test_hf_pipeline.py

from transformers import AutoModelForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from transformers import pipeline
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
import os
import torch
from transformers.pipelines.base import KeyDataset

os.environ["HF_HOME"] = "/mnt/home/cached/"
os.environ["TORCH_HOME"] = "/mnt/home/cached/"

class ListDataset(Dataset):
     def __init__(self, original_list):
        self.original_list = original_list
     def __len__(self):
        return len(self.original_list)

     def __getitem__(self, i):
        return self.original_list[i]

def v1():
    # summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    # documents = ["Document 1 text...", "Document 2 text...", "..."]
    # https://discuss.huggingface.co/t/progress-bar-for-hf-pipelines/20498
    summarizer = pipeline('summarization')

    documents = ["An apple a day, keeps the doctor away said someone very smart long time ago but people didn't listen", 
            "When one buys a lot of real estate he tends to splurge and not worry about the costs which is dangerous",
            "An apple a day, keeps the lawer away said someone very smart long time ago but people didn't listen", 
            "When one buys a lot of real estate she tends to splurge and not worry about the long-term costs which is dangerous"]
    summarizer(documents, min_length=5, max_length=15)

    progress_bar = tqdm(total=len(documents), desc="Summarizing", ncols=100)

    summaries = []
    for doc in documents:
        summary = summarizer(doc, max_length=130, min_length=30, do_sample=False)[0]
        summaries.append(summary['summary_text'])
        progress_bar.update(1)

    progress_bar.close()


    for i, summary in enumerate(summaries):
        print(f"Document {i+1} summary: {summary}")


def fix_generation_pipeline():
    generator = pipeline('text-generation', model='gpt2')
    sample = generator('test test', pad_token_id=generator.tokenizer.eos_token_id)

    # https://github.com/huggingface/transformers/issues/19853#issuecomment-1290759818
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

def main():
    # NOTE: SA: TODO This works! 
    # https://discuss.huggingface.co/t/batch-inference-using-open-source-llms/42570
    # https://colab.research.google.com/drive/1_h8NvcIpoYvcOVOOyD25tkdkDix-0TIq?usp=sharing
    
    output_key = "summary_text"
    pipeline_type = "summarization"
    results = []
    dataset = load_dataset("multi_x_science_sum", split="test").select(list(range(20)))
    generator = pipeline(pipeline_type, max_length=100, batch_size=4)
    # generator = pipeline('text-generation', max_length=512,  model=model, tokenizer=tokenizer)
    for outputs in tqdm(generator(KeyDataset(dataset, "abstract"))):
        results.extend([out[output_key] for out in outputs])
        # print([out["summary_text"] for out in outputs])
    print(results)
    print(f"Total list of length: {len(results)}")


def using_for_loop(model):
    # https://huggingface.co/docs/transformers/generation_strategies
    # Something similar to: 
    generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id)
    # generation_config = GenerationConfig.from_pretrained("/tmp", "translation_generation_config.json")
    translation_generation_config = GenerationConfig(
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token=model.config.pad_token_id,
    )
    translation_generation_config.save_pretrained("/tmp", "translation_generation_config.json")

    results = []
    batch_size = 100
    dataset = load_dataset("multi_x_science_sum", split="test").select(list(range(20)))
    for chunk in tqdm(range(test_df.shape[0] // batch_size + 1)):
        descr = test_df[batch_size * chunk: (batch_size+1) * chunk]['description'].to_list()
        res = pipeline(descr)
        results += res



def llama_pipeline(model_name = "meta-llama/Llama-2-7b-chat-hf", model_max_length: int = 1500, truncation: bool = True, load_in_4bit: bool = True, batch_size: int=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length, truncation=truncation)
    tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation 
    tokenizer.padding_side = "right"
    tokenizer.pad_token = "<PAD>"
    generation_config = GenerationConfig.from_pretrained(model_name)
    print(generation_config)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.float16, 
                                    bnb_4bit_use_double_quant=True)
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


if __name__ == "__main__":
    # main()
    llama_pipeline()

