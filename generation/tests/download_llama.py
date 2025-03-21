# Run as python -m tests.download_llama
# PYTHONPATH=. python tests/download_llama.py

from transformers import AutoTokenizer
from autoreview.models.toolkit_utils import set_env_variables
set_env_variables(do_hf_login=True)
from datasets import load_dataset, set_caching_enabled
import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def download_llama():
    dataset = load_dataset("multi_x_science_sum", split="test")
    model = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=3500, truncation=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    system_prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?'
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]


# download_llama()

def download_gptq():
    model_name_or_path = "TheBloke/Llama-2-70B-GPTQ"
    # To use a different branch, change revision
    # For example: revision="gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    prompt = "Tell me about AI"
    prompt_template=f'''{prompt}
    '''
    print("\n\n*** Generate:")
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    print(tokenizer.decode(output[0]))
    # Inference can also be done using transformers' pipeline
    print("*** Pipeline:")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    print(pipe(prompt_template)[0]['generated_text'])

# download_gptq()


def togetherai_llama():
    
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", device_map="auto", torch_dtype=torch.float16)
    # trust_remote_code=True,

    input_context = "Your text here"
    input_ids = tokenizer.encode(input_context, return_tensors="pt").to('cuda')
    output = model.generate(input_ids, max_length=128, temperature=0.7)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)

togetherai_llama()


# def parse_args() -> argparse.Namespace:
#     """
#     Argument parser function
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-m",
#         "--model_name",
#         default="meta-llama/Llama-2-7b-chat-hf",
#         help="Model name",
#     )
#     parser.add_argument(
#         "-s",
#         "--save_dir",
#         default="/datadrive2/shubham/outputs",
#         help="Path to save dir",
#     )
#     parser.add_argument(
#         "-d",
#         "--dataset_name",
#         default="multi_x_science_sum",
#         help="Dataset name",
#     )

#     args = parser.parse_args()
#     return args


# if __name__ == "__main__":
#     config = parse_args()
#     # llama_trainer = LlamaV2Finetune(config)
#     # llama_trainer.llama_finetune(save_dir=config.save_dir)