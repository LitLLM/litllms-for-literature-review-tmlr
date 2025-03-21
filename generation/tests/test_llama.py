# https://huggingface.co/blog/llama2
import os

from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login
import json
from typing import Any, List
import pathlib

def load_json(path: str):
    """
    This function opens and JSON file path
    and loads in the JSON file.

    :param path: Path to JSON file
    :type path: str
    :return: the loaded JSON file
    :rtype: dict
    """
    with open(path, "r",  encoding="utf-8") as file:
        json_object = json.load(file)
    return json_object


parent_dir = pathlib.Path(__file__).parent.parent.resolve()
file_path = f"{parent_dir}/autoreview/models/resources/hf_token.json"
with open(file_path, "r",  encoding="utf-8") as file:
    config = json.load(file)    
access_token_read = config["HF_TOKEN"]

login(token=access_token_read)


seed = 42
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)


model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=3500, truncation=True)
# {'model_max_length': 3500}
# truncation=True, max_length=512
# tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_len=512)
# model_max_length / max_len

# https://stackoverflow.com/questions/67849833/how-to-truncate-input-in-the-huggingface-pipeline
# tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512,'return_tensors':'pt'}

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)


system_prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?'
texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
# result = pipe(f"<s>[INST] {prompt} [/INST]")

# Solved - max_new_tokens is always overshadow by max_length https://github.com/huggingface/transformers/issues/13983
# max_new_tokens https://github.com/huggingface/transformers/issues/21369
# https://github.com/huggingface/transformers/issues/4501
# tokenizer=('distilbert-base-uncased', {'model_max_length': 128}) /  truncation=True
# LlamaTokenizer has no pad token https://github.com/huggingface/transformers/issues/22312
# Llama-2-hf non stopping token generation https://github.com/huggingface/transformers/issues/24994

# For generic class, see 
# https://github.com/liltom-eth/llama2-webui/blob/main/llama2_wrapper/model.py#L166
# https://github.com/facebookresearch/llama/blob/main/llama/generation.py
sequences = pipeline(
    texts,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False,
    max_new_tokens=300,
    temperature= 0.9
)

# truncation=True    
print(sequences)
for seq in sequences:
    print(f"Result: {seq[0]['generated_text']}")



# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map=device_map, torch_dtype=torch.bfloat16)
# text = "..."
# inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
