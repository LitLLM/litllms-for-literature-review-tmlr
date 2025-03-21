
# Run as python -m tests.test_together_ai
# or
# PYTHONPATH=. python tests/test_together_ai.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoreview.models.ml_utils import get_gpu_memory
import torch
import os
os.environ["HF_HOME"] = "/mnt/home/cached/"
os.environ["TORCH_HOME"] = "/mnt/home/cached/"


tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K",device_map="auto", trust_remote_code=False, torch_dtype=torch.float16)

get_gpu_memory()

input_context = "Who is Barack Obama?"
input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()
output = model.generate(input_ids, max_length=128, temperature=0.7)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)


get_gpu_memory()