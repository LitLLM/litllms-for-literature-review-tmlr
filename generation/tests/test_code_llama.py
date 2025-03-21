# Run as python -m tests.test_code_llama
# PYTHONPATH=. python tests/test_code_llama.py

# https://huggingface.co/blog/codellama#how-to-use-code-llama
from autoreview.models.toolkit_utils import set_env_variables
set_env_variables(do_hf_login=True)
from transformers import AutoTokenizer
import transformers
import torch


tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'def fibonacci(',
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=100,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

