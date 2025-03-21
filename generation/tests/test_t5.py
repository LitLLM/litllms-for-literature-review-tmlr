# Run as python -m tests.test_t5
# PYTHONPATH=. python tests/test_t5.py
# https://huggingface.co/docs/transformers/model_doc/longt5
# https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary#how-to-in-python
import os
os.environ["HF_HOME"] = "/mnt/home/cached/"
os.environ["TORCH_HOME"] = "/mnt/home/cached/"

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

model_name = "google/long-t5-tglobal-base"
# model_name = "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
dataset = load_dataset("scientific_papers", "pubmed", split="validation")
model = (
    LongT5ForConditionalGeneration.from_pretrained(model_name)
    .to("cuda")
    .half()
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_answers(batch):
    inputs_dict = tokenizer(
        batch["article"], max_length=16384, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=2)
    batch["predicted_abstract"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return batch


result = dataset.map(generate_answers, batched=True, batch_size=2)
rouge = evaluate.load("rouge")
rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"])