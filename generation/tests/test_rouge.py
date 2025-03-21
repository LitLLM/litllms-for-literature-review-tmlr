# https://huggingface.co/spaces/evaluate-metric/rouge
import evaluate
from datasets import load_metric


# scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
# https://github.com/allenai/PRIMER/blob/main/utils/compute_scores.py#L73
# scores = scorer.score('The quick brown fox jumps over the lazy dog', 'The quick brown dog jumps on the log.')


rouge = evaluate.load('rouge')
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = rouge.compute(predictions=predictions,
                         references=references)
print(results)

hf_metric = load_metric("rouge")
score = hf_metric.compute(predictions=predictions,
                            references=references, 
                            use_aggregator=False,
                            use_stemmer=True)
print(f"HF datasets scores: {score}")

print("\n-------------------------------------------------\n")

rouge = evaluate.load('rouge')
predictions = ["hello goodbye", "ankh morpork"]
references = ["goodbye", "general kenobi"]
results = rouge.compute(predictions=predictions,
                         references=references,
                         use_aggregator=False)
# print(list(results.keys()))
# ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
print(results)
# print(results["rouge1"])
# [0.5, 0.0]
score = hf_metric.compute(predictions=predictions,
                            references=references, 
                            use_aggregator=False,
                            use_stemmer=True)
print(f"\nHF datasets scores: {score}")

print("\n-------------------------------------------------\n")

rouge = evaluate.load('rouge')
predictions = ["hello goodbye", "ankh morpork"]
references = ["goodbye", "general kenobi"]
results = rouge.compute(predictions=predictions,
                         references=references,
                         use_aggregator=True)
# print(list(results.keys()))
# ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
print(results)
# print(results["rouge1"])
# 0.25
score = hf_metric.compute(predictions=predictions,
                            references=references, 
                            use_aggregator=True,
                            use_stemmer=True)
print(f"\nHF datasets scores using aggregator: {score}")

