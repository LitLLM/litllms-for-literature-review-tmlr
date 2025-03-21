from datasets import load_metric
from typing import Any
from rouge_score import rouge_scorer
import evaluate
import argparse
import pickle as pkl


class Metrics:
    """
    This class allows to calculate metrics
    """

    def __init__(self, config: Any):
        self.name = "metrics"
        self.config = config
        # HF dataset metric
        self.metric = load_metric(config["metric"])
        # HF eval metric
        self.rouge_eval_hf = evaluate.load(config["metric"])

    def load_metric(self, metric):
        """
        This loads HF datasets metrics
        """
        hf_metric = load_metric(metric)
        return hf_metric

    def compute_score(self, result, use_stemmer: bool = True, use_aggregator: bool = True, print_scores: bool = True):
        # This is HF datasets metric
        score = self.metric.compute(predictions=result["preds"],
                                    references=result["refs"], 
                                    use_aggregator=use_aggregator,
                                    use_stemmer=use_stemmer)
        if print_scores:
            print(score)
            print("\n-------------------------------------------------\n")
            print(f"HF dataset library metrics: ")
            print("Rouge 1: ", score['rouge1'][0])
            print("Rouge 2: ", score['rouge2'][0])
            print("Rouge L: ", score['rougeL'][0])
        rouge_scores_datasets_hf = {"rouge1": score['rouge1'][0], 
                        "rouge2": score['rouge2'][0],
                        "rougeL": score['rougeL'][0]}
        
        # Using HF evaluate library
        rouge_scores_eval_hf = self.rouge_eval_hf.compute(predictions=result["preds"], references=result["refs"], 
                                                          use_aggregator=use_aggregator, use_stemmer=use_stemmer)
        if print_scores:
            print("\n-------------------------------------------------\n")
            print(f"Use aggregator: {use_aggregator} and use stemmer: {use_stemmer}")
            print("Evaluate library metrics: ", rouge_scores_eval_hf)

        return score, rouge_scores_datasets_hf, rouge_scores_eval_hf
    

def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--result_path",
        default="/tmp",
        help="Path to results",
    )

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    parsed_args = parse_args()
    scorer = Metrics(config={"metric": "rouge"})
    results = pkl.load(open(parsed_args.result_path,'rb'))
    scorer.compute_score(results)
