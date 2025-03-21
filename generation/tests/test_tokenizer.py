import argparse
from functools import partial
from datasets import load_dataset
from transformers import (
    AutoTokenizer)
from autoreview.models.llama2_finetune import preprocess_dataset, compute_length, create_prompt_formats

# def compute_length(example, col_name):
#     # https://huggingface.co/learn/nlp-course/chapter5/3?fw=pt#creating-new-columns
#     return {f"{col_name}_length": len(example[col_name].split())}

def get_pandas_percentile(df, colname):
    df[colname].describe(percentiles=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    return

def get_column_stats(df,column_name,to_dict = False):
	if to_dict:
		return df[column_name].value_counts().to_dict()
	else: 
		return df[column_name].value_counts()


def length_stats(dataset, col_name: str = "related_work"):
    dataset = dataset.map(partial(compute_length, col_name=col_name))
    # https://discuss.huggingface.co/t/copy-columns-in-a-dataset-and-compute-statistics-for-a-column/22157/11
    mean = dataset.with_format("pandas")[f"{col_name}_length"].mean()
    print(mean)
    return dataset


def main(args, split = "train", max_length=1000, seed: int=42):
    dataset = load_dataset(args.dataset_name, split=split)
    for col_name in ["related_work", "abstract", "text"]:
        dataset = length_stats(dataset=dataset, col_name=col_name)

    dataset = dataset.map(partial(create_prompt_formats, eval=True, truncate_context=True, max_len_context=1500))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=max_length, truncation=True)

    dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
    # for col_name in ["related_work", "abstract", "text"]:
    #     dataset = length_stats(dataset=dataset, col_name=col_name)


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="meta-llama/Llama-2-7b-hf",
        help="Model name",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        default="/mnt/home/results/llama2/",
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
    config = parse_args()
    main(config)
