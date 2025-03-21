import argparse


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model name",
    )
    # can be "meta-llama/Llama-2-7b-hf", gpt-3.5-turbo", "gpt-4"
    parser.add_argument(
        "-mo",
        "--mode",
        default="chatgpt",
        help="ChatGPT or Llama v2",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default="/results/auto_review/",
        help="Path to save dir",
    )
    parser.add_argument(
        "-p",
        "--prompt_type",
        default="research_template",
        help="Type of prompt to choose from all the prompt templates",
    )
    parser.add_argument(
        "-t",
        "--max_tries",
        default=5,
        help="Number of retries we want the ChatGPT to provide the correct JSON output",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="multi_x_science_sum",
        help="Dataset name",
    )
    parser.add_argument(
        "--small_dataset",
        default=False,
        help="Filter dataset",
    )
    parser.add_argument(
        "--use_langchain",
        default=True,
        help="Use langchain",
    )
    parser.add_argument(
        "--small_dataset_num",
        default=2,
        help="Number of examples to filter",
    )
    parser.add_argument(
        "-z",
        "--zero_shot",
        default=True,
        help="Use in zero shot setting",
    )
    parser.add_argument(
        "--use_llama_chat",
        default=False,
        help="Llama chat prompt",
    )
    parser.add_argument(
        "--load_in_4bit",
        default=False,
        help="Quantized load in 4 bit",
    )

    args = parser.parse_args()
    return args
