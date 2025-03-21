import os
import re
import openai
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


from .vllm import infer_vllm

# Set openai credentials
openai.api_key = os.environ.get("OPENAI_API_KEY")


def extract_ids(papers_string):
    pattern = r"([a-f0-9]{40}):"
    matches = re.findall(pattern, papers_string)
    return matches


def parse_papers_reranking(combined_papers):
    # Check if the input is NaN
    if pd.isna(combined_papers):
        return np.nan
    pattern = r"ID: (.*?) - Title: (.*?)"
    matches = re.findall(pattern, combined_papers)
    return [f"{match[0].strip()} {match[1]}" for match in matches]


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        # print(matches)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def run_llm_api(
    json_data,
    gen_engine="gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> str:
    """
    This function actually calls the OpenAI API
    Models such as 'gpt-4o-mini' and 'gpt-4o-mini' are available
    :param json_data:
    :return:
    """
    if "gpt" in gen_engine:
        return run_openai_api(json_data, gen_engine, max_tokens, temperature)
    elif "llama" in gen_engine:
        return infer_vllm(
            prompt=json_data["prompt"],
            max_tokens=max_tokens,
            end_point=gen_engine,
            temperature=temperature,
        )


def run_openai_api(
    json_data,
    gen_engine="gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> str:
    """
    This function actually calls the OpenAI API
    Models such as 'gpt-4o-mini' and 'gpt-4o-mini' are available
    :param json_data:
    :return:
    """
    openai_client = openai.OpenAI()
    completion = openai_client.chat.completions.create(
        model=gen_engine,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": f"{json_data['system_prompt']}"},
            {"role": "user", "content": f"{json_data['prompt']}"},
        ],
    )
    return completion.choices[0].message.content
