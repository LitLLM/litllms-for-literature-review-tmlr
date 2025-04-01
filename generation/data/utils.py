import os
import psutil
import argparse
import linecache
from datasets import load_dataset
import pandas as pd
import re

def get_memory_usage():
    # Get the current process ID
    pid = os.getpid()
    # Get the memory usage information for the current process
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    # Convert memory usage from bytes to gigabytes
    memory_gb = memory_info.rss / (1024 ** 3)  # 1 GB = 1024^3 bytes
    # Print the total memory used in GB with two decimal places
    print(f"Total memory used by the Python process: {memory_gb:.2f} GB")

def process_ids_s2_paper_dataset(sample, aid_prefix = "ARXIV:", mid_prefix = "MAG:"):
    aid = sample["aid"]
    mid = sample["mid"]
    references = sample["ref_abstract"]  # abstract
    ref_mids = references["mid"]
    # Filter and add MAG prefix for s2
    filtered_ref_mids = list(map(lambda item: f"{mid_prefix}item", filter(lambda item: item != "", ref_mids)))
    # filtered_list = list(filter(lambda item: item != "", ref_mids))
    sample["s2_arxiv"] = f"{aid_prefix}{aid}"
    sample["s2_mid"] = f"{mid_prefix}{mid}"
    sample["s2_cite_mids"] = create_line_cite(filtered_ref_mids)
    return sample

def create_line_cite(cite_list):
    ref_str = ",".join(cite_list)
    return ref_str

def get_s2orc_dir(config, folder_name: str = "s2orc"):
    s2orc_dir = f"{config.data_dir}/{folder_name}"
    return s2orc_dir

def replace_cite_text_with_mxs_cite(para_text, original_cite_text,  mxs_cite, trailing_space: bool = True, remove_extra_space: bool = True):
    if trailing_space:
        mxs_cite = f"{mxs_cite} "
    new_para_text = para_text.replace(original_cite_text, mxs_cite)
    # Adding trailing space can add extra spaces. Remove them
    if remove_extra_space:
        new_para_text = re.sub(' +', ' ', new_para_text)
    return new_para_text


def get_hf_dataset(dataset_name, split: str="test", small_dataset: bool = False, redownload: bool = False):
    if redownload:
        dataset = load_dataset(dataset_name, split=split, download_mode='force_redownload')
    else:
        dataset = load_dataset(dataset_name, split=split)
    if small_dataset:
        dataset = dataset.select(list(range(3)))
    # set_caching_enabled(False)
    return dataset

def load_hf_dataset(processed_jsonl_path):
    dataset = load_dataset('json', data_files=processed_jsonl_path)
    print(dataset[0])
    desired_id = "2964227312"
    filtered_data = dataset.filter(lambda example: example['mid'] == desired_id)
    print("Filtered data:",filtered_data)



def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--savedir",
        default="results/auto_review/dataset/new_dataset/",
        help="Save directory path",
    )
    parser.add_argument(
        "-hf",
        "--hf_savedir",
        default="results/auto_review/dataset/new_dataset/hf_datasets",
        help="Save directory path",
    )
    parser.add_argument(
        "-da",
        "--data_dir",
        default="results/auto_review/dataset/full_dataset/",
        help="Directory path where s2orc is saved",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="multi_x_science_sum",
        help="Dataset name",
    )
    parser.add_argument(
        "-do",
        "--filter_s2orc",
        default=True,
        help="Dataset name",
    )
    # SA: TODO change filter_s2orc
    parser.add_argument(
        "-mx",
        "--multi_x_science_text",
        default=False,
        help="Dataset name",
    )

    args = parser.parse_args()
    return args


def load_jsonl_as_hf(jsonl_gz_file, streaming: bool = False):
    from datasets import load_dataset
    data_files = {'train': jsonl_gz_file}
    dataset = load_dataset('json', data_files=data_files, split='train', streaming=streaming)
    print(next(iter(dataset)))


def read_specific_line(filepath, index):
    """
    https://stackoverflow.com/a/2081861
    """
    # Note: this is 0-index
    particular_line = linecache.getline(filepath, index) 
    return particular_line




def append_hf_data_text(sample, df, get_value_filtered_rows):
    """
    # text_obj = {"aid", "mid", "text_except_rw", "all_para_text", "title", "abstract"}
    """
    result = []
    # print(sample)
    aid = sample["aid"]
    mid = sample["mid"]
    references = sample["ref_abstract"]  # abstract
    ref_mids = references["mid"]

    # filtered_rows_aid = df.loc[df['aid'] == aid]
    filtered_rows_aid = df.filter[lambda row: row['aid'] == aid]
    # filtered_rows_mid = df.loc[df['mid'] == mid]
    filtered_rows_mid = df.filter[lambda row: row['mid'] == mid]

    if not filtered_rows_aid.empty:
        text_except_rw, title, total_words, found = get_value_filtered_rows(filtered_rows_aid)
    elif not filtered_rows_mid.empty:
        text_except_rw, title, total_words, found = get_value_filtered_rows(filtered_rows_mid)
    else:
        title = ""
        text_except_rw = ""
        found = 0
        total_words = 0

    sample["text_except_rw"] = text_except_rw
    sample["title"] = title
    sample["found"] = found
    sample["total_words"] = total_words
    return pd.Series([title, text_except_rw, found, total_words])
