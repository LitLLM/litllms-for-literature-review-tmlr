#!/usr/bin/env python3
# Copyright (c) ServiceNow Research and its affiliates.
"""
Defining utils for ML models here
"""

import json
from typing import Any, List
import pathlib
import pandas as pd
import torch
import datetime
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time
import re
from huggingface_hub import InferenceClient
# from text_generation import Client


def load_json(path: str) -> Any:
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


def write_to_file(path: str, input_text: str) -> None:
    """
    This function opens a file and writes the given input
    string onto the file.

    :param path: Path to the file
    :param input_text: The text to be written on the file
    """
    with open(path, "w", encoding="utf-8") as open_file:
        open_file.write(input_text)
    open_file.close()


def load_api_key(file_path: str = None) -> str:
    """
    Loads the api key from json file path

    :param file_path:
    :return:
    """
    cur_dir = pathlib.Path(__file__).parent.resolve()
    # Load config values
    if not file_path:
        # Default file path
        file_path = f"{cur_dir}/resources/config.json"
    config = load_json(file_path)
    api_key = config["OPENAI_API_KEY"]

    return api_key


def write_excel_df(
    df_list: List,
    sheet_name_list: List,
    writer: pd.ExcelWriter = None,
    close_writer: bool = False,
    save_file_path: str = None,
    append_mode: bool = False,
):
    """
    Save a list of df in different sheets in one excel file.
    Args:
        writer:
        df_list:
        sheet_name_list:
        close_writer:
        save_file_path:
        append_mode:

    https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without \
    -overwriting-data-using-pandas
    https://www.geeksforgeeks.org/how-to-write-pandas-dataframes-to-multiple-excel-sheets/


    Returns:
    """
    if save_file_path:
        if append_mode:
            writer = pd.ExcelWriter(save_file_path, mode="a", engine="xlsxwriter")
        else:
            writer = pd.ExcelWriter(save_file_path, engine="xlsxwriter")
    # Write each dataframe to a different worksheet
    assert len(df_list) == len(sheet_name_list)
    for index in range(len(df_list)):
        df_list[index].to_excel(writer, sheet_name=sheet_name_list[index])
    # Close the Pandas Excel writer and output the Excel file.
    if close_writer:
        writer.close()
    return


def load_all_prompts(file_path: str = None) -> str:
    """
    Loads the api key from json file path

    :param file_path:
    :return:
    """
    cur_dir = pathlib.Path(__file__).parent.resolve()
    # Load prompts from file
    if not file_path:
        # Default file path
        file_path = f"{cur_dir}/resources/prompts.json"
    prompts = load_json(file_path)

    return prompts

def get_gpu_memory(model = None):
    # https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    free_reserved = r-a  # free inside reserved
    f = t-a  
    print(f"---------------------------MERMORY USAGE--------------------------------------\n")
    now = datetime.datetime.now()
    print("Current date and time : ", now.strftime("%Y-%m-%d %H:%M:%S"))
    get_gpu_name()
    print(f"""Total memory: {t/1024.0/1024.0:.2f}MB, Reserved: {r/1024.0/1024.0:.2f}MB, 
          Allocated: {a/1024.0/1024.0:.2f}MB, Free: {f/1024.0/1024.0:.2f}MB""")
    print(f"------------------------------------------------------------------------------\n")
    if model:
        get_model_mem_footprint(model)


def get_model_mem_footprint(model):
    try: 
        print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    except:
        print("Not able to get the memory footprint of the model")


def get_gpu_name():
    # https://stackoverflow.com/a/48152675
    device_names = torch.cuda.get_device_name()
    device_counts = torch.cuda.device_count()
    print(f"Running {device_counts} GPUs of type {device_names}")

# Turns a dictionary into a class
class Dict2Class(object):      
    def __init__(self, my_dict):          
        for key in my_dict:
            setattr(self, key, my_dict[key])


def concurrent_requests(texts, config, func_name, max_workers:int =10, headers= False):
    print(f"Loading the client from {config.service_url}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        preds = []
        timings = []
        for text in texts:
            futures.append(executor.submit(func_name, text=text, service_url=config.service_url, headers=headers))        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            out, rt = future.result()
            timings.append(rt)
            out = postprocess_output(out)
            preds.append(out)
    return preds

def hf_tgi(text, service_url, headers: bool = False, EAI_TOKEN: str = ""):
    """
    Using this for results
    """
    # inference_args = {"prompt": "What is DL?", "max_new_tokens": 400, "temperature": 1.0}
    # print(client.text_generation(**inference_args))

    st = time.monotonic()
    if headers:
        client = InferenceClient(model=service_url,headers={"Authorization": f"Bearer {EAI_TOKEN}"})
    else:
        client = InferenceClient(model=service_url)

    try:
        results = client.text_generation(text, max_new_tokens=500, do_sample=True, stream=False, top_k=10, 
                                        seed=1337, temperature=0.6, repetition_penalty=1.2)
    except Exception as e:       
        results = ""    
        print(f"Got the exception as: {e}")
    dt = time.monotonic() - st
    return results, dt

def postprocess_output(input_string):
    # Remove extra spaces
    output_string = re.sub(r'\s+', ' ', input_string)
    # Remove extra newline characters
    output_string = re.sub(r'\n+', '\n', output_string)
    # Remove leading and trailing whitespace
    output_string = output_string.strip()
    return output_string
