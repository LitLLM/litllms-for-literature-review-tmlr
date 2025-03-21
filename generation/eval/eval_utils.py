import concurrent.futures
from tqdm import tqdm
import time
from huggingface_hub import InferenceClient
import json, os
import pathlib
from typing import Any
import random
from autoreview.models.data_utils import get_sentences_spacy, find_strings_with_word_positions

def concurrent_requests(func_name, texts, config):
    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        preds = []
        timings = []

        for text in texts:
            futures.append(executor.submit(func_name, text=text, service_url=config.service_url))        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            out, rt = future.result()
            timings.append(rt)
            preds.append(out)
    return preds

def find_avg_across_two_col(df, col1, col2, metric_col1, metric_col2, value):
    # Filter rows where either 'Model 1' or 'Model 2' is 'XYZ'
    filtered_df = df[(df[col1] == value) | (df[col2] == value)]
    # average_score = filtered_df[metric_col].mean()
    # Calculate the average scores for 'Model XYZ' by combining scores from both columns
    average_score = (filtered_df[metric_col1] + filtered_df[metric_col2]).mean()

    return average_score


def dump_jsonl(filepath, answers):
    with open(filepath, "w") as f:
        table = [json.dumps(ans) for ans in answers]
        f.write("\n".join(table))


def load_eval_prompts(file_path: str = None) -> str:
    """
    Loads the api key from json file path

    :param file_path:
    :return:
    """
    cur_dir = pathlib.Path(__file__).parent.resolve()
    # Load prompts from file
    if not file_path:
        # Default file path
        file_path = f"{cur_dir}/resources/eval_prompts.json"
    prompts = load_json(file_path)

    return prompts


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


def shuffle_and_select():

    # Assuming you have a list of JSON lines
    json_lines = [
        '{"id": 1, "data": "A"}',
        '{"id": 2, "data": "B"}',
        '{"id": 3, "data": "C"}',
        '{"id": 4, "data": "D"}',
        '{"id": 5, "data": "E"}'
    ]

    # Generate a shuffled list of indices
    indices = list(range(len(json_lines)))
    random.shuffle(indices)

    # Subset a specific number of lines (e.g., the first 3 lines) from the original json_lines
    subset_count = 3
    subset_indices = indices[:subset_count]

    # Filter JSON lines based on the shuffled indices for the subset
    shuffled_subset_json_lines = [json_lines[i] for i in subset_indices]

    # Now you have the JSON lines shuffled in 'shuffled_subset_json_lines'
    for line in shuffled_subset_json_lines:
        data = json.loads(line)
        print(data)


def select_data_columns(example, cols_to_keep):
    # Replace 'column1' and 'column2' with the names of the columns you want to select
    # dataset = dataset.map(partial(select_data_columns, cols_to_keep=["abstract", "related_work"]))
    return_dict = {}
    for colname in cols_to_keep:
        return_dict[f"{colname}"] = example[f"{colname}"]
    return return_dict

def get_total_words(example):
    example[f"abstract_length"] = len(example["abstract_length"].split())
    example[f"rw_length"] = len(example["rw_length"].split())
    return example

def is_length_within_range(example, min_length=100, max_length=700):
    abstract = example['abstract']
    related_work = example['related_work']
    return len(abstract) >= min_length and len(abstract) <= max_length and len(related_work) >= min_length and len(related_work) <= max_length

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

def read_model_outputs(savedir, out_name, dataset_name):
    filepath = f"{savedir}/{out_name}_{dataset_name}.jsonl"
    # response_dict = {}
    # with open(filepath) as f:
    #     for line in f:
    #         r = json.loads(line)
    #         response_dict["model_id"] = r["model_id"]
    #         response_dict["response"] = r["response"]
    result_list = get_json_list(filepath)
    return result_list

def append_to_jsonl(data, file_path, do_json_dump: bool = True, mode="a"):
    with open(file_path, 'a') as jsonl_file:
        if do_json_dump:
            json_string = json.dumps(data)
        else:
            json_string = data
            # TypeError: can't concat str to bytes. TODO find a better way
        jsonl_file.write(json_string + '\n')



def flatten_ref_papers(sample, nlp_spacy=None):
    references = sample["ref_abstract"]  # abstract
    citations = references["cite_N"]
    abstracts = references["abstract"]
    ref_text = ""
    num_words = []
    cite_list = []
    for index in range(len(citations)):
        # We only use references which have abstracts
        if abstracts[index] != "":
            individual_abstract = abstracts[index]
            ctx_length = len(individual_abstract.split())
            num_words.append(ctx_length)
            # ref_text = f"Reference {citations[index]}: {individual_abstract}\n" # Wrong
            ref_text = f"{ref_text}Reference {citations[index]}: {individual_abstract}\n\n"
            cite_list.append(citations[index])

    sample["cite_list"] = cite_list
    sample["num_cites"] = len(cite_list)
    sample["ref_text"] = ref_text
    sample["num_words"] = num_words
    sample["total_ref_words"] = sum(num_words)

    return sample

def find_cite_in_reference(cite, reference):
    """
    https://chat.openai.com/share/0538fe60-eb06-4206-bc72-5d97a4d34dcb 
    """
    found_strings = []
    for s in cite:
        if s in reference:
            found_strings.append(s)
    missing_cite = len(cite) - len(found_strings) 
    return missing_cite


def get_data():
    dataset = Dataset.from_dict(result_json)
    Dataset.from_pandas(pd.DataFrame(data=data))

def create_source_data(self, savedir, model_list, num_sample=25):
    preds = []
    human_jsons = self.read_model_outputs(savedir=savedir, out_name="ref_data")
    for model_name in model_list:
        response_jsons = self.read_model_outputs(savedir=savedir, out_name=model_name)
        list2_selected_fields = [{"id": obj["id"], "value": obj["value"]} for obj in list2]

        sampled_jsons = random.sample(response_jsons, num_sample)
        preds.append(sampled_jsons)
    
    sampled_jsons = random.sample(human_jsons, num_sample)
    for human_line in sampled_jsons:
        line = {"example_id": human_line["example_id"], "model_id": human_line["model_id"], "response": human_line["gt_related_work"]}
        preds.append(line)
    source_data_filepath = f"{savedir}/source_samples_{self.dataset_name}.jsonl"
    dump_jsonl(source_data_filepath, preds)
