import spacy
from functools import partial
from transformers import AutoTokenizer
import datasets
from datasets import set_caching_enabled, Dataset, load_dataset
import pandas as pd
import pickle as pkl
import jsonlines
import json
import re
import os
from autoreview.models.ml_utils import get_gpu_memory, load_all_prompts, Dict2Class
from autoreview.models.pipeline import MDSGen


def create_llama_chat_prompt(complete_prompt):
    # Llama chat expected prompt
    # NOTE: this is for chat model, have to verify for non chat models
    llama_prompt = f"[INST] <<SYS>>\n{complete_prompt}\n<</SYS>>\n\n[/INST]"
    # SA: TODO https://colab.research.google.com/drive/1mV9sAY4QBKLmS58dpFGHgwCXQKRASR31?usp=sharing#scrollTo=qf1qxbiF-x6p
    # prompt = f"[INST] <<SYS>>\n{complete_prompt}\n<</SYS>>\n\nWrite a function that reverses a string. [/INST]"
    return llama_prompt

def llama_first_query_no_sys(prompt):
    # https://huggingface.co/blog/codellama#conversational-instructions
    prompt = f"<s>[INST] {prompt.strip()} [/INST]"
    return prompt

def llama_first_query_with_sys(prompt, sys_msg):
    # https://huggingface.co/blog/codellama#conversational-instructions
    prompt = f"<s><<SYS>>\\n{sys_msg}\\n<</SYS>>\\n\\n{prompt}"
    return prompt


def flatten_ref_papers_from_citations(sample, col_name = "ref_abstract", one_line_baseline: bool = False, mapped: bool = False,
                                      nlp_spacy=None, truncate_cite: bool = True, max_len_cite: int = 250, print_debug: bool = False):
    references = sample["ref_abstract"]  # abstract
    citations = references["cite_N"]
    abstracts = references["abstract"]
    ref_text = ""
    num_words = []
    for index in range(len(citations)):
        # We only use references which have abstracts
        if abstracts[index] != "":
            if one_line_baseline:
                    one_line = select_n_lines(abstracts[index], nlp_spacy=nlp_spacy)
                    ref_text = f"{ref_text} {one_line}"
            else:
                individual_abstract = abstracts[index]
                ctx_length = len(individual_abstract.split())
                individual_abstract = truncate_single_paper_or_input(individual_abstract, max_len_cite=max_len_cite, truncate_cite=truncate_cite,
                                                                      print_debug=print_debug)
                # if truncate_cite and ctx_length > max_len_cite:
                #     print(f"Found single abstract of length {ctx_length}. Truncating to {max_len_cite}")
                #     individual_abstract = " ".join(individual_abstract.split()[:max_len_cite])
                num_words.append(min(ctx_length, max_len_cite))
                # ref_text = f"Reference {citations[index]}: {individual_abstract}\n" # Wrong
                ref_text = f"{ref_text}Reference {citations[index]}: {individual_abstract}\n"
                

    sample["ref_text"] = ref_text
    sample["num_words"] = num_words
    sample["total_ref_words"] = sum(num_words)

    if mapped:
        return sample
    return ref_text


def truncate_single_paper_or_input(input_data, max_len_cite: int = 250, truncate_cite: bool = True, single_paper: bool= True, print_debug: bool = False):
    ctx_length = len(input_data.split())
    if truncate_cite and ctx_length > max_len_cite:
        if print_debug:
            if single_paper:
                print(f"Found single abstract of length {ctx_length}. Truncating to {max_len_cite}")
            else:
                print(f"Found total ctx of length {ctx_length}. Truncating to {max_len_cite}. Not truncating base prompt or any plan if provided.")    
        input_data = " ".join(input_data.split()[:max_len_cite])
    # ctx_length = len(complete_input.split())
    # if truncate_text and ctx_length > max_len_ctx:
    #     print(f"Found total ctx of length {ctx_length}. Truncating to {max_len_ctx}. Not truncating base prompt or plan.")
    #     complete_input = " ".join(complete_input.split()[:max_len_ctx])

    return input_data


def create_generation_plan(sample, col_ref = "ref_abstract", col_gt_rw = "related_work", mapped: bool = False, nlp_spacy=None):
    gt_related_work = sample[col_gt_rw]
    gt_sentences = get_sentences_spacy(gt_related_work, nlp_spacy=nlp_spacy)
    references = sample[col_ref]  # refs
    citations = references["cite_N"]
    abstracts = references["abstract"]
    num_gt_lines = len(gt_sentences)
    num_gt_words = len(gt_related_work.split())
    plan = create_prefix_template(num_gt_lines, num_gt_words)

    citations = [cite for cite, abs in zip(citations, abstracts) if abs]

    for index, sent in enumerate(gt_sentences):
        # cite_list = find_strings_in_reference(citations, sent)
        cite_list, _ = find_strings_with_word_positions(citations, sent)
        # If empty, dont cite anything
        if cite_list:
            line_template = create_line_cite_template(cite_list, index+1)  # List are 0 index
            plan = f"{plan} {line_template}"
    if mapped:
        sample["plan"] = plan
        sample["num_gt_lines"] = num_gt_lines
        return sample
    return plan, num_gt_lines

def create_prefix_template(num_lines, num_words):
    plan_template_prefix = f"Please generate the related work in {num_lines} lines using max {num_words} words. "
    return plan_template_prefix

def create_line_cite_template(cite_list, line_num):
    cite_str = " ".join(cite_list)
    line_template = f"Please cite {cite_str} on line {line_num}."
    return line_template

def find_strings_in_reference(strings, reference):
    """
    https://chat.openai.com/share/0538fe60-eb06-4206-bc72-5d97a4d34dcb 
    """
    found_strings = []
    for s in strings:
        if s in reference:
            found_strings.append(s)
    return found_strings

def find_strings_with_word_positions(strings, reference):
    """
    https://chat.openai.com/share/0538fe60-eb06-4206-bc72-5d97a4d34dcb
    """
    found_strings = []
    words = reference.split()
    # Create a set of words for faster lookup
    word_set = set(words)
    found_strings = [s for s in strings if s in word_set]
    word_positions = [words.index(s) + 1 for s in found_strings]  # Adding 1 for word count

    return found_strings, word_positions



def create_model_input(sample, one_line_baseline: bool = False, plan_based_gen: bool = False, 
                       prefix: str = "", nlp_spacy=None, suffix: str = "",
                       truncate_text: bool = True, max_len_ctx: int = 1300, max_len_cite: int = 250, 
                       use_full_text = False, print_debug: bool = False):
    original_abstract = sample["abstract"]
    abstract_text = ""
    # print(f"One line baseline set to {one_line_baseline}")
    if one_line_baseline:
        # print(f"One line baseline set to {one_line_baseline}")
        one_line = select_n_lines(original_abstract, nlp_spacy=nlp_spacy)
        abstract_text = f"{one_line}"
    else: 
        abstract_text = f"Main Abstract: {original_abstract}\n"
    ref_text = flatten_ref_papers_from_citations(sample, one_line_baseline=one_line_baseline, nlp_spacy=nlp_spacy, max_len_cite=max_len_cite, print_debug=print_debug)
    complete_input = f"{abstract_text}\n{ref_text}"
    if prefix != "":
        complete_input = f"{prefix}: {complete_input}"
    
    if use_full_text:
        text_except_rw = sample["text_except_rw"]
        paper_text = f"Paper text: {text_except_rw}\n"
        complete_input = f"{complete_input}\n{paper_text}"

    complete_input = truncate_single_paper_or_input(complete_input, max_len_cite=max_len_ctx, truncate_cite=truncate_text, single_paper=False, print_debug=print_debug)
    # ctx_length = len(complete_input.split())
    # if truncate_text and ctx_length > max_len_ctx:
    #     print(f"Found total ctx of length {ctx_length}. Truncating to {max_len_ctx}. Not truncating base prompt or plan.")
    #     complete_input = " ".join(complete_input.split()[:max_len_ctx])

    if suffix != "":
        complete_input = f"{complete_input} {suffix}"

    if plan_based_gen:
        plan, num_gt_lines = create_generation_plan(sample, col_ref = "ref_abstract", col_gt_rw = "related_work", nlp_spacy=nlp_spacy)
        complete_input = f"{complete_input} \nPlan: {plan}"
        sample["num_gt_lines"] = num_gt_lines
        sample["plan"] = plan
    sample["text"] = complete_input
    gt_related_work = sample["related_work"]
    num_gt_words = len(gt_related_work.split())
    sample["num_gt_words"] = num_gt_words

    return sample

def get_complete_mapped_prompt(sample, base_prompt: str, data_column: str = "text", suffix: str = "Related work:\n", 
                               use_llama_chat: bool = False, data: str = "", mapped: bool = True, model_type: str = "", 
                               word_based: bool = False):
    """
    """
    if not data:
        data = sample[data_column]
    if word_based:
        num_gt_words = sample["num_gt_words"]
        base_prompt = base_prompt.format(num_gt_words=num_gt_words)        
    complete_prompt = f"{base_prompt}\n```{data}```"
    if use_llama_chat:
        # complete_prompt = f"<s>[INST] {complete_prompt.strip()} [/INST]"
        if model_type == "upstage":
            complete_prompt = f"### User:\n{complete_prompt.strip()}\n\n### Assistant:\n"
        elif model_type == "oa":
            system_message = "You are a helpful assitant"
            complete_prompt = f"""<|system|>{system_message}</s><|prompter|>{complete_prompt.strip()}</s><|assistant|>"""
        else:
            # complete_prompt = f"[INST]\{complete_prompt.strip()}\n[/INST]\n\n"
            complete_prompt = f"<s>[INST] {complete_prompt.strip()} [/INST]"

    if suffix:
        complete_prompt = f"{complete_prompt}\n{suffix}"

    if mapped:
        sample[data_column] = complete_prompt
        return sample
    else:
        return complete_prompt


def get_sentences_spacy(text, nlp_spacy=None):
    """
    Split sentences using spacy
    https://stackoverflow.com/questions/46290313/how-to-break-up-document-by-sentences-with-spacy
    """
    if not nlp_spacy:
        nlp_spacy = spacy.load('en_core_web_sm')
    else:
        sentence_list = list(nlp_spacy(text).sents)
        # nlp_text = [sent.text.strip() for sent in nlp_text.sents]
        sentence_list = [sent.text.strip() for sent in sentence_list]

    return sentence_list

def select_n_lines(text: str, num_lines: int = 1, nlp_spacy=None, delimiter: str = ".", trailing_dot: bool = True):
    """
    Selects first n lines from text
    """
    if nlp_spacy:
        split_lines = get_sentences_spacy(text, nlp_spacy=nlp_spacy)
    else:
        # TODO: SA: one line cant delimit by "."  e.g. / 1. 
        split_lines = text.split(delimiter)
        if trailing_dot:
            split_lines = [f"{single_line}." for single_line in split_lines]
    if num_lines == 1:
        return_lines = split_lines[0]
    else:
        return_lines = split_lines[:num_lines]
    return return_lines

def create_prompt_formats(sample, eval: bool = False, truncate_context: bool = True, max_len_context: int = 1100, use_llama_chat: bool = True):
    """
    SA: Format various fields of the sample ('instruction', 'context', 'response') in the original dataset
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    # features: ['aid', 'mid', 'abstract', 'related_work', 'ref_abstract']
    # references = row["ref_abstract"]  # ref abstract
    # citations = references["cite_N"]
    # abstracts = references["abstract"]
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INSTRUCTION_SPEC = f"""You will be provided with an abstract and other references papers as Input. 
    Given the abstract and the relevant papers, provide the Response a related work section for a research paper citing the sources correctly."""
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    references = sample["ref_abstract"]  # abstract
    citations = references["cite_N"]
    abstracts = references["abstract"]
    original_abstract = sample["abstract"]
    input_text = f"Main Abstract: {original_abstract}"
    for index in range(len(citations)):
        input_text = (
            f"{input_text} \n Reference {citations[index]}: {abstracts[index]}"
        )

    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{INSTRUCTION_SPEC}"
    input_context = f"{INPUT_KEY}\n{input_text}"
    response = f"{RESPONSE_KEY}\n{sample['related_work']}"
    end = f"{END_KEY}"
    response_prompt = f"{RESPONSE_KEY}\n"
    if truncate_context:
        ctx_length = len(input_context.split())
        if ctx_length > max_len_context:
            print(f"Found context of length {ctx_length}. Truncating to {max_len_context}")
            input_context = " ".join(input_context.split()[:max_len_context])
    if eval:
        parts = [part for part in [instruction, input_context, response_prompt] if part]
    else:
        parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)

    if use_llama_chat:
        formatted_prompt = create_llama_chat_prompt(formatted_prompt)
    
    sample["text"] = formatted_prompt

    return sample


def compute_length(example, col_name):
    # https://huggingface.co/learn/nlp-course/chapter5/3?fw=pt#creating-new-columns
    example[f"{col_name}_length"] = len(example[col_name].split())
    return example


def length_stats(dataset, col_name: str = "related_work"):
    dataset = dataset.map(partial(compute_length, col_name=col_name))
    # https://discuss.huggingface.co/t/copy-columns-in-a-dataset-and-compute-statistics-for-a-column/22157/11
    mean = dataset.with_format("pandas")[f"{col_name}_length"].mean()
    print(mean)


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True
    )

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str, shuffle: bool = True, eval: bool = False):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    original_columns=['aid', 'mid', 'abstract', 'related_work', 'ref_abstract']
    # SA: Can also do
    # original_columns = dataset.column_names
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats) #, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['aid', 'mid', 'abstract', 'related_work', 'ref_abstract', 'text']
    )

    print(f"Dataset size before filtering: {dataset.shape}")
    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)
    print(f"Dataset size after filtering: {dataset.shape}")
    if shuffle:
        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed)

    return dataset


def get_pandas_percentile(df, col_name_list):
    # https://gist.github.com/shubhamagarwal92/13e2c41d09156c3810740d7697a883d1
    # https://stackoverflow.com/questions/34026089/create-dataframe-from-another-dataframe-describe-pandas
    describe_df = df[col_name_list].describe(percentiles=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    return describe_df

def convert_pandas_to_hf(df_pandas):
    # https://stackoverflow.com/questions/71102654/huggingface-datasets-convert-a-dataset-to-pandas-and-then-convert-it-back
    dataset_from_pandas = Dataset.from_pandas(df_pandas)
    return dataset_from_pandas

def convert_hf_to_pandas(dataset):
    df = dataset.to_pandas()
    return df

def pkl_dump(pkl_file_path, object_list):
    with open(pkl_file_path, 'wb') as fp:
        pkl.dump(object_list, fp)
    print(f"Saving the object at {pkl_file_path}")


def read_jsonl_to_df(jsonl_file, lines: bool = True):
    df = pd.read_json(jsonl_file, lines=lines)
    return df

def save_dataset_to_jsonl(dataset, jsonl_file):
    # https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/save_load_dataset.ipynb#scrollTo=8PZbm6QOAtGO
    from datasets import load_dataset
    # dataset = load_dataset("squad")
    for split, dataset in dataset.items():
        dataset.to_json(jsonl_file)

def hf_dataset_to_jsonl(dataset, output_file, fields=None):
    """
    Converts specific fields from a Hugging Face dataset to JSON Lines (JSONL) format.

    Args:
        dataset (datasets.Dataset): The Hugging Face dataset object.
        fields (list): A list of field names to extract from each example.
        output_file (str): The path to the output JSONL file.

    Returns:
        None
    """
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for example in dataset:
            if not fields:
                json_string = json.dumps(example)
            else:
                # Extract specific fields from the example
                selected_fields = {field: example[field] for field in fields}                
                # Convert the selected fields to a JSON string
                json_string = json.dumps(selected_fields)
            
            # Write the JSON string to the JSONL file with a newline separator
            jsonl_file.write(json_string + '\n')

def df_jsonl(dataframe, output_file, fields=None):
    """
    Converts specific fields from a Pandas DataFrame to JSON Lines (JSONL) format.

    Args:
        dataframe (pd.DataFrame): The Pandas DataFrame.
        fields (list): A list of column names to extract from the DataFrame.
        output_file (str): The path to the output JSONL file.

    Returns:
        None
    """
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for _, row in dataframe.iterrows():
            # Extract specific fields from the row
            selected_fields = {field: row[field] for field in fields}            
            # Convert the selected fields to a JSON string
            json_string = json.dumps(selected_fields)            
            # Write the JSON string to the JSONL file with a newline separator
            jsonl_file.write(json_string + '\n')


def list_of_dicts_to_jsonl(data_list, output_file):
    """
    Converts a list of dictionaries to JSON Lines (JSONL) format.

    Args:
        data_list (list): A list of dictionaries to convert to JSONL format.
        output_file (str): The path to the output JSONL file.

    Returns:
        None
    """
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for data_dict in data_list:
            # Convert the dictionary to a JSON string
            json_string = json.dumps(data_dict)
            
            # Write the JSON string to the JSONL file with a newline separator
            jsonl_file.write(json_string + '\n')


def dump_to_jsonl(item_list, jsonl_file):
    # item_list = [{'a': 1, 'b': 2}, {'a': 123, 'b': 456}]
    with jsonlines.open(jsonl_file, 'w') as writer:
        writer.write_all(item_list)

def pkl_load(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


def postprocess_output(input_string):
    # Remove extra spaces
    output_string = re.sub(r'\s+', ' ', input_string)
    # Remove extra newline characters
    output_string = re.sub(r'\n+', '\n', output_string)
    # Remove leading and trailing whitespace
    output_string = output_string.strip()
    return output_string

def pretty_print_preds(preds):
    print("\nOutput\n".join(map(str, preds)))


def get_base_name_from_hf_path(hf_path):
    """
    Can be something like: 
    Eg. hf_path:  multi_x_science, shubhamagarwal92/rw_2308_filtered
    """

    base_name = os.path.split(hf_path)[1]
    return base_name


def metrics_wrapper(model_name, savedir, preds, refs, dataset, generated_plan=[]):
    print(f"Total responses are {len(preds)}")
    config={"model_name": model_name} # Used for saving file
    config = Dict2Class(config)
    score_pipeline = MDSGen(config, inference=False)
    dataset = pd.DataFrame(dataset)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if generated_plan:
        print("Saving metrics with generated plan")
        score_pipeline.calculate_metrics(preds, refs, dataset, savedir, plans=generated_plan)
    else:
        print("Saving metrics with no generated plan")
        score_pipeline.calculate_metrics(preds, refs, dataset, savedir)


def get_hf_dataset(dataset_name, small_dataset: bool = False, split: str= "test", redownload: bool = False):
    if redownload:
        dataset = load_dataset(dataset_name, split=split, download_mode='force_redownload')
    else:
        dataset = load_dataset(dataset_name, split=split)
    hf_column_names = dataset.column_names
    if "ref_abstract_full_text" in hf_column_names:
        dataset = dataset.remove_columns(['ref_abstract_full_text_original', 'ref_abstract_full_text', "ref_abstract_original"])
    if small_dataset:
        dataset = dataset.select(list(range(5)))
    set_caching_enabled(False)
    return dataset


# def generate_prompt(data_point):
# return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
# Write a response that appropriately completes the request.  # noqa: E501
# ### Instruction:
# {data_point["instruction"]}
# ### Input:
# {data_point["input"]}
# ### Response:
# {data_point["output"]}"""

