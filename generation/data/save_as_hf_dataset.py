from datasets import Dataset, load_dataset, set_caching_enabled, DatasetDict
from utils import parse_args, get_hf_dataset, replace_cite_text_with_mxs_cite
from functools import partial
from huggingface_hub import login as hf_login
import os
import pandas as pd
import json
import copy


def get_value_filtered_rows(filtered_rows):
    text_except_rw = filtered_rows['text_except_rw'].values[0]
    title = filtered_rows['title'].values[0]

    if isinstance(title, list):
        title = title[0]
    found = 1
    total_words = len(text_except_rw.split())
    return text_except_rw, title, total_words, found 

def convert_to_dict(df_test):
    """
    https://discuss.huggingface.co/t/save-datasetdict-to-huggingface-hub/12075/4
    https://stackoverflow.com/questions/72499850/how-to-load-two-pandas-dataframe-into-hugginfaces-dataset-object
    https://discuss.huggingface.co/t/from-pandas-dataframe-to-huggingface-dataset/9322
    
    """
    dataset_dic = DatasetDict({"test": Dataset.from_pandas(df_test)})

def push_df_to_hub(dataset, repo_path: str="shubhamagarwal92/multi_x_science_test_full", do_hf_login: bool = True):
    """
    https://huggingface.co/docs/datasets/upload_dataset
    """
    HF_TOKEN = os.environ['HF_WRITE_TOKEN']
    if do_hf_login:
        hf_login(token=HF_TOKEN)    
    # dataset = load_dataset("stevhliu/demo", token=True)
    dataset.push_to_hub(repo_path)

def get_pandas_percentile(df, col_name_list):
    # https://gist.github.com/shubhamagarwal92/13e2c41d09156c3810740d7697a883d1
    # https://stackoverflow.com/questions/34026089/create-dataframe-from-another-dataframe-describe-pandas
    describe_df = df[col_name_list].describe(percentiles=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    return describe_df

def append_data_text(row, df_to_filter):
    # desired_id = "2964227312" # 1609.05140
    # aid = "1405.5867"
    # filtered_rows = df.loc[df['mid'] == desired_id]
    # filtered_value = filtered_rows['text_except_rw'].values[0]
    # print(filtered_value)
    aid = row["aid"]
    mid = row["mid"]
    filtered_rows_aid = df_to_filter[df_to_filter['aid'] == aid]
    filtered_rows_mid = df_to_filter[df_to_filter['mid'] == mid]
    if not filtered_rows_aid.empty:
        text_except_rw, title, total_words, found = get_value_filtered_rows(filtered_rows_aid)
    elif not filtered_rows_mid.empty:
        text_except_rw, title, total_words, found = get_value_filtered_rows(filtered_rows_mid)
    else:
        title = ""
        text_except_rw = ""
        found = 0
        total_words = 0
    return pd.Series([title, text_except_rw, found, total_words])

def check_if_list(text_field):
    if isinstance(text_field, list):
        text_field = text_field[0]
    return text_field


class DataCreator: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        print(f"Defining class {self.class_name}")

    def append_mxs_hf_dataset(self, split: str = "test", small_dataset: bool = False):
        processed_jsonl_path = f"{self.config.savedir}/processed_text-mxs-{split}.jsonl"
        df = pd.read_json(processed_jsonl_path, lines=True, dtype={"mid": str, "aid": str})

        dataset = get_hf_dataset(self.config.dataset_name, split=split, small_dataset=small_dataset)
        set_caching_enabled(False)

        df_pandas = dataset.to_pandas()
        df_pandas[["title", "text_except_rw", "found", "total_words"]] = df_pandas.apply(append_data_text, df_to_filter=df, axis=1)
        all_df = get_pandas_percentile(df_pandas, ["found", "total_words"])
        print(all_df)
        total_found = sum(df_pandas["found"])
        print(f"Total % found: {total_found/len(df_pandas)*100}")
        
        dataset = Dataset.from_pandas(df_pandas)
        dataset = dataset.remove_columns(['found'])
        dataset_dic = DatasetDict({"test": dataset})
        # print(dataset[160])
        push_df_to_hub(dataset_dic)


    def create_recent_hf_dataset(self, savedir, arxiv_prefix: str = "2308"):
        # processed_jsonl_path = f"{self.config.savedir}/processed_text-mxs-{split}.jsonl"
        processed_jsonl_path = f"{savedir}/processed_text-{arxiv_prefix}.jsonl"
        refs_jsonl_path = f"{savedir}/processed_text-{arxiv_prefix}_refs_papers.jsonl"
        refs_df = pd.read_json(refs_jsonl_path, lines=True, dtype={"mid": str, "aid": str, "corpusid": str})
        print(f"Length of refs: {len(refs_df)}")
        print(refs_df.columns)
        # "aid", "mid", "text_except_rw", "all_para_text", "title", "abstract", "paras", "corpusid"
        # "related_work", "references"
        # "mxs_cite", "matched_paper_id", "original_cite_text"
        latest_papers = [json.loads(l) for l in open(processed_jsonl_path, "r").readlines()]
        print(f"Length of latest_papers: {len(latest_papers)}")
        one_para_list = []
        all_para_list = []
        one_para_list_filtered = []
        all_para_list_filtered = []
        for paper in latest_papers:
            # Skip where we dont have relataed work section
            if not "paras" in paper:
                continue            
            aid = paper["aid"]
            mid = paper["mid"]
            abstract = check_if_list(paper["abstract"])
            text_except_rw = paper["text_except_rw"]
            title = check_if_list(paper["title"])
            corpusid = paper["corpusid"]
            # paras is a list 
            # references is a list
            # "ref_abstract" - "abstract", "cite_N", "mid"
            paras = paper["paras"]            
            if not paras:
                # print("Empty paragraphs")
                continue

            # We write for individual as well as all paragraphs 
            all_para_obj = {"aid": aid, "mid": mid, "abstract": abstract, "corpusid": corpusid, 
                            "text_except_rw": text_except_rw, "title": title}
            whole_rw_cite_N = []
            whole_rw_cite_N_original = []
            whole_rw_abstract_refs = []
            whole_rw_corpusid_refs = []
            whole_rw_all_para_text_refs = []
            whole_rw_filter_record = False
            whole_rw_original_related_work = ""
            whole_rw_related_work = ""
            whole_rw_total_cites = 0

            # We do for multiple paras as in multi-x-science
            # This could have multiple paras of same paper
            for one_para in paras:
            # one_para = paras[0]
                original_related_work = one_para["related_work"]
                references = one_para["references"] 
                cite_N = []
                cite_N_original = []
                abstract_refs = []
                corpusid_refs = []
                all_para_text_refs = []
                filter_record = False
                related_work = original_related_work
                for bibref in references:
                    mxs_cite = bibref["mxs_cite"]
                    matched_paper_id = str(bibref["matched_paper_id"])
                    original_cite_text = bibref["original_cite_text"]
                    related_work = replace_cite_text_with_mxs_cite(related_work, original_cite_text=original_cite_text, mxs_cite=mxs_cite)
                    filtered_rows = refs_df[refs_df['corpusid'] == matched_paper_id]
                    if not filtered_rows.empty:
                        all_para_text = filtered_rows['all_para_text'].values[0]                
                        abstract_ref = filtered_rows['abstract'].values[0]
                    else:
                        filter_record = True
                        whole_rw_filter_record = True
                        all_para_text = ""
                        abstract_ref = ""
                    cite_N.append(mxs_cite)
                    corpusid_refs.append(matched_paper_id)
                    cite_N_original.append(original_cite_text)
                    all_para_text_refs.append(all_para_text)
                    abstract_refs.append(abstract_ref)

                ref_abstract = {"abstract": abstract_refs, "cite_N": cite_N, "corpursid": corpusid_refs}
                ref_abstract_original = {"abstract": abstract_refs, "cite_N": cite_N_original, "corpursid": corpusid_refs}
                ref_abstract_full_text = copy.deepcopy(ref_abstract)
                ref_abstract_full_text["all_para_text"] = all_para_text_refs                    
                ref_abstract_full_text_original = copy.deepcopy(ref_abstract_original)
                ref_abstract_full_text_original["all_para_text"] = all_para_text_refs                    
                total_cites = len(references)
                paper_obj = copy.deepcopy(all_para_obj)
                paper_obj["related_work"] = related_work
                paper_obj["original_related_work"] = original_related_work            
                paper_obj["ref_abstract"] = ref_abstract
                paper_obj["ref_abstract_original"] = ref_abstract_original
                paper_obj["ref_abstract_full_text"] = ref_abstract_full_text
                paper_obj["ref_abstract_full_text_original"] = ref_abstract_full_text_original
                paper_obj["total_cites"] = total_cites

                # Keep only records with atleast one citation
                if total_cites > 0: 
                    # No citations text found -- dont include
                    all_empty = all(element == "" for element in abstract_refs)
                    if not all_empty:
                        one_para_list.append(paper_obj)
                    if not filter_record:
                        one_para_list_filtered.append(paper_obj)

                # Extend for the whole rw -- all paras
                whole_rw_cite_N.extend(cite_N)
                whole_rw_cite_N_original.extend(cite_N_original)
                whole_rw_abstract_refs.extend(abstract_refs)
                whole_rw_corpusid_refs.extend(corpusid_refs)
                whole_rw_all_para_text_refs.extend(all_para_text_refs)                
                whole_rw_original_related_work = f"{whole_rw_original_related_work}{original_related_work}\n"
                whole_rw_related_work = f"{whole_rw_related_work}{related_work}\n"
                whole_rw_total_cites += total_cites

            ref_abstract = {"abstract": whole_rw_abstract_refs, "cite_N": whole_rw_cite_N, "corpursid": whole_rw_corpusid_refs}
            ref_abstract_original = {"abstract": whole_rw_abstract_refs, "cite_N": whole_rw_cite_N_original, "corpursid": whole_rw_corpusid_refs}
            ref_abstract_full_text = copy.deepcopy(ref_abstract)
            ref_abstract_full_text["all_para_text"] = whole_rw_all_para_text_refs                    
            ref_abstract_full_text_original = copy.deepcopy(ref_abstract_original)
            ref_abstract_full_text_original["all_para_text"] = all_para_text_refs                    
            whole_rw_paper_obj = copy.deepcopy(all_para_obj)
            whole_rw_paper_obj["related_work"] = whole_rw_related_work
            whole_rw_paper_obj["original_related_work"] = whole_rw_original_related_work            
            whole_rw_paper_obj["ref_abstract"] = ref_abstract
            whole_rw_paper_obj["ref_abstract_original"] = ref_abstract_original
            whole_rw_paper_obj["ref_abstract_full_text"] = ref_abstract_full_text
            whole_rw_paper_obj["ref_abstract_full_text_original"] = ref_abstract_full_text_original
            whole_rw_paper_obj["total_cites"] = whole_rw_total_cites
            # Keep only records with atleast one citation
            if whole_rw_total_cites > 0: 
                # No citations text found -- dont include
                all_empty = all(element == "" for element in whole_rw_abstract_refs)
                if not all_empty:
                    all_para_list.append(whole_rw_paper_obj)
                if not whole_rw_filter_record:
                    all_para_list_filtered.append(whole_rw_paper_obj)


        one_para_df = pd.DataFrame(one_para_list)
        one_para_df_filtered = pd.DataFrame(one_para_list_filtered)
        all_para_df = pd.DataFrame(all_para_list)
        all_para_df_filtered = pd.DataFrame(all_para_list_filtered)

        print(f"Length of one para df: {len(one_para_df)}")
        print(f"Length of filtered one para df: {len(one_para_df_filtered)}")
        print(f"Length of all para df: {len(all_para_df)}")
        print(f"Length of filtered all para df: {len(all_para_df_filtered)}")
        print(one_para_df_filtered.iloc[0]["related_work"])

        self.convert_to_hf_upload(df_pandas=one_para_df, repo_name="rw_2308")
        self.convert_to_hf_upload(df_pandas=one_para_df_filtered, repo_name="rw_2308_filtered")
        # self.convert_to_hf_upload(df_pandas=all_para_df, repo_name="rw_2308_all_paras")
        # self.convert_to_hf_upload(df_pandas=all_para_df_filtered, repo_name="rw_2308_filtered_all_paras")

    def convert_to_hf_upload(self, df_pandas, repo_name, repo_prefix = "shubhamagarwal92/"):
        dataset = Dataset.from_pandas(df_pandas)
        # dataset = dataset.remove_columns(['found'])
        dataset_dic = DatasetDict({"test": dataset})
        repo_path = f"{repo_prefix}{repo_name}"
        push_df_to_hub(dataset_dic, repo_path)
        hf_data_save_dir = f"{self.config.hf_savedir}/{repo_name}"
        if not os.path.exists(hf_data_save_dir):
            os.makedirs(hf_data_save_dir)
        dataset_dic.save_to_disk(hf_data_save_dir)



if __name__ == "__main__":
    parsed_args = parse_args()
    downloader = DataCreator(parsed_args)
    if parsed_args.multi_x_science_text:
        print(f"Processing the MXS extension")
        downloader.append_mxs_hf_dataset()
    downloader.create_recent_hf_dataset(parsed_args.savedir)
