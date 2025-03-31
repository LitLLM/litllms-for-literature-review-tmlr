# SA: TODO Refactor big time. 
import json
import os
import argparse
from datasets import load_dataset, set_caching_enabled
import pickle as pkl
import psutil
from functools import partial
import linecache
import gzip
import json
import time
import datetime
from tqdm import tqdm
import re
import pandas as pd
from utils import process_ids_s2_paper_dataset, parse_args


class DataDownloader: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        print(f"Defining class {self.class_name}")
        # Defining global variables where we can save all the corpus ids of the references
        self.references_corpus_list = []

    def get_hf_dataset(self, dataset_name, split: str="test", small_dataset: bool = False):
        dataset = load_dataset(dataset_name, split=split)
        if small_dataset:
            dataset = dataset.select(list(range(3)))
        # set_caching_enabled(False)
        return dataset

    def get_all_aid_mid_mxs(self, split: str = "test", small_dataset: bool = False, use_s2_paper_prefix: bool = False):
        """
        Get all MXS (Multi-X-Science) ids
        """
        dataset = self.get_hf_dataset(self.config.dataset_name, split=split, small_dataset=small_dataset)
        if use_s2_paper_prefix:
            dataset = dataset.map(partial(process_ids_s2_paper_dataset))
            s2_arxiv = dataset["s2_arxiv"]
            s2_mid = dataset["s2_mid"]
            s2_cite_mids = dataset["s2_cite_mids"]
            pkl_dump = {"s2_arxiv": s2_arxiv, "s2_mid": s2_mid, "s2_cite_mids": s2_cite_mids}
        else:
            arxiv = []
            mid = []
            ref_mids = []
            # TODO: Clearly we can use map here
            for row in tqdm(dataset):                
                arxiv.append(row["aid"])
                mid.append(row["mid"])
                references = row["ref_abstract"]  # abstract
                ref_mid_line = references["mid"]
                for ref in ref_mid_line:
                    if ref != "":
                        ref_mids.append(ref)
            pkl_dump = {"arxiv": arxiv, "mid": mid, "ref_mids": ref_mids}
        print(f"Total aids: {len(arxiv)}")
        print(f"Total mid: {len(mid)}")
        print(f"Total ref_mids: {len(ref_mids)}")
        print(f"Sample ref mids: {ref_mids[:3]}")

        pkl_path = self.get_mxs_pkl_path(self.config.savedir, split=split)
        self.save_pkl_dump(pkl_dump, pkl_path)

    def get_mxs_pkl_path(self, savedir: str, split: str="test"):
        pkl_path = f"{savedir}/id_list_{split}.pkl"
        return pkl_path

    def save_pkl_dump(self, pkl_obj, pkl_file_path):
        with open(pkl_file_path, 'wb') as fp:
            pkl.dump(pkl_obj, fp)
        print(f"Saved at {pkl_file_path}")
    
    def pkl_load(self, pkl_file_path):
        print(f"Reading file from {pkl_file_path}")

        with open(pkl_file_path, 'rb') as f:
            pkl_obj = pkl.load(f)
        return pkl_obj

    def find_related_work_section(self, sectionheader_text):
        related_work_string = ["Related work", "Literature Review", "Background"]
        index = -1  # Default value if not found

        for search_string in related_work_string:
            # Iterate through the list of strings to search in
            for i, s in enumerate(sectionheader_text):
                if search_string.lower() in s.lower():
                # if search_string.lower() == s.lower():
                    index = i
                    break
        return index

    def find_cite_in_refs(self):
        # Filter "related work" by doing regex on text of sectionheader (https://github.com/allenai/s2-folks/blob/main/examples/python/s2ag_datasets/sample-datasets.py#L24)
        # Find "related work" annotation (r_start, r_end) using sectionheader
        # Text would be from r_end to start of next section
        # For each bibref item (b_start, b_end),  filter all those bibref with (b_start, b_end) in (r_start, r_end), ie. r_start < b_start and b_end < r_end   -- get corresponding ref_id
        # For these ref_id , use bibentry to find matched_paper_id
        # Use the matched_paper_id as corpus_id to filter the text again from second round of s2orc data traversal
        return

    def is_recent_pub(self, arxiv_id, arxiv_prefix: str = "2308"):
    # def is_recent_pub(self, docs, index, arxiv_prefix: str = "2308"):
    #     if docs[index]['externalids']['arxiv'] and docs[index]['externalids']['arxiv'][:3] == arxiv_prefix:
    #         return True
        # First four chars represent year-month
        if arxiv_id and arxiv_id[:4] == arxiv_prefix:
            return True

    def append_to_jsonl(self, data, file_path, do_json_dump: bool = True, mode="a"):
        with open(file_path, 'a') as jsonl_file:
            if do_json_dump:
                json_string = json.dumps(data)
            else:
                json_string = data
                # TypeError: can't concat str to bytes. TODO find a better way
            jsonl_file.write(json_string + '\n')

    def filter_s2orc(self, savedir, split: str = "test", data_type: str = "s2orc", arxiv_prefix: str = "2308", num_files = 30):
        """
        Here we filter by aid, mid from MXS dataset.
        We also filter by latest publications -- arxiv_prefix Aug-23 (2308)
        """
        st = time.monotonic()
        now = datetime.datetime.now()
        print(f"Calling the main function of {self.class_name} at {now}")

        pkl_path = self.get_mxs_pkl_path(self.config.savedir, split=split)
        mxs_ids = self.pkl_load(pkl_path)
        mxs_arxiv = mxs_ids["arxiv"]
        mxs_mid = mxs_ids["mid"]
        mxs_ref_mids = mxs_ids["ref_mids"]
        
        total_found_papers = 0  # Across all chunks
        total_new_papers = 0 # Across all chunks
        completed = 30
        for file_index in range(num_files):
            id_list = []
            no_external_id_list = []
            found_papers = 0
            new_papers = 0
            if file_index < completed:
                download_path = f"{self.config.data_dir}/{data_type}/{data_type}-part{file_index}.jsonl.gz"
                print(f"Reading this chunk from {download_path}")                
                # download_path = f"{self.config.data_dir}/{data_type}_copy/{data_type}-part{index}.jsonl"
                # with open(download_path, "r") as f:
                with gzip.open(download_path, 'rb') as gzipped_file:
                    for line_num, line in enumerate(gzipped_file):
                        docs = json.loads(line)
                        corpusid = docs['corpusid']
                        # print(docs.keys(), line_num)
                        # print(docs['externalids'])
                        externalids = docs['externalids']
                        # SA: TODO TODO TODO -- this filter is bad linenum changes -- 
                        # Also if we dont have externalids, we can still get the text
                        if not externalids:
                            # print(f"No external id found for {corpusid}")
                            no_external_id_list.append(corpusid)
                            continue
                        arxiv_id = externalids.get("arxiv", None)
                        mag = externalids.get("mag", None)
                        # Saving the ids for easy future references
                        id_obj = {
                            "line_num": line_num,
                            "corpusid": corpusid,
                            "arxiv": arxiv_id,
                            "mag": mag,
                            "file_path": download_path, 
                            "corpus_file_name": f"{data_type}-part{file_index}.jsonl.gz"
                        }
                        id_list.append(id_obj)

                        # Appending raw Multi X Science papers
                        if arxiv_id and arxiv_id in mxs_arxiv or mag and mag in mxs_mid or mag and mag in mxs_ref_mids:
                            # jsonl_path = f"{savedir}/{data_type}-part{file_index}.jsonl"
                            mxs_jsonl_path = f"{savedir}/{data_type}-mxs-{split}.jsonl"
                            self.append_to_jsonl(data=docs, file_path=mxs_jsonl_path)
                            found_papers += 1                            
                        # Appending raw latest 2308 papers
                        if arxiv_id and self.is_recent_pub(arxiv_id=arxiv_id):
                            jsonl_path = f"{savedir}/{arxiv_prefix}_papers.jsonl"
                            self.append_to_jsonl(data=docs, file_path=jsonl_path)
                            new_papers += 1

                        if line_num % 25000 == 0:
                            print(f"Processed {line_num+1} records")
                print(f"Total records: {len(id_list)}")
                print(f"No external ids found for: {len(no_external_id_list)}")
                print(f"Saving {new_papers} latest papers at: {savedir}/{arxiv_prefix}_papers.jsonl")
                print(f"Saving {found_papers} mxs papers at: {savedir}/{data_type}-mxs-{split}.jsonl")
                dt = time.monotonic() - st
                print("Time elapsed now", dt)
                total_found_papers += found_papers
                total_new_papers += new_papers

                pkl_path = f"{savedir}/{data_type}-part{file_index}_id_list.pkl"
                pkl_dump = {f"part{file_index}": id_list}
                self.save_pkl_dump(pkl_dump, pkl_path)
                print(f"Total records: {len(id_list)}")
                print(id_list[0])
                print(f"Time after one file: {datetime.datetime.now()}")
        dt = time.monotonic() - st
        print("Time elapsed for all files: ", dt)
        print(f"Total found papers: {total_found_papers}")
        print(f"Total new papers: {total_new_papers}")

        return

    def text_of(self, type, text, annotations):
        return [text[int(a['start']):int(a['end'])] for a in annotations.get(type, '')]

    def text_from_annotations(self, text, ann):        
        """
        given ann['start'], ann['end'] -- return the text
        """
        return text[int(ann['start']):int(ann['end'])]

    def get_ids_from_doc(self, docs):
        externalids = docs['externalids']
        # corpusid = docs["corpusid"]
        # if not externalids:
        #     # print(f"No external id found for {corpusid}")
        #     continue
        arxiv_id = externalids.get("arxiv", None)
        mag = externalids.get("mag", None)
        return arxiv_id, mag


    def process_raw_filtered_data(self, savedir, split: str = "test", data_type: str = "s2orc", num_files = 30, 
                         find_citations: bool = False, arxiv_prefix: str = "2308", delexicalized: bool = True,
                         only_one: bool = False, process_latest_refs: bool = False):
        """
        We use this function to create both datasets:
        1. Given filtered list of jsonl -- get full text 
        2. Given filtered list of jsonl of recent papers -- get related work, citations and corresponding ids
        find_citations 
            True -- create a new dataset -- need to find all citations and align with text
            False -- Work with MXS
        delexicalized
            True -- replace (X et al) with bibref1
            False -- Citations as it is
        only_one
            True - only first para
            False - list of all paras with citations
        """
        # SA: TODO: change file name
        if find_citations:
            jsonl_path = f"{savedir}/{arxiv_prefix}_papers.jsonl"
        elif process_latest_refs:
            jsonl_path = f"{savedir}/raw_references_s2_corpus_id-{arxiv_prefix}.jsonl"
        else:
            jsonl_path = f"{savedir}/{data_type}-mxs-{split}.jsonl"
            # mxs_jsonl_path = f"{savedir}/{data_type}-mxs-{split}-copy.jsonl"            
        found_rw = 0
        with open(jsonl_path, "r") as jsonl_file:
            for line_num, line in enumerate(jsonl_file):
                citations_obj = {}
                docs = json.loads(line)
                text = docs['content']['text']
                corpusid = docs['corpusid']
                annotations = {k: json.loads(v) for k, v in docs['content']['annotations'].items() if v}

                if "paragraph" not in annotations:
                    print("No paragraph information. Skipping this paper!")
                    continue
                if "title" not in annotations:
                    print("No title information!")
                    title = ""
                else:
                    title = self.text_of("title", text, annotations)[0]
                    # title_start = int(annotations["title"][0]["start"])
                    # title_end = int(annotations["title"][0]["end"])
                    # title = text[title_start: title_end]
                    # abstract = annotations("abstract")
                aid, mid = self.get_ids_from_doc(docs)
                # SA: TODO check if this is sorted
                first_para_start = int(annotations["paragraph"][0]["start"])
                last_para_end = int(annotations["paragraph"][-1]["end"])
                # print(first_para_start, last_para_end)
                # print(type(first_para_start), type(last_para_end))
                # print(len(text))
                all_para_text = text[first_para_start:last_para_end]
                # # TODO - check - take first element -- returned as list
                # We didnt find any section named "Related work", but this could be ref paper
                if "sectionheader" not in annotations:
                    if find_citations:
                        print("No section header information. Cant find citations. Skipping!")
                        continue                    
                    # Paper could be the reference paper in MXS. So maybe we just need full text
                    # all_para_text contains all para
                    text_except_rw= ""
                else:
                    # SA: TODO check if this is sorted
                    sectionheader_ann = annotations["sectionheader"]
                    sectionheader_text = self.text_of("sectionheader", text, annotations)
                    index_rw_header = self.find_related_work_section(sectionheader_text)
                    total_sections = len(sectionheader_ann)
                    # print(sectionheader_text)                   
                    # We didnt find any section named "Related work", but this could be ref paper
                    if index_rw_header == -1:
                        text_except_rw= ""
                        print(f"Found no Related work section for aid: {aid}")
                    else:
                        found_rw += 1
                        last_section = False
                        # print(index_rw_header)
                        # We need text of paper, from start of first section (paragraph) to r_start, r_end to end of paragraph
                        # find r_start, r_end, next_section_start
                        # Main text: first_para_start:r_start , next_section_start:last_para_end
                        rw_sectionheader_start = int(annotations["sectionheader"][index_rw_header]["start"])
                        text_before_rw = text[first_para_start:rw_sectionheader_start]
                        # Related work is the last section
                        if index_rw_header+1 == total_sections:
                            text_except_rw = text_before_rw
                            last_section = True
                        else:
                            # Related work is in the middle
                            # SA: TODO Check if RW is not at the start
                            # Hopefully it would still be in list
                            next_sectionheader_start = int(annotations["sectionheader"][index_rw_header+1]["start"])
                            text_after_rw = text[next_sectionheader_start:last_para_end]
                            text_except_rw = text_before_rw + text_after_rw
                        # Now we are finding citations and aligning them with references and text
                        if find_citations:
                            if "bibentry" not in annotations:
                                print("No bibentry information. Cant find citations. Skipping!")
                                continue
                            if "bibref" not in annotations:
                                print("No bibref information. Cant find citations. Skipping!")
                                continue                            
                            
                            paras = self.filter_related_work_section_with_citations(text, annotations, index_rw_header, delexicalized, only_one, last_section)
                            # abstract
                            citations_obj = {"paras": paras}
                if "abstract" in annotations:
                    # TODO - check - take first element -- returned as list
                    abstract = self.text_of("abstract", text, annotations)[0]
                else:
                    abstract = ""
                text_obj = {"aid": aid, "mid": mid, "text_except_rw": text_except_rw, "all_para_text": all_para_text, "title": title, 
                            "abstract": abstract, "corpusid": corpusid}                
                # Write in different files
                if find_citations:
                    # New dataset
                    processed_jsonl_path = f"{savedir}/processed_text-{arxiv_prefix}.jsonl"
                    # Also add the citations obj in the text object to save
                    text_obj = self.update_dict(text_obj, citations_obj)
                elif process_latest_refs:
                    processed_jsonl_path = f"{savedir}/processed_text-{arxiv_prefix}_refs_papers.jsonl"
                else:                    
                    processed_jsonl_path = f"{savedir}/processed_text-mxs-{split}.jsonl"                    
                # Appending for each line    
                self.append_to_jsonl(data=text_obj, file_path=processed_jsonl_path)
        print(f"Found related work sections for total {found_rw} documents")
        print(f"Saving the processed text at: {processed_jsonl_path}")
        if find_citations:
            pkl_dump = {"ref_corpus_id": self.references_corpus_list}
            print(f"Total references to find: {len(self.references_corpus_list)}")
            pkl_path = f"{savedir}/references_s2_corpus_id-{arxiv_prefix}.pkl"
            self.save_pkl_dump(pkl_dump, pkl_path)

    def second_traversal_get_reference_papers(self, savedir, data_type="s2orc", arxiv_prefix="2308", num_files = 30):
        """
        Filter reference papers from main corpus
        """
        # Where we save their data
        refs_save_jsonl_path = f"{savedir}/raw_references_s2_corpus_id-{arxiv_prefix}.jsonl"
        refs_pkl_path = f"{savedir}/references_s2_corpus_id-{arxiv_prefix}.pkl"
        ref_ids = self.pkl_load(refs_pkl_path)["ref_corpus_id"]
        print(f"Total ref ids: {len(ref_ids)}") # 69373
        ref_ids = [int(item) for item in ref_ids if item != ""]
        # ref_ids = list(filter(lambda item: item != "", ref_ids))
        # print(f"Total filtered ref ids {len(ref_ids)}") # 51471
        # print(f"Ref ids: {ref_ids[:3]}")
        # print(f"Ref ids type: {type(ref_ids[1])}")
        found_papers = 0
        print(f"Time now: {datetime.datetime.now()}")
        for file_index in range(num_files):
            id_list_pkl_path = f"{savedir}/{data_type}-part{file_index}_id_list.pkl"
            id_list = self.pkl_load(id_list_pkl_path)[f"part{file_index}"] # {f"part{file_index}": id_list}
            ids_df = pd.DataFrame(id_list) # list of jsonl  
            # print(ids_df)
            indices = ids_df[ids_df['corpusid'].isin(ref_ids)].index
            found_lines = ids_df.loc[indices, 'line_num'].tolist()
            del ids_df
            print(f"Total ids found: {len(found_lines)}")
            download_path = f"{self.config.data_dir}/{data_type}/{data_type}-part{file_index}.jsonl.gz"
            print(f"Reading this chunk from {download_path}")                
            with gzip.open(download_path, 'rb') as gzipped_file:
                for line_num, line in enumerate(gzipped_file):
                    if line_num in found_lines:
                        docs = json.loads(line)
                        corpusid = docs['corpusid']
                        assert corpusid in ref_ids, f"{corpusid} is not in the list."
                        self.append_to_jsonl(data=docs, file_path=refs_save_jsonl_path)
                        found_papers += 1
            print(f"Time after one file: {datetime.datetime.now()}")
        print(f"Saving total {found_papers} found papers at: {refs_save_jsonl_path}")


    def update_dict(self, big_dict, small_dict):
        big_dict.update(small_dict)
        return big_dict

    def filter_related_work_section_with_citations(self, text, annotations, index_rw_header, delexicalized: bool = True, 
                                                   only_one: bool = False, last_section: bool = False):
        """
        We found the RW section header indices -- start/end
        
        Need to filter texts
        Need to filter paragraphs
        """
        
        paras = self.filter_related_work_by_paras(text, annotations, index_rw_header, last_section)
        # print(paras)
        # Sometimes paras is empty -> TODO: helps in debugging
        # if only_one:
        if only_one and paras:
            return paras[-1]
        return paras

    def filter_related_work_by_paras(self, text, annotations, index_rw_header, last_section):
        paragraphs = annotations["paragraph"]
        rw_sectionheader_start = int(annotations["sectionheader"][index_rw_header]["start"])
        if last_section:
            # We will use this as the end of RW since it is last section
            next_sectionheader_start = int(annotations["paragraph"][-1]["end"])  
        else:
            # We will use the start of next section as the end of RW
            next_sectionheader_start = int(annotations["sectionheader"][index_rw_header+1]["start"])


        # # cite_N, corpus_id
        # Bibrefs are places when cited in text. Bibentries are the name of the papers and details
        # Bibref
        # (Agarwal et al.) showed ....
        # ------------------------------
        # Bibentries
        # References:
        # Shubham Agarwal and others, "A cool paper", Arxiv 2023. -- Bibentry
        bibrefs = annotations["bibref"]
        bibentries = annotations["bibentry"]
        related_work_paragraphs = []
        for paragraph in paragraphs:
            # RW found when paragraph starts and ends between the rw_sectionheader and next_sectionheader
            if rw_sectionheader_start < paragraph["start"] and paragraph["end"] < next_sectionheader_start:
                found_bibref = []
                # TODO: This can be faster by keeping the index of bibref already searched previously
                for bibref in bibrefs:
                    # If bibref starts and ends within paragraph content
                    if paragraph["start"] < bibref["start"] and bibref["start"] < paragraph["end"]:
                        # print(bibref)
                        # SA: TODO handle exception
                        if "attributes" in bibref and "ref_id" in bibref["attributes"]:
                            ref_id = bibref["attributes"]["ref_id"]
                            matched_paper_id, mxs_cite = self.find_cite_corpus_id(ref_id, bibentries)
                            self.references_corpus_list.append(matched_paper_id)
                        else:
                            matched_paper_id, mxs_cite = None, "@cite_999"
                        original_cite_text = self.text_from_annotations(text=text, ann=bibref)
                        bibref_obj = {"mxs_cite": mxs_cite, "matched_paper_id": matched_paper_id, "original_cite_text": original_cite_text}
                        found_bibref.append(bibref_obj)
                para_text = self.text_from_annotations(text=text, ann=paragraph)
                # Can do delexicalized here: replace_cite_text_with_mxs_cite()
                related_work_paragraph_obj = {"related_work": para_text, "references": found_bibref}
                related_work_paragraphs.append(related_work_paragraph_obj)
        return related_work_paragraphs


    def find_cite_corpus_id(self, ref_id, bibentries):
        """
        id, matched_paper_id -- bibentry
        ref_id -- bibref
        return matched_paper_id
        """
        # if annotations["bibentry"][index]["attributes"]["id"] == ["bibref"][index2]["attributes"]["ref_id"]:
        #     annotations["bibentry"][0]["attributes"]["matched_paper_id"]
        # Use a list comprehension to find the matched_paper_id
        matching_entries = [entry for entry in bibentries if entry['attributes']['id'] == ref_id]
        # Returned as a list. Use first element. (We dont care about the full reference text, just the id)
        # print(matching_entries[0]['attributes'])
        matched_paper_id = matching_entries[0]['attributes'].get('matched_paper_id', "")
        mxs_cite = self.get_mxs_style_cite(ref_id=ref_id)

        return matched_paper_id, mxs_cite

    def get_mxs_style_cite(self, ref_id):
        """
        Get element as @cite_3 from b3 or XX3
        """
        cite_num = self.get_num_from_alphanumeric(ref_id)
        return f"@cite_{cite_num}"

    def get_num_from_alphanumeric(self, alphanumeric_str):
        """
        Get num from alphanumeric string
        """
        # Return last element
        num_list = re.findall(r'(\d+)', alphanumeric_str)
        return num_list[-1]


if __name__ == "__main__":
    parsed_args = parse_args()
    downloader = DataDownloader(parsed_args)

    splits = ["test", "train"]
    
    for split in splits:
        if parsed_args.filter_s2orc:
            # Step 1: Get all mids aids from MXS dataset, 2308 entries. 
            # id_list_test.pkl
            downloader.get_all_aid_mid_mxs(split)
            # Step 2: Filter S2ORC huge corpus for found ids as well as recent papers
            # s2orc-part*_id_list.pkl
            # s2orc-mxs-test.jsonl
            # 2308_papers.jsonl        
            downloader.filter_s2orc(parsed_args.savedir)
        # Step 3: Process the filtered dump to extract full text
        # processes_text
        if parsed_args.multi_x_science_text:
            print("Processing for Multi-X-Science dataset")
            downloader.process_raw_filtered_data(parsed_args.savedir, split=split)
        # Step 4: For recent papers, align with references
        print("Processing only for new dataset")
        downloader.process_raw_filtered_data(parsed_args.savedir, find_citations=True)
        
        # Step 5
        downloader.second_traversal_get_reference_papers(parsed_args.savedir)
        # Eg: Working for https://arxiv.org/pdf/2308.08443.pdf
        
        # Step 6
        downloader.process_raw_filtered_data(parsed_args.savedir, process_latest_refs=True)
        print("All done!")