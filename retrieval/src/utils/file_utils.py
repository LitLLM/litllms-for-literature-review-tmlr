"""
This module contains utility functions related to dealing with files.
"""

import os
import tarfile
import pandas as pd
import os
import logging


def parse_arxiv_id_from_paper_url(url):
    """
    This function parses the arxiv id from the paper url
    """
    arxiv_id = url.split("/")[-1]
    if arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    return arxiv_id


def read_csv(file_path, num_rows=None):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Read df.shape: {df.shape}")
    if num_rows:  # Return only the first num_rows
        return df.iloc[:num_rows]
    return df


def extract_tex_files(tar_filename, output_dir="extracted_files"):
    # Extract the tar.gz file
    if tar_filename:
        try:
            with tarfile.open(tar_filename, "r:gz") as tar:
                tar.extractall(path=output_dir)
            # print(f"Extracted files to {output_dir}")
            return output_dir
        except Exception as e:
            print(f"Failed to extract files for {tar_filename}: {e}")
            return None
    return None


def find_tex_files(directory):

    # Find all .tex files in the directory
    tex_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
    return tex_files


def read_tex_files(tex_files):
    # Read and concatenate all .tex files
    full_text = ""
    try:
        for tex_file in tex_files:
            with open(tex_file, "r", encoding="utf-8") as file:
                full_text += file.read() + "\n"
        return full_text
    except Exception as e:
        print(f"Failed to read .tex files: {e}")
        return "None"


def save_processed_papers(papers, output_dir, paper_index, prompt_num):
    df = pd.DataFrame(papers) if papers else pd.DataFrame()
    os.makedirs(
        f"{output_dir}/processing_paper{paper_index}",
        exist_ok=True,
    )
    df.to_csv(
        f"{output_dir}/processing_paper{paper_index}/processing_prompt{prompt_num}.csv",
        index=False,
    )


def save_recommended_papers(papers, output_dir, arxiv_id, id_counter):
    df = pd.DataFrame(papers) if papers else pd.DataFrame()
    directory_path = f"{output_dir}/recommended_papers_{id_counter}"
    os.makedirs(directory_path, exist_ok=True)
    df.to_csv(f"{directory_path}/recommended_for_{arxiv_id}.csv", index=False)


def setup_logging(log_file="faiss_handler.log"):
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("Logging setup complete.")


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        logging.info(f"Creating directory {directory}")
        os.makedirs(directory)
