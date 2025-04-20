# https://github.com/allenai/s2-folks/blob/main/examples/python/s2ag_datasets/sample-datasets.py#L13
# https://api.semanticscholar.org/datasets/v1/release/2023-07-25/dataset/s2orc
# https://github.com/allenai/s2-folks/issues/127
import requests
from urllib.parse import quote
import json
import os
import urllib
import argparse


S2_API_KEY=os.environ['S2_API_KEY']

# prepare progressbar
def show_progress(block_num, block_size, total_size):
    """
    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """
    print(round(block_num * block_size / total_size *100,2), end="\r")


class DataDownloader: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config
        print(f"Defining class {self.class_name}")

    def get_release_info(self):
        print(f'Using API Key: {os.getenv("S2_API_KEY")}')
        # Get info about the latest release
        latest_release = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest").json()
        print(latest_release['README'])
        print(latest_release['release_id'])
        # Get info about past releases
        dataset_ids = requests.get("http://api.semanticscholar.org/datasets/v1/release").json()
        earliest_release = requests.get(f"http://api.semanticscholar.org/datasets/v1/release/{dataset_ids[0]}").json()
        # Print names of datasets in the release
        print("\n".join(d['name'] for d in latest_release['datasets']))
        # abstracts, authors, citations, embeddings-specter_v1, embeddings-specter_v2, paper-ids, 
        # papers, publication-venues, s2orc, tldrs
        # Print README for one of the datasets
        print(latest_release['datasets'][2]['README'])

    def main(self, savedir, data_type: str ="s2orc"):
        print(f"Calling the main function of {self.class_name}")
        print(f'Using API Key: {os.getenv("S2_API_KEY")}')
        if S2_API_KEY == "":
            raise ValueError("You havent set API key!")
        # Get info about the papers dataset
        # papers = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers",
        #                     headers={'X-API-KEY': os.getenv("S2_API_KEY")}).json()
        s2_dataset = requests.get(f"http://api.semanticscholar.org/datasets/v1/release/latest/dataset/{data_type}",
                            headers={'X-API-KEY': os.getenv("S2_API_KEY")}).json()

        # Only has files field
        data_files = s2_dataset['files']
        print(f"Total files for {data_type}: {len(s2_dataset['files'])}")
        # 30 files of 1.7G each
        # Download the first part of the dataset

        save_data_path = f"{savedir}/{data_type}"
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        
        completed = -1
        for index, single_chunk in enumerate(data_files):
            if index > completed:
                download_path = f"{save_data_path}/{data_type}-part{index}.jsonl.gz"
                print(f"Downloading and saving this chunk as {download_path}")
                urllib.request.urlretrieve(single_chunk, download_path, show_progress)
        return


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--savedir",
        default="results/auto_review/dataset/full_dataset/",
        help="Save directory path",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    downloader = DataDownloader(parsed_args)
    downloader.get_release_info()
    downloader.main(parsed_args.savedir)

# https://github.com/allenai/s2-folks/blob/main/examples/python/s2ag_datasets/get_sample_files.sh#L9
# "gunzip -q /mnt/colab_public/results/auto_review/dataset/full_dataset/*/*.gz"
# gunzip -q */*.gz
# gunzip -q full_dataset/*.gz