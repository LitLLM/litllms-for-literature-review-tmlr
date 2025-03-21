import json
import requests
import os
import csv
from time import sleep
import re
import numpy as np
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process FAISS index queries")
    parser.add_argument(
        "--batch_start", type=int, required=True, help="Start index of the query batch"
    )
    parser.add_argument(
        "--batch_end", type=int, required=True, help="End index of the query batch"
    )
    parser.add_argument(
        "--api_key", type=str, required=True, help="Semantic Scholar API key"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        required=True,
        help="Path to the directory containing query results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output CSV file to save processed data",
    )
    return parser.parse_args()


def clean_corpus_id(corpus_id):
    corpus_id_str = str(corpus_id)
    cleaned_id = re.sub(r"\D", "", corpus_id_str)

    if cleaned_id.isdigit():
        return cleaned_id
    else:
        return None


def fetch_paper_data(corpus_id, api_key, max_retries=5):
    cleaned_corpus_id = clean_corpus_id(corpus_id)

    if cleaned_corpus_id is None:
        return None

    url = f"https://api.semanticscholar.org/graph/v1/paper/CorpusId:{cleaned_corpus_id}"
    params = {
        "fields": "paperId,url,title,abstract,year,citationCount,isOpenAccess,fieldsOfStudy,publicationDate,journal"
    }
    headers = {"x-api-key": api_key}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limit exceeded, sleeping for 60 seconds...")
                sleep(60)
                continue
            else:
                print(f"Error {response.status_code} for CorpusId {cleaned_corpus_id}")
                return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            sleep(10)  # Wait before retrying
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            sleep(10)
        except Exception as e:
            print(f"Error {e} occured")
            sleep(20)

    print(
        f"Failed to retrieve paper data after {max_retries} attempts for CorpusId {cleaned_corpus_id}"
    )
    return None


def process_results_directory(
    directory_path, api_key, output_file, batch_start, batch_end
):
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["abstract_index", "combined_full_specter_papers"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for query_num in tqdm(
            range(batch_start, batch_end + 1), desc="Processing Queries"
        ):
            filename = f"query_{query_num}_results.json"
            file_path = os.path.join(directory_path, filename)

            if not os.path.exists(file_path):
                writer.writerow(
                    {
                        "abstract_index": query_num,
                        "combined_full_specter_papers": np.nan,
                    }
                )
                continue

            with open(file_path, "r") as file:
                results = json.load(file)

            full_specter_papers = []
            for index, result in enumerate(
                tqdm(
                    results, desc=f"Fetching Papers for Query {query_num}", leave=False
                )
            ):
                corpus_id = result["corpus_id"]
                paper_data = fetch_paper_data(corpus_id, api_key)

                if paper_data:
                    abstract = paper_data.get("abstract")
                    if abstract is None:
                        paper_info = (
                            f"New Order: {index + 1} - "
                            f"Original Order: {index + 1} - "
                            f"ID: {np.nan} - "
                            f"Title: {np.nan} - "
                            f"Abstract: {np.nan} - "
                            f"Citation Count: {np.nan}"
                        )
                    else:
                        paper_info = (
                            f"New Order: {index + 1} - "
                            f"Original Order: {index + 1} - "
                            f"ID: {paper_data.get('paperId', np.nan)} - "
                            f"Title: {paper_data.get('title', np.nan)} - "
                            f"Abstract: {abstract} - "
                            f"Citation Count: {paper_data.get('citationCount', np.nan)}"
                        )
                    full_specter_papers.append(paper_info)
                else:
                    paper_info = (
                        f"New Order: {index + 1} - "
                        f"Original Order: {index + 1} - "
                        f"ID: {np.nan} - "
                        f"Title: {np.nan} - "
                        f"Abstract: {np.nan} - "
                        f"Citation Count: {np.nan}"
                    )
                    full_specter_papers.append(paper_info)

            combined_info = (
                "; ".join(full_specter_papers) if full_specter_papers else np.nan
            )
            writer.writerow(
                {
                    "abstract_index": query_num,
                    "combined_full_specter_papers": combined_info,
                }
            )

    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()

    process_results_directory(
        directory_path=args.directory_path,
        api_key=args.api_key,
        output_file=args.output_file,
        batch_start=args.batch_start,
        batch_end=args.batch_end,
    )
