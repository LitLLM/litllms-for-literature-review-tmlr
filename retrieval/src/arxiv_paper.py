import arxiv
import pandas as pd
import argparse
import os
import logging
from tqdm import tqdm
from datetime import datetime
import re
import utils
from collections import namedtuple

class ArxivWrapper:
    """
    A wrapper for the arXiv API.

    Attributes:
        query (str): The query string for the search.
        max_results (int): The maximum number of results to return.
        output_file (str): The name of the output file.

    """

    def __init__(self, query, max_results, output_file):
        """
        The constructor for ArxivWrapper class.

        Parameters:
            query (str): The query string for the search.
            max_results (int): The maximum number of results to return.
            output_file (str): The name of the output file.
        """
        self.query = query
        self.max_results = max_results
        self.output_file = output_file

    def search_papers(self):
        """
        Search for papers using the arXiv API and return a DataFrame.

        Returns:
            df (DataFrame): A DataFrame containing the search results.
        """
        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        df_raw = pd.DataFrame()

        pbar = tqdm(total=self.max_results)  # Initialize the progress bar

        for result in search.results():
            entry_id = result.entry_id
            uid = entry_id.split(".")[-1]
            title = result.title
            date_published = result.published
            abstract = result.summary

            # perform keyword based openalex search with this paper and ensure it has at least 100 search results
            llm_config = namedtuple("llm_config", ["gen_engine", "max_tokens", "temperature", "n_keywords"])
            llm_config.gen_engine = "gpt-4o-mini"
            llm_config.max_tokens = 300
            llm_config.temperature = 0.2
            llm_config.n_keywords = 3
            keywords = utils.get_keywords_for_s2_search(abstract, llm_config)
            num_results = 0
            for query in keywords['queries']:
                # use openalex as s2 requires a key
                openalex_results = utils.get_openalex_candidates(query, 100)
                num_results += len(openalex_results)
                if num_results >= 100:
                    break
            if num_results < 100:
                continue
            result_dict = {
                "uid": uid,
                "title": title,
                "date_published": date_published,
                "abstract": abstract,
            }

            df_raw = df_raw._append(result_dict, ignore_index=True)

            pbar.update(1)  # Update the progress bar

        pbar.close()  # Close the progress bar

        df = df_raw.copy()
        df["id"] = df.apply(self.create_id, axis=1)
        print(f"Total papers retrieved: {len(df)}")
        return df

    def create_id(self, row):
        """
        Create a new id for each row in the DataFrame.

        Parameters:
            row (Series): A row in the DataFrame.

        Returns:
            id (str): The new id.
        """
        yearmonth = row["date_published"].strftime("%y%m")
        uid = re.sub(
            "v\d+", "", row["uid"]
        )  # remove 'v' followed by any number from the uid
        return f"{yearmonth}.{uid}"

    def save_dataframe(self, df):
        """
        Apply the create_id function to the DataFrame and save it.

        Parameters:
            df (DataFrame): The DataFrame to save.
        """
        df["id"] = df.apply(self.create_id, axis=1)
        os.makedirs("dataset", exist_ok=True)
        output_path = os.path.join("dataset", self.output_file)
        df.to_csv(output_path, index=False)


def main(args):
    arxiv_wrapper = ArxivWrapper(args.query, args.max_results, args.output_file)
    df = arxiv_wrapper.search_papers()
    print(df.head())
    arxiv_wrapper.save_dataframe(df)


def get_args():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Search Arxiv for papers.")
    parser.add_argument(
        "--query", type=str, default="artificial intelligence", help="Query string"
    )
    parser.add_argument(
        "--max_results", type=int, default=2000, help="Maximum number of results"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output_lit_arxiv.csv",
        help="Output file name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
