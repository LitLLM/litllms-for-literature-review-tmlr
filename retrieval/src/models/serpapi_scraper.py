import pandas as pd
import argparse
import os
import logging
from tqdm import tqdm
from datetime import datetime
import re
from serpapi import GoogleSearch


class SerpAPIWrapper:
    """
    A wrapper for the OpenAlex API.

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
        self.serp_api_key = self.get_api_key()

    def get_api_key(self):
        """
        Get SERP API Key
        """
        SERP_API_KEY = os.environ["SERP_API_KEY"]
        if SERP_API_KEY is None:
            raise EnvironmentError("SERP API KEY is not defined.")
        # This sets globally serp api key
        GoogleSearch.SERP_API_KEY = SERP_API_KEY
        return SERP_API_KEY

    def search_papers(self, query, site_filter: str = "site:arxiv.org"):
        """
        Search for papers using the arXiv API and return a DataFrame.

        Returns:
            df (DataFrame): A DataFrame containing the search results.
        """

        final_query = f"{query} {site_filter}"

        search = GoogleSearch({"q": final_query, "api_key": self.serp_api_key})
        results = search.get_dict()
        print(results.keys())
        organic_results = results["organic_results"]
        print(f"Total results: {len(organic_results)}")
        for result in organic_results[:2]:
            print(result["link"])
            print(result)

        return


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Search Arxiv for papers.")
    parser.add_argument(
        "--query", type=str, default="transformers", help="Query string"
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

    wrapper = SerpAPIWrapper(args.query, args.max_results, args.output_file)
    df = wrapper.search_papers(args.query)
    print(df.head())
