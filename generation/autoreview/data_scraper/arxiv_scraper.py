#!/usr/bin/env python3
# Copyright (c) ServiceNow Inc. and its affiliates.

"""
This file defines Arxiv scraper
"""

import argparse
from typing import Any, Dict
import logging
import pathlib
import json
import pandas as pd
from os import path
from tqdm import tqdm
import arxiv
from autoreview.data_scraper.utils import write_excel_df


class ArxivScraper:
    """
    This class allows to scrape Arxiv for related papers
    """

    def __init__(self, config):
        self.name = "arxiv_scraper"
        self.config = config

    def scrape_arxiv(self, query) -> Any:
        search = arxiv.Search(
            query=f"\"{query}\"",
            max_results=config.max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        response_data = []
        for result in tqdm(search.results()):
            json_line = {
                "title": result.title,
                "summary": result.summary,
                "primary_category": result.primary_category,
                "entry_id": result.entry_id,
                "authors": result.authors,
                "pdf_url": result.pdf_url
            }
            # "published": result.published,

            response_data.append(json_line)
        print(f"Total number of responses are: {len(response_data)}")
        response_df = pd.DataFrame(response_data)

        # for _, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        #    do something
        cur_dir = pathlib.Path(__file__).parent.resolve()
        xls_file_path = path.join(cur_dir, f'results.xlsx')
        write_excel_df(df_list=[response_df], sheet_name_list=["test"],
                       save_file_path=xls_file_path, close_writer=True)

        return


def parse_args() -> argparse.Namespace:
    """
    Parse cli arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topic", default="Multimodal dialog", help="Paper topic")
    parser.add_argument("-r", "--max_results", default=10, help="Maximum results")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    logging.basicConfig(level=logging.INFO)
    scraper = ArxivScraper(config)
    scraper.scrape_arxiv(query=config.topic)
