"""
Usage:
python litrank/paper_manager.py --n_candidates 100 --n_queries 500 --n_keywords 3 --search_engine specter --gen_engine gpt-4o-mini --rerank_method llm+embedding --temperature 0.2 --max_tokens 10000
"""

import os
import re
import json
import utils
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from faiss_exp import s2_data


from collections import namedtuple

# set environment variables
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["DATASETS_BASE_DIR"] = "./dataset"
logging.basicConfig(filename="rerank_log.txt", level=logging.INFO)


from models.reranker import Reranker
from arxiv_paper import main as arxiv_main


class PaperManager:
    """
    A class to manage the retrieval and reranking of papers
    """

    def __init__(self, config):
        """
        Initializes the PaperManager class
        config: A namedtuple containing the configuration parameters
        """
        self.config = config
        self.paper_index = 0
        self.output_dir = f"f-k={config.n_candidates}-n={config.n_queries}/{config.search_engine}/seed={config.seed}"
        self.config.n_keywords = (
            config.n_keywords if config.search_engine == "s2" else 1
        )
        self.extracted_s2_queries = []
        self.s2_query_reasoning = []
        self.query_reasoning_verification = []
        self.specter_index_path = os.getenv("SPECTER_INDEX_PATH")

        self.output_dir = os.path.join(os.getenv("DATASETS_BASE_DIR"), self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Model name: {self.config.gen_engine}")

        if self.config.rerank_method == "embedding":
            self.emb_reranker = Reranker(config)

    def get_candidates(self, query_paper_info, query_index):
        if self.config.search_engine == "s2":
            return self.get_s2_candidates(query_paper_info, query_index)
        elif self.config.search_engine == "specter":
            return self.get_specter_candidates(query_paper_info, query_index)
        elif self.config.search_engine == "openalex":
            return self.get_openalex_candidates(query_paper_info, query_index)
        elif self.config.search_engine == "s2+specter":
            s2_candidates = self.get_s2_candidates(query_paper_info, query_index)
            specter_candidates = self.get_specter_candidates(
                query_paper_info, query_index
            )
            # combine the two lists and do a quick reranking on top of it
            candidates = s2_candidates + specter_candidates
            org_reranking_type = self.config.reranking_prompt_type
            org_max_tokens = self.config.max_tokens
            self.config.reranking_prompt_type = "basic_ranking"
            self.config.max_tokens = 2000
            reranked_candidates = self.rerank_candidates(
                query_paper_info["abstract"], candidates, save_spreadsheet=False
            )
            self.config.reranking_prompt_type = org_reranking_type
            self.config.max_tokens = org_max_tokens
            return reranked_candidates[: self.config.n_candidates]

    def get_specter_candidates(self, query_paper_info, query_index):
        """
        This function takes in an abstract, generates keywords for that abstract and
        then fetches the top k papers from the specter embeddings index
        """
        print("Fetching candidates from the Specter index...")
        try:
            cache_path = os.path.join(
                self.specter_index_path, f"query_{query_index}_results.json"
            )
            results = json.load(open(cache_path))
            results = results[: self.config.n_candidates]
            candidates = []
            for result in tqdm(
                results,
                desc=f"Fetching Candidates for query {query_index}",
                leave=False,
            ):
                corpus_id = result["corpus_id"]
                candidates.append(
                    s2_data.fetch_paper_data(
                        corpus_id, api_key=os.getenv("S2_API_KEY")
                    ),
                )
            candidates = [
                {
                    "paper_id": paper.get("paperId", "Unknown"),
                    "title_paper": paper.get("title", "No title"),
                    "abstract": paper.get("abstract", "Abstract not available"),
                    "citation_count": paper.get("citationCount", "Unknown"),
                    "publication_date": paper.get("publicationDate", "Unknown"),
                    "external_ids": paper.get("externalIDs", []),
                }
                for paper in candidates
                if paper
            ]
            # save processed papers
            utils.save_processed_papers(
                papers=candidates,
                output_dir=self.output_dir,
                paper_index=query_index,
                prompt_num=1,
            )
            return candidates
        except FileNotFoundError as e:
            print(f"Error in reading Specter index: {e}")
            return []

    def get_openalex_candidates(self, query_paper_info, query_index):
        """
        This function takes in an abstract, generates keywords for that abstract and
        then fetches the top k papers from the openalex embeddings index
        """
        abstract = query_paper_info["abstract"]
        published_date = query_paper_info["publicationDate"]

        query = query_paper_info["title"]
        papers = utils.get_openalex_candidates(query, self.config.n_candidates)
        utils.save_processed_papers(
            papers=papers,
            output_dir=self.output_dir,
            paper_index=query_index,
            prompt_num=1,
        )
        return papers

    def get_s2_candidates(self, query_paper_info, query_index):
        """
        This function takes in an abstract, generates keywords for that abstract and
        then queries the Semantic Scholar API to get a list of papers
        for each keyword generated.
        """
        abstract = query_paper_info["abstract"]
        published_date = query_paper_info["publicationDate"]
        papers = []
        # Step 1: Get keywords for the abstract
        query_data = utils.get_keywords_for_s2_search(abstract, self.config)
        self.extracted_s2_queries = queries = (
            query_data["queries"] if query_data else None
        )
        self.s2_query_reasoning = query_data.get(
            "reasoning", ["No reasoning available"]
        )
        if not queries:
            print("No queries generated for the abstract")
            return []

        # Step 1.1: verify that the extracted sentences in the reasoniings
        # are present in the abstract
        self.query_reasoning_verification = utils.check_extractive(
            reasonings=self.s2_query_reasoning, source_text=abstract
        )

        # Step 2: Fetch candidates for each query and save
        all_processed_papers = []
        for i, query in enumerate(queries):
            print(f"Query {i+1}: {query}")
            papers = utils.fetch_candidates_s2(
                query=query,
                query_publication_date=published_date,
                max_retries=3,
                n_candidates=self.config.n_candidates,
            )
            if not papers:
                print(f"No papers found for query {query}")
                continue
            all_processed_papers.extend(papers)

            # save processed papers
            utils.save_processed_papers(
                papers=papers,
                output_dir=self.output_dir,
                paper_index=query_index,
                prompt_num=i + 1,
            )
        self.paper_index += 1  # Increment paper index
        return all_processed_papers

    def rerank_candidates(self, abstract, papers, save_spreadsheet=True):
        """
        This function takes in an abstract and a list of candidate papers
        and reranks the list of candidate papers
        """
        print(f"Reranking {len(papers)} papers based on the abstract.")
        # print temperature
        print(f"Temperature: {self.config.temperature}")
        max_retries = 3
        for _ in range(max_retries):
            ordered_papers = self.attempt_reranking(abstract, papers, save_spreadsheet)
            if ordered_papers:
                return ordered_papers

    def attempt_reranking(self, abstract, papers, save_spreadsheet=True):
        # accumulated reasons for reranking
        ranking, reason, arxiv_ids = utils.rerank_candidates_batch(
            abstract, papers, self.config
        )

        # Extract new order from response.
        extracted_order = [
            int(s)
            for s in re.findall(r"\d+", ranking)
            if 1 <= int(s) <= self.config.n_candidates
        ]
        print(f"Extracted new order: {extracted_order}")
        print(f"Extracted reason: {reason}")

        # verify that the extracted sentences in the reasoniings
        # are present in the referred paper
        self.ranking_reasoning_verification = []
        # TODO: remove this hack by restricting the model
        # in the prompt to generate reasoning in a specific format
        delimiter = (
            "\n" if len(reason.split("\n")) > len(reason.split("\n\n")) else "\n\n"
        )
        for idx, reasoning in enumerate(reason.split(delimiter)):
            try:
                # extract sentence within quotes
                extracted_sentence = re.search(r'"(.*?)"', reasoning)
                if not extracted_sentence:
                    # try with single quotes
                    extracted_sentence = re.search(r"'(.*?)'", reasoning)
                extracted_sentence = extracted_sentence.group(1).strip()
                self.ranking_reasoning_verification.append(
                    extracted_sentence in papers[extracted_order[idx] - 1]["abstract"]
                )
            except Exception as e:
                logging.error(f"Error in extracting reasoning verification: {e}")
                self.ranking_reasoning_verification.append(False)

        # export the ranking and reason to an excel file
        if save_spreadsheet:
            utils.save_spreadsheet(
                abstract,
                papers,
                ranking,
                reason,
                arxiv_ids,
                self.ranking_reasoning_verification,
                self.extracted_s2_queries,
                self.s2_query_reasoning,
                self.query_reasoning_verification,
                self.output_dir,
                self.config,
            )
        # sanitize the extracted order (remove duplicates and out of range values, etc.)
        final_new_order = utils.sanitize_paper_order(extracted_order, len(papers))

        # Reorder papers based on the final new order
        ordered_papers = [papers[i - 1] for i in final_new_order if i <= len(papers)]

        # Update paper details with new order information
        for i, paper in enumerate(ordered_papers, start=1):
            paper.update(
                {
                    "original_order": papers.index(paper) + 1,
                    "new_order": i,  # New order based on sorted position.
                    "title_paper": paper.get("title_paper", "No title"),
                    "paper_id": paper.get("paper_id", "Unknown"),
                    "abstract": paper.get("abstract", "Abstract not available"),
                    "citations": paper.get("citationCount", 0),
                    "external_ids": paper.get("externalIds", []),
                }
            )
        return ordered_papers  # Return the result if the API request was successful

    def format_references(self, ref):
        """
        This function takes in a string of references and returns a formatted string of references
        """
        if isinstance(ref, list):
            if not ref or pd.isna(ref).all():
                return "Invalid reference entry"
        elif pd.isna(ref):
            return "Invalid reference entry"

        formatted_refs = []
        for item in ref:
            if isinstance(item, dict):
                paper_id = item.get("paperId", "Unknown")
                title = item.get("title", "No title")
                formatted_refs.append(f"{paper_id}: {title}")
            else:
                return "Invalid reference entry"

        return "; ".join(formatted_refs)

    def get_ground_truth(self, df, output_file_name):
        """
        This function takes in a df and returns a list of ground truth papers
        """
        print("Getting ground truth...")
        df_all_paper_data = pd.DataFrame()
        for index, row in df.iterrows():
            paper_data = utils.get_paper_data(row["Id"])
            # print(paper_data)
            df_paper_data = pd.DataFrame([paper_data])
            df_all_paper_data = pd.concat(
                [df_all_paper_data, df_paper_data], ignore_index=True
            )

        print(df_all_paper_data.head())
        df_all_paper_data["cited_references"] = df_all_paper_data["references"].apply(
            self.format_references
        )
        output_path = os.path.join(
            os.getenv("DATASETS_BASE_DIR"), "ground_truth", output_file_name
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_all_paper_data.to_csv(output_path, index=False)
        return df_all_paper_data

    def get_citation_count(self, paper_id):
        """
        This function takes in a paper id and returns the citation count
        """
        try:
            paper_data = utils.get_paper_data(paper_id)
            citation_count = paper_data.get("citationCount", "Unknown")
            return citation_count
        except ValueError as e:
            print(f"An error occurred in get_citation_count: {e}")
            return None

    def update_paper_index_based_on_existing_directories(self, base_file_path):
        max_index = -1
        pattern = re.compile(r"processing_paper(\d+)")
        for item in os.listdir(base_file_path):
            match = pattern.match(item)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)
        self.paper_index = max_index + 1 if max_index >= 0 else 0

    def fetch_candidates_and_rerank(
        self, search_space_df, base_file_path=None, num_rows=None
    ):
        aggregated_candidate_papers = []
        aggregated_reranked_papers = []
        base_file_path = base_file_path or self.output_dir
        # self.update_paper_index_based_on_existing_directories(base_file_path)

        for index, row in search_space_df.iterrows():
            file_path = os.path.join(
                base_file_path, f"processing_paper{index}", "combined.csv"
            )
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"Found existing file: {file_path}")
                candidate_papers = self.read_combined_csv(file_path, num_rows)
            else:
                print(f"Processing query paper {index+1} of {len(search_space_df)}")
                print("No of KW queries", self.config.n_keywords)
                try:
                    candidate_papers = self.get_candidates(row, query_index=index)
                    print(
                        f"Retrieved {len(candidate_papers)} papers for paper {index+1}"
                    )
                except Exception as e:
                    print(f"Error in fetching candidates for paper {index+1}: {e}")
                    continue

                # if self.config.n_keywords != 1:
                self.combine_csv_files(
                    index,
                    num_files=self.config.n_keywords,
                    total_rows=self.config.n_candidates,
                    base_file_path=base_file_path,
                )
                candidate_papers = self.read_combined_csv(file_path, num_rows)
                # self.combine_csv_files(index, num_files=self.config.n_keywords, total_rows=self.config.n_candidates, base_file_path=base_file_path)
                # candidate_papers = self.read_combined_csv(file_path, num_rows)

            # Check if the file exists
            file_exists_cited = f"{self.output_dir}/cited_papers_aggregated-n={config.n_queries}-k={config.n_candidates}-model={config.gen_engine}-search_engine={config.search_engine}-rerank_prompt_type={config.reranking_prompt_type}.csv"

            file_exists_reranked = f"{self.output_dir}/reranked_papers_aggregated-n={config.n_queries}-k={config.n_candidates}-model={config.gen_engine}-search_engine={config.search_engine}-rerank_prompt_type={config.reranking_prompt_type}.csv"

            if os.path.exists(file_exists_cited) and (
                (index + 1)
                in pd.read_csv(file_exists_cited)["abstract_index"].values.tolist()
            ):
                print(f"Abstarct {index+1} already processed. Skipping...")
                df_candidate_papers_aggregated = pd.read_csv(file_exists_cited)
                df_reranked_papers_aggregated = pd.read_csv(file_exists_reranked)
                continue

            self.process_and_aggregate_papers(
                index,
                row["abstract"],
                candidate_papers,
                aggregated_candidate_papers,
                aggregated_reranked_papers,
            )
            df_candidate_papers_aggregated = pd.DataFrame(aggregated_candidate_papers)
            df_reranked_papers_aggregated = pd.DataFrame(aggregated_reranked_papers)
            utils.save_cited_and_aggregated_papers(
                df_candidate_papers_aggregated,
                df_reranked_papers_aggregated,
                self.config,
                self.output_dir,
            )
        return df_candidate_papers_aggregated, df_reranked_papers_aggregated

    def read_combined_csv(self, file_path, num_rows=None):
        try:
            candidate_papers = pd.read_csv(file_path, index_col=False, nrows=num_rows)
            return candidate_papers.to_dict("records")
        except pd.errors.EmptyDataError:
            print(f"File {file_path} is empty or contains no data.")
            return []

    def combine_csv_files(
        self, paper_index, num_files=3, total_rows=100, base_file_path=None
    ):
        """
        This function takes in a paper index, the number of files, and the total number of rows, and combines the CSV files.
        """
        base_file_path = base_file_path or self.output_dir
        columns = [
            "paper_id",
            "title_paper",
            "abstract",
            "citation_count",
            "external_ids",
        ]
        rows_per_file = total_rows // num_files
        dfs = []
        unique_ids = set()
        total_rows_collected = 0

        for j in range(1, num_files + 1):
            file_path = os.path.join(
                self.output_dir,
                f"processing_paper{paper_index}",
                f"processing_prompt{j}.csv",
            )
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    df = pd.read_csv(file_path)
                    rows_to_read = min(rows_per_file, total_rows - total_rows_collected)
                    df = df[~df["paper_id"].isin(unique_ids)]
                    df = df.dropna(subset=["abstract"])
                    df = df.head(rows_to_read)
                    unique_ids.update(df["paper_id"])
                    total_rows_collected += len(df)
                    dfs.append(df)
                except pd.errors.EmptyDataError:
                    print(f"File {file_path} is empty or contains no data.")

        if dfs:
            df_all = pd.concat(dfs)
            df_all.drop_duplicates(subset="paper_id", keep="first", inplace=True)

            if len(df_all) < total_rows:
                for j in range(1, num_files + 1):
                    if total_rows_collected >= total_rows:
                        break
                    file_path = os.path.join(
                        self.output_dir,
                        f"processing_paper{paper_index}",
                        f"processing_prompt{j}.csv",
                    )
                    additional_df = pd.read_csv(file_path)
                    additional_df = additional_df[
                        ~additional_df["paper_id"].isin(unique_ids)
                    ]
                    additional_df = additional_df.dropna(subset=["abstract"])
                    rows_needed = total_rows - total_rows_collected
                    additional_df = additional_df.head(rows_needed)
                    unique_ids.update(additional_df["paper_id"])
                    total_rows_collected += len(additional_df)
                    dfs.append(additional_df)

                df_all = pd.concat(dfs)
                df_all.drop_duplicates(subset="paper_id", keep="first", inplace=True)
        else:
            df_all = pd.DataFrame(columns=columns)

        print(f"Saving combined CSV for paper {paper_index}")
        out_dir = os.path.join(self.output_dir, f"processing_paper{paper_index}")
        os.makedirs(out_dir, exist_ok=True)
        df_all.to_csv(os.path.join(out_dir, "combined.csv"), index=False)

    def process_and_aggregate_papers(
        self,
        index,
        abstract,
        candidate_papers,
        aggregated_candidate_papers,
        aggregated_reranked_papers,
    ):
        # print(f"Processing and aggregating papers for abstract {abstract}")
        if self.config.rerank_method == "llm":
            rerank_papers = self.rerank_candidates(abstract, candidate_papers)
        elif self.config.rerank_method == "embedding":
            rerank_papers = self.emb_reranker.rerank_papers(abstract, candidate_papers)
        else:
            rerank_papers = None

        if rerank_papers is None:
            return
        combined_info = "; ".join(
            [
                f"{paper.get('paper_id', 'Unknown ID')}: {paper.get('title_paper', 'No Title')} - Abstract: {paper.get('abstract', 'No abstract available')}"
                for paper in candidate_papers
            ]
        )

        combined_reranked_info = "; ".join(
            [
                f"New Order: {paper['new_order']} - Original Order: {paper['original_order']} - ID: {paper.get('paper_id', 'Unknown ID')} - Title: {paper.get('title_paper', 'No title')} - Abstract: {paper.get('abstract', 'No abstract available')} - Citation Count: {paper.get('citation_count', 'Unknown')} - External IDs: {paper.get('external_ids', [])}"
                for paper in rerank_papers
            ]
        )

        aggregated_candidate_papers.append(
            {"abstract_index": index + 1, "combined_candidate_papers": combined_info}
        )
        aggregated_reranked_papers.append(
            {
                "abstract_index": index + 1,
                "combined_reranked_papers": combined_reranked_info,
            }
        )

    def recommend_papers(self, df):
        # df = pd.read_csv("./litrank/dataset/arxiv_papers/rw_2308.csv")
        all_recommendations = []
        aggregated_candidate_papers = []
        aggregated_reranked_papers = []
        for id_counter, arxiv_id in enumerate(df["Id"]):
            processed_papers = self.process_recommended_papers(
                arxiv_id, self.output_dir, id_counter
            )
            print(processed_papers)
            all_recommendations.append(processed_papers)
            print(
                f"Retrieved and processed {len(processed_papers)} recommendations for paper {arxiv_id}"
            )
            # Check if there are any recommended papers
            if processed_papers:
                # Extract the abstract from the first recommended paper
                abstract = processed_papers[0]["abstract"]
                # Call process_and_aggregate_papers with the necessary arguments
                self.process_and_aggregate_papers(
                    id_counter,
                    abstract,
                    processed_papers,
                    aggregated_candidate_papers,
                    aggregated_reranked_papers,
                )
            else:
                print(f"No recommendations found for paper {arxiv_id}")
        df_candidate_papers_aggregated = pd.DataFrame(aggregated_candidate_papers)
        df_reranked_papers_aggregated = pd.DataFrame(aggregated_reranked_papers)
        print(df_candidate_papers_aggregated.head())
        print(df_reranked_papers_aggregated.head())

        # save the aggregated dataframes to CSV files
        df_candidate_papers_aggregated.to_csv(
            f"{self.output_dir}/candidate_papers_aggregated_rec1.csv", index=False
        )
        df_reranked_papers_aggregated.to_csv(
            f"{self.output_dir}/reranked_papers_aggregated_rec1.csv", index=False
        )

    def process_recommended_papers(self, arxiv_id, output_dir, id_counter):
        papers = utils.get_recommendations_from_s2(
            arxiv_id, n_candidates=self.config.n_candidates
        )
        # save recommeded papers
        utils.save_recommended_papers(papers, output_dir, arxiv_id, id_counter)
        return papers

    def main(self):
        utils.set_seed(self.config.seed)
        gt_file_name = f"ground_truth_latest_500_final_sqmq.csv"
        # gt_file_name = f"ground_truth_latest_{self.config.n_queries}.csv"
        gt_path = os.path.join(
            os.getenv("DATASETS_BASE_DIR"), "ground_truth", gt_file_name
        )
        # Define the named tuple
        Args = namedtuple("Args", ["query", "max_results", "output_file"])
        arxiv_args = Args(
            query="artificial intelligence",
            max_results=500,
            output_file="output_lit_arxiv500.csv",
        )

        kb_path = os.path.join("./dataset", "arxiv_papers", arxiv_args.output_file)
        print(f"kb_path: {kb_path}")

        if not os.path.exists(kb_path):
            print("Creating kb_path file...")
            os.makedirs(os.path.dirname(kb_path), exist_ok=True)
            # this will create that kb_path file
            arxiv_main(arxiv_args)

        print(f"Reading {kb_path}...")
        df = pd.read_csv(kb_path)
        df = df.rename(columns={"id": "Id"}).drop_duplicates(subset="Id")
        print(df.shape)
        df = df.iloc[: self.config.n_queries]
        print(df.shape)

        # fetch/create GT file (gt citations)
        if os.path.exists(kb_path) and os.path.exists(gt_path):
            search_space_df = utils.read_csv(gt_path, self.config.n_queries)
        else:
            search_space_df = self.get_ground_truth(df, gt_file_name)
            search_space_df.to_csv(gt_path, index=False)

        # use direct s2 recommendataion or our approach
        if self.config.recommend:
            self.recommend_papers(df)
        else:
            df_candidate_papers_aggregated, df_reranked_papers_aggregated = (
                self.fetch_candidates_and_rerank(
                    search_space_df,
                    base_file_path=self.output_dir,
                    num_rows=self.config.n_candidates,
                )
            )

            print(df_candidate_papers_aggregated.head())
            print(df_reranked_papers_aggregated.head())


def parse_args() -> argparse.Namespace:
    """
    Parse cli arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--n_candidates",
        type=int,
        default=100,
        help="No. of papers to suggest for every query abstract",
    )
    parser.add_argument(
        "-n",
        "--n_queries",
        type=int,
        default=500,
        help="No. of query papers for which to suggest related works for",
    )

    parser.add_argument(
        "-q",
        "--n_keywords",
        type=int,
        default=3,
        help="No. of queries to generate for S2 by summarizing every query paper",
    )
    parser.add_argument(
        "-r", "--recommend", action="store_true", help="run the recommendation function"
    )
    parser.add_argument(
        "-e",
        "--skip_extractive_check",
        action="store_true",
        help="run the recommendation function",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="truncate search space (useful for testing)",
    )
    parser.add_argument(
        "-m",
        "--gen_engine",
        default="llama-3.1-405b",  # also supports openAI models like gpt-4o-mini
        help="LLM backbone to use",
    )
    parser.add_argument(
        "--rerank_method",
        default="llm",  # options: llm, embedding, llm+embedding
        help="Method to use for reranking (LLM/Embedding/LLM+Embedding)",
    )

    parser.add_argument(
        "-s",
        "--search_engine",
        default="s2",  # options: s2, specter, s2+specter
        help="Where to fetch the candidates for a query paper",
    )

    parser.add_argument(
        "--reranking_prompt_type",
        default="debate_ranking_abstract",  # options: debate_ranking, reasoned_ranking
        help="the type of prompt to use for reranking",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for LLM models",
    )

    parser.add_argument(
        "--max_tokens", type=int, default=10000, help="Max tokens for LLM"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument("--use_pdf", action="store_true", help="Use PDFs for reranking")
    parser.add_argument(
        "--use_full_text", action="store_true", help="Use full text for reranking"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    manager = PaperManager(config)
    manager.main()
