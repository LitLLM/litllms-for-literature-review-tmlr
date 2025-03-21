"""
This module contains utility functions for interacting with the arXiv and semantic scholar (S2) API.
"""

import re
import os
import time
import requests
from requests.exceptions import ConnectionError, Timeout

from .file_utils import extract_tex_files, find_tex_files, read_tex_files

S2_API_KEY = os.getenv("S2_API_KEY")


def get_arxiv_id_by_title_or_abstract(
    search_text: str, search_type: str = "title"
) -> str:
    base_url = "http://export.arxiv.org/api/query"

    # Determine the search query based on the search type
    if search_type == "title":
        query = f"search_query=ti:{search_text}&start=0&max_results=1"
    elif search_type == "abstract":
        query = f"search_query=abs:{search_text}&start=0&max_results=1"
    else:
        return "Error: Invalid search type. Use 'title' or 'abstract'."
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}?{query}")

            # Check for successful status code
            if response.status_code == 200:
                break

            # Handle non-200 responses (server-side errors, etc.)
            else:
                print(f"Error: Received status code {response.status_code}")

        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")

            # Exponential backoff strategy
            wait_time = (2**attempt) + 1  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                return None

    # Use re to extract the Arxiv ID. It is enclosed inside <id> tags and starts with 'http://arxiv.org/abs/'
    arxiv_id = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", response.text)
    arxiv_id = arxiv_id.group(1) if arxiv_id else None
    return arxiv_id


def download_arxiv_source(paper_id):
    # Construct the URL for the source files
    url = f"https://arxiv.org/e-print/{paper_id}"
    response = requests.get(url)

    if response.status_code == 200:
        tar_filename = f"{paper_id}.tar.gz"
        with open(tar_filename, "wb") as file:
            file.write(response.content)
        # print(f"Downloaded source files: {tar_filename}")
        return tar_filename
    else:
        print(f"Failed to download source files: {response.status_code}")
        return None


def get_paper_data(id=None, paper_url=None):
    """
    Retrieves data of one paper based on ID or URL.
    :param id: Paper ID (e.g., 'ARXIV:2106.15928')
    :param paper_url: URL of the paper (optional if ID is provided)
    :return: JSON response with paper data
    """
    if id:
        formatted_id = f"ARXIV:{id}"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{formatted_id}"
    elif paper_url:
        url = f"https://api.semanticscholar.org/graph/v1/paper/URL:{paper_url}"
    else:
        return "Error: Either 'id' or 'paper_url' must be provided"
    fields = "title,url,abstract,citationCount,journal,isOpenAccess,fieldsOfStudy,year,publicationDate,journal,references"
    rsp = requests.get(
        url, headers={"X-API-KEY": S2_API_KEY}, params={"fields": fields}
    )
    rsp = rsp.json()
    # add full txt if it exists
    fname = f"./dataset/arxiv_papers_full_text/{id}.txt"
    if os.path.exists(fname):
        with open(fname, "r") as f:
            rsp["full_text"] = f.read()
    else:
        rsp["full_text"] = "Full text not available"
    return rsp


def read_or_download_text(arxiv_id):
    if os.path.exists(f"./dataset/arxiv_papers_s2_full/{arxiv_id}.txt"):
        print(f"{arxiv_id} already exists. Reading the file...")
        full_text = open(
            f"./dataset/arxiv_papers_s2_full/{arxiv_id}.txt",
            "r",
        ).read()
    else:
        print(f"{arxiv_id} doesn't already exist. Downloading full text...")
        # download the paper
        tar_filename = download_arxiv_source(arxiv_id)
        if tar_filename:
            extracted_dir = extract_tex_files(tar_filename)
            if extracted_dir:
                tex_files = find_tex_files(extracted_dir)
                if tex_files:
                    full_text = read_tex_files(tex_files)
                    # remove the extracted files
                    os.system(f"rm -rf {extracted_dir}")
                    os.system(f"rm {tar_filename}")

                    # write the full text to a file
                    with open(
                        f"./dataset/arxiv_papers_s2_full/{arxiv_id}.txt",
                        "w",
                        encoding="utf-8",
                    ) as file:
                        file.write(full_text)
    return full_text


def download_pdf(arxiv_id):
    if not os.path.exists(f"dataset/arxiv_papers_pdf/{arxiv_id}.pdf"):
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        os.system(f"curl -o ./dataset/arxiv_papers_pdf/{arxiv_id}.pdf {url}")
        # make sure the file is not corrupted
        import PyPDF2

        try:
            with open(
                f"./dataset/arxiv_papers_pdf/{arxiv_id}.pdf",
                "rb",
            ) as pdf_file:
                PyPDF2.PdfReader(pdf_file)
        except Exception as e:
            print(f"Error in reading PDF: {e}")
            os.system(f"rm ./dataset/arxiv_papers_pdf/{arxiv_id}.pdf")
            return None
    return f"{arxiv_id}.pdf"


def process_abstract_inverted_index(abstract_inverted_index):
    """
    Convert the abstract_inverted_index from OpenAlex to a readable abstract text.

    Parameters:
        abstract_inverted_index (dict): The abstract_inverted_index from OpenAlex

    Returns:
        str: The readable abstract text
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return "Abstract not available"

    # Create a list of words with their positions
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    # Sort by position
    word_positions.sort()

    # Join words to form the abstract
    abstract = " ".join(word for _, word in word_positions)

    return abstract


def get_openalex_candidates(
    query=None, n_candidates=10, query_publication_date="2024", max_retries=3
):
    fields = "title,url,abstract,citationCount,journal,isOpenAccess,fieldsOfStudy,year,publicationDate,journal,references"
    for attempt in range(max_retries):
        try:
            rsp = query_openalex_paper_search_api(
                query, n_candidates, fields, query_publication_date
            )
            if rsp is None:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1) - 1
                    time.sleep(wait_time)
                    print(f"Request failed. Retrying...")
                    continue
                else:
                    print(f"Request failed. No more retries.")
                    return []

            results = rsp.json()
            papers = results.get("results", [])
            papers = papers[:n_candidates]
            processed_papers = []
            for paper in papers:
                processed_papers.append(
                    {
                        "paper_id": paper.get("id", "Unknown"),
                        "title_paper": paper.get(
                            "display_name", paper.get("title", "No title")
                        ),
                        "abstract": process_abstract_inverted_index(
                            paper.get("abstract_inverted_index", {})
                        ),
                        "citation_count": paper.get("cited_by_count", 0),
                        "publication_date": paper.get("publication_date", "Unknown"),
                        "external_ids": paper.get("ids", {}),
                    }
                )
            return processed_papers
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1) - 1
                time.sleep(wait_time)
                print(f"Request failed with error {e}. Retrying...")
            else:
                print(f"Request failed with error {e}. No more retries.")
                return []


def sanitize_query(query: str) -> str:
    """
    Sanitize a paper title by removing problematic characters.

    Args:
        title: The title to sanitize

    Returns:
        Sanitized title
    """
    if not query:
        return None
    chars_to_remove = ".:'\",()!?"
    translation_table = str.maketrans("", "", chars_to_remove)
    return query.translate(translation_table)


def query_openalex_paper_search_api(
    query, n_candidates, fields, query_publication_date="2024"
):
    """
    query: query string to send to the OpenAlex search API
    n_candidates: number of papers to retrieve for the given query
    fields: fields to retrieve for each paper
    """
    query = sanitize_query(query)
    # Construct API URL with proper query parameters
    base_url = "https://api.openalex.org/works"
    headers = {}

    # Add polite pool identifier if email is provided
    headers["User-Agent"] = f"mailto:yourstruly7@duck.com"

    # Prepare query parameters
    params = {
        "search": query,
        "sort": "relevance_score:desc",  # Sort by relevance
        "per-page": n_candidates,
    }

    try:
        # Use exponential backoff for the API request
        def make_request():
            return requests.get(base_url, params=params, headers=headers)

        response = make_request()
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"Error in querying OpenAlex: {e}")
        return None


def fetch_candidates_s2(
    max_retries=3, n_candidates=10, query=None, query_publication_date="2024"
):
    fields = "title,url,abstract,citationCount,journal,isOpenAccess,fieldsOfStudy,year,publicationDate,journal,references"
    for attempt in range(max_retries):
        try:
            rsp = query_s2_paper_search_api(
                query, n_candidates, fields, query_publication_date
            )
            # rsp.raise_for_status()
            results = rsp.json()
            total = results["total"]
            if not total:
                print("No matches found. Please try another query.")
                return []
            print(f"Found {total} results. Showing up to {n_candidates}.")
            papers = results.get("data", [])
            papers = papers[:n_candidates]
            processed_papers = [
                {
                    "paper_id": paper.get("paperId", "Unknown"),
                    "title_paper": paper.get("title", "No title"),
                    "abstract": paper.get("abstract", "Abstract not available"),
                    "citation_count": paper.get("citationCount", "Unknown"),
                    "publication_date": paper.get("publicationDate", "Unknown"),
                    "external_ids": paper.get("externalIDs", []),
                }
                for paper in papers
            ]
            return processed_papers
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1) - 1
                time.sleep(wait_time)
                print(f"Request failed with error {e}. Retrying...")
            else:
                print(f"Request failed with error {e}. No more retries.")
                return []


def query_s2_paper_search_api(query, n_papers, fields, query_publication_date="2024"):
    """
    query: query string to send to the S2 search API
    n_papers: number of papers to retrieve for the given query
    fields: fields to retrieve for each paper
    """
    return requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": n_papers,
            "fields": fields,
            "publicationDateOrYear": f":{query_publication_date}",
        },
    )


def get_recommendations_from_s2(arxiv_id: str, n_candidates=20, max_retries=3):
    fields = "title,url,abstract,citationCount,journal,isOpenAccess,fieldsOfStudy,publicationDate,year,journal"
    query_id = f"ArXiv:{arxiv_id}"

    for _ in range(max_retries):
        try:
            print(f"Getting recommendations for paper {query_id}...")
            rsp = requests.post(
                "https://api.semanticscholar.org/recommendations/v1/papers/",
                json={
                    "positivePaperIds": [query_id],
                },
                params={"fields": fields, "limit": n_candidates},
            )
            results = rsp.json()
            papers = results["recommendedPapers"]

        except KeyError:
            print(
                f"No 'recommendedPapers' in the response for arxiv_id: {arxiv_id}. Retrying..."
            )
            time.sleep(2)  # wait for 2 seconds before retrying
            papers = []
    print(
        f"Failed to get 'recommendedPapers' for arxiv_id: {arxiv_id} after {max_retries} attempts."
    )
    if not papers:
        with open("failed_arxiv.txt", "a") as f:
            f.write(f"{arxiv_id}\n")
    processed_papers = [
        {
            "paper_id": paper.get("paperId", "Unknown"),
            "title_paper": paper.get("title", "No title"),
            "abstract": paper.get("abstract", "Abstract not available"),
            "citation_count": paper.get("citationCount", "Unknown"),
            "publication_date": paper.get("publicationDate", "Unknown"),
            "external_ids": paper.get("externalIds", []),
        }
        for paper in papers
    ]
    return processed_papers


def get_paper_details_s2(paper_id):
    # Define the fields you want to include in the response
    fields = "paperId,title,abstract,citationCount,references,citations,authors,publicationDate,fieldsOfStudy,externalIDs"
    url = f"https://api.semanticscholar.org/v1/paper/{paper_id}?fields={fields}"

    try:
        response = requests.get(url)
        # response.raise_for_status()  # Check for HTTP request errors
        return response.json()  # Parse and return the JSON response
    except Exception as e:
        print(f"Error fetching details for paper ID {paper_id}: {e}")
        return None  # Return None or an appropriate error response


def test_openalex_integration():
    """
    Test function to verify the OpenAlex integration.
    """
    print("Testing OpenAlex integration...")
    query = "machine learning"
    n_candidates = 5
    query_publication_date = "2023"

    papers = get_openalex_candidates(
        query=query,
        n_candidates=n_candidates,
        query_publication_date=query_publication_date,
    )

    if not papers:
        print("Error: No papers returned from OpenAlex.")
        return False

    print(f"Successfully retrieved {len(papers)} papers from OpenAlex.")
    print("\nSample paper:")
    sample_paper = papers[0]
    for key, value in sample_paper.items():
        if key == "abstract":
            print(f"{key}: {value[:200]}...")  # Print first 200 chars of abstract
        else:
            print(f"{key}: {value}")

    return True


if __name__ == "__main__":
    # Run the test function
    test_openalex_integration()
