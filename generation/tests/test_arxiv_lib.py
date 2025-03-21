# https://vitalflux.com/python-scraper-code-search-arxiv-latest-papers/
# https://pypi.org/project/arxiv/
import arxiv
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
search = arxiv.Search(
    query="\"FeastNet\"",
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance
)
# sort_order = arxiv.SortOrder.Descending
# SortCriterion.Relevance
# SortCriterion.SubmittedDate
# submittedDate, lastUpdatedDate

# query = "healthcare AND \"machine learning\""
# query="ti: History for visual dialog",
# query = "\"logistic regression\""
# query="\"FeastNet\""
# query="\"Multimodal dialog\""

for result in search.results():
    print('Title: ', result.title, '\nDate: ', result.published, '\nId: ',
          result.entry_id, '\nDOI: ', result.doi,
          '\nAuthors: ', result.authors,
          '\nPrimary_category: ', result.primary_category,
          '\nSummary: ', result.summary, '\nURL: ', result.pdf_url, '\n\n')

search = arxiv.Search(id_list=["1605.08386v1"])
paper = next(search.results())
print(paper.title)

if not os.path.exists("./test_output"):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./test_output")

# paper = next(arxiv.Search(id_list=["1605.08386v1"]).results())
# Download the PDF to the PWD with a default filename.
# paper.download_pdf()
# Download the PDF to the PWD with a custom filename.
# paper.download_pdf(filename="downloaded-paper.pdf")
# Download the PDF to a specified directory with a custom filename.
paper.download_pdf(dirpath="./test_output", filename="downloaded-paper.pdf")

paper = next(arxiv.Search(id_list=["1605.08386v1"]).results())
# Download the archive to the PWD with a default filename.
# paper.download_source()
# # Download the archive to the PWD with a custom filename.
# paper.download_source(filename="downloaded-paper.tar.gz")
# Download the archive to a specified directory with a custom filename.
paper.download_source(dirpath="./test_output", filename="downloaded-paper.tar.gz")

big_slow_client = arxiv.Client(
    page_size=100,
    delay_seconds=10,
    num_retries=5
)

# Prints 1000 titles before needing to make another request.
for result in big_slow_client.results(arxiv.Search(query="multimodal")):
    print(result.title)
