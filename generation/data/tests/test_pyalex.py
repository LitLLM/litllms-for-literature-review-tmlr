# pip install pyalex
import os, subprocess
import pyalex
from pyalex import Works
from pathlib import Path
import urllib.request
import requests


pyalex.config.email = "mail@example.com"


def extract_pdf(url, output_path):
    """ Extract PDF based on URL

    Args:
        url (string): link to PDF 
        output_path (string): Path to output PDF file

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    # command = f"wget -w 3 --random-wait -q -O {output_path} {url}"
    command = f"wget -w 3 --random-wait --no-check-certificate -O {output_path} '{url}'"
    subprocess.call(command, shell=True)
    
    if os.path.exists(output_path):
        return True
    return False 


# "1528079221", "2156109180", "1591263692"
# the work to extract the referenced works of
work = Works()["W1591263692"]
# print(work)
mid = "W1591263692"
open_access = work["open_access"]
best_oa_location = work["best_oa_location"]
is_oa = best_oa_location["is_oa"]
pdf_url = best_oa_location["pdf_url"]
print(best_oa_location)
license = best_oa_location["license"]

print(pdf_url)

# urllib.request.urlretrieve(pdf_url, "filename.pdf")
extract_pdf(pdf_url, f"out.pdf")
# filename = Path('metadata.pdf')
# url = 'http://www.hrecos.org//images/Data/forweb/HRTVBSH.Metadata.pdf'
# response = requests.get(pdf_url)
# filename.write_bytes(response.content)


# Use https://codebeautify.org/python-formatter-beautifier#google_vignette


# ["open_access"]["oa_url"]
# ["best_oa_location"]

#         "is_oa": True,
#         "landing_page_url": "http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.9785",
#         "pdf_url": "http://ijcai.org/Past%20Proceedings/IJCAI-95-VOL%201/pdf/122.pdf",
#         "source": {}
# "license"

# https://github.com/J535D165/pyalex#cited-publications-referenced-works
# refs = Works()[w["referenced_works"]]
# print(len(refs))
# # print("Trying")
# print(refs[0])
# print(refs[0].keys())


# dict_keys(['id', 'doi', 'title', 'display_name', 'publication_year', 'publication_date', 'ids', 'language', 
# 'primary_location', 'type', 'type_crossref', 'open_access', 'authorships', 'countries_distinct_count', 
# 'institutions_distinct_count', 'corresponding_author_ids', 'corresponding_institution_ids', 'apc_list', 
# 'apc_paid', 'cited_by_count', 'biblio', 'is_retracted', 'is_paratext', 'concepts', 'mesh', 'locations_count', '
# locations', 'best_oa_location', 'sustainable_development_goals', 'grants', 'referenced_works_count', 'referenced_works', 
# 'related_works', 'ngrams_url', 'abstract_inverted_index', 'cited_by_api_url', 'counts_by_year', 'updated_date', 'created_date'])