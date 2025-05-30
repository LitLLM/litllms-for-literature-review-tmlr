# https://github.com/allenai/s2-folks/blob/main/examples/python/find_and_recommend_papers/find_papers.py
import os
import re
import pandas as pd

import requests

S2_API_KEY = os.getenv('S2_API_KEY')
result_limit = 10


def main():
    basis_paper = find_basis_paper()
    find_recommendations(basis_paper)


def find_basis_paper():
    papers = None
    while not papers:
        query = input('Find papers about what: ')
        if not query:
            continue
        fields = 'title,url,abstract,citationCount,journal,isOpenAccess,fieldsOfStudy,year,journal'

        rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                           headers={'X-API-KEY': S2_API_KEY},
                           params={'query': query, 'limit': result_limit, 'fields': fields})
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        if not total:
            print('No matches found. Please try another query.')
            continue

        print(f'Found {total} results. Showing up to {result_limit}.')
        papers = results['data']
        print(papers)
        df = pd.DataFrame(papers)
        print(df)
        print(df["abstract"])
        print(df["citationCount"])
        print_papers(papers)
        sort_by = "Citations"
        if sort_by == "Citations":
            df.sort_values(by="citationCount", ascending=False)
        elif sort_by == "Year":
            df.sort_values(by="year", ascending=False)
        papers_list = df.to_dict(orient='records')

        # papers_list = df.values.tolist()
        print(papers_list)

    selection = ''
    while not re.fullmatch('\\d+', selection):
        selection = input('Select a paper # to base recommendations on: ')
    return results['data'][int(selection)]


def find_paper(url):
    url_query = f"URL:{url}"
    r = requests.post(
        'https://api.semanticscholar.org/graph/v1/author/batch',
        params={'fields': 'name,hIndex,citationCount'},
        json={"ids":["1741101", "1780531"]}
    )
    print(json.dumps(r.json(), indent=2))


def get_paper(paper_id):
    rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}',
                       headers={'X-API-KEY': S2_API_KEY},
                       params={'fields': 'title,authors'})
    rsp.raise_for_status()
    return rsp.json()



def find_recommendations(paper):
    print(f"Up to {result_limit} recommendations based on: {paper['title']}")
    rsp = requests.get(f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper['paperId']}",
                       headers={'X-API-KEY': S2_API_KEY},
                       params={'fields': 'title,url', 'limit': 10})
    rsp.raise_for_status()
    results = rsp.json()
    print_papers(results['recommendedPapers'])


def print_papers(papers):
    for idx, paper in enumerate(papers):
        print(f"{idx}  {paper['title']} {paper['url']} {paper['citationCount']}")


if __name__ == '__main__':
    main()