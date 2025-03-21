import json
from ..api_utils import (
    get_arxiv_id_by_title_or_abstract,
    read_or_download_text,
    download_pdf,
)

from . import related_work_prompts as rwp
from . import keyword_prompts as sp
from . import ranking_prompts as rp
from typing import Dict, Union
import time


def load_prompt(prompt_type: str) -> Union[str, Dict]:
    if prompt_type == "all":
        prompts = {
            "basic_template": rwp.get_prompt(prompt_type="basic"),
            "plan_template": rwp.get_prompt(prompt_type="plan"),
            "summarization_template": sp.get_prompt(prompt_type="basic"),
            "role_template": rp.get_role_prompt(prompt_type="basic"),
            "ranking_template": rp.get_ranking_prompt(prompt_type="basic"),
            "ranking_reasoning_template": rp.get_ranking_prompt(
                prompt_type="reasoning"
            ),
        }
        return prompts
    elif prompt_type == "basic":
        # return a basic prompt for writing the related work
        return rwp.get_prompt(prompt_type="basic")
    elif prompt_type == "plan":
        # return a prompt with a plan for writing the related work
        return rwp.get_prompt(prompt_type="plan")
    elif prompt_type == "summary":
        # return a prompt for summarization
        return sp.get_prompt(prompt_type="basic")
    elif prompt_type == "reasoned_summary":
        # return a prompt for summarization with reasoning
        return sp.get_prompt(prompt_type="reasoning")
    elif prompt_type == "role":
        # return a prompt for the role of a research assistant
        return rp.get_role_prompt(prompt_type="basic")
    elif prompt_type == "basic_ranking":
        # return a prompt for ranking papers
        return rp.get_ranking_prompt(prompt_type="basic")
    elif prompt_type == "reasoned_ranking":
        return rp.get_ranking_prompt(prompt_type="reasoning")
    elif prompt_type == "reasoned_ranking_full":
        return rp.get_ranking_prompt(prompt_type="reasoning_full")
    elif prompt_type == "reason_then_ranking_full":
        return rp.get_ranking_prompt(prompt_type="reason_then_ranking_full")
    elif prompt_type == "reasoned_ranking_pdf":
        return rp.get_ranking_prompt(prompt_type="ranking_reasoning_pdf")
    elif prompt_type == "debate_ranking":
        return rp.get_ranking_prompt(prompt_type="debate_ranking")
    elif prompt_type == "debate_ranking_abstract":
        return rp.get_ranking_prompt(prompt_type="debate_ranking_abstract")


def format_abstracts_as_references(
    papers, use_pdf=False, use_full_text=False, fetch_arxiv_ids=True
):
    # cite_list = ["@cite_1", "@cite_2", "@cite_3"]
    """
    [1]: {abstract} \n [2]: {abstract}
    """
    arxiv_ids = []
    cite_text = " "
    for _ in range(5):
        try:
            arxiv_id_cache = json.load(open("dataset/arxiv_paper_ids.json", "r"))
            break
        except Exception as e:
            print("Error in loading arxiv_id_cache. Retrying...")
            print(e)
            print(time.sleep(5))
            continue
    for index, paper in enumerate(papers):
        # citation = f"@cite_{index+1}"
        citation = f"{index+1}"
        abstract = paper.get("abstract", "Abstract not available")
        title = paper.get("title_paper", "No title")
        if not abstract:
            abstract = "Abstract not available"

        if not title:
            title = "No title"

        if not fetch_arxiv_ids:
            cite_text += f"[{citation}]: Abstract: {title}\n{abstract}\n"
            continue
        # try finiding it in the cache
        arxiv_id = None
        found_in_cache = False
        if title != "No title":
            arxiv_id = arxiv_id_cache.get(title, None)
            found_in_cache = arxiv_id is not None
        if not arxiv_id:
            arxiv_id = get_arxiv_id_by_title_or_abstract(
                abstract.strip(), search_type="abstract"
            )
        if not arxiv_id:
            arxiv_id = get_arxiv_id_by_title_or_abstract(
                title.strip(), search_type="title"
            )
        if arxiv_id:
            # update the caching system
            if not found_in_cache:
                with open("dataset/arxiv_paper_ids.json", "w") as f:
                    arxiv_id_cache[title] = arxiv_id
                    json.dump(arxiv_id_cache, f, indent=4)

            if (not use_pdf) and use_full_text:
                # get the full text of the paper
                print(f"Getting full text for paper {index+1}/{len(papers)}...")
                paper["full_text"] = read_or_download_text(arxiv_id)
            elif use_pdf:
                paper["external_ids"] = download_pdf(arxiv_id)
                if paper["external_ids"] is None:
                    print(f"Error in downloading PDF for arxiv_id: {arxiv_id}")
                    continue
        else:
            print("No arxiv id found with abstract+title search")
            print(f"for paper {index+1}/{len(papers)}")
            print(f"Title: {title}")
            print(f"Abstract: {abstract}")
            print("Skipping this paper...")
            continue
        arxiv_ids.append(arxiv_id)
        if not use_pdf:
            # print(f"Formatting paper {index+1}/{len(papers)}...")
            cite_text += f"[{citation}]: Abstract: {title}\n{abstract}\n"
            if "full_text" in paper:
                # print(f"Adding full text to paper {index+1}/{len(papers)}...")
                cite_text += f"Full text: {paper['full_text']}\n"
        else:
            cite_text += f"[{citation}]: {paper['external_ids']}\n"
    return cite_text, arxiv_ids


def format_prompt(base_prompt, abstract, cite_text, plan=""):
    """
    This prompt formats the abstract and cite_text
    """
    if plan:
        data = f"Abstract: {abstract} \n {cite_text} \n Plan: {plan}"
    else:
        data = f"Abstract: {abstract} \n {cite_text}"
    complete_prompt = f"{base_prompt}\n```{data}```"
    return complete_prompt
