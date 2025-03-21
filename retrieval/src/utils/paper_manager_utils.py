import os
import re
import json
from .llm_utils import run_llm_api, extract_html_tags
from .prompt_utils import format_abstracts_as_references, load_prompt


def get_keywords_for_s2_search(abstract, config):
    print(f"Num keyword queries: {config.n_keywords}")
    # generate keywords for the abstract
    summarization_system_template, summarization_prompt = load_prompt(
        prompt_type="reasoned_summary"
    )
    summarization_prompt = summarization_prompt.format(
        n_keywords=config.n_keywords, abstract=abstract
    )
    json_data = {
        "prompt": summarization_prompt,
        "system_prompt": summarization_system_template,
    }

    # add some retries in case the API fails
    for _ in range(3):
        try:
            query = run_llm_api(
                json_data,
                gen_engine=config.gen_engine,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            query = query.replace("```json", "").replace("```", "")
            query_data = json.loads(query)
            print(
                f"LLM summarized keyword query to be used for S2 API: \n {query_data}"
            )
            return query_data
        except Exception as e:
            print(f"Failed to generate keywords for S2 search: {e}")
            if _ == 4:
                print("Failed to generate keywords for S2 search. Skipping...")
                return None


def check_extractive(reasonings, source_text):
    exists = []
    for reasoning in reasonings:
        # extract sentence within quotes
        extracted_sentence = re.search(r'"(.*?)"', reasoning)
        if not extracted_sentence:
            # try with single quotes
            extracted_sentence = re.search(r"'(.*?)'", reasoning)
        if not extracted_sentence:
            exists.append(False)
        else:
            extracted_sentence = extracted_sentence.group(1).strip()
            if is_within_edit_distance(extracted_sentence, source_text):
                exists.append(True)
            else:
                exists.append(False)
    return exists


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def is_within_edit_distance(string, paragraph, max_distance=5):
    if not string or not paragraph:
        return False
    string_length = len(string)
    paragraph_length = len(paragraph)

    for i in range(paragraph_length - string_length + 1):
        substring = paragraph[i : i + string_length]
        distance = levenshtein_distance(string, substring)
        if distance <= max_distance:
            return True
    return False


def run_reranking_prompt(abstract, candidate_papers, config):
    ranking_system_template, ranking_prompt = load_prompt(
        prompt_type=config.reranking_prompt_type
    )
    cite_text, arxiv_ids = format_abstracts_as_references(
        candidate_papers,
        use_pdf=config.use_pdf,
        use_full_text=config.use_full_text,
        fetch_arxiv_ids=config.reranking_prompt_type != "basic_ranking",
    )
    # print(ranking_prompt)
    json_data = {
        "prompt": ranking_prompt.format(
            query_abstract=abstract, reference_papers=cite_text
        ),
        "system_prompt": ranking_system_template,
    }

    if config.use_pdf:
        json_data["arxiv_ids"] = arxiv_ids

    llm_output = run_llm_api(
        json_data,
        gen_engine=config.gen_engine,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    return llm_output, arxiv_ids


def rerank_candidates_batch(abstract, candidate_papers, config):
    arxiv_ids = []
    if "debate_ranking" in config.reranking_prompt_type:
        accumulated_reasons = []
        papers_to_remove = []
        probs = []
        batch_size = 10
        for p in range(0, len(candidate_papers), batch_size):
            print(
                f"Reranking batch {p+1} to {p+batch_size} (batch_size={batch_size})..."
            )
            arg_against = None
            arg_for = None
            for _ in range(3):
                p_batch = candidate_papers[p : p + batch_size]
                response, arxiv = run_reranking_prompt(abstract, p_batch, config)
                if not arxiv:
                    papers_to_remove.append(p)
                    continue

                arg_for = extract_html_tags(response, ["arguments_for"])
                arg_against = extract_html_tags(response, ["arguments_against"])
                # first check if there are the required arguments
                if (
                    "arguments_for" not in arg_for
                    or "arguments_against" not in arg_against
                ):
                    print("Arguments not found in response. Retrying...")
                    continue
                if not config.skip_extractive_check:
                    # extract reasoning from the llm response
                    arg_for = arg_for["arguments_for"]
                    arg_against = arg_against["arguments_against"]
                    # check extractiveness of the reasoning
                    check_passed = []
                    for k in range(len(arg_against)):
                        if check_extractive([arg_for[k]], p_batch[k]["abstract"])[
                            0
                        ] and check_extractive(
                            [arg_against[k]], p_batch[k]["abstract"]
                        ):
                            check_passed.append(True)
                        else:
                            check_passed.append(False)
                    if False not in check_passed:
                        break
                    else:
                        print(response)
                        print(
                            f"Reasoning for batch {p+1} is not extractive. {check_passed}. Retrying..."
                        )
            for j in range(len(arg_against)):
                try:
                    prob = extract_html_tags(response, ["probability"])["probability"][
                        j
                    ]
                    prob = float(prob.split(":")[-1].strip())
                except Exception as e:
                    print(f"Failed to extract probability from response: {e}")
                    prob = 0.0
                if not arg_against or not arg_for:
                    arguments = f"Arguments for [{j+1}]: no arguments generated for and against \nProbability: 0"
                    prob = 0.0
                else:
                    arguments = f"Arguments for [{j+1}]: {arg_for[j]}\nArguments against [{j+1}]: {arg_against[j]}\nProbability: {prob}"
                accumulated_reasons.append(arguments)
                probs.append(
                    (j + (((p + 1) // 5) * 5), prob)
                )  # p is batch id (0-indexed)
                arxiv_ids.extend(arxiv)
        # sort the papers based on the probability
        probs.sort(key=lambda x: x[1], reverse=True)
        ranking = [p[0] + 1 for p in probs]
        ranking = " > ".join([f"[{i}]" for i in ranking])
        reason = "\n\n".join(accumulated_reasons)
    elif "reasoned" in config.reranking_prompt_type:
        accumulated_reasons = []
        papers_to_remove = []
        probs = []
        batch_size = 1
        for p in range(0, len(candidate_papers), batch_size):
            print(
                f"Reranking batch {p+1} to {p+batch_size} (batch_size={batch_size})..."
            )
            for _ in range(3):
                p_batch = candidate_papers[p : p + batch_size]
                response, arxiv = run_reranking_prompt(abstract, p_batch, config)
                if not arxiv:
                    papers_to_remove.append(p)
                    arguments = None
                    continue
                # it reasoning from the llm response
                arguments = extract_html_tags(response, ["arguments"])
                if "arguments" not in arguments:
                    print("Arguments not found in response. Retrying...")
                    continue

                arguments = arguments["arguments"]
                # check extractiveness of the reasoning
                if check_extractive(arguments, p_batch[0]["abstract"])[0]:
                    break
                else:
                    print(f"Reasoning for batch {p+1} is not extractive. Retrying...")
            try:
                prob = extract_html_tags(response, ["probability"])["probability"][0]
                prob = float(prob.split(":")[-1].strip())
            except Exception as e:
                print(f"Failed to extract probability from response: {e}")
                prob = 0.0
            if not arguments:
                arguments = (
                    f"Arguments for [{p+1}]: no arguments generated\nProbability: 0"
                )
                prob = 0.0
            else:
                arguments = (
                    f"Arguments for [{p+1}]: {arguments[0]}\nProbability: {prob}"
                )
            accumulated_reasons.append(arguments)
            probs.append((p, prob))
        probs.sort(key=lambda x: x[1], reverse=True)
        ranking = [p[0] + 1 for p in probs]
        ranking = " > ".join([f"[{i}]" for i in ranking])
        reason = "\n\n".join(accumulated_reasons)
    else:  # basic reranking
        ranking, arxiv_ids = run_reranking_prompt(abstract, candidate_papers, config)
        reason = "No reason for basic reranking"

    if not ranking or not reason:
        print(
            f"Failed to extract ranking and reason from response. Skipping reranking..."
        )
        ranking = " > ".join([f"[{i+1}]" for i in range(len(candidate_papers))])
        reason = "This list is not reranked"
    return ranking, reason, arxiv_ids


def sanitize_paper_order(extracted_order, n_candidates):
    # Initialize a list to hold the final new order, ensuring no duplicates and limiting to 100 items
    final_new_order = []
    seen = set()
    # Keep track of unique items and their order
    # Apply GPT-4 output to reorder, falling back to original order if not specified
    for item in extracted_order:
        if item not in seen:
            final_new_order.append(item)
            seen.add(item)

    # If the final list is shorter than 100, fill in with original order,
    # skipping already included
    if len(final_new_order) < 100:
        for i in range(1, n_candidates + 1):
            if i not in seen:
                final_new_order.append(i)
    print(f"Final new order: {final_new_order}")
    return final_new_order


def save_cited_and_aggregated_papers(
    df_cited_papers_aggregated, df_reranked_papers_aggregated, config, output_dir
):
    # Check if the file exists
    file_exists_cited = os.path.isfile(
        f"{output_dir}/cited_papers_aggregated-n={config.n_queries}-k={config.n_candidates}-model={config.gen_engine}-search_engine={config.search_engine}-rerank_prompt_type={config.reranking_prompt_type}.csv"
    )
    file_exists_reranked = os.path.isfile(
        f"{output_dir}/reranked_papers_aggregated-n={config.n_queries}-k={config.n_candidates}-model={config.gen_engine}-search_engine={config.search_engine}-rerank_prompt_type={config.reranking_prompt_type}.csv"
    )

    # Save the DataFrame, appending if file exists
    df_cited_papers_aggregated.to_csv(
        f"{output_dir}/cited_papers_aggregated-n={config.n_queries}-k={config.n_candidates}-model={config.gen_engine}-search_engine={config.search_engine}-rerank_prompt_type={config.reranking_prompt_type}.csv",
        mode="a",
        header=not file_exists_cited,
        index=False,
    )

    df_reranked_papers_aggregated.to_csv(
        f"{output_dir}/reranked_papers_aggregated-n={config.n_queries}-k={config.n_candidates}-model={config.gen_engine}-search_engine={config.search_engine}-rerank_prompt_type={config.reranking_prompt_type}.csv",
        mode="a",
        header=not file_exists_reranked,
        index=False,
    )
