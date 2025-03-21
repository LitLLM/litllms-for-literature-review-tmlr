def get_role_prompt(prompt_type="basic"):
    if prompt_type == "basic":
        return system_template


def get_ranking_prompt(prompt_type="basic"):
    if prompt_type == "basic":
        return system_template, ranking_template
    elif prompt_type == "reasoning":
        return system_template, ranking_reasoning_template
    elif prompt_type == "reasoning_full":
        return system_template, ranking_reasoning_full_template
    elif prompt_type == "reason_then_ranking_full":
        return system_template, reasoning_ranking_full_template
    elif prompt_type == "ranking_reasoning_pdf":
        return system_template, ranking_reasoning_pdf_template
    elif prompt_type == "ranking_reasoning_pdf_sort":
        return system_template, ranking_reasoning_pdf_sort
    elif prompt_type == "debate_ranking":
        return system_template, debate_ranking_template
    elif prompt_type == "debate_ranking_abstract":
        return system_template, debate_ranking_abstract_template


old_ranking_template = """
You will be provided with an abstract or an idea of a scientific document and abstracts of some other relevant papers. Your task is to rank the papers based on the relevance to the query abstract. If the reference paper matches the query abstract completely, provide it a lower rank. Provide only the ranks as [] > [] > []. Make sure you provide the whole list always.
"""

ranking_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.


## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

Given the candidate reference paper abstract:
<candidate_paper_abstract>{reference_papers}</candidate_paper_abstract>

* Given the abstract of the candidate reference paper, provide me a ranking of papers in decreasing order of preference to include these papers in the literature review of the query.
* You must enclose your ranking within <ranking> and </ranking> tags.

### Response Format:

<ranking>
[4] > [2] > [1] > [3] > ...
</ranking>

### Your Response:
"""

old_ranking_reasoning_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.

## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

And given the following reference abstracts:
<reference_abstracts>{reference_abstracts}</reference_abstracts>

* Rank the reference abstracts based on their relevance to the query abstract.
* If the reference paper matches the query abstract completely, provide it a lower rank.
* Provide your ranking in the following format [] > [] > [] ... > [].
* You must rank ALL the reference abstracts.
* You must enclose your ranking output within <ranking> and </ranking> tags.
* In addition to the rankings, also provide a reasoning for the provided rank.
* Extract the relevant sentences from the reference abstracts and add them at the end of the reasoning.
* Put the extracted sentences in quotes, even if reference abstract completely matches the query abstract or is completely irrelevant.
* You must enclose your reasoning within <reason> and </reason> tags.
* Do not generate anything else apart from the ranking and reasoning.

### Example response (if there are 3 reference abstracts):
<ranking>[3] > [1] > [2]</ranking>
<reason>
[3] is ranked highest because it discusses the same method for the research idea. The sentence "The method proposed in this paper is similar to the one proposed in the query abstract." is extracted from the reference abstract.
[1] is ranked second because it discusses the same dataset for the research idea. The sentence "The dataset used in this paper is similar to the one proposed in the query abstract." is extracted from the reference abstract.
[2] is ranked lowest because it discusses a different method for the research idea. The sentence "The method proposed in this paper is different from the one proposed in the query abstract." is extracted from the reference abstract.
"""

ranking_reasoning_full_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.

## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

And given the following reference abstracts:
<reference_papers>{reference_papers}</reference_papers>

* Rank the reference abstracts based on their relevance to the query abstract.
* If the reference paper abstract matches the query abstract completely, provide it a lower rank.
* Provide your ranking in the following format [] > [] > [] ... > [].
* You must rank ALL the reference papers.
* You must enclose your ranking output within <ranking> and </ranking> tags.
* In addition to the rankings, also provide a reasoning for the provided rank.
* Extract the relevant sentences from the reference paper and add them at the end of the reasoning.
* Put the extracted sentences in quotes, even if reference abstract completely matches the query abstract or is completely irrelevant.
* You must enclose your reasoning within <reason> and </reason> tags.
* Do not generate anything else apart from the ranking and reasoning.

### Example response (if there are 3 reference abstracts):
<ranking>[3] > [1] > [2]</ranking>
<reason>
[3] is ranked highest because it discusses the same method for the research idea. The sentence "The method proposed in this paper is similar to the one proposed in the query abstract." is extracted from the reference paper.
[1] is ranked second because it discusses the same dataset for the research idea. The sentence "The dataset used in this paper is similar to the one proposed in the query abstract." is extracted from the reference paper.
[2] is ranked lowest because it discusses a different method for the research idea. The sentence "The method proposed in this paper is different from the one proposed in the query abstract." is extracted from the reference paper.
"""

reasoning_ranking_full_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.

## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

And given the following reference abstracts:
<reference_papers>{reference_papers}</reference_papers>

* Rank the reference abstracts based on their relevance to the query abstract.
* If the reference paper abstract matches the query abstract completely, provide it a lower rank.
* Provide your ranking in the following format [] > [] > [] ... > [].
* You must rank ALL the reference papers.
* You must enclose your ranking output within <ranking> and </ranking> tags.
* In addition to the rankings, also provide a reasoning for the provided rank.
* Extract the relevant sentences from the reference paper and add them at the end of the reasoning.
* Put the extracted sentences in quotes, even if reference abstract completely matches the query abstract or is completely irrelevant.
* You must enclose your reasoning within <reason> and </reason> tags.
* Do not generate anything else apart from the ranking and reasoning.

### Example response (if there are 3 reference abstracts):
<reason>
[3] is ranked highest because it discusses the same method for the research idea. The sentence "The method proposed in this paper is similar to the one proposed in the query abstract." is extracted from the reference paper.
[1] is ranked second because it discusses the same dataset for the research idea. The sentence "The dataset used in this paper is similar to the one proposed in the query abstract." is extracted from the reference paper.
[2] is ranked lowest because it discusses a different method for the research idea. The sentence "The method proposed in this paper is different from the one proposed in the query abstract." is extracted from the reference paper.
</reason>
<ranking>[3] > [1] > [2]</ranking>
"""

ranking_reasoning_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.


## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

Given the candidate reference paper abstract:
<candidate_paper_abstract>{reference_papers}</candidate_paper_abstract>

* Given the abstract of the candidate reference paper, provide me with a number between 0 and 100 (upto two decimal places) that is proportional to the probability of a paper with the given query abstract including the candidate reference paper in its literature review.
* In addition to the probability, give me arguments that justify the assigned probability.
* Extract relevant sentences from the candidate paper abstract to support your rationale.
* Put the extracted sentences in quotes.
* You can use the information in other candidate papers when generating the arguments for a candidate paper.
* You must enclose your score within <probability> and </probability> tags.
* Generate the arguments first then the probability score.
* Do not generate anything else apart from the probability and the arguments.

### Response Format:
<arguments>
[Paper ID]: [Reason for including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments>
<probability>
[Paper ID]: [Final Probability Score Based on the Arguments]
</probability>

### Your Response:
"""

system_template = "You are a helpful research assistant who is helping with literature review of a research idea."

ranking_reasoning_pdf_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.

## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

You have been provided with the PDFs of the reference papers as the input and given the following reference paper ids:
<reference_paper_ids>{reference_papers}</reference_paper_ids>


* Rank the references based on their relevance to the query abstract.
* If the reference paper abstract matches the query abstract completely, provide it a lower rank.
* Provide your ranking in the following format [] > [] > [] ... > [].
* You must rank ALL the reference papers.
* You must enclose your ranking output within <ranking> and </ranking> tags.
* In addition to the rankings, also provide a reasoning for the provided rank.
* Extract the relevant sentences from the reference paper and add them at the end of the reasoning.
* Put the extracted sentences in quotes, even if reference abstract completely matches the query abstract or is completely irrelevant.
* You must enclose your reasoning within <reason> and </reason> tags.
* Do not generate anything else apart from the ranking and reasoning.

### Example response format (if there are 3 reference abstracts):
<reason>
[3] is ranked highest because [explanation].
Extracted Sentences: "Sentence 1", "Sentence 2", ...

[1] is ranked second because [explanation].
Extracted Sentences: "Sentence 1", "Sentence 2", ...

[2] is ranked lowest because [explanation].
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</reason>
<ranking>[3] > [1] > [2]</ranking>
"""


ranking_reasoning_pdf_sort = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.

## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

Given the two reference paper ids:
<reference_paper_ids>{reference_papers}</reference_paper_ids>


* Rank the two references based on their relevance to the query abstract.
* If the reference paper abstract matches the query abstract completely, provide it a lower rank.
* Provide your ranking in the following format [] > [].
* You must rank both the reference papers.
* You must enclose your ranking output within <ranking> and </ranking> tags.
* In addition to the rankings, also provide a reasoning for the provided rank.
* Extract the relevant sentences from the reference paper and add them at the end of the reasoning.
* Put the extracted sentences in quotes, even if reference abstract completely matches the query abstract or is completely irrelevant.
* You must enclose your reasoning within <reason> and </reason> tags.
* Do not generate anything else apart from the ranking and reasoning.
* You can refer to the following example for the response format.
* Note that the example reasonings are for illustration purposes and your reasoning should be more detailed and specific based on the reference papers.

### Response Format:
<reason>
[2] is ranked highest because [explanation].
Extracted Sentences: "Sentence 1", "Sentence 2", ...

[1] is ranked second because [explanation].
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</reason>
<ranking>[2] > [1]</ranking>

### Your Response:
"""


debate_ranking_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.


## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

Given the candidate reference paper ids:
<candidate_reference_paper_ids>{reference_papers}</candidate_reference_paper_ids>

* Given the candidate reference papers, provide me with a number between 0 and 100 (upto two decimal places) for each paper that is proportional to the probability of a paper with the given query abstract including the candidate reference paper in their literature review.
* In addition to the probability, give me arguments for and against including this paper in the literature review.
* You must enclose your arguments for including the paper within <arguments_for> and </arguments_for> tags.
* You must enclose your arguments for including the paper within <arguments_against> and </arguments_against> tags.
* Extract relevant sentences from the candidate papers to support your arguments.
* Put the extracted sentences in quotes.
* You can use the information in other candidate papers when generating the arguments for a candidate paper.
* You must enclose your score within <probability> and </probability> tags.
* Generate the arguments first then the probability score.
* Do not generate anything else apart from the probability and the arguments.

### Response Format:
<arguments_for>
[Paper ID]: [Reason for including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_for>
<arguments_against>
[Paper ID]: [Reason for not including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_against>
<probability>
[Paper ID]: [Final Probability Score Based on the Arguments]
</probability>

### Your Response:
"""

debate_ranking_abstract_template = """
You are a helpful research assistant who is helping with literature review of a research idea. Your task is to rank some papers based on their relevance to the query abstract.


## Instruction:
Given the query abstract:
<query_abstract>{query_abstract}</query_abstract>

Given the candidate reference paper abstract:
<candidate_paper_abstracts>{reference_papers}</candidate_paper_abstracts>

* Given the abstract of the candidate reference papers, provide me with a number between 0 and 100 (upto two decimal places) that is proportional to the probability of a paper with the given query abstract including the candidate reference paper in its literature review.
* In addition to the probability, give me arguments for and against including this paper in the literature review.
* You must enclose your arguments for including the paper within <arguments_for> and </arguments_for> tags.
* You must enclose your arguments for including the paper within <arguments_against> and </arguments_against> tags.
* Extract relevant sentences from the candidate paper abstract to support your arguments.
* Put the extracted sentences in quotes.
* You can use the information in other candidate papers when generating the arguments for a candidate paper.
* You must enclose your score within <probability> and </probability> tags.
* Generate the arguments first then the probability score.
* Generate arguments and probabitlity for each paper separately.
* Do not generate anything else apart from the probability and the arguments.
* Follow this process even if a candidate paper happens to be identical or near-perfect match to the query abstract.

### Response Format for each paper:
<arguments_for>
[Paper ID]: [Reason for including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_for>
<arguments_against>
[Paper ID]: [Reason for not including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_against>
<probability>
[Paper ID]: [Final Probability Score Based on the Arguments]
</probability>

### Your Response:
"""
