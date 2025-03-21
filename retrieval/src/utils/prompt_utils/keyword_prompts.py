def get_prompt(prompt_type="basic"):
    if prompt_type == "basic":
        return system_template, summarization_template
    elif prompt_type == "reasoning":
        return system_template, summary_reasoning_template


summarization_template = """
You are a helpful research assistant who is helping with literature review of a research idea. You will be provided with an abstract of a scientific document. Your task is to frame queries that would be used to search an academic search engine and retrieve relevant papers.

Here is the abstract:
{abstract}

## Instruction:
* Each query should not be more than 5 keywords. Please write the query in a similar fashion as a human would use search engine.
* Please generate {n_keywords} search queries.
* Please make sure to generate different search queries using a variety of key words so as to get maximum papers that could be cited.
* Please return a JSON with a key "queries" that has the list of queries:
"""

summary_reasoning_template = """
You are a helpful research assistant who is helping with literature review of a research idea. You will be provided with an abstract of a scientific document. Your task is to frame queries that would be used to search an academic search engine and retrieve relevant papers.

Here is the abstract:
{abstract}

## Instruction:
* Each query should not be more than 5 keywords. Please write the query in a similar fashion as a human would use search engine.
* Please generate {n_keywords} search queries.
* Please make sure to generate different search queries using a variety of key words so as to get maximum papers that could be cited.
* In addition to the queries, also provide a reasoning for the generated queries.
* Extract the relevant sentences from the abstract that justify your reasoning.
* Put the extracted sentences in quotes and put them at the end of each of your reasonings.
* Example reasoning:
"The query is framed to get papers that discuss the method proposed in the abstract. The sentence 'The method proposed in this paper is similar to the one proposed in the query abstract.' is extracted from the abstract."
* Please return a JSON with a key "queries" that has the list of queries and a "reasoning" key that has the reasoning for the queries.
* Do not generate anything else apart from the JSON.

### Response:
"""

system_template = "You are a helpful research assistant who is helping with literature review of a research idea."
