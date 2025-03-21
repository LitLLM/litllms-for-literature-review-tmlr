# lit-rank
LLM Reranking

## Setup
Install conda/miniconda then create an environment

```
conda create -n litllm-retrieval python=3.11 -y
pip install -r requirements.txt
```

1. Run `src/arxiv_paper.py` (currently don't need to run explicitly)
    - It downloads latest n papers(=2000) from ArXiv and stores them in dataset/arxiv_papers/{filename}.csv
    - It queries for "artificial intelligence" (cs.AI) papers
    - There are other params we can tweak like the search query, and the name of the output file

2. Run `python src/paper_manager.py --n_candidates 100 --n_queries 500 --n_keywords 3 --search_engine s2 --gen_engine gpt-4o-mini --rerank_method llm+embedding`
