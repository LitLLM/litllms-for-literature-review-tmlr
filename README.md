# LitLLMs, LLMs for Literature Review: Are we there yet?

https://arxiv.org/abs/2412.15249

## Abstract

Literature reviews are an essential component of scientific research, but they remain time-intensive and challenging to write, especially due to the recent influx of research papers. This paper explores the zero-shot abilities of recent Large Language Models (LLMs) in assisting with the writing of literature reviews based on an abstract. We decompose the task into two components: 1. Retrieving related works given a query abstract, and 2. Writing a literature review based on the retrieved results. We analyze how effective LLMs are for both components.

For retrieval, we introduce a novel two-step search strategy that first uses an LLM to extract meaningful keywords from the abstract of a paper and then retrieves potentially relevant papers by querying an external knowledge base. Additionally, we study a prompting-based re-ranking mechanism with attribution and show that re-ranking doubles the normalized recall compared to naive search methods, while providing insights into the LLM's decision-making process.

In the generation phase, we propose a two-step approach that first outlines a plan for the review and then executes steps in the plan to generate the actual review. To evaluate different LLM-based literature review methods, we create test sets from arXiv papers using a protocol designed for rolling use with newly released LLMs to avoid test set contamination in zero-shot evaluations. We release this evaluation protocol to promote additional research and development in this regard.

Our empirical results suggest that LLMs show promising potential for writing literature reviews when the task is decomposed into smaller components of retrieval and planning.

## Project Page

Visit our project page: [https://litllm.github.io](https://litllm.github.io)

## Usage Instructions

Please refer to the README of `retrieval` and `generation` for instructions to run the retrieval and generation experiments, respectively.

## Contributors
Shubham Agarwal¹²³*, Gaurav Sahu¹⁴*, Abhay Puri¹*, Issam H. Laradji¹⁵, Krishnamurthy DJ Dvijotham¹, Jason Stanley¹, Laurent Charlin²³⁶, Christopher Pal¹²⁷⁶

### Affiliations
- [¹ServiceNow Research](https://github.com/serviceNow/)
- ²Mila - Quebec AI Institute
- ³HEC Montreal
- ⁴University of Waterloo
- ⁵University of British Columbia
- ⁶Canada CIFAR AI Chair
- ⁷Polytechnique Montreal

*Equal Contribution


## Citations

If you found this work useful, please cite:

```bibtex
@article{agarwal2024llms,
  title={LitLLMs, LLMs for Literature Review: Are we there yet?},
  author={Agarwal*, Shubham and Sahu*, Gaurav and Puri*, Abhay and Laradji, Issam H and Dvijotham, Krishnamurthy DJ and Stanley, Jason and Charlin, Laurent and Pal, Christopher},
  journal={arXiv preprint arXiv:2412.15249},
  year={2024}
}

@article{agarwal2024litllm,
  title={Litllm: A toolkit for scientific literature review},
  author={Agarwal*, Shubham and Sahu*, Gaurav and Puri*, Abhay and Laradji, Issam H and Dvijotham, Krishnamurthy DJ and Stanley, Jason and Charlin, Laurent and Pal, Christopher},
  journal={arXiv preprint arXiv:2402.01788},
  year={2024}
}
```
