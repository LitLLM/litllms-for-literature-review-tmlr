import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# from models.faiss_indexer import EmbeddingIndexer
import matplotlib.pyplot as plt


class Reranker:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="specter2", set_active=True
        )
        print(sum(p.numel() for p in self.model.parameters()))
        print(self.model.num_parameters())

        self.batch_size = 16
        # self.faiss_indexer = EmbeddingIndexer(
        #     dimension=768, index_path=self.config.faiss_index_path
        # )

    def search_with_faiss(self, query_abstract, top_k=100):
        # Convert the single abstract to embeddings
        query_embeddings = (
            self.get_embeddings([{"abstract": query_abstract}]).detach().numpy()
        )
        # Use FAISS to find the top K similar items
        distances, indices, corpus_id = self.faiss_indexer.search_index(
            query_embeddings, top_k
        )
        # Create a list of dictionaries with the same structure as in rerank_papers
        scores = []
        for i in range(len(indices[0])):
            score = {}
            score["index"] = indices[0][i]
            score["distance"] = distances[0][i]
            score["corpus_id"] = corpus_id[0][i]
            # Optionally, add the original paper data if available
            # score['paper'] = self.papers[score['index']]
            scores.append(score)
        # Sort scores by distance (ascending, because lower distance means higher similarity)
        sorted_scores = sorted(scores, key=lambda x: x["distance"])
        # Convert to DataFrame and add 'original_order' and 'new_order' columns
        df_scores = pd.DataFrame(sorted_scores)
        df_scores["original_order"] = df_scores.index + 1
        df_scores["new_order"] = range(1, len(df_scores) + 1)
        # Convert DataFrame back to list of dictionaries
        dict_scores = df_scores.to_dict("records")
        return dict_scores

    def get_embeddings(self, papers_dict, title=False):
        # Split papers_dict into batches
        batches = [
            papers_dict[i : i + self.batch_size]
            for i in range(0, len(papers_dict), self.batch_size)
        ]
        print(batches)
        all_embeddings = []
        for batch in batches:
            if title:
                # both abstract and title
                text_batch = [
                    d["title"] + self.tokenizer.sep_token + (d.get("abstract") or "")
                    for d in batch
                ]
            else:
                # only abstract
                text_batch = [d["abstract"] for d in batch]
            inputs = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            output = self.model(**inputs)
            embeddings = output.last_hidden_state[:, 0, :]
            # print(embeddings.shape)
            all_embeddings.append(embeddings)
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def plot_results(self, results, filename):
        # Extract distances from results
        distances = [result["distance"] for result in results]
        # Create a list of indices for the x-axis
        indices = range(1, len(distances) + 1)
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(indices, distances, marker="o")
        plt.title("Distances of Top K Results")
        plt.xlabel("Rank")
        plt.ylabel("Distance")
        plt.savefig(filename)
        plt.show()

    def get_cosine_similarity(self, query, cand_papers):
        """
        https://github.com/allenai/scirepeval/blob/main/reviewer_matching.py#L63
        """
        # scores = {cid: cosine_similarity(cand_papers[cid], query).flatten() for cid in cand_papers}
        cand_papers = cand_papers.detach().numpy()
        query = query.detach().numpy()
        scores = cosine_similarity(cand_papers, query).flatten()
        # print(scores)
        return scores

    def rerank_papers(self, query, cand_papers):
        """
        query : Abstract Papers
        cand_papers : List of cited papers
        """
        if query == "" or pd.isna(query) or len(cand_papers) == 0:
            print("No query or papers found")
            return []
        q_emb = self.get_embeddings([{"abstract": query}])
        print(q_emb.shape)
        # print(cand_papers[0]['abstract'])
        scores = []
        paper_batches = [
            cand_papers[i : i + self.batch_size]
            for i in range(0, len(cand_papers), self.batch_size)
        ]

        for batch in paper_batches:
            cand_papers_emb = self.get_embeddings(batch)
            for i, ref in enumerate(batch):
                ref_score = ref
                ref_score["score"] = self.get_cosine_similarity(
                    q_emb, cand_papers_emb[i].unsqueeze(0)
                )
                # append to scores
                scores.append(ref_score)

        # df_scores = pd.DataFrame(scores, sorted(scores, key=lambda x: x['score'], reverse=True))
        # df_scores = pd.DataFrame(sorted(scores, key=lambda x: x['score'], reverse=True))
        # print(df_scores)
        # return df_scores
        # Sort scores and create DataFrame
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        df_scores = pd.DataFrame(sorted_scores)
        # Add 'original_order' and 'new_order' columns
        df_scores["original_order"] = df_scores.index + 1
        df_scores["new_order"] = range(1, len(df_scores) + 1)

        print(df_scores)
        dict_scores = df_scores.to_dict("records")
        print(dict_scores)
        return dict_scores

    def main(self):
        df = pd.read_csv(self.config.data_path)
        # print(df)
        # df[‘old’_order] = df.index


def parse_args() -> argparse.Namespace:
    """
    Parse cli arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="/mnt/colab_public/datasets/abhay/litrank/dataset/arxiv_papers/output_lit_arxiv2k.csv",
        help="Data path",
    )
    # add faiss index path
    parser.add_argument(
        "-f",
        "--faiss_index_path",
        type=str,
        default="/mnt/colab_public/results/auto_review/dataset/embeddings_index/index_1.index",
        help="FAISS index path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    # Example usage
    reranker = Reranker(config)
    text = "The thermodynamic properties of AgB2 and AuB2 compounds in AlB2 and OsB2-type structures are investigated from first-principles calculations based on density functional theory (DFT) using projector augmented waves (PAW) potentials within the generalized gradient approximation (GGA) for modeling exchange-correlation effects, respectively. Specifically, using the quasi-harmonic Debye model, the effects of pressure and temperature, up to 100 GPa and 1400 K, on the bulk modulus, Debye temperature, thermal expansion, heat capacity and the Gruneisen parameter are calculated successfully and trends are discussed."
    embeddings = reranker.get_embeddings(text)
    print(embeddings)
    # result = reranker.search_with_faiss(text)
    # print(result)
    # reranker.plot_results(result, "results.png")
    reranker.main()
