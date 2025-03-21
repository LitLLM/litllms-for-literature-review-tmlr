# run this script to index embeddings and search for similar embeddings using FAISS
# python embedding_indexer.py --action search --index_path /path/to/index/index.faiss --dimension 768 --query_file /path/to/query.jsonl --top_k 100
# python embedding_indexer.py --action index --data_dir /path/to/embeddings --index_path /path/to/index/index.faiss --dimension 768 --use_gpu
import os

os.environ["KMP_AFFINITY"] = "disabled"
import argparse
import json
import numpy as np
import faiss
import os
import gzip
import ast
from tqdm import tqdm


class EmbeddingIndexer:
    def __init__(self, index_path, dimension, use_gpu=False):
        self.index_path = index_path
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.corpus_id = []
        self.index = self.init_index()

    def init_index(self):
        index = None
        try:
            if os.path.exists(self.index_path):
                index = faiss.read_index(self.index_path)
        except RuntimeError as e:
            print(
                f"Error reading index from {self.index_path}. The file might not exist, be corrupted, or there could be a permissions issue."
            )
            raise e  # re-raise the exception to stop execution
        if index is None:
            print("Creating new index...")
            index = faiss.IndexFlatL2(self.dimension)
            # normalize the vectors to unit length before adding to the index
        if self.use_gpu:
            print("Moving index to GPU...")
            # Automatically use all available GPUs
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Distribute the index across all GPUs
            index = faiss.index_cpu_to_all_gpus(index, co=co)  # Simpler call
        return index

    def add_embeddings(self, embeddings):
        print("Normalizing embeddings")
        faiss.normalize_L2(embeddings)
        print("Adding embeddings to index")
        self.index.add(embeddings)
        print(f"Total embeddings in index: {self.index.ntotal}")
        print("Saving index to disk")
        # If the index is a GPU index, convert it back to a CPU index before saving
        try:
            self.index = faiss.index_gpu_to_cpu(self.index)  # Convert to CPU index
        except RuntimeError as e:
            print(f"Error converting index to CPU. The index might not be on GPU.")
        faiss.write_index(self.index, self.index_path)
        print(f"Index saved to {self.index_path}")

    # def load_and_index(self, file_path):
    #     print(f"Loading embeddings from {file_path}")
    #     embeddings = []
    #     open_func = gzip.open if file_path.endswith('.gz') else open
    #     try:
    #         with open_func(file_path, 'rt', encoding='utf-8') as f:
    #             for line in f:
    #                 data = json.loads(line)
    #                 vector = json.loads(data['vector'])
    #                 embeddings.append(vector)
    #     except EOFError:
    #         print(f"Error reading file {file_path}. The file may be corrupted or incomplete.")
    #     embeddings = np.array(embeddings, dtype='float32')
    #     self.add_embeddings(embeddings)

    def load_and_index(self, file_path):
        print(f"Loading embeddings from {file_path}")
        embeddings = []
        ids = []
        open_func = gzip.open if file_path.endswith(".gz") else open
        try:
            with open_func(file_path, "rt", encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading embeddings"):
                    data = json.loads(line)
                    # can we calculate the entire size of the data?
                    # print(len(data))
                    # print data for one and type of data
                    # one for data in data
                    # print(data)  # print the entire dictionary
                    # print(type(data.get('vector')))  # print the type of the value associated with 'vector'
                    # print(data.keys())
                    # vector = np.array(data['vector'], dtype='float32')
                    vector_string = data["vector"]
                    vector_list = ast.literal_eval(vector_string)
                    vector = np.array(vector_list, dtype="float32")
                    embeddings.append(vector)
                    ids.append(
                        data["corpusid"]
                    )  # Assuming 'corpus_id' is a field in each line
        except EOFError:
            print(
                f"Error reading file {file_path}. The file may be corrupted or incomplete."
            )
        embeddings = np.array(embeddings, dtype="float32")
        self.corpus_id.extend(ids)  # Append new ids to the corpus_id list
        self.add_embeddings(embeddings)

    # def search_index(self, query_embeddings, k):
    #     print("Searching for the most similar embeddings")
    #     D, I = self.index.search(query_embeddings, k)
    #     return D, I

    def search_index(self, query_embeddings, k):
        print("Searching for the most similar embeddings")
        D, I = self.index.search(query_embeddings, k)  # D: distances, I: indices
        # Map indices to corpus_id
        print("Mapping indices to corpus_id")
        # self.corus_ids
        print(self.corpus_id)
        print(len(self.corpus_id))
        corpus_ids = [
            [self.corpus_id[idx] for idx in query_result] for query_result in I
        ]
        return D, I, corpus_ids


def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Embedding Indexer and Searcher")
    parser.add_argument(
        "--action",
        type=str,
        choices=["index", "search"],
        required=True,
        help="Action to perform: 'index' or 'search'",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing JSONL embedding files for indexing",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Path to save/load the FAISS index",
    )
    parser.add_argument(
        "--dimension", type=int, required=True, help="Dimension of the embeddings"
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for indexing")
    parser.add_argument(
        "--query_file", type=str, help="File containing query embeddings for searching"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top similar results to retrieve for a search query",
    )
    parser.add_argument(
        "--use_ann", action="store_true", help="Use Approximate Nearest Neighbor search"
    )
    return parser.parse_args()


def load_query_embeddings(file_path):
    print(f"Loading query embeddings from {file_path}")
    embeddings = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data["embedding"])
    return np.array(embeddings, dtype="float32")


def main():
    args = parse_arguments()
    indexer = EmbeddingIndexer(
        index_path=args.index_path, dimension=args.dimension, use_gpu=args.use_gpu
    )
    if args.action == "index":
        if not args.data_dir:
            raise ValueError("Data directory must be specified for indexing action")
        for file_name in os.listdir(args.data_dir):
            if file_name.endswith(".jsonl") or file_name.endswith(
                ".jsonl.gz"
            ):  # Check for both .jsonl and .jsonl.gz files
                file_path = os.path.join(args.data_dir, file_name)
                indexer.load_and_index(file_path)
        print("Indexing complete.")
    elif args.action == "search":
        if not args.query_file:
            raise ValueError("Query file must be specified for search action")
        query_embeddings = load_query_embeddings(args.query_file)
        D, I = indexer.search_index(query_embeddings, args.top_k)
        print(f"Distances: {D}\nIndices: {I}")


if __name__ == "__main__":
    main()
