import os
import json
import numpy as np
import faiss
import gzip
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import ast
import logging
import torch
import time


class FAISSHandler:
    def __init__(self, dimension, use_gpu=False):
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.model = None
        self.tokenizer = None
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self):
        if not self.use_gpu:
            return False
        try:
            return faiss.get_num_gpus() > 0 and torch.cuda.is_available()
        except Exception as e:
            logging.warning(
                f"Error checking GPU availability: {e}. Falling back to CPU."
            )
            return False

    def load_model(self):
        if self.model is None or self.tokenizer is None:
            logging.info(
                f"Loading SPECTER v2 model and tokenizer on {'GPU' if self.gpu_available else 'CPU'}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
            self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            self.model.load_adapter(
                "allenai/specter2", source="hf", load_as="specter2", set_active=True
            )
            if self.gpu_available:
                self.model = self.model.cuda()
            self.model.eval()
            logging.info("SPECTER v2 model loaded successfully")

    def create_index(self, input_file, output_file):
        logging.info(f"Creating FAISS index for {input_file}")
        index = faiss.IndexFlatL2(self.dimension)

        if self.gpu_available:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logging.info("Using GPU for indexing")
            except Exception as e:
                logging.warning(
                    f"Failed to use GPU for indexing: {e}. Falling back to CPU."
                )
                self.gpu_available = False

        index = faiss.IndexIDMap(index)

        embeddings = []
        corpus_ids = []
        try:
            with gzip.open(input_file, "rt", encoding="utf-8") as f:
                for line in tqdm(f, desc="Processing JSONL"):
                    data = json.loads(line)
                    vector = ast.literal_eval(data["vector"])
                    vector = np.array(vector, dtype="float32")
                    embeddings.append(vector)
                    corpus_ids.append(data["corpusid"])

        except EOFError:
            logging.error(
                f"Error: {input_file} is corrupted or incomplete. Skipping this file."
            )
            return

        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        corpus_id_array = np.array(corpus_ids).astype("int64")
        index.add_with_ids(embeddings, corpus_id_array)

        if self.gpu_available:
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, output_file)
        logging.info(f"Index saved to {output_file}")

    def generate_embedding(self, abstract):
        self.load_model()
        inputs = self.tokenizer(
            [abstract],
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        if self.gpu_available:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
        embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding


def search_index_file(args):
    index_file, query_embedding, k, use_gpu = args
    logging.info(f"Searching FAISS index {index_file}")

    try:
        start_time = time.time()

        index = faiss.read_index(index_file)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logging.info("Using GPU for search")
            except Exception as e:
                logging.warning(f"Failed to use GPU for search: {e}. Using CPU.")

        # Search the index and capture the results
        D, I = index.search(query_embedding, k)

        end_time = time.time()

        time_taken = end_time - start_time
        logging.info(f"FAISS search completed in {time_taken:.4f} seconds")

        # Return the results and time taken
        return [
            {"distance": float(d), "corpus_id": int(i), "index_file": index_file}
            for d, i in zip(D[0], I[0])
        ], time_taken

    except Exception as e:
        logging.error(f"Error searching index {index_file}: {e}")
        return [], 0
