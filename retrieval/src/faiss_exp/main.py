import argparse
import os
import logging
import json
import heapq
import pandas as pd
from faiss_handler import FAISSHandler, search_index_file
from utils.file_utils import setup_logging, ensure_directory_exists

def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Index Creation and Search")
    parser.add_argument("--action", choices=['index', 'search'], required=True, help="Action to perform: 'index' or 'search'")
    parser.add_argument("--input_dir", type=str, help="Directory containing input JSONL.GZ files for indexing")
    parser.add_argument("--output_dir", type=str, help="Directory to save the FAISS index files")
    parser.add_argument("--index_dir", type=str, help="Directory containing FAISS index files (for searching)")
    parser.add_argument("--query_file", type=str, help="Path to the JSON file containing queries (for searching)")
    parser.add_argument("--dimension", type=int, default=768, help="Dimension of the embeddings")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for indexing/searching")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top results to retrieve per query")
    parser.add_argument("--intermediate_dir", type=str, help="Directory to store intermediate results")
    parser.add_argument("--final_dir", type=str, help="Directory to store final aggregated results")
    parser.add_argument("--num_files", type=int, default=20, help="Number of files to process")
    parser.add_argument("--file_range", type=str, nargs='+', help="List of input files for this job")
    parser.add_argument("--batch_start", type=int, help="Start index of the query batch")
    parser.add_argument("--batch_end", type=int, help="End index of the query batch")
    
    args = parser.parse_args()

    if args.action == 'search' and (not args.intermediate_dir or not args.final_dir):
        parser.error('--intermediate_dir and --final_dir are required for search action')
    
    return args

def main():
    setup_logging()
    args = parse_arguments()

    handler = FAISSHandler(args.dimension, args.use_gpu)
    handler.load_model() 

    if args.action == 'index':
        logging.info("Starting FAISS index creation")
        ensure_directory_exists(args.output_dir)

        input_files = args.file_range

        if not input_files:
            logging.error("No files provided in --file_range. Exiting.")
            return

        for i, input_file in enumerate(input_files, start=1):
            input_path = input_file  # Use file passed in --file_range
            base_name = os.path.basename(input_file).replace('.jsonl.gz', '')  # Get the base name of the input file
            output_path = os.path.join(args.output_dir, f"{base_name}_index.faiss")  # Use base name in output file
            logging.info(f"Creating index for {input_file} -> {output_path}")
            handler.create_index(input_path, output_path)
            logging.info("FAISS index creation completed.")

    elif args.action == 'search':
        ensure_directory_exists(args.intermediate_dir)
        ensure_directory_exists(args.final_dir)

        logging.info(f"Loading queries from {args.query_file}")
        with open(args.query_file, 'r') as f:
            queries = json.load(f)

        # Process only the specified batch of queries
        batch_queries = queries[args.batch_start:args.batch_end]

        index_files = []
        for root, dirs, files in os.walk(args.index_dir):
            for file in files:
                if file.endswith('.faiss'):
                    index_files.append(os.path.join(root, file))

        index_files = sorted(index_files)
        
        for query_id, abstract in enumerate(batch_queries, start=args.batch_start):
            if not abstract or pd.isna(abstract):
                logging.warning(f"Query {query_id} has a NaN or empty abstract. Skipping.")
                continue

            logging.info(f"Processing query {query_id}")
            intermediate_query_dir = os.path.join(args.intermediate_dir, f"query_{query_id}")
            ensure_directory_exists(intermediate_query_dir)

            query_embedding = handler.generate_embedding(abstract)

            all_results = []
            for index_file in index_files:
                logging.info(f"Searching index file: {index_file}")
                try:
                    results, search_time = search_index_file((index_file, query_embedding, args.top_k, handler.gpu_available))
                    logging.info(f"Search time for {index_file}: {search_time:.4f} seconds")
                    all_results.extend(results)

                    # Save intermediate results
                    intermediate_file = os.path.join(intermediate_query_dir, f"{os.path.basename(index_file)}_results.json")
                    with open(intermediate_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    logging.info(f"Intermediate results saved to {intermediate_file}")
                except Exception as e:
                    logging.error(f"Error processing index file {index_file}: {e}")

            logging.info(f"Total results collected: {len(all_results)}")

            # Filter valid results
            valid_results = [result for result in all_results if isinstance(result, dict) and 'distance' in result]
            logging.info(f"Valid results after filtering: {len(valid_results)}")

            if not valid_results:
                logging.warning(f"No valid results found for query {query_id}")
                continue

            # Sort and keep the top K results
            top_k_results = heapq.nsmallest(args.top_k, valid_results, key=lambda x: x['distance'])

            # Save the final aggregated results
            final_output_file = os.path.join(args.final_dir, f"query_{query_id}_results.json")
            with open(final_output_file, 'w') as f:
                json.dump(top_k_results, f, indent=2)

            logging.info(f"Final aggregated results saved for query {query_id} to {final_output_file}")

if __name__ == "__main__":
    main()
