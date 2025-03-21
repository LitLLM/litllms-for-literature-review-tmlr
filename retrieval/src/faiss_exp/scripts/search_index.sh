#!/bin/bash
# This script submits FAISS search jobs in batches.
# 
# Parameters:
# - num_jobs: Total number of jobs.
# - total_queries: Total queries to process.
# - queries_per_job: Number of queries per job (calculated as total_queries / num_jobs).
# - index_dir: Directory containing FAISS index files.
# - query_file: File containing the queries to be processed.
# - intermediate_dir: Directory to save intermediate results.
# - final_dir: Directory to save final results.
# - dimension: Dimensionality of the embeddings.
# - top_k: Number of top results to retrieve for each query.
# 
# For each job, the script calculates the range of queries to process, 
# and submits a job that runs a Python script to perform FAISS search on the given batch of queries.

num_jobs=50
total_queries=500
queries_per_job=$((total_queries / num_jobs))
index_dir="/path/to/your/index/dir"
query_file="/path/to/query/json"
intermediate_dir="/path/to/dir"
final_dir="/path/to/final/dir"
dimension=768
top_k=100

for ((i=0; i<num_jobs; i++)); do
    batch_start=$((i * queries_per_job))
    
    if [ $i -eq $((num_jobs - 1)) ]; then
        batch_end=$total_queries
    else
        batch_end=$((batch_start + queries_per_job))
    fi

    echo "Submitting job for queries from $batch_start to $batch_end"

    python /mnt/home/lit-rank/data/faiss_gpu/main.py \
        --action search \
        --index_dir "$index_dir" \
        --query_file "$query_file" \
        --intermediate_dir "$intermediate_dir" \
        --final_dir "$final_dir" \
        --dimension $dimension \
        --top_k $top_k \
        --batch_start $batch_start \
        --batch_end $batch_end \
done
